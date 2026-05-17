(function () {
  const app = window.avtApp;
  if (!app) return;

  const MODEL_ROOT = "/static/models/browser/";
  const DEFAULT_FPS = 25;
  let manifest = null;
  let featureSession = null;
  let classifierSession = null;
  let activeProvider = null;

  function providers() {
    return navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  }

  function softmax2(a, b) {
    const max = Math.max(a, b);
    const ea = Math.exp(a - max);
    const eb = Math.exp(b - max);
    const total = ea + eb;
    return [ea / total, eb / total];
  }

  async function loadManifest() {
    if (manifest) return manifest;
    const response = await fetch(`${MODEL_ROOT}manifest.json`);
    if (!response.ok) throw new Error("Could not load browser model manifest");
    manifest = await response.json();
    return manifest;
  }

  async function loadModels() {
    if (!window.ort) throw new Error("ONNX Runtime Web is not available");
    const config = await loadManifest();
    const selectedProviders = providers();
    activeProvider = selectedProviders[0];
    app.els.clientInferenceBtn.disabled = true;
    app.els.clientInferenceStatus.textContent = `Loading browser models with ${activeProvider}...`;
    app.appendTerminalLine(`[browser] loading ONNX models via ${selectedProviders.join(" -> ")}`);

    try {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
      featureSession = featureSession || (await ort.InferenceSession.create(`${MODEL_ROOT}${config.models.videomaeFeatureExtractor}`, {
        executionProviders: selectedProviders,
      }));
      app.appendTerminalLine("[browser] VideoMAE feature extractor ready");
      classifierSession = classifierSession || (await ort.InferenceSession.create(`${MODEL_ROOT}${config.models.anomalyClassifier}`, {
        executionProviders: selectedProviders,
      }));
      app.appendTerminalLine("[browser] anomaly classifier ready");

      const test = new Float32Array(1 * config.maxFrames * config.featureDim);
      const started = performance.now();
      await classifierSession.run({ features: new ort.Tensor("float32", test, [1, config.maxFrames, config.featureDim]) });
      const elapsed = performance.now() - started;
      app.els.clientInferenceStatus.textContent = `Browser models loaded on ${activeProvider}. Classifier test ${elapsed.toFixed(1)} ms.`;
    } finally {
      app.els.clientInferenceBtn.disabled = false;
    }
  }

  function videoReady(video) {
    return new Promise((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error("Could not load video metadata"));
    });
  }

  function seek(video, time) {
    return new Promise((resolve, reject) => {
      const done = () => {
        video.removeEventListener("seeked", done);
        resolve();
      };
      video.addEventListener("seeked", done, { once: true });
      video.onerror = () => reject(new Error("Could not seek uploaded video"));
      video.currentTime = Math.min(Math.max(time, 0), Math.max(video.duration - 0.02, 0));
    });
  }

  async function captureFrame(video, canvas, time, size) {
    await seek(video, time);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    canvas.width = size;
    canvas.height = size;
    ctx.drawImage(video, 0, 0, size, size);
    return ctx.getImageData(0, 0, size, size);
  }

  async function clipTensor(video, canvas, startTime, config) {
    const size = config.imageSize;
    const data = new Float32Array(1 * config.clipLen * 3 * size * size);
    const frameStep = config.frameSkip / DEFAULT_FPS;

    for (let frame = 0; frame < config.clipLen; frame += 1) {
      const image = await captureFrame(video, canvas, startTime + frame * frameStep, size);
      const pixels = image.data;
      const frameOffset = frame * 3 * size * size;
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const pixelIndex = (y * size + x) * 4;
          const outIndex = y * size + x;
          data[frameOffset + outIndex] = (pixels[pixelIndex] / 255 - config.imageMean[0]) / config.imageStd[0];
          data[frameOffset + size * size + outIndex] = (pixels[pixelIndex + 1] / 255 - config.imageMean[1]) / config.imageStd[1];
          data[frameOffset + 2 * size * size + outIndex] = (pixels[pixelIndex + 2] / 255 - config.imageMean[2]) / config.imageStd[2];
        }
      }
    }

    return new ort.Tensor("float32", data, [1, config.clipLen, 3, size, size]);
  }

  function buildClipStarts(duration, config) {
    const span = (config.clipLen * config.frameSkip) / DEFAULT_FPS;
    const stride = config.clipStride / DEFAULT_FPS;
    const starts = [];
    for (let start = 0; start + span <= duration; start += stride) {
      starts.push(start);
    }
    if (!starts.length) starts.push(0);
    return starts;
  }

  async function extractFeatures(video, canvas, config) {
    const starts = buildClipStarts(video.duration, config);
    const features = [];
    const started = performance.now();
    for (let i = 0; i < starts.length; i += 1) {
      app.setWorkflowStep("features", "running", `Browser feature ${i + 1}/${starts.length}`);
      app.appendTerminalLine(`[browser-features] extracting feature ${i + 1}/${starts.length} t=${starts[i].toFixed(2)}s`);
      const tensor = await clipTensor(video, canvas, starts[i], config);
      const outputs = await featureSession.run({ pixel_values: tensor });
      features.push(new Float32Array(outputs.features.data));
      app.appendTerminalLine(`[browser-features] feature ${i + 1} ready shape=(768,)`);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    app.appendTerminalLine(`[browser-features] completed ${features.length} features in ${((performance.now() - started) / 1000).toFixed(2)}s`);
    return { features, starts };
  }

  async function predictFeatureWindow(featureRows, threshold, config, label) {
    const input = new Float32Array(1 * config.maxFrames * config.featureDim);
    const rows = featureRows.length > config.maxFrames ? featureRows.slice(-config.maxFrames) : featureRows;
    for (let row = 0; row < rows.length; row += 1) {
      input.set(rows[row], row * config.featureDim);
    }
    const outputs = await classifierSession.run({
      features: new ort.Tensor("float32", input, [1, config.maxFrames, config.featureDim]),
    });
    const logits = outputs.logits.data;
    const [probNormal, probAnomaly] = softmax2(logits[0], logits[1]);
    const prediction = probAnomaly >= threshold ? "ANOMALY" : "NORMAL";
    app.appendTerminalLine(`[browser-calc] ${label}: normal=${probNormal.toFixed(4)} threat=${probAnomaly.toFixed(4)} => ${prediction}`);
    return {
      prob_normal: probNormal,
      prob_anomaly: probAnomaly,
      prediction,
      confidence: Math.max(probNormal, probAnomaly),
    };
  }

  async function scoreTimeline(features, starts, duration, threshold, config) {
    const timeline = [];
    const segmentClips = config.segmentClips;
    const step = Math.max(1, Math.floor(segmentClips / 2));
    for (let start = 0; start < Math.max(1, features.length - segmentClips + 1); start += step) {
      const end = Math.min(start + segmentClips, features.length);
      const result = await predictFeatureWindow(features.slice(start, end), threshold, config, `segment_${timeline.length + 1}`);
      timeline.push({
        start: Number(starts[start].toFixed(2)),
        end: Number(Math.min(starts[end - 1] + (config.clipLen * config.frameSkip) / DEFAULT_FPS, duration).toFixed(2)),
        prob_anomaly: result.prob_anomaly,
        prediction: result.prediction,
      });
    }
    return timeline;
  }

  function segmentUnionSeconds(segments) {
    const intervals = segments.map((seg) => [Number(seg.start), Number(seg.end)]).sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const [start, end] of intervals) {
      if (!merged.length || start > merged[merged.length - 1][1]) merged.push([start, end]);
      else merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], end);
    }
    return merged.reduce((sum, [start, end]) => sum + Math.max(0, end - start), 0);
  }

  function scoreAtTime(timeline, time) {
    return (
      timeline.find((segment) => Number(segment.start) <= time && time <= Number(segment.end)) ||
      timeline.reduce((best, segment) => {
        const current = Math.min(Math.abs(time - segment.start), Math.abs(time - segment.end));
        const previous = Math.min(Math.abs(time - best.start), Math.abs(time - best.end));
        return current < previous ? segment : best;
      }, timeline[0])
    );
  }

  async function frameSamples(video, timeline) {
    if (!timeline.length) return [];
    const canvas = document.createElement("canvas");
    const samples = [];
    const duration = video.duration;
    const times = [0, duration * 0.25, duration * 0.5, duration * 0.75];
    const peak = timeline.reduce((best, segment) => (segment.prob_anomaly > best.prob_anomaly ? segment : best), timeline[0]);
    times.unshift((peak.start + peak.end) / 2);
    const used = new Set();
    for (const time of times) {
      const key = time.toFixed(2);
      if (used.has(key)) continue;
      used.add(key);
      await captureFrame(video, canvas, time, 360);
      const score = scoreAtTime(timeline, time);
      samples.push({
        time: Number(time.toFixed(2)),
        score: score.prob_anomaly,
        prediction: score.prediction,
        image: canvas.toDataURL("image/jpeg", 0.74),
      });
      if (samples.length >= 6) break;
    }
    return samples.sort((a, b) => a.time - b.time);
  }

  async function analyzeInBrowser() {
    const file = app.els.videoFile.files[0];
    if (!file) return;
    app.startWorkflow(file);
    app.els.uploadResult.textContent = "Starting browser-side analysis...";
    app.els.timeline.innerHTML = "";
    app.els.visualSummary.innerHTML = "";
    app.els.analysisMetrics.innerHTML = "";
    app.els.frameGallery.innerHTML = "";

    const processingStarted = performance.now();
    let url = null;
    try {
      const config = await loadManifest();
      await loadModels();
      app.appendTerminalLine("[browser] decoding video locally");
      app.setWorkflowStep("frames", "running", "Reading browser video metadata");

      url = URL.createObjectURL(file);
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      video.src = url;
      await videoReady(video);
      app.appendTerminalLine(`[browser-video] duration=${video.duration.toFixed(2)}s assumed_fps=${DEFAULT_FPS}`);

      const canvas = document.createElement("canvas");
      const featureStarted = performance.now();
      const { features, starts } = await extractFeatures(video, canvas, config);
      const featureSeconds = (performance.now() - featureStarted) / 1000;

      app.setWorkflowStep("scoring", "running", "Scoring in browser");
      const threshold = Number(app.els.thresholdInput.value);
      const scoringStarted = performance.now();
      const overall = await predictFeatureWindow(features, threshold, config, "overall");
      const timeline = await scoreTimeline(features, starts, video.duration, threshold, config);
      const scoringSeconds = (performance.now() - scoringStarted) / 1000;
      const anomalySegments = timeline.filter((segment) => segment.prob_anomaly >= threshold);
      const peakScore = Math.max(...timeline.map((segment) => segment.prob_anomaly), overall.prob_anomaly);
      const peakSegment = timeline.reduce((best, segment) => (segment.prob_anomaly > best.prob_anomaly ? segment : best), timeline[0]);
      const averageScore = timeline.reduce((sum, segment) => sum + segment.prob_anomaly, 0) / Math.max(timeline.length, 1);
      const anomalySeconds = segmentUnionSeconds(anomalySegments);
      const samples = await frameSamples(video, timeline);
      const operationalScore = Math.max(overall.prob_anomaly, peakScore);
      const operational = {
        prob_anomaly: operationalScore,
        prob_normal: 1 - operationalScore,
        prediction: operationalScore >= threshold ? "ANOMALY" : "NORMAL",
        confidence: Math.max(operationalScore, 1 - operationalScore),
        basis: peakScore >= overall.prob_anomaly ? "peak_segment" : "whole_video",
      };
      const result = {
        filename: file.name,
        duration: video.duration,
        fps: DEFAULT_FPS,
        clips: features.length,
        threshold,
        overall,
        operational,
        timeline,
        anomaly_segments: anomalySegments,
        peak_score: peakScore,
        peak_segment: peakSegment,
        frame_samples: samples,
        metrics: {
          frames: Math.round(video.duration * DEFAULT_FPS),
          fps: DEFAULT_FPS,
          duration_seconds: Number(video.duration.toFixed(2)),
          clips: features.length,
          features: features.length,
          feature_dim: config.featureDim,
          timeline_segments: timeline.length,
          anomaly_segments: anomalySegments.length,
          anomaly_seconds: Number(anomalySeconds.toFixed(2)),
          anomaly_coverage: anomalySeconds / Math.max(video.duration, 0.001),
          average_score: averageScore,
          peak_score: peakScore,
          threshold,
          processing_seconds: Number(((performance.now() - processingStarted) / 1000).toFixed(3)),
          phase_times: {
            read_video_seconds: 0,
            feature_extraction_seconds: Number(featureSeconds.toFixed(3)),
            scoring_seconds: Number(scoringSeconds.toFixed(3)),
          },
        },
      };

      app.appendTerminalLine(`[browser-done] completed prediction=${operational.prediction} threat=${operationalScore.toFixed(4)}`, "complete");
      app.setWorkflowStep("completed", "complete", "Browser scoring completed");
      app.completeWorkflow(result);
      app.renderUpload(result);
    } catch (err) {
      app.appendTerminalLine(`[browser-error] ${err.message}`, "error");
      app.failWorkflow(err.message);
      app.els.uploadResult.textContent = err.message;
    } finally {
      if (url) URL.revokeObjectURL(url);
    }
  }

  const browserAnalyzeBtn = document.getElementById("browserAnalyzeBtn");
  if (browserAnalyzeBtn) {
    browserAnalyzeBtn.addEventListener("click", analyzeInBrowser);
  }

  window.avtBrowserInference = {
    analyzeInBrowser,
    loadModels,
  };
})();
