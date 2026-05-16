from pathlib import Path
import sys

from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vad_platform.detector import ViolenceDetectionService


app = Flask(
    __name__,
    static_folder=str(ROOT / "web" / "static"),
    template_folder=str(ROOT / "web" / "templates"),
)
service = ViolenceDetectionService(project_root=ROOT)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/api/health")
def health():
    return jsonify(service.health())


@app.post("/api/reset")
def reset():
    service.reset()
    return jsonify({"ok": True})


@app.post("/api/live-frame")
def live_frame():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image field"}), 400
    return jsonify(
        service.process_live_frame(
            image_data,
            threshold=payload.get("threshold"),
            request_focus_screen=bool(payload.get("focusScreen")),
        )
    )


@app.post("/api/analyze-video")
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "Upload field must be named video"}), 400

    threshold = request.form.get("threshold", type=float)
    video_file = request.files["video"]
    result = service.analyze_uploaded_video(video_file, threshold=threshold)
    status = 200 if "error" not in result else 400
    return jsonify(result), status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
