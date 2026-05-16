"""
SRU++-based Video Anomaly Detection Training Module
Fixed & optimized for RTX 4060 GPU
Key fix: srupp_training.py was missing `parser = argparse.ArgumentParser(...)` in main()
"""

import logging
import os
import argparse
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import gc

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
try:
    from sru import SRUpp
    SRUPP_IMPORT_ERROR = None
except Exception as exc:
    SRUpp = None
    SRUPP_IMPORT_ERROR = exc
from torch.nn import CrossEntropyLoss, Module, Dropout, Linear
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class SRUppModel(Module):

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super(SRUppModel, self).__init__()
        if SRUpp is None:
            raise RuntimeError(
                "SRU++ is unavailable. Install optional training dependencies with "
                "`pip install -r requirements-training.txt` and, on Windows, install "
                "MSVC Build Tools plus CUDA toolkit if you need SRU CUDA kernels. "
                f"Original import error: {SRUPP_IMPORT_ERROR}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = kwargs.get('num_layers', 2)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_classes = kwargs.get('num_classes', 2)
        self.proj_size = kwargs.get('proj_size', 784)

        self.srupp_layers = SRUpp(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            proj_size=self.proj_size,
            dropout=kwargs.get('dropout_prob', 0.0),
            bidirectional=self.bidirectional,
            layer_norm=kwargs.get('layer_norm', False),
            highway_bias=kwargs.get('highway_bias', 0.0),
            rescale=kwargs.get('rescale', True),
            nn_rnn_compatible_return=kwargs.get('nn_rnn_compatible_return', False),
            proj_input_to_hidden_first=kwargs.get('proj_input_to_hidden_first', False),
            normalize_after=kwargs.get('normalize_after', False),
        )

        self.dropout = Dropout(kwargs.get('dropout_layer_prob', 0.2))
        output_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.linear = Linear(in_features=output_size, out_features=self.num_classes)
        self.l2_reg_lambda = kwargs.get('l2_reg_lambda', 1e-5)

    def forward(self, x):
        # SRU++ returns (output_states, hidden_states, cell_states)
        output_states, _, _ = self.srupp_layers(x)
        output = self.linear(self.dropout(output_states[-1]))
        return output

    def l2_regularization(self):
        l2_reg = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.l2_reg_lambda * l2_reg

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'SRU++',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'proj_size': self.proj_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'bidirectional': self.bidirectional,
            'l2_reg_lambda': self.l2_reg_lambda,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class VADTrainerPlusPlus:

    def __init__(self,
                 model: SRUppModel,
                 device: torch.device,
                 save_dir: str = 'artifacts/models',
                 log_dir: str = 'artifacts/logs'):

        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'epoch_times': []
        }

        log_filename = os.path.join(log_dir, f'srupp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def prepare_data(self,
                     embeddings_path: str,
                     labels_path: str,
                     test_size: float = 0.2,
                     batch_size: int = 32,
                     random_state: int = 42) -> Tuple[DataLoader, DataLoader]:

        print("Loading data for SRU++ training...")
        file_embeddings = np.load(embeddings_path)
        file_labels = np.load(labels_path)

        if len(file_embeddings) != len(file_labels):
            raise ValueError("Embeddings and labels length mismatch.")

        # Handle 2D embeddings (num_videos, features) -> expand to 3D
        if len(file_embeddings.shape) == 2:
            print(f"[INFO] 2D embeddings {file_embeddings.shape} -> expanding to 3D...")
            file_embeddings = np.expand_dims(file_embeddings, axis=1)

        if len(file_embeddings.shape) != 3:
            raise ValueError(f"Expected 3D [videos, frames, features], got {file_embeddings.shape}")

        print(f"Embeddings shape: {file_embeddings.shape}")
        print(f"Labels shape: {file_labels.shape}")
        print(f"Unique labels: {np.unique(file_labels)}")

        file_labels = file_labels.astype(np.int64)
        file_embeddings = file_embeddings.astype(np.float32)

        x_train, x_test, y_train, y_test = train_test_split(
            file_embeddings, file_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=file_labels
        )

        train_embeddings = torch.from_numpy(x_train)
        train_labels = torch.from_numpy(y_train)
        test_embeddings = torch.from_numpy(x_test)
        test_labels = torch.from_numpy(y_test)

        print(f'Train: {train_embeddings.shape} | Test: {test_embeddings.shape}')

        train_data = TensorDataset(train_embeddings, train_labels)
        test_data = TensorDataset(test_embeddings, test_labels)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,
                                  pin_memory=True, num_workers=2)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size,
                                 pin_memory=True, num_workers=2)

        del file_embeddings, file_labels, x_train, x_test, y_train, y_test
        gc.collect()

        return train_loader, test_loader

    def train(self,
              train_loader: DataLoader,
              epochs: int = 100,
              learning_rate: float = 0.001,
              optimizer_type: str = 'adam',
              save_best_model: bool = True) -> Dict[str, List[float]]:

        criterion = CrossEntropyLoss()

        if optimizer_type.lower() == 'adam':
            optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        print(f"\nStarting SRU++ training for {epochs} epochs on {self.device}...")
        best_accuracy = 0.0

        for epoch in range(epochs):
            self.model.train()
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            epoch_start = datetime.now()

            progress_bar = tqdm(enumerate(train_loader),
                                desc=f"Epoch {epoch+1}/{epochs}",
                                total=len(train_loader))

            for i, (videos, labels) in progress_bar:
                videos = videos.permute(1, 0, 2).to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(videos)
                loss = criterion(outputs, labels)
                loss += self.model.l2_regularization()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()

                progress_bar.set_postfix({
                    'loss': f'{total_loss/(i+1):.4f}',
                    'acc': f'{100*total_correct/total_samples:.2f}%'
                })

            scheduler.step()
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * total_correct / total_samples

            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_accuracy'].append(accuracy)
            self.training_history['epoch_times'].append(epoch_time)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.1f}s")
            self.logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")

            if save_best_model and accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model('best_srupp_model.pth', epoch + 1, accuracy)
                print(f"  --> Best model saved (acc: {accuracy:.2f}%)")

        return self.training_history

    def evaluate(self, test_loader: DataLoader, show_plots: bool = False) -> Dict[str, Any]:
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Evaluating"):
                videos = videos.permute(1, 0, 2).to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(videos)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        accuracy = 100 * np.mean(all_labels == all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)

        num_classes = all_probabilities.shape[1]
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(all_labels, all_probabilities[:, 1])
            avg_precision = average_precision_score(all_labels, all_probabilities[:, 1])
        else:
            fpr, tpr, roc_auc = [], [], 0.0
            precision, recall, avg_precision = [], [], 0.0

        results = {
            'test_accuracy': accuracy,
            'confusion_matrix': cm,
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
            'precision': precision, 'recall': recall,
            'average_precision': avg_precision
        }

        print(f"\n=== SRU++ Evaluation ===")
        print(f"Test Accuracy: {accuracy:.2f}%")
        if num_classes == 2:
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        if show_plots:
            self.plot_evaluation_results(results)

        return results

    def plot_evaluation_results(self, results: Dict[str, Any]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("SRU++ Confusion Matrix")

        if len(results['fpr']) > 0:
            axes[1].plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
                         label=f'AUC = {results["roc_auc"]:.2f}')
            axes[1].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
            axes[1].set_title('ROC Curve')
            axes[1].legend()

            axes[2].step(results['recall'], results['precision'], color='b', alpha=0.2, where='post')
            axes[2].fill_between(results['recall'], results['precision'], step='post', alpha=0.2, color='b')
            axes[2].set_title(f'PR Curve (AP={results["average_precision"]:.2f})')

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'srupp_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plots saved: {plot_path}")
        plt.show()

    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        axes[0].plot(epochs, self.training_history['train_loss'], 'b-')
        axes[0].set_title('SRU++ Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].grid(True)

        axes[1].plot(epochs, self.training_history['train_accuracy'], 'r-')
        axes[1].set_title('SRU++ Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'srupp_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Training history saved: {plot_path}")
        plt.show()

    def save_model(self, filename: str, epoch: int, accuracy: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'model_info': self.model.get_model_info(),
            'training_history': self.training_history
        }
        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Model saved: {save_path}")

    def load_model(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        print(f"Loaded from epoch {checkpoint['epoch']} | acc: {checkpoint['accuracy']:.2f}%")
        return checkpoint


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


def main():
    # BUG FIX: original srupp_training.py was missing this line
    parser = argparse.ArgumentParser(description="SRU++ Video Anomaly Detection Training")

    # Data
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (32 recommended for RTX 4060)')

    # Model
    parser.add_argument('--input_size', type=int, default=2048,
                        help='Feature dim of your .npy embeddings')
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--proj_size', type=int, default=784)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2,
                        help='2 for binary, 12 for multiclass UCF-Crime')
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--dropout_layer_prob', type=float, default=0.2)
    parser.add_argument('--l2_reg_lambda', type=float, default=1e-5)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])

    # Output
    parser.add_argument('--save_dir', type=str, default='artifacts/models')
    parser.add_argument('--log_dir', type=str, default='artifacts/logs')
    parser.add_argument('--show_plots', action='store_true')

    args = parser.parse_args()
    device = setup_device()

    model = SRUppModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        proj_size=args.proj_size,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        bidirectional=args.bidirectional,
        dropout_prob=args.dropout_prob,
        dropout_layer_prob=args.dropout_layer_prob,
        l2_reg_lambda=args.l2_reg_lambda
    )

    print("\nModel Info:")
    print(json.dumps(model.get_model_info(), indent=2))

    trainer = VADTrainerPlusPlus(model, device, args.save_dir, args.log_dir)

    train_loader, test_loader = trainer.prepare_data(
        args.embeddings_path, args.labels_path,
        args.test_size, args.batch_size
    )

    trainer.train(train_loader, args.epochs, args.learning_rate, args.optimizer)
    trainer.plot_training_history()
    results = trainer.evaluate(test_loader, args.show_plots)

    trainer.save_model(
        f'final_srupp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
        args.epochs, results['test_accuracy']
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
