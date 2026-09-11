"""
Speech Emotion Recognition Pipeline
------------------------------------
Trains a CNN + BiLSTM + Transformer model to classify emotions from speech
audio, using RAVDESS and CREMA-D as source datasets. Includes adaptive
noise-aware preprocessing, log-mel feature extraction with deltas, and
both in-dataset and cross-dataset evaluation.
"""

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Fix RNG state across random/numpy/torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Config:
    """Central place for paths and hyperparameters used throughout the pipeline."""

    DATASETS = {
        'RAVDESS': 'data/RAVDESS/',
        'CREMA-D': 'data/CREMA-D/',
    }

    PREPROC_DIR = "preproc/"
    FEATURES_DIR = "features/"
    MODEL_DIR = "models/"
    EXPERIMENTS_DIR = "experiments/"
    RESULTS_DIR = "results/"

    # Audio / feature settings
    SAMPLE_RATE = 16000
    N_MELS = 80
    N_FFT = 2048
    HOP_LENGTH = 160     # ~10ms hop at 16kHz
    WIN_LENGTH = 400     # ~25ms window at 16kHz
    MAX_FRAMES = 300     # fixed sequence length after pad/truncate
    SNR_THRESHOLD_DB = 10
    USE_ADAPTIVE_PREPROC = True

    # Model architecture
    CNN_CHANNELS = [64, 128, 256]
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_HEADS = 8
    D_MODEL = 256

    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    EARLY_STOP_PATIENCE = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Labels
    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']
    NUM_EMOTIONS = len(EMOTIONS)

    # Maps dataset-specific codes onto the shared label set above.
    # RAVDESS uses numeric codes, CREMA-D uses 3-letter codes.
    EMOTION_LOOKUP = {
        '01': 'neutral', '02': 'neutral',   # RAVDESS "calm" folds into neutral
        '03': 'happy', '04': 'sad', '05': 'angry',
        '06': 'fear', '07': 'disgust', '08': None,  # surprise has no home in our label set
        'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad',
        'ANG': 'angry', 'FEA': 'fear', 'DIS': 'disgust',
    }


def parse_ravdess_filename(data_dir):
    """
    Walk a RAVDESS directory and pull emotion/speaker info out of filenames.
    Format: 03-01-06-01-02-01-12.wav (emotion is the 3rd field, actor id is the 7th).
    """
    rows = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith('.wav'):
                continue
            fields = fname.split('-')
            if len(fields) < 7:
                continue

            emotion_code = fields[2]
            actor_id = fields[6].split('.')[0]
            emotion = Config.EMOTION_LOOKUP.get(emotion_code)

            if emotion and emotion in Config.EMOTIONS:
                rows.append({
                    'file_path': os.path.join(root, fname),
                    'emotion': emotion,
                    'speaker_id': f"RAVDESS_{actor_id}",
                    'dataset': 'RAVDESS'
                })
    return pd.DataFrame(rows)


def parse_cremad_filename(data_dir):
    """
    Walk a CREMA-D directory and pull emotion/speaker info out of filenames.
    Format: 1001_DFA_ANG_XX.wav (speaker id first, emotion is the 3rd field).
    """
    rows = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith('.wav'):
                continue
            fields = fname.split('_')
            if len(fields) < 3:
                continue

            speaker_id, emotion_code = fields[0], fields[2]
            emotion = Config.EMOTION_LOOKUP.get(emotion_code)

            if emotion and emotion in Config.EMOTIONS:
                rows.append({
                    'file_path': os.path.join(root, fname),
                    'emotion': emotion,
                    'speaker_id': f"CREMAD_{speaker_id}",
                    'dataset': 'CREMA-D'
                })
    return pd.DataFrame(rows)


def build_combined_metadata():
    """Load every configured dataset, merge them, and write a single metadata CSV."""
    frames = []

    for name, path in Config.DATASETS.items():
        if not os.path.exists(path):
            print(f"  Skipping {name} - path not found")
            continue

        print(f"Loading {name}...")
        if name == 'RAVDESS':
            df = parse_ravdess_filename(path)
        elif name == 'CREMA-D':
            df = parse_cremad_filename(path)
        else:
            continue

        frames.append(df)
        print(f"  Loaded {len(df)} files")

    metadata = pd.concat(frames, ignore_index=True)

    label_to_id = {label: idx for idx, label in enumerate(Config.EMOTIONS)}
    metadata['emotion_code'] = metadata['emotion'].map(label_to_id)

    os.makedirs('data', exist_ok=True)
    metadata.to_csv('data/metadata.csv', index=False)

    print(f"\nTotal files: {len(metadata)}")
    print(f"Emotion distribution:\n{metadata['emotion'].value_counts()}")
    return metadata


class AdaptivePreprocessor:
    """Noise-aware audio cleanup: SNR estimation, spectral subtraction, VAD, gain control."""

    @staticmethod
    def estimate_snr(audio, sr):
        """Rough SNR estimate using the first 100ms of audio as a noise reference."""
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if len(trimmed) == 0:
            return 0

        signal_power = np.mean(trimmed ** 2)
        noise_power = np.mean(audio[:int(0.1 * sr)] ** 2)

        if noise_power == 0:
            return 100

        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def spectral_subtraction(audio, sr):
        """Subtract an estimated noise spectrum from the signal spectrum."""
        noise_clip = audio[:int(0.1 * sr)]

        spec = librosa.stft(audio)
        noise_spec = librosa.stft(noise_clip)
        noise_mag = np.abs(noise_spec).mean(axis=1, keepdims=True)

        magnitude = np.abs(spec)
        phase = np.angle(spec)

        # Floor at 10% of original magnitude so we don't over-subtract into silence.
        cleaned_magnitude = np.maximum(magnitude - 1.5 * noise_mag, 0.1 * magnitude)
        cleaned_spec = cleaned_magnitude * np.exp(1j * phase)

        return librosa.istft(cleaned_spec)

    @staticmethod
    def normalize_rms(audio, target_rms=0.1):
        """Scale audio to a target RMS level so loudness is consistent across clips."""
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        return audio

    @staticmethod
    def trim_silence(audio, sr):
        """Drop leading/trailing silence via a simple energy-based VAD."""
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=20,
            frame_length=2048,
            hop_length=512
        )
        return trimmed

    @staticmethod
    def process(file_path, save_path=None):
        """Run the full cleanup chain on a single audio file."""
        audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)

        if Config.USE_ADAPTIVE_PREPROC:
            snr = AdaptivePreprocessor.estimate_snr(audio, sr)
            if snr < Config.SNR_THRESHOLD_DB:
                audio = AdaptivePreprocessor.spectral_subtraction(audio, sr)

            audio = AdaptivePreprocessor.normalize_rms(audio)
            audio = AdaptivePreprocessor.trim_silence(audio, sr)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            import soundfile as sf
            sf.write(save_path, audio, sr)

        return audio, sr


class FeatureExtractor:
    """Turns raw audio into log-mel spectrograms (optionally with delta features)."""

    @staticmethod
    def log_mel_spectrogram(audio, sr):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            win_length=Config.WIN_LENGTH,
            n_mels=Config.N_MELS,
            fmin=0,
            fmax=sr // 2
        )
        return librosa.power_to_db(mel, ref=np.max)

    @staticmethod
    def with_deltas(mel_spec):
        """Stack the spectrogram with its first and second order derivatives."""
        delta = librosa.feature.delta(mel_spec)
        delta2 = librosa.feature.delta(mel_spec, order=2)
        return np.stack([mel_spec, delta, delta2], axis=0)

    @staticmethod
    def extract(audio, sr, use_deltas=True):
        mel_spec = FeatureExtractor.log_mel_spectrogram(audio, sr)

        if use_deltas:
            return FeatureExtractor.with_deltas(mel_spec)
        return mel_spec[np.newaxis, :]


class EmotionDataset(Dataset):
    """Loads audio on the fly, applies preprocessing/augmentation, and extracts features."""

    def __init__(self, dataframe, augment=False, use_deltas=True):
        self.df = dataframe.reset_index(drop=True)
        self.augment = augment
        self.use_deltas = use_deltas

    def __len__(self):
        return len(self.df)

    def _augment(self, audio):
        """Randomly apply noise injection, time stretch, and pitch shift."""
        if np.random.rand() > 0.5:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise

        if np.random.rand() > 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)

        if np.random.rand() > 0.5:
            steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=Config.SAMPLE_RATE, n_steps=steps)

        return audio

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio, sr = AdaptivePreprocessor.process(row['file_path'])

        if self.augment:
            audio = self._augment(audio)

        features = FeatureExtractor.extract(audio, sr, self.use_deltas)

        # Pad or truncate along the time axis so every sample has the same length.
        if features.shape[-1] > Config.MAX_FRAMES:
            features = features[..., :Config.MAX_FRAMES]
        elif features.shape[-1] < Config.MAX_FRAMES:
            pad_amount = Config.MAX_FRAMES - features.shape[-1]
            features = np.pad(features, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')

        return torch.FloatTensor(features), row['emotion_code']


class CNNEncoder(nn.Module):
    """Three-block convolutional front end that turns spectrograms into local features."""

    def __init__(self, in_channels=3):
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels if i == 0 else Config.CNN_CHANNELS[i - 1],
                    Config.CNN_CHANNELS[i],
                    kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(Config.CNN_CHANNELS[i]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            for i in range(len(Config.CNN_CHANNELS))
        ])

    def forward(self, x):
        # x: (batch, channels, freq, time)
        for block in self.blocks:
            x = block(x)
        return x


class EmotionRecognitionModel(nn.Module):
    """CNN encoder -> BiLSTM -> Transformer encoder -> classification head."""

    def __init__(self, num_emotions=6, in_channels=3):
        super().__init__()

        self.cnn = CNNEncoder(in_channels=in_channels)

        # After 3 maxpool(2) layers, freq/time dims shrink by a factor of 8 each.
        cnn_out_freq = Config.N_MELS // (2 ** len(Config.CNN_CHANNELS))
        self.cnn_out_dim = Config.CNN_CHANNELS[-1] * cnn_out_freq

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if Config.LSTM_NUM_LAYERS > 1 else 0
        )

        self.lstm_to_transformer = nn.Linear(Config.LSTM_HIDDEN_SIZE * 2, Config.D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.D_MODEL * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.TRANSFORMER_LAYERS)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(Config.D_MODEL, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_emotions)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.cnn(x)                          # (B, 256, F', T')
        x = x.permute(0, 3, 1, 2)                 # (B, T', 256, F')
        x = x.reshape(batch_size, x.size(1), -1)  # (B, T', 256*F')

        x, _ = self.lstm(x)                       # (B, T', 512)
        x = self.lstm_to_transformer(x)           # (B, T', D_MODEL)
        x = self.transformer(x)                   # (B, T', D_MODEL)

        x = x.permute(0, 2, 1)                    # (B, D_MODEL, T')
        x = self.global_pool(x).squeeze(-1)        # (B, D_MODEL)

        return self.classifier(x)


class Trainer:
    """Handles the training loop, validation, checkpointing, and early stopping."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.best_val_acc = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_accs = []
        self.val_f1s = []

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0

        progress = tqdm(self.train_loader, desc='Training')
        for features, labels in progress:
            features = features.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})

        return running_loss / len(self.train_loader)

    def _run_validation(self):
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.config.DEVICE)
                outputs = self.model(features)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()

                preds.extend(batch_preds)
                targets.extend(labels.numpy())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        return acc, f1, preds, targets

    def fit(self):
        print(f"\nTraining on {self.config.DEVICE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"{'='*60}")

            train_loss = self._train_one_epoch()
            val_acc, val_f1, _, _ = self._run_validation()

            self.train_losses.append(train_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)

            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val F1-Score: {val_f1:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_path = os.path.join(self.config.MODEL_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, checkpoint_path)
                print(f"  New best model saved (acc: {val_acc:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            self.scheduler.step()


def evaluate_model(model, data_loader, config, label="Test"):
    """Run inference over a loader and report accuracy, F1, and a full classification report."""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc=f'Evaluating {label}'):
            features = features.to(config.DEVICE)
            outputs = model(features)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()

            preds.extend(batch_preds)
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    cm = confusion_matrix(targets, preds)

    print(f"\n{'='*60}")
    print(f"{label} Results")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(targets, preds, target_names=config.EMOTIONS, digits=4))

    return acc, f1, cm, preds, targets


def plot_training_curves(trainer, config):
    """Save loss/accuracy/F1 curves from a completed training run."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(trainer.train_losses, label='Train Loss', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trainer.val_accs, label='Validation Accuracy', color='green', linewidth=2)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(trainer.val_f1s, label='Validation F1-Score', color='orange', linewidth=2)
    axes[2].set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/training_curves.png", dpi=300, bbox_inches='tight')
    print("Saved training curves")


def plot_confusion_matrix(cm, config, title="Confusion Matrix"):
    """Save a heatmap confusion matrix to the results directory."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=config.EMOTIONS,
        yticklabels=config.EMOTIONS,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    filename = title.lower().replace(' ', '_') + '.png'
    plt.savefig(f"{config.RESULTS_DIR}/{filename}", dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")


def plot_dataset_comparison(results_by_dataset, config):
    """Bar chart comparing accuracy/F1 across each source dataset."""
    datasets = list(results_by_dataset.keys())
    accuracies = [results_by_dataset[d]['accuracy'] for d in datasets]
    f1_scores = [results_by_dataset[d]['f1'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accuracies, width, label='Accuracy', color='skyblue')
    ax.bar(x + width / 2, f1_scores, width, label='F1-Score', color='coral')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_DIR}/dataset_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved dataset comparison")


def main():
    set_seed(42)
    config = Config()

    for path in [config.PREPROC_DIR, config.FEATURES_DIR, config.MODEL_DIR,
                 config.EXPERIMENTS_DIR, config.RESULTS_DIR]:
        os.makedirs(path, exist_ok=True)

    print("=" * 80)
    print("SPEECH EMOTION RECOGNITION - TRAINING PIPELINE")
    print("=" * 80)

    # Load and merge datasets, mapping each one onto a shared emotion label set.
    print("\nLoading datasets with standardized labels...")
    metadata = build_combined_metadata()

    # Split by speaker so the same voice never appears in both train and test.
    print("\nSplitting dataset by speaker...")
    speakers = metadata['speaker_id'].unique()
    train_speakers, holdout_speakers = train_test_split(speakers, test_size=0.3, random_state=42)
    val_speakers, test_speakers = train_test_split(holdout_speakers, test_size=0.5, random_state=42)

    train_df = metadata[metadata['speaker_id'].isin(train_speakers)]
    val_df = metadata[metadata['speaker_id'].isin(val_speakers)]
    test_df = metadata[metadata['speaker_id'].isin(test_speakers)]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Build datasets/loaders. Only the training set gets augmentation.
    print("\nBuilding datasets with adaptive preprocessing...")
    train_dataset = EmotionDataset(train_df, augment=True, use_deltas=True)
    val_dataset = EmotionDataset(val_df, augment=False, use_deltas=True)
    test_dataset = EmotionDataset(test_df, augment=False, use_deltas=True)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                              num_workers=4, pin_memory=True)

    print("\nBuilding CNN + BiLSTM + Transformer model...")
    model = EmotionRecognitionModel(num_emotions=config.NUM_EMOTIONS, in_channels=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining with augmentation and early stopping...")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()

    print("\nLoading best checkpoint for evaluation...")
    checkpoint = torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'), map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")

    print("\nRunning evaluation...")
    test_acc, test_f1, test_cm, test_preds, test_labels = evaluate_model(
        model, test_loader, config, "Overall Test Set"
    )

    # Break results down per source dataset.
    results_by_dataset = {}
    for dataset_name in metadata['dataset'].unique():
        subset = test_df[test_df['dataset'] == dataset_name]
        if len(subset) == 0:
            continue

        subset_loader = DataLoader(
            EmotionDataset(subset, augment=False, use_deltas=True),
            batch_size=config.BATCH_SIZE,
            num_workers=4
        )
        acc, f1, cm, _, _ = evaluate_model(model, subset_loader, config, dataset_name)
        results_by_dataset[dataset_name] = {'accuracy': acc, 'f1': f1, 'confusion_matrix': cm}

    # Leave-one-dataset-out cross-dataset generalization check.
    if len(metadata['dataset'].unique()) > 1:
        print("\nRunning cross-dataset evaluation...")
        for held_out in metadata['dataset'].unique():
            source_datasets = [d for d in metadata['dataset'].unique() if d != held_out]
            cross_train_df = metadata[metadata['dataset'].isin(source_datasets)]
            cross_test_df = metadata[metadata['dataset'] == held_out]

            if len(cross_test_df) <= 20:
                continue

            print(f"\nTrain on: {source_datasets} | Test on: {held_out}")

            cross_model = EmotionRecognitionModel(
                num_emotions=config.NUM_EMOTIONS, in_channels=3
            ).to(config.DEVICE)

            cross_train_loader = DataLoader(
                EmotionDataset(cross_train_df, augment=True, use_deltas=True),
                batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4
            )
            cross_val_loader = DataLoader(
                EmotionDataset(cross_test_df.sample(min(200, len(cross_test_df))),
                                augment=False, use_deltas=True),
                batch_size=config.BATCH_SIZE, num_workers=4
            )

            # Shorter run for this ablation - just enough epochs to compare generalization.
            ablation_config = Config()
            ablation_config.NUM_EPOCHS = 3
            cross_trainer = Trainer(cross_model, cross_train_loader, cross_val_loader, ablation_config)

            print("Training cross-dataset model (3 epochs)...")
            cross_trainer.fit()

            cross_test_loader = DataLoader(
                EmotionDataset(cross_test_df, augment=False, use_deltas=True),
                batch_size=config.BATCH_SIZE, num_workers=4
            )
            evaluate_model(cross_model, cross_test_loader, config, f"Cross-Dataset: {held_out}")

    print("\nGenerating plots...")
    plot_training_curves(trainer, config)
    plot_confusion_matrix(test_cm, config, "Overall Test Confusion Matrix")

    for dataset_name, results in results_by_dataset.items():
        plot_confusion_matrix(results['confusion_matrix'], config, f"{dataset_name} Confusion Matrix")

    if len(results_by_dataset) > 1:
        plot_dataset_comparison(results_by_dataset, config)

    summary = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'best_val_acc': float(checkpoint['val_acc']),
        'best_val_f1': float(checkpoint['val_f1']),
        'per_dataset': {
            name: {'accuracy': float(res['accuracy']), 'f1': float(res['f1'])}
            for name, res in results_by_dataset.items()
        },
        'config': {
            'model': 'CNN+BiLSTM+Transformer',
            'cnn_channels': config.CNN_CHANNELS,
            'lstm_hidden': config.LSTM_HIDDEN_SIZE,
            'lstm_layers': config.LSTM_NUM_LAYERS,
            'transformer_layers': config.TRANSFORMER_LAYERS,
            'transformer_heads': config.TRANSFORMER_HEADS,
            'epochs_trained': len(trainer.train_losses),
            'adaptive_preprocessing': config.USE_ADAPTIVE_PREPROC
        }
    }

    with open(f"{config.RESULTS_DIR}/results_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Model saved to: {config.MODEL_DIR}")
    print("\nGenerated files:")
    print("  - training_curves.png")
    print("  - confusion matrices (per dataset)")
    print("  - dataset_comparison.png")
    print("  - results_summary.json")
    print("  - best_model.pth")


if __name__ == "__main__":
    main()