<details>
<summary><b>Team Roles & Individual Contributions</b></summary>

### Arshia: Team Lead, Model Architecture & Evaluation

* Led the project: set the overall architecture direction (CNN + BiLSTM + Transformer), made the call to unify RAVDESS and CREMA-D into one shared six-class label set instead of training separate per-dataset models, and reviewed integration between the data pipeline and the model.
* Wrote the evaluation code (`evaluate_model`), including the per-dataset breakdown and the leave-one-dataset-out cross-dataset check, so the results showed more than one blended accuracy number.
* Built the plotting and reporting pipeline (`plot_training_curves`, `plot_confusion_matrix`, `plot_dataset_comparison`, `results_summary.json`) that the whole results section is based on.
* Wrote the final README and report covering why two datasets were combined and what the cross-dataset results actually mean.

### Anjan: Data Engineering & Preprocessing

* Pulled in RAVDESS and CREMA-D and handled the fact that they encode emotion labels completely differently in their filenames: wrote `parse_ravdess_filename` and `parse_cremad_filename`, and `build_combined_metadata` to merge both into one metadata table.
* Built the adaptive preprocessing step (`AdaptivePreprocessor`): per-clip SNR estimation, spectral subtraction only when a clip is actually noisy, RMS normalization, and silence trimming.
* Wrote the feature extraction and dataset loading (`FeatureExtractor`, `EmotionDataset`), including the log-mel + delta + delta-delta features and the training-only augmentation (noise, time stretch, pitch shift).

### Vasu: Model & Training

* Built the model (`CNNEncoder`, `EmotionRecognitionModel`): CNN front end for local spectral patterns, BiLSTM for how that evolves over time, Transformer encoder on top for longer-range context before the classification head.
* Wrote the `Trainer` class: AdamW optimizer, cosine annealing with warm restarts, gradient clipping, early stopping on validation accuracy, and best-checkpoint saving.
* Set the training configuration (batch size 16, learning rate 1e-4, up to 10 epochs, early stop patience 2) and ran the full training pipeline end to end.



</details>
