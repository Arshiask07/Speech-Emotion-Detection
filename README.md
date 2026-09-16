# Speech Emotion Recognition

Classifies a speech clip into one of six emotions (neutral, happy, sad, angry, fear, disgust) using a CNN + BiLSTM + Transformer stack trained from scratch on RAVDESS and CREMA-D, combined and split by speaker.

## How it works

```
Raw audio (16kHz)
  -> adaptive preprocessing (SNR check, optional spectral subtraction, RMS normalize, trim silence)
  -> log-mel spectrogram + delta + delta-delta (80 mel bins, 300 frames)
  -> CNN encoder (3 blocks: 64 -> 128 -> 256 channels, each Conv2d(k=3) + BatchNorm + ReLU + MaxPool2)
  -> reshape, feed into BiLSTM (2 layers, 256 hidden units, dropout 0.3)
  -> Linear projection to d_model=256
  -> Transformer encoder (4 layers, 8 heads, feedforward 1024, dropout 0.1)
  -> global average pool over time
  -> classifier (256 -> 256 -> 128 -> 6, dropout 0.3 before each of the first two)
```

Preprocessing only reaches for spectral subtraction when a clip's estimated SNR (first 100ms treated as noise reference) drops below 10dB; clean clips skip it. Subtraction floors at 10% of original magnitude so it doesn't over-subtract into silence. After that: RMS-normalize to 0.1, trim silence at top_db=20.

Training-set clips get augmented, each of the three independently at 50% chance: Gaussian noise (std 0.005), time stretch (rate 0.9-1.1), pitch shift (-2 to +2 semitones). Validation and test never touch this.

Features are padded/truncated to a fixed 300 frames along the time axis so batches stack cleanly.

## Data

RAVDESS and CREMA-D aren't included in the repo. Download them yourself and place at `data/RAVDESS/` and `data/CREMA-D/`:

| Dataset | Source |
|---|---|
| RAVDESS | zenodo.org/record/1188976 |
| CREMA-D | github.com/CheyneyComputerScience/CREMA-D |

The parser expects each dataset's native filename format (RAVDESS: `03-01-06-01-02-01-12.wav`, emotion is field 3; CREMA-D: `1001_DFA_ANG_XX.wav`, emotion is field 3 split on `_`). Renamed files get silently skipped, not errored on.

RAVDESS's "calm" code folds into neutral, "surprised" is dropped entirely since CREMA-D has no equivalent class. Splits are done by `speaker_id`, not by clip: `train_test_split` on unique speakers, 70/15/15 train/val/test, so the same voice never crosses splits.

## Training

Batch size 16, up to 10 epochs, AdamW at lr 1e-4 (weight decay 1e-5), gradient clipped to norm 1.0, CosineAnnealingWarmRestarts (T_0=10, T_mult=2). Early stopping on validation accuracy, patience 2. Best checkpoint (by val accuracy) is saved to `models/best_model.pth` and reloaded for final evaluation.

After the main model, the script also trains a separate model per leave-one-dataset-out split (train on one dataset, test on the other) for 3 epochs each, to check cross-dataset generalization.

## Known issues

- Cross-dataset (leave-one-dataset-out) results are computed and printed to console via `evaluate_model()`, but never written to `results_summary.json` or plotted, unlike every other evaluation in the script. If you want those numbers after the run finishes, they're only in whatever captured the stdout.
- `preproc/`, `features/`, and `experiments/` get created by `main.py` at startup but nothing is ever written into them. Preprocessing and feature extraction both happen on the fly in `EmotionDataset.__getitem__`, not cached to disk.
- `requirements.txt` doesn't list `soundfile`, which `AdaptivePreprocessor.process()` imports, but only inside the `if save_path:` branch, which nothing in `main()` actually triggers. Doesn't break the default run.
- Best validation accuracy (0.442) came in lower than the final test accuracy (0.570) on the run in `results/`. Val and test are different speaker subsets by construction, so this isn't necessarily a bug.

## Results

From `results/results_summary.json`, one full run, 10 epochs, no early stop triggered (val accuracy was still rising at epoch 10, see `training_curves.png`):

| Split | Accuracy | F1 (weighted) |
|---|---|---|
| Overall test | 57.0% | 0.561 |
| RAVDESS only | 52.9% | 0.510 |
| CREMA-D only | 57.8% | 0.564 |
| Best validation | 44.2% | 0.434 |

Random guessing across 6 balanced classes is ~17%, so this is a solid multiple of chance for a model with no pretrained audio backbone.

Per-class recall on the overall test set, computed directly from `overall_test_confusion_matrix.png`: angry is the clear best (197/228, 86.4%), sad is solid (152/228, 66.7%), neutral middling (129/216, 59.7%), disgust and happy weaker (47.4% and 42.1%), fear is worst (91/228, 39.9%), mostly lost to sad (73 of fear's clips predicted as sad).

That fear/sad confusion is dataset-dependent: on RAVDESS alone, fear recall is actually the strongest class at 78.1% (25/32); on CREMA-D alone it drops to 33.7% (66/196), with 71 fear clips predicted as sad. Happy shows the same pattern in reverse: weak on RAVDESS (9/32, 28.1%) despite CREMA-D dragging the combined number up.

Cross-dataset numbers aren't in the saved results (see Known issues), but from console output on the same run: training on CREMA-D and testing on RAVDESS held up reasonably (~31% accuracy); training on RAVDESS and testing on CREMA-D collapsed to near-constant "neutral" predictions (~17% accuracy, close to chance). RAVDESS alone (24 actors, studio conditions) is a narrower dataset than CREMA-D (91 actors), so a model trained only on it has less to generalize from.

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Trains, evaluates, runs the per-dataset and cross-dataset breakdowns, and writes every plot plus `results_summary.json`. `SpeechEmotionDetectionRunon.ipynb` runs the same thing from Colab with Drive mounted, if you don't want to deal with local GPU setup.

## Files

```
main.py                                    full pipeline: data loading, preprocessing,
                                            model, training, evaluation, plotting
SpeechEmotionDetectionRunon.ipynb          Colab notebook version
requirements.txt
results/
  training_curves.png                      loss / val accuracy / val F1 per epoch
  overall_test_confusion_matrix.png
  ravdess_confusion_matrix.png
  crema-d_confusion_matrix.png
  dataset_comparison.png
  results_summary.json
```

`data/`, `models/`, `preproc/`, `features/`, `experiments/` are gitignored. Datasets are large and downloadable separately, the checkpoint regenerates from a re-run, and the last three are unused scaffolding (see Known issues).

## If picking this back up

- Persist cross-dataset eval results to `results_summary.json` and add confusion matrix plots for them, same as the in-dataset runs get.
- Fear vs. sad is the model's biggest weakness, and it's worse on CREMA-D specifically. Worth a closer look at whether it's an acoustic overlap issue or a CREMA-D labeling/quality issue before assuming it's the former.
- The RAVDESS-to-CREMA-D generalization collapse to near-constant "neutral" is the sharpest limitation in the whole project. Three epochs on ~1,250 clips from one actor pool isn't much to generalize from, so a longer cross-dataset run would be worth trying before concluding the architecture itself can't generalize.

