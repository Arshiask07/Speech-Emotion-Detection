\# Speech Emotion Recognition



A deep learning model that listens to a short clip of speech and predicts the speaker's emotion — neutral, happy, sad, angry, fear, or disgust — using nothing but the audio itself.



The core idea is a CNN + BiLSTM + Transformer stack: convolutions pick up local acoustic texture, the BiLSTM tracks how that texture evolves across the clip, and the Transformer lets the model weigh distant parts of the utterance against each other before making a decision. It's trained on two public datasets, RAVDESS and CREMA-D, combined and split by speaker so no voice appears in both training and test.



\## Why two datasets instead of one



A model trained and tested on a single dataset tends to look better than it actually is — it partly learns that dataset's specific actors, microphones, and recording room, not emotion itself. Combining RAVDESS (professional actors, studio conditions) and CREMA-D (a much larger, more demographically varied actor pool) forces the model to find features that hold up across both. This repo also runs a leave-one-dataset-out check — train on one corpus, test entirely on the other — specifically to expose how much performance drops when the model meets a completely unfamiliar recording setup.



\## How it works



```

Raw audio (16 kHz)

&#x20;  -> Adaptive preprocessing

&#x20;       - estimate SNR; only apply spectral subtraction if the clip is actually noisy

&#x20;       - RMS normalization for consistent loudness

&#x20;       - trim leading/trailing silence

&#x20;  -> Log-mel spectrogram + delta + delta-delta (80 mel bins, 300 frames)

&#x20;  -> CNN encoder (3 blocks: 64 -> 128 -> 256 channels)

&#x20;  -> Bidirectional LSTM (2 layers, 256 hidden units)

&#x20;  -> Transformer encoder (4 layers, 8 heads, d\_model=256)

&#x20;  -> Global average pooling

&#x20;  -> Fully connected classifier -> 6 emotion classes

```



The preprocessing step is adaptive on purpose — it only reaches for spectral subtraction when a clip's estimated signal-to-noise ratio drops below 10 dB, so clean recordings aren't needlessly degraded by noise reduction they don't need.



\## Datasets



| Dataset | What it is | Source |

|---|---|---|

| RAVDESS | 24 professional actors, North American English, studio-quality | \[zenodo.org/record/1188976](https://zenodo.org/record/1188976) |

| CREMA-D | 91 actors, wide range of ages and ethnic backgrounds | \[github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) |



Both are public and free to download but not included in this repo — together they're a few gigabytes of audio. Grab them from the links above and place them at `data/RAVDESS/` and `data/CREMA-D/`. The loader expects each dataset's standard filename format (RAVDESS: `03-01-06-01-02-01-12.wav`; CREMA-D: `1001\_DFA\_ANG\_XX.wav`) to pull out emotion labels and speaker IDs — if you rename files, the parser will silently skip them.



\## Running it



```bash

pip install -r requirements.txt

python main.py

```



That trains the model with early stopping, evaluates on a held-out test set, runs the per-dataset breakdown and the cross-dataset generalization check, and writes every plot plus a JSON summary to `results/`.



If you'd rather not deal with local GPU setup, `SpeechEmotionRecognition\_Colab.ipynb` mounts Google Drive, installs everything, and runs `main.py` from there — that's how this project was actually trained.



\## Results



Trained for 10 epochs on the full combined dataset (8,690 clips after filtering) — no early stopping triggered, validation accuracy was still climbing at epoch 10.



| Split | Accuracy | F1 (weighted) |

|---|---|---|

| Overall test set | 57.0% | 0.561 |

| RAVDESS only | 52.9% | 0.510 |

| CREMA-D only | 57.8% | 0.564 |

| Cross-dataset: train CREMA-D, test RAVDESS | 30.9% | 0.268 |

| Cross-dataset: train RAVDESS, test CREMA-D | 17.4% | 0.085 |



Random guessing on 6 balanced classes sits around 17%, so the in-domain numbers are a solid multiple of chance for a model with no pretrained audio backbone behind it — everything here is learned from scratch on these two datasets.



\*\*Per-class behavior (overall test set):\*\* anger is by far the easiest class to recognize (F1 0.76) — it has the most distinctive acoustic signature of the six. Sad and neutral both land around 0.52-0.58. Fear is the hardest class (F1 0.50, recall only 0.40) and is most often confused with sad; disgust gets pulled toward neutral and angry. That pattern lines up with what's generally reported in speech emotion research — fear and sadness share a lot of low-energy, low-pitch-variance acoustic ground, so a model without extra prosodic cues will genuinely struggle to tell them apart.



\*\*The cross-dataset gap is the most interesting limitation here.\*\* A model trained only on RAVDESS collapses almost entirely to predicting "neutral" when tested on CREMA-D. This isn't a bug — three epochs on \~1,250 clips from a single, fairly homogeneous actor pool just isn't enough signal to generalize to a dataset with a completely different set of speakers and recording conditions. It's a reasonable illustration of why cross-corpus generalization remains a genuinely hard, open problem in this field, rather than something a bigger model alone would fix.



Full numbers for every run are in `results/results\_summary.json`; training curves and confusion matrices for each split are the PNGs alongside it.



\## Repository structure



```

.

├── main.py                                  # full training + evaluation pipeline

├── SpeechEmotionRecognition\_Colab.ipynb      # Colab notebook used to train this

├── requirements.txt

├── results/

│   ├── training\_curves.png

│   ├── overall\_test\_confusion\_matrix.png

│   ├── ravdess\_confusion\_matrix.png

│   ├── crema-d\_confusion\_matrix.png

│   ├── dataset\_comparison.png

│   └── results\_summary.json

└── README.md

```



`data/`, `models/`, `preproc/`, `features/`, and `experiments/` are intentionally left out of version control. The datasets are large and publicly downloadable (see above), the trained checkpoint (`best\_model.pth`) is regenerable by re-running `main.py` and would push the repo well past a reasonable size, and `preproc/`, `features/`, and `experiments/` are just empty scaffolding directories the script creates but doesn't currently write anything into.



\## implementation notes 


\- Splits are done by speaker ID, not by individual clip. If they were split by clip, the same actor's voice could show up in both train and test, and the model could partly cheat by recognizing the speaker rather than the emotion.

\- Labels are unified across both datasets into one shared six-class set. RAVDESS's "calm" category folds into "neutral" since CREMA-D has no equivalent, and RAVDESS's "surprised" clips are dropped entirely since CREMA-D doesn't include that emotion.

\- Augmentation (noise injection, time stretching, pitch shifting) only applies to the training split, each independently at a 50% chance per clip, so the model sees a genuinely varied version of the training data without ever touching validation or test audio.

