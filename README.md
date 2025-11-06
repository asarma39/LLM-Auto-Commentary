LLM-Based Automatic Cricket Commentary Using Umpire Sensor Data

This repo contains our work-in-progress implementation for an end-to-end system that:

Takes wearable sensor + UWB distance data from cricket umpires.

Classifies the umpire signal (e.g. boundary, wide, no-ball).

(Planned) Detects events from continuous streams.

(Planned) Generates natural-language cricket commentary using an LLM.

Right now, we’ve built a clean data pipeline + a first baseline classifier that gets decent performance and is easy to extend.

What’s Implemented So Far
1. ViSig Data Loader

File: src/data_loading/load_visig.py

Capabilities:

Loads all .mat files containing umpire signal trials.

Parses:

acc_mat – accelerometer data

gyro_mat – gyroscope data

dist_mat – pairwise UWB distances between sensor nodes

rawt – timestamps

Normalizes shapes and converts everything to time-major format.

Extracts:

label (e.g. boundary4, wide, out, etc.)

participant_id from filename suffix (e.g. _1, _2, …).

Provides:

ViSigSample dataclass to hold one trial.

load_visig_mat(...) to load a single file.

load_visig_dataset(...) to load all .mat files under a directory.

get_label_distribution(...) for quick label counts.

to_flat_sequence(...) to convert each trial into a (T, F) matrix.

Current notes:

We’ve validated on our dataset:

80 samples

10 labels (boundary4, boundary6, cancelcall, deadball, legbye, noball, out, penaltyrun, shortrun, wide)

Balanced: 8 samples per class

Each sample is a multivariate time series:

T timesteps (varies per sample)

F = 195 features per timestep:

90 from acc_mat

90 from gyro_mat

15 from upper-triangular dist_mat

2. PyTorch Dataset Wrapper

File: src/data_loading/cricket_dataset.py

Capabilities:

build_label_mapping(...)

Deterministic label → index mapping (sorted lexicographically).

CricketSignalsDataset

Wraps a list of ViSigSample into something you can feed to PyTorch.

Uses to_flat_sequence(...) internally.

Outputs:

x: tensor of shape (max_len, feature_dim)

Center-cropped if sequence is longer than max_len

Padded with pad_value at the end if shorter

y: scalar class index

Exposes:

num_classes

feature_dim (inferred from data)

create_cricket_datasets(...)

Loads all samples from a given root.

Builds a shared label mapping.

Creates one CricketSignalsDataset and splits into:

train / val / test via random_split (70% / 15% / 15% by default).

This gives us a single source of truth for model-ready data.

3. Baseline Sequence Classifier (1D CNN)

Files:

src/models/seq_cnn.py

src/training/train_seq_classifier.py

Key ideas:

We treat each signal as a time series, not an image.

We use a 1D CNN over time:

Input: (batch, seq_len, feature_dim)

Conv1d over time dimension to learn local temporal patterns.

Global max pooling over time for shift invariance.

Linear layer → class logits.

Reasoning:

Small dataset (~80 sequences) → need a simple, low-parameter model.

1D CNNs are standard for multivariate time-series classification.

This is an intentional, interpretable baseline, not the final architecture.

train_seq_classifier.py:

Loads data using create_cricket_datasets.

Builds SimpleCricketCNN with:

input_dim = feature_dim

num_classes = dataset.num_classes

Trains with:

Adam (lr = 1e-3 default)

Cross-entropy loss

Early stopping on validation accuracy

Saves best checkpoint to:

models/checkpoints/visig_simple_cnn.pt

Prints:

train/val metrics per epoch

final test accuracy

We’ve seen ~60–70% test accuracy with a 2-layer CNN on this tiny dataset, which is a solid sanity-checked baseline (random would be 10%).

Data Format (For Everyone’s Intuition)

Each .mat file ≈ one labeled umpire gesture.

Inside:

acc_mat: (6, 15, N)

gyro_mat: (6, 15, N)

dist_mat: (6, 6, N)

rawt: (N,)

Where:

6 = number of sensor nodes

15 = 5 consecutive IMU readings × 3 axes (pre-grouped)

N = timesteps in this trial

We convert to:

ViSigSample:
    acc  -> (T, 6, 15)
    gyro -> (T, 6, 15)
    dist -> (T, 6, 6)
    t    -> (T,)


Then to_flat_sequence(sample) → (T, 195).

From there, models only see a clean multivariate sequence.

Repository Structure (Suggested)

If not already, we’re organizing as:

project-root/
  README.md
  .env                         # optional, can store VISIG_ROOT here
  models/
    checkpoints/
  src/
    __init__.py
    data_loading/
      __init__.py
      load_visig.py
      cricket_dataset.py
    models/
      __init__.py
      seq_cnn.py
    training/
      __init__.py
      train_seq_classifier.py
  notebooks/
    00_eda.ipynb               # lengths, label dist, sanity plots, etc.
  data/
    # (we do NOT commit .mat here; stored locally)

Setup Instructions
1. Clone & create environment
git clone <repo-url>
cd <repo-name>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt


If requirements.txt doesn’t exist yet, minimal set:

numpy
scipy
torch
torchvision
torchaudio
tqdm


(LLM + commentary dependencies can be added later.)

2. Point to the data

We use an environment variable VISIG_ROOT so paths aren’t hardcoded.

Example:

export VISIG_ROOT="/path/to/visig/data/cricket"


or add to .env:

VISIG_ROOT=/path/to/visig/data/cricket


Each .mat file should live somewhere under this directory.

How to Run Things
1. Quick sanity check: loader

From project root:

export VISIG_ROOT="/path/to/visig/data/cricket"

python -m src.data_loading.load_visig


What it does:

Loads all .mat files.

Prints:

number of samples

label distribution

shapes for the first sample

If that looks reasonable, you’re good.

2. Create datasets & inspect shapes
python -m src.data_loading.cricket_dataset


What it does:

Uses VISIG_ROOT.

Builds train/val/test splits.

Prints:

split sizes

example x shape (should be (max_len, feature_dim))

example label index.

3. Train the baseline CNN
python -m src.training.train_seq_classifier


Requirements:

VISIG_ROOT set.

Runs with defaults:

max_len = 400

batch_size = 8

lr = 1e-3

num_epochs = 50

patience = 8

During training, you’ll see logs like:

Using device: cuda
Input dim: 195, num_classes: 10
Dataset sizes -> train: 56, val: 12, test: 12
Epoch 001: train_loss=..., val_acc=..., best_val_acc=...
...
Saved best model to models/checkpoints/visig_simple_cnn.pt
Test accuracy: 0.67


Interpretation:

If test accuracy ~0.6–0.7:

Baseline working.

If test accuracy ~0.1:

Something’s wrong (check VISIG_ROOT, shapes, labels).

How This Fits the Bigger Project

Right now we’ve completed:

✅ Clean ingestion of .mat files into a typed Python representation.

✅ Standardized (T, F) feature representation for each trial.

✅ PyTorch Dataset + random train/val/test splitting.

✅ Simple, well-structured 1D CNN baseline with early stopping.

✅ Verified that the sensor data is predictive of the signal labels.

Next major milestones (for us / for teammates):

Better evaluation

Confusion matrix.

Subject-wise splits (train on some umpires, test on others).

Continuous stream simulation

Concatenate multiple trials + idle.

Sliding window + classifier → event detection.

LLM-based commentary generation

Map detected events to short commentary lines.

Add simple templates as a fallback (to keep it robust).

End-to-end demo

“Given this sensor stream, show detected events + auto-generated commentary.”