# Learning Schedules for Text-to-Image GANs

This repository contains the code for our EE641 final project on **training schedules for text-to-image GANs**. Rather than changing the architecture, we study how *time-varying hyperparameters* (generator noise amplitude, discriminator augmentation probability, and regularization strength) affect GAN stability and performance when trained on MS-COCO 2017.

We implement and compare three families of schedulers:

- **Annealed** – hand-designed cosine schedules,
- **Learnable** – monotone learnable schedules via a softmax–CDF parameterization,
- **Adaptive** – a small controller on top of an annealed baseline, reacting to loss and gradient statistics.

The code allows you to train a CLIP-conditioned GAN with each scheduler, compute FID / Inception Score / CLIPScore, and reproduce the figures and tables in our report.

---

## 1. Repository Structure and Major Files

After unzipping, the project code lives under:

```text
ee641_final_project_MengyueShen_YaoxuanLiu-main2/
└── gan_schedulers_project/
    ├── README.md                # This file
    ├── requirements.txt         # Python dependencies
    ├── train.py                 # Main training & evaluation script for one scheduler
    ├── run_all_phases.py        # Convenience script to run all three schedulers
    ├── datasets/
    │   ├── __init__.py
    │   └── coco_text_image.py   # MS-COCO text–image dataset wrapper
    ├── models/
    │   ├── __init__.py
    │   ├── clip_text_encoder.py # CLIPTextEncoder: OpenCLIP wrapper (text + preprocess)
    │   └── generator_discriminator.py
    │       # ConditionalGenerator / ConditionalDiscriminator implementations
    └── schedulers/
        ├── __init__.py
        ├── base.py              # BaseScheduler (abstract interface)
        ├── annealed.py          # Hand-crafted cosine annealed scheduler
        ├── learnable_monotone.py# Learnable monotone scheduler (softmax–CDF)
        └── adaptive_annealed.py # Adaptive scheduler on top of annealed baseline
```

At runtime, the `train.py` script will also create:

```text
gan_schedulers_project/
└── results/
    ├── checkpoints/
    │   └── gan_<scheduler>.pt          # Saved weights for each scheduler
    ├── <scheduler>/
    │   ├── loss_curves_<scheduler>.png
    │   ├── schedule_curves_<scheduler>.png
    │   ├── samples_grid_<scheduler>.png
    │   ├── training_history_<scheduler>.json
    │   └── metrics_<scheduler>.json
    └── summary_metrics.json            # (from run_all_phases.py) aggregated results
```

---

## 2. Setup and Installation

### 2.1. Environment

Recommended:

- Python **3.9–3.11**
- CUDA-capable GPU (for reasonable training time)
- A working C/C++ toolchain if you install `pycocotools` from source

Create and activate a virtual environment (example with `venv`):

```bash
cd gan_schedulers_project

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows (PowerShell/CMD)
```

### 2.2. Install Python dependencies

Dependencies are listed in `requirements.txt`:

```text
torch
torchvision
numpy
matplotlib
tqdm
Pillow
pycocotools
open_clip_torch
scipy
```

Install them with:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `pycocotools` may require system headers (e.g., `gcc`, `python-dev` on Linux).
- `open_clip_torch` will download CLIP weights on first use; this may require internet or a pre-populated cache.
- `scipy` is used for FID computation (matrix square roots).

---

## 3. Dataset and Data Layout

The code uses **MS-COCO 2017** captions for training and evaluation.

The `CocoTextImageDataset` in `datasets/coco_text_image.py` expects the following directory layout:

```text
<data_root>/
├── train2017/                       # Training images
├── val2017/                         # Validation images
└── annotations/
    ├── captions_train2017.json      # COCO captions annotations (train)
    └── captions_val2017.json        # COCO captions annotations (val)
```

For example, if you put the dataset under `./data/COCO2017`, it should look like:

```text
gan_schedulers_project/
├── train.py
├── ...
└── data/
    └── COCO2017/
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── captions_train2017.json
            └── captions_val2017.json
```

Data loading details:

- `datasets.coco_text_image.CocoTextImageDataset`:
  - uses `pycocotools.COCO` to load caption annotations,
  - builds a list of samples with fields `{ "image_path": ..., "caption": ... }`,
  - returns `(image_tensor, caption_string)` in `__getitem__`,
  - optional `max_samples` can be used to subsample for debugging.
- `coco_collate_fn` stacks image tensors into a batch and returns a list of raw caption strings.

Image preprocessing in `train.py`:

- `transforms.Resize((image_size, image_size))` (default `image_size = 256`),
- `transforms.ToTensor()`,
- `transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`, mapping pixel values to `[-1, 1]`.

---

## 4. How to Run the Code

### 4.1. Single-scheduler training (train.py)

From inside `gan_schedulers_project/`:

```bash
python train.py   --data_root ./data/COCO2017   --scheduler_type annealed   --image_size 256   --batch_size 32   --epochs 12   --device cuda   --eval   --eval_samples 5000   --results_dir results
```

**Command-line arguments (train.py)**:

- `--data_root` (str, default `./data/COCO2017`)  
  Root folder of COCO 2017 (`train2017/`, `val2017/`, `annotations/`).

- `--scheduler_type` (str, default `annealed`)  
  Which scheduler to use:  
  - `"annealed"` – cosine annealed schedules,  
  - `"learnable"` – learnable monotone schedules,  
  - `"adaptive"` – adaptive controller + annealed baseline.

- `--image_size` (int, default `256`)  
  Target image resolution; must be compatible with generator/discriminator (`64, 128, 256` supported in `generator_discriminator.py`, but the training script uses 256 by default).

- `--batch_size` (int, default `32`)  
  Training batch size.

- `--epochs` (int, default `10` / changed in `run_all_phases.py` to 12)  
  Number of epochs.

- `--device` (str, default `"cuda"`)  
  `"cuda"` or `"cpu"`. Internally, `torch.cuda.is_available()` is checked; if CUDA is not available, it falls back to CPU.

- `--eval` (flag)  
  If provided, runs full evaluation at the end of training (FID, Inception Score, CLIPScore) using `evaluate_metrics`.

- `--eval_samples` (int, default `5000`)  
  Number of fake samples to generate for evaluation. We used 5,000 for reported results.

- `--results_dir` (str, default `"results"`)  
  Output directory for checkpoints, metrics, and plots.

**What train.py does:**

1. Builds COCO train/val datasets and `DataLoader`s with `coco_collate_fn`.
2. Instantiates:
   - `CLIPTextEncoder` (OpenCLIP model + tokenizer + preprocess),
   - `ConditionalGenerator` and `ConditionalDiscriminator`,
   - the chosen scheduler (`AnnealedScheduler`, `LearnableMonotoneScheduler`, or `AdaptiveAnnealedScheduler`).
3. Creates optimizers:
   - `optim_G = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))`,
   - `optim_D = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))`,
   - `optim_S = Adam(scheduler parameters, lr=1e-3)` if the scheduler has learnable parameters.
4. Trains for `num_epochs`:
   - For each batch:
     - computes normalized progress `u = step / (total_steps - 1)`,
     - queries the scheduler for `noise_sigma`, `augment_p`, `reg_lambda`,
     - encodes captions with CLIP,
     - generates fake images, applies data augmentations, and trains D/G with `BCEWithLogitsLoss`,
     - logs `loss_D`, `loss_G`, and the scheduler outputs to `history`,
     - tracks simple gradient norms of G and D in `state_for_sched` for adaptive scheduling.
5. At the end of training:
   - saves checkpoint in `results/checkpoints/gan_<scheduler_type>.pt`,
   - calls `save_training_curves(...)` to write:
     - `loss_curves_<scheduler_type>.png`,
     - `schedule_curves_<scheduler_type>.png`,
     - `training_history_<scheduler_type>.json`,
   - calls `sample_and_save_images(...)` to create a grid of generated samples:
     - `samples_grid_<scheduler_type>.png`.

6. If `--eval` is set:
   - calls `evaluate_metrics(...)` to compute FID, Inception Score, and CLIPScore on the validation set (up to `--eval_samples` examples),
   - saves `metrics_<scheduler_type>.json` under `results/<scheduler_type>/`,
   - prints the metrics to stdout.

The `train(...)` function returns a `metrics` dictionary (empty if `run_eval` is `False`).

---

### 4.2. Running all three schedulers (run_all_phases.py)

To reproduce the full comparison:

```bash
python run_all_phases.py
```

This script:

- sets `data_root = "./data/COCO2017"` and `results_dir = "results"`,
- iterates over `sched_types = ["annealed", "learnable", "adaptive"]`,
- for each scheduler, calls:

  ```python
  metrics = train(
      data_root=data_root,
      scheduler_type=sched,
      image_size=256,
      batch_size=32,
      num_epochs=12,
      device="cuda",
      run_eval=True,
      n_eval_samples=5000,
      results_dir=results_dir,
  )
  ```

- aggregates all metrics into `results/summary_metrics.json`:

  ```json
  {
    "annealed":  { "FID": ..., "InceptionScore": ..., "CLIPScore": ..., ... },
    "learnable": { "FID": ..., "InceptionScore": ..., "CLIPScore": ..., ... },
    "adaptive":  { "FID": ..., "InceptionScore": ..., "CLIPScore": ..., ... }
  }
  ```

This is the easiest way for someone else to run the full experiment suite and obtain the numbers for the table in the report.

---

### 4.3. Using a saved checkpoint for inference

Once training finishes, you will have checkpoints like:

```text
results/checkpoints/gan_annealed.pt
results/checkpoints/gan_learnable.pt
results/checkpoints/gan_adaptive.pt
```

A typical inference-only script would:

1. Recreate `CLIPTextEncoder`, `ConditionalGenerator`, and `ConditionalDiscriminator` with the same hyperparameters used in `train.py` (e.g., `noise_dim=128`, `base_channels=64`, `image_size=256`).
2. Load the saved `state_dict`s:

   ```python
   ckpt = torch.load("results/checkpoints/gan_annealed.pt", map_location=device)
   G.load_state_dict(ckpt["G"])
   D.load_state_dict(ckpt["D"])
   scheduler.load_state_dict(ckpt["scheduler"])  # optional for inference
   ```

3. For new text prompts:
   - use `CLIPTextEncoder` to encode a list of text strings into features,
   - sample noise from `torch.randn(batch_size, 128, device=device)`,
   - feed `(noise, text_features)` into `G` to obtain fake images in `[-1, 1]`,
   - rescale to `[0, 1]` or `[0, 255]` for visualization.

The logic is very similar to what `sample_and_save_images(...)` does inside `train.py`.

---

## 5. Code Organization

We follow a clear separation of concerns:

- **Model definitions**:
  - `models/generator_discriminator.py`  
    - `ConditionalGenerator`: upsampling CNN conditioned on CLIP text features.  
    - `ConditionalDiscriminator`: downsampling CNN that fuses image and text.
  - `models/clip_text_encoder.py`  
    - `CLIPTextEncoder`: wraps `open_clip` to provide:
      - `.model` (CLIP model),
      - `.preprocess` (image transforms),
      - `.tokenizer` (text tokenizer),
      - `.encode_text(texts: List[str])` and `.forward(texts)`.

- **Training script (entry point)**:
  - `train.py`  
    - `train(...)`: main training loop for a single scheduler.  
    - `evaluate_metrics(...)`: FID, Inception Score, CLIPScore evaluation.  
    - `save_training_curves(...)`: log curves and write PNG/JSON.  
    - `sample_and_save_images(...)`: generate and save image grids.  
    - `if __name__ == "__main__":` block with `argparse` for CLI use.

- **Evaluation and analysis code**:
  - Included inside `train.py`:
    - Inception-based FID and Inception Score,
    - CLIPScore via cosine similarity of CLIP image/text features.

- **Data loading and preprocessing utilities**:
  - `datasets/coco_text_image.py`:
    - `CocoTextImageDataset`: wraps COCO 2017 images + captions.
    - `coco_collate_fn`: collate function that returns `(batch_images, list_of_captions)`.

- **Configuration / hyperparameters**:
  - Most configuration is in code, not separate YAML:
    - noise_dim, base_channels, image_size are set in `train(...)` when instantiating models,
    - optimizer settings and learning rates are defined in `train(...)`,
    - scheduler ranges (start/end values) are defined in each scheduler’s `__init__`.

- **Schedulers**:
  - `schedulers/base.py`: defines the base `BaseScheduler` interface.
  - `schedulers/annealed.py`: cosine annealed scheduler:
    - maps normalized progress `u` to `noise_sigma`, `augment_p`, `reg_lambda`.
  - `schedulers/learnable_monotone.py`:
    - maintain learnable logits over `k_bins`,
    - convert to a monotone curve via a softmax–CDF construction,
    - enforce monotonicity while learning the curve shape.
  - `schedulers/adaptive_annealed.py`:
    - wraps an `AnnealedScheduler`,
    - uses a small MLP controller to rescale `noise_sigma`, `augment_p`, `reg_lambda` based on:
      - EMA of generator and discriminator losses,
      - EMA of gradient norms,
      - current progress `u`.

---

## 6. Code Documentation Notes

The code aims to be compact and readable for someone familiar with PyTorch and GANs. The most “non-obvious” parts correspond directly to the project’s main ideas:

- **Learnable monotone schedules**  
  Implemented in `schedulers/learnable_monotone.py` using:
  - learnable logits → softmax → discrete PDF → cumulative sum → CDF,
  - interpolation over `u ∈ [0, 1]` to get a monotone curve,
  - rescaling to parameter-specific ranges.

- **Adaptive schedules**  
  Implemented in `schedulers/adaptive_annealed.py`:
  - base annealed schedule + controller MLP,
  - controller input: EMA of losses and gradient norms + progress `u`,
  - controller output: multiplicative scaling factors for each parameter.

- **Metric computation**  
  Implemented in `train.py`:
  - Inception-based FID and Inception Score,
  - CLIPScore via cosine similarity of CLIP image/text features.

---

## 7. Reproducibility

The goal is not bit-for-bit determinism, but **similar performance** when re-running the code with the same settings.

### 7.1. Random seeds

By default, we rely on PyTorch’s and NumPy’s standard randomness. For stronger reproducibility, you can add the following at the top of `train.py`:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

This will fix the Python, NumPy, and PyTorch RNGs.

### 7.2. Hyperparameter configurations

Key hyperparameters used in the experiments reported in the final report:

- **Model** (in `train(...)`):
  - `noise_dim = 128`,
  - `base_channels = 64`,
  - `image_size = 256`,
  - CLIP model: `ViT-B-32`, pretrained on `"laion2b_s34b_b79k"` (via `open_clip.create_model_and_transforms`).

- **Training**:
  - `batch_size = 32`,
  - `num_epochs = 12` (when using `run_all_phases.py`),
  - optimizer:
    - Generator: `Adam(lr=2e-4, betas=(0.5, 0.999))`,
    - Discriminator: `Adam(lr=2e-4, betas=(0.5, 0.999))`,
    - Scheduler (if learnable): `Adam(lr=1e-3)`.

- **Schedulers**:
  - `AnnealedScheduler` (`annealed.py`):
    - cosine annealing from:
      - noise: `noise_start=1.0 → noise_end=0.3`,
      - data augmentation: `aug_start=0.0 → aug_end=0.8`,
      - regularization: `reg_start=0.1 → reg_end=0.0`.

  - `LearnableMonotoneScheduler` (`learnable_monotone.py`):
    - `k_bins=16`,
    - `min_noise=0.3, max_noise=1.0`,
    - `min_aug=0.0, max_aug=0.8`,
    - `min_reg=0.0, max_reg=0.1`.

  - `AdaptiveAnnealedScheduler` (`adaptive_annealed.py`):
    - base scheduler: default `AnnealedScheduler`,
    - controller MLP:
      - `hidden_dim=32`,
      - scale range `[min_scale=0.5, max_scale=2.0]`,
    - EMA momentum for state signals (losses and gradient norms) as defined in the file.

### 7.3. Training procedures and schedules

For each scheduler type:

- Total steps: `total_steps = num_epochs * len(train_loader)`.
- At each step:
  - `u = step / (total_steps - 1)`,
  - the scheduler is called with `(step, total_steps, state_for_sched)` to produce:
    - `noise_sigma`, `augment_p`, `reg_lambda`, and `u`,
  - the generator uses `noise_sigma` to scale the Gaussian noise,
  - the discriminator uses `augment_p` within `apply_augmentation` (in `train.py`),
  - `reg_lambda` scales an additional regularization term for D (simple L2 penalty on the fake images in this implementation),
  - after backpropagation, gradient norms are measured via `compute_grad_norm` and stored into `state_for_sched` for the next scheduler call (adaptive schedulers only).

### 7.4. Data preprocessing steps

- **Images**:
  - resized to `(image_size, image_size)` using bilinear interpolation,
  - converted to tensors, and normalized to `[-1, 1]` via `(x - 0.5) / 0.5`.

- **Captions**:
  - raw COCO captions are used as strings,
  - inside `CLIPTextEncoder`, tokenization is handled by `open_clip.get_tokenizer(model_name)`:
    - `.tokenizer(texts)` returns token IDs,
    - `model.encode_text(tokens)` returns text feature vectors,
    - features are L2-normalized.

- **Inception metrics**:
  - images are upsampled to `(299, 299)`,
  - rescaled from `[-1, 1]` back to `[0, 1]`,
  - passed through `torchvision.models.inception_v3` to get:
    - logits (for Inception Score),
    - features (for FID).

### 7.5. Code used to generate figures and results

All key results in the report can be regenerated with:

- **Metric computation**: `evaluate_metrics(...)` in `train.py`  
  → produces FID, Inception Score, CLIPScore for a given trained model.

- **Training curves and schedule plots**: `save_training_curves(...)`  
  → writes `loss_curves_<scheduler>.png`, `schedule_curves_<scheduler>.png`, and `training_history_<scheduler>.json` under `results/<scheduler>/`.

- **Sample image grids**: `sample_and_save_images(...)`  
  → writes `samples_grid_<scheduler>.png` under `results/<scheduler>/`.

- **Aggregated metrics across schedulers**: `run_all_phases.py`  
  → runs all three schedulers and saves `summary_metrics.json`.

By re-running `run_all_phases.py` with the same dataset and hyperparameters, another user should obtain qualitatively similar curves and metrics.

---

## 8. Special Technical Requirements and Tips

- A **GPU** is strongly recommended; training on CPU is possible but extremely slow.
- The first run may download:
  - CLIP weights (via `open_clip_torch`),
  - Inception weights (via `torchvision.models.inception_v3(weights=...)`).
- If you see a CUDA OOM error:
  - reduce `--batch_size`,
  - optionally reduce `--image_size`.
- Always run scripts from the `gan_schedulers_project/` directory so that relative paths for `data_root` and `results_dir` resolve correctly.

---

This README is intended to be self-contained:  
someone familiar with PyTorch and deep learning should be able to set up the environment, prepare COCO 2017, run all three schedulers, and regenerate both the quantitative metrics and qualitative figures used in our final report without needing to guess any hidden configurations.
