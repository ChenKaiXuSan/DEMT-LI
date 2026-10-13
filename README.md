# DEMT-LI

Code for *DEMT-LI: A Deep Learning Framework for Dimensional Expansion Method for
Time-Series Lung Imaging* (manuscript under review).

DEMT-LI adds a sequential coronal 2D image-surrogate encoder to an RMSim-style
3D CNN–ConvLSTM3D deformation model. The network predicts dense displacement fields
(DVFs) that warp a reference 4D-CT phase into later respiratory phases. The study is
a retrospective, CT-only proof of concept on six POPI 4D-CT cases. It is not validated
for cine-MRI, clinical tumour localization or end-to-end MR-linac use.

## Protocol notes (read before reusing the results)

* **Inputs.** During both training and evaluation, the 3D encoder receives the full
  three-phase CT sequence (50%, 70%, 90%). Only the 50% volume is warped by the
  spatial transformer. The evaluation is therefore **not** a reference-volume-only
  inference protocol. The 2D surrogate encoder sees frames 50/70% in the encoder pass
  and 70/90% in the decoder pass (see `demt_li/legacy/image_2d`).
* **Cross-validation.** Six outer patient-level folds with a 4–1–1 split. Fold *i*
  tests patient *i*, uses patient *i* mod 6 + 1 for internal validation and trains on
  the remaining four. All three methods use the same splits. The internal-validation
  loss drives the learning-rate schedule and checkpoint selection. The test patient is
  loaded once, after the best checkpoint is fixed.
* **Loss.** Voxelwise MSE plus λ times the mean squared spatial DVF gradient, with
  λ = 1. Both terms are averaged over the three phases. See `demt_li/losses.py`.
* **Training.** Adam (zero weight decay), 200 epochs, batch size 1, seed 42, FP32.
  ReduceLROnPlateau (factor 0.1, patience 10) monitors the internal-validation loss
  divided by its value before training. There is no dropout and no early stopping.
* **Method-specific settings, kept from the original implementations:**

  | Method | Learning rate | Patient order | Augmentation |
  | --- | --- | --- | --- |
  | `image_2d` (DEMT-LI) | 1e-3 | shuffled each epoch | none |
  | `rpm_1d` | 1e-3 | shuffled each epoch | none |
  | `no_surrogate` | 1e-4 | fixed | random horizontal flip of each axial slice (p = 0.5) |

* **Statistics.** Phases are averaged within each patient before summarizing (n = 6).
  Paired comparisons use exact sign-flip tests, percentile patient-bootstrap 95% CIs
  (10,000 resamples, seed 20260817) and Holm correction within each family.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10
pip install -r requirements.txt
python -m pytest -q
```

The reported runs used an NVIDIA RTX 3090 (24 GB), PyTorch 2.12.0 and CUDA 13.0.
One training job needs about 12 GB of GPU memory and roughly 1.6 h for 200 epochs.

## Data

1. **POPI 4D-CT and landmarks.** These are public data from CREATIS
   (Vandemeulebroucke et al., ICCR 2007). Download them with:

   ```bash
   python -m demt_li.popi --output data/popi
   ```

2. **Surrogate inputs.**
   * `2DCT-<patient>/{50,70,90}.png`: one fixed coronal frame per phase and patient
     (the 2D surrogate).
   * A 6 × 10 CSV of diaphragm-apex positions: one row per patient, one column per
     phase from 0% to 90% (the 1D surrogate).

   > TODO before release: state where these 18 PNGs and the CSV can be obtained.

3. **Build the model inputs.**

   ```bash
   python -m demt_li.prepare_data --popi-root data/popi --png-root <png dir> \
       --rpm-csv <rpm csv> --output data/inputs
   ```

   CT preprocessing keeps DICOM slices 10–137 and crops 320 × 320 around the slice
   centre. It then resizes each slice to 128 × 128 (torchvision bilinear, antialiased)
   and keeps Hounsfield units, matching the original data loader. Rebuilding from the
   public POPI archives reproduces the volumes used in the paper to within 5e-4 HU
   (float32 rounding).

## Running the experiments

| Paper item | Command |
| --- | --- |
| Training, 3 methods × 6 folds | `python -m demt_li.run_all --data data/inputs --output runs/main --gpus 0 1 2 3` |
| One fold | `python -m demt_li.train --method image_2d --fold 1 --data data/inputs --output runs/main` |
| Tables II–III, Fig. 10 (image fidelity and paired tests) | `python -m demt_li.summarize --run runs/main --data data/inputs` |
| Figs. 12–14 (Experiment 2, regional-input sensitivity) | `python -m demt_li.regional_input --run runs/main --data data/inputs --output results/regional_input` |
| Tables V–VI, Figs. 20–22 (DVF plausibility, landmark TRE, resolution) | `python -m demt_li.auxiliary --run runs/main --popi-root data/popi --dicom-cache data/inputs/dicom --output results/auxiliary` |
| Fig. 23 (octant MAE) | `python -m demt_li.octant_error --run runs/main --data data/inputs --output results/octant` |
| Table VII (model-only latency) | `python -m demt_li.latency --checkpoint runs/main/image_2d/fold_01/best.pt --data data/inputs --output results/latency.json` |

Each fold directory contains `config.json`, `epochs.jsonl`, `best.pt`, `last.pt`,
`selection.json` and `test_prediction.npz` (the prediction plus the DVF of the held-out
patient).

### Reported image-fidelity results (patient-level mean ± SD, n = 6)

| Method | SSIM | PSNR (dB) | MAE (HU) |
| --- | --- | --- | --- |
| No surrogate | 0.8984 ± 0.0408 | 35.61 ± 2.07 | 32.13 ± 8.54 |
| 1D RPM surrogate | 0.8942 ± 0.0423 | 34.99 ± 2.07 | 34.94 ± 9.86 |
| DEMT-LI (2D image surrogate) | 0.9445 ± 0.0284 | 39.86 ± 2.16 | 21.14 ± 4.98 |

Reproducibility: loading the reported checkpoints with this code reproduces the saved
predictions and DVFs exactly. A 3-epoch retraining check against the reported run
matched the initial validation loss, patient order and learning rate exactly. Loss
values differed at about the 1e-5 relative level, the same size as the difference
between two repeated runs of this code. The cause is that some CUDA backward kernels
(`grid_sample`, trilinear upsampling) are not bitwise deterministic, so retrained
weights will not be bit-identical to the reported checkpoints.

## Repository layout

```
demt_li/
  legacy/          original network code, unchanged (see legacy/README.md)
  network.py       model loading, per-method settings, input tensors, inference
  losses.py        MSE + spatial DVF-gradient objective
  protocol.py      4-1-1 folds and data I/O
  preprocessing.py CT / PNG / RPM preprocessing
  popi.py          POPI download helper
  prepare_data.py, train.py, run_all.py, summarize.py,
  regional_input.py, auxiliary.py, octant_error.py, latency.py
tests/             unit tests (no data or GPU required)
```

## Acknowledgements

The deformation backbone follows RMSim (D. Lee et al., *Phys. Med. Biol.* 68:045009,
2023). The spatial transformer follows VoxelMorph (G. Balakrishnan et al., *IEEE TMI*
38:1788–1800, 2019). The data come from the POPI-model (J. Vandemeulebroucke et al.,
ICCR 2007).

> TODO before release: add the license and citation.
