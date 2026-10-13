#!/usr/bin/env bash
# End-to-end pipeline. Run from the repository root.
#   bash project/script.sh <png dir> <rpm csv>
set -euo pipefail

PNG_ROOT=${1:?path to 2DCT-<patient>/<phase>.png surrogate frames}
RPM_CSV=${2:?path to the 6 x 10 RPM CSV}
GPUS=${GPUS:-"0 1 2 3"}
RUN=runs/main

# 1. Public POPI 4D-CT and landmarks, then model inputs.
python -m project.dataloader.popi --output data/popi
python -m project.dataloader.prepare_data --popi-root data/popi --png-root "$PNG_ROOT" \
    --rpm-csv "$RPM_CSV" --output data/inputs

# 2. 3 methods x 6 folds (4-1-1), then Tables II-III.
python -m project.main --data data/inputs --output "$RUN" --gpus $GPUS

# 3. Secondary analyses.
python -m project.evaluation.regional_input --run "$RUN" --data data/inputs --output "$RUN/results/regional_input"
python -m project.evaluation.auxiliary --run "$RUN" --popi-root data/popi \
    --dicom-cache data/inputs/dicom --output "$RUN/results/auxiliary"
python -m project.evaluation.octant_error --run "$RUN" --data data/inputs --output "$RUN/results/octant"
python -m project.evaluation.latency --checkpoint "$RUN/image_2d/fold_01/best.pt" --data data/inputs \
    --output "$RUN/results/latency.json"
