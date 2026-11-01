# UTA-RLDD dataset

This project is designed around the University of Texas at Arlington
Real-Life Drowsiness Dataset (UTA-RLDD).

## Official sources

- Project page: https://sites.google.com/view/utarldd/home
- Google Drive folder: https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl

## What we verified

The official page is publicly readable. The Drive folder listing is also
publicly visible and contains these ZIP files:

| File | Approximate size |
| --- | --- |
| Fold1_part1.zip | 12 GB |
| Fold1_part2.zip | 10.88 GB |
| Fold2_part1.zip | 10.56 GB |
| Fold2_part2.zip | 11.36 GB |
| Fold3_part1.zip | 11.93 GB |
| Fold3_part2.zip | 7.55 GB |
| Fold4_part1.zip | 11.58 GB |
| Fold4_part2.zip | 13.35 GB |
| Fold5_part1.zip | 10.16 GB |
| Fold5_part2.zip | 11.95 GB |

Total size reported by the authors: **111.3 GB**, 180 videos, 60 subjects.

No ZIP file was downloaded into this project automatically. Each archive is
several gigabytes, and a full copy is not required to run the live webcam
demo.

## Labels (do not invent others)

Each video is labeled by the participant as one of:

- `0` = alert
- `5` = low vigilant
- `10` = drowsy

The 60 subjects are split into five folds of 12 people for cross-validation.

## How to add a subset

1. Open the Drive folder in a browser (Google account recommended).
2. Download the smallest part first: `Fold3_part2.zip` (~7.55 GB).
3. Extract it into `dataset/raw/` so the fold folder is preserved.
4. Inspect what was extracted:

```
python backend/dataset_loader.py --inspect
```

5. Train only after videos are present:

```
python backend/train.py --max-videos 6 --frame-stride 10
python backend/evaluate.py --max-videos 6
```

## Live application without the dataset

The Flask webcam system uses MediaPipe facial landmarks, personalized
calibration, and temporal risk scoring. It does **not** need UTA-RLDD
videos to start. Training and official 5-fold evaluation stay pending
until at least one extracted fold is available under `dataset/raw/`.
