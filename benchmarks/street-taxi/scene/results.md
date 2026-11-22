# Segmentation benchmark: BotFlow demo street footage

- Ground truth: 42 images, 55 masks (37 images / 52 masks scored; 5 calibration images excluded)
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU; SAM 3 float16
- Matching: one-to-one by score at mask IoU >= 0.5
- Calibration judge: scene (Qwen3-VL), app thresholds [0.3, 0.45, 0.6, 0.75, 0.85], agreement 0.6

| Prompt | Calibrated threshold | Mask AP | Mask AP50 | Box AP50 | F1 @0.5 (plain SAM 3) | F1 @calibrated (BotFlow) | 95% CI | Mask IoU of matches | s / image | Peak GPU MB |
|---|---|---|---|---|---|---|---|---|---|---|
| taxi | 0.45 | 0.975 | 0.993 | 0.993 | 0.912 | 0.889 | [0.8095, 0.96] | 0.966 | 0.666 | 2144 |
| taxi car | 0.3 | 0.951 | 0.962 | 0.962 | 0.619 | 0.424 | [0.3571, 0.4955] | 0.967 | 0.74 | 2163 |

## Precision / recall by threshold

### Prompt "taxi"

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.3 | 0.667 | 1.000 | 0.800 | 52 | 26 | 0 |
| 0.4 | 0.743 | 1.000 | 0.853 | 52 | 18 | 0 |
| 0.5 | 0.839 | 1.000 | 0.912 | 52 | 10 | 0 |
| 0.6 | 0.897 | 1.000 | 0.946 | 52 | 6 | 0 |
| 0.7 | 0.946 | 1.000 | 0.972 | 52 | 3 | 0 |
| 0.8 | 0.961 | 0.942 | 0.952 | 49 | 2 | 3 |
| 0.9 | 0.977 | 0.827 | 0.896 | 43 | 1 | 9 |

Calibration (vlm, 69.0 s): 0.3: 4/5, 0.45: 4/5, 0.6: 4/5, 0.75: 3/5, 0.85: 2/5 -> calibrated

### Prompt "taxi car"

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.3 | 0.269 | 1.000 | 0.424 | 52 | 141 | 0 |
| 0.4 | 0.359 | 1.000 | 0.528 | 52 | 93 | 0 |
| 0.5 | 0.448 | 1.000 | 0.619 | 52 | 64 | 0 |
| 0.6 | 0.578 | 1.000 | 0.732 | 52 | 38 | 0 |
| 0.7 | 0.776 | 1.000 | 0.874 | 52 | 15 | 0 |
| 0.8 | 0.845 | 0.942 | 0.891 | 49 | 9 | 3 |
| 0.9 | 0.896 | 0.827 | 0.860 | 43 | 5 | 9 |

Calibration (vlm, 85.3 s): 0.3: 5/5, 0.45: 4/5, 0.6: 4/5, 0.75: 4/5, 0.85: 4/5 -> calibrated

