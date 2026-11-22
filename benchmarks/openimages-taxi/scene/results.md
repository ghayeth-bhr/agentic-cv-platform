# Segmentation benchmark: Open Images V7 taxis (corrected)

- Ground truth: 98 images, 162 masks (93 images / 156 masks scored; 5 calibration images excluded)
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU; SAM 3 float16
- Matching: one-to-one by score at mask IoU >= 0.5
- Calibration judge: scene (Qwen3-VL), app thresholds [0.3, 0.45, 0.6, 0.75, 0.85], agreement 0.6

| Prompt | Calibrated threshold | Mask AP | Mask AP50 | Box AP50 | F1 @0.5 (plain SAM 3) | F1 @calibrated (BotFlow) | 95% CI | Mask IoU of matches | s / image | Peak GPU MB |
|---|---|---|---|---|---|---|---|---|---|---|
| taxi | 0.45 | 0.835 | 0.919 | 0.908 | 0.832 | 0.824 | [0.7799, 0.8783] | 0.935 | 0.632 | 2749 |
| taxi car | 0.6 | 0.819 | 0.899 | 0.890 | 0.664 | 0.723 | [0.6639, 0.7847] | 0.932 | 0.639 | 2782 |

## Precision / recall by threshold

### Prompt "taxi"

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.3 | 0.569 | 0.955 | 0.713 | 149 | 113 | 7 |
| 0.4 | 0.689 | 0.936 | 0.793 | 146 | 66 | 10 |
| 0.5 | 0.770 | 0.904 | 0.832 | 141 | 42 | 15 |
| 0.6 | 0.832 | 0.891 | 0.861 | 139 | 28 | 17 |
| 0.7 | 0.884 | 0.827 | 0.854 | 129 | 17 | 27 |
| 0.8 | 0.924 | 0.776 | 0.843 | 121 | 10 | 35 |
| 0.9 | 0.968 | 0.583 | 0.728 | 91 | 3 | 65 |

Calibration (vlm, 61.5 s): 0.3: 5/5, 0.45: 5/5, 0.6: 5/5, 0.75: 5/5, 0.85: 3/5 -> calibrated

### Prompt "taxi car"

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.3 | 0.350 | 0.974 | 0.515 | 152 | 282 | 4 |
| 0.4 | 0.425 | 0.974 | 0.591 | 152 | 206 | 4 |
| 0.5 | 0.503 | 0.974 | 0.664 | 152 | 150 | 4 |
| 0.6 | 0.582 | 0.955 | 0.723 | 149 | 107 | 7 |
| 0.7 | 0.705 | 0.949 | 0.809 | 148 | 62 | 8 |
| 0.8 | 0.791 | 0.923 | 0.852 | 144 | 38 | 12 |
| 0.9 | 0.877 | 0.731 | 0.797 | 114 | 16 | 42 |

Calibration (vlm, 98.2 s): 0.3: 5/5, 0.45: 5/5, 0.6: 5/5, 0.75: 5/5, 0.85: 5/5 -> calibrated

