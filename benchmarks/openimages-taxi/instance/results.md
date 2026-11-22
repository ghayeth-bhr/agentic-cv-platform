# Segmentation benchmark: Open Images V7 taxis (corrected)

- Ground truth: 98 images, 162 masks (93 images / 156 masks scored; 5 calibration images excluded)
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU; SAM 3 float16
- Matching: one-to-one by score at mask IoU >= 0.5
- Calibration judge: instance (Qwen3-VL), app thresholds [0.3, 0.45, 0.6, 0.75, 0.85], agreement 0.6

| Prompt | Calibrated threshold | Mask AP | Mask AP50 | Box AP50 | F1 @0.5 (plain SAM 3) | F1 @calibrated (BotFlow) | 95% CI | Mask IoU of matches | s / image | Peak GPU MB |
|---|---|---|---|---|---|---|---|---|---|---|
| taxi | 0.75 | 0.835 | 0.919 | 0.908 | 0.832 | 0.844 | [0.7902, 0.8854] | 0.936 | 0.684 | 2749 |
| taxi car | refused (no_acceptable_threshold) | 0.819 | 0.899 | 0.890 | 0.664 | no labels |  | 0.932 | 0.689 | 2782 |

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

Calibration (vlm, 86.5 s): 0.3: 3/5, 0.45: 3/5, 0.6: 3/5, 0.75: 4/5, 0.85: 2/5 -> calibrated

Judge audit (25 images, 62 masks vs ground truth): accuracy 0.7778, caught 7 of 16 wrong masks, wrongly rejected 3 real ones, unsure 8.

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

Calibration (vlm, 156.2 s): 0.3: 1/5, 0.45: 1/5, 0.6: 1/5, 0.75: 2/5, 0.85: 2/5 -> no_acceptable_threshold

Judge audit (25 images, 88 masks vs ground truth): accuracy 0.7973, caught 25 of 35 wrong masks, wrongly rejected 5 real ones, unsure 14.

