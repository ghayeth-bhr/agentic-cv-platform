# Benchmarks

How well does BotFlow label a dataset, and does its agent make good decisions?
These benchmarks answer that with hand-labeled ground truth, on one consumer
GPU (RTX 4050 Laptop, 6 GB), using the same code the app runs.

## What is measured

The models are well known (SAM 3, Qwen3-VL), so their generic scores are not
repeated here. What is measured is BotFlow's own work on top of them:

| Question | Metric |
|---|---|
| How good are the masks BotFlow produces? | COCO mask AP, AP50, and mask IoU of each matched object |
| Does the agent's calibration pick a good threshold? | Precision, recall and F1 at the calibrated threshold, compared with plain SAM 3 at its default 0.5 and with the best threshold in hindsight |
| Can the vision-language judge be trusted? | Each mask the judge sees is compared with the ground truth (judge audit) |
| Does it run on a laptop? | Seconds per image and peak GPU memory |

Matching is one-to-one by score at mask IoU >= 0.5. F1 intervals are 95%
bootstrap intervals over images. Calibration uses the app's own grid
(0.3, 0.45, 0.6, 0.75, 0.85), its 60% agreement rule, and 5 sample images that
are then **excluded from scoring**, because a user's dataset is never judged
on the images the agent tuned on.

## Datasets

| | Images | Masks | Ground truth |
|---|---|---|---|
| **Open Images taxis** (main) | 98 | 162 | [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) validation and test masks, drawn entirely by hand by professional annotators. Only images where every taxi is masked are kept. One mask was removed after review: a police car labeled as a taxi in `134f39599bbc5848.jpg`. Taxis from many cities, most of them not yellow. |
| **Street footage** (in-domain) | 42 | 55 | Frames from the demo video, labeled by BotFlow and then corrected by hand. Because the labels started from BotFlow's own output, this set favors BotFlow; it shows behavior on real CCTV-style footage, not an independent score. |

Prompts are `taxi` and `taxi car`. Colour words such as "yellow taxi" are
deliberately not used, since most Open Images taxis are not yellow.

## Results

### Mask quality

| Dataset | Prompt | Mask AP | Mask AP50 | Mask IoU of matched taxis |
|---|---|---|---|---|
| Open Images | taxi | **0.835** | **0.919** | **0.935** |
| Open Images | taxi car | 0.819 | 0.899 | 0.932 |
| Street footage | taxi | 0.975 | 0.993 | 0.966 |
| Street footage | taxi car | 0.951 | 0.962 | 0.967 |

When BotFlow finds a taxi, its mask overlaps the hand-drawn one by about 93%
on independent photos.

### Calibration: the agent's threshold choice

The judge that scores each calibration threshold was changed after the first
benchmark run. The old judge looked at one image with every mask drawn on it;
the new one looks at each mask on its own crop and asks whether that object is
the target (`AGENTIC_CV_VLM_JUDGE=instance`, now the default).

| Dataset | Prompt | Plain SAM 3 (0.5) | Old judge | New judge | Best threshold in hindsight |
|---|---|---|---|---|---|
| Open Images | taxi | F1 0.832 | 0.45, F1 0.824 | **0.75, F1 0.844** | 0.6, F1 0.861 |
| Open Images | taxi car | F1 0.664 | 0.6, F1 0.723 | **refused the prompt** | 0.8, F1 0.852 |
| Street footage | taxi | F1 0.912 | 0.45, F1 0.889 | 0.45, F1 0.889 | 0.7, F1 0.972 |
| Street footage | taxi car | F1 0.619 | **0.3, F1 0.424** (141 wrong cars) | **refused the prompt** | 0.8, F1 0.891 |

- **"taxi car" is a bad prompt**: it also marks ordinary cars. The old judge
  accepted it, and on the street footage would have labeled 141 wrong cars.
  The new judge refuses every threshold, so the run stops and asks the user
  instead of building a wrong dataset.
- With a good prompt, the new judge does as well as or better than plain SAM 3
  on Open Images, and the same as the old judge on the street footage.
- **Still open:** the calibrated threshold is often lower than the best one
  (0.45 against 0.7 on the street footage). The agreement rule picks the middle
  of the thresholds that tie for best agreement, and with 5 samples many tie.

### Judge audit

Each mask the judge would see (score >= 0.3, top 6 per image) is compared with
the ground truth: a mask is right when it overlaps a hand-drawn mask at IoU >= 0.5.

| Dataset | Prompt | Masks | Agrees with ground truth | Wrong masks caught | Real taxis rejected |
|---|---|---|---|---|---|
| Open Images (25 images) | taxi | 62 | 78% | 7 of 16 | 3 of 38 |
| Open Images (25 images) | taxi car | 88 | 80% | 25 of 35 | 5 of 39 |
| Street footage (15 images) | taxi | 30 | 90% | 3 of 6 | 0 of 23 |
| Street footage (15 images) | taxi car | 61 | 71% | 21 of 37 | 1 of 22 |

The judge rarely rejects a real taxi. It misses about half of the wrong masks
when they are duplicates or fragments of a real taxi: the crop shows a taxi,
but the mask does not count as a separate correct object.

### Speed and memory

| | Open Images | Street footage |
|---|---|---|
| SAM 3, seconds per image (median) | 0.63 to 0.69 | 0.67 to 0.80 |
| Peak GPU memory (SAM 3, float16) | 2,749 to 2,782 MiB | 2,144 to 2,163 MiB |
| Calibration, 5 images x 5 thresholds | 62 to 156 s | 69 to 130 s |

Calibration time is mostly the VLM judging. It was measured with another
BotFlow server holding about 1.3 GB of the same GPU, so it is an upper bound.

## Files

```
benchmarks/
  openimages-taxi/{scene,instance}/results.{json,md}   # scene = old judge, instance = new judge
  street-taxi/{scene,instance}/results.{json,md}
```

Each `results.json` holds every number above, the calibration attempts per
threshold, the excluded calibration images, and the environment.

## Reproduce

Needs the `ml` extra, an NVIDIA GPU, and Ollama with `qwen3-vl:4b`.

```bash
# 1. Build the Open Images taxi set (downloads ~210 MB of mask archives).
uv run python scripts/build_openimages_taxi_set.py .cache/openimages data/openimages-taxi

# 2. Import it in BotFlow (Datasets -> Import dataset), review it, remove the
#    police-car mask in 134f39599bbc5848.jpg, approve all, and export COCO.

# 3. Run the benchmark on the export, once per judge.
AGENTIC_CV_VLM_JUDGE=instance uv run python scripts/benchmark_segmentation.py \
  --gt openimages-taxi-approved-coco.zip --prompt taxi --prompt "taxi car" \
  --judge-audit 25 --out benchmarks/openimages-taxi/instance
AGENTIC_CV_VLM_JUDGE=scene uv run python scripts/benchmark_segmentation.py \
  --gt openimages-taxi-approved-coco.zip --prompt taxi --prompt "taxi car" \
  --out benchmarks/openimages-taxi/scene
```

SAM 3 inference is deterministic on the same GPU; the VLM judge runs at
Ollama's defaults, so a verdict can occasionally differ between runs.
