# API Reference

> Version generated on 2025-07-04

This document provides an exhaustive overview of the public surface of the repository – command-line entry-points, Python modules, classes, and helper functions.  Each symbol documented below is considered _public_ and stable.  Internal/private helpers (whose names start with an underscore or are clearly implementation details) are **not** guaranteed to stay unchanged.

---

## Table of Contents

1. Quick-Start
2. Command-Line Interface (`main.py`)
3. Python Packages & Modules
   1. `nets` – Model Architectures
   2. `utils.dataset` – Data-loading & Augmentation
   3. `utils.util` – Utility Helpers (losses, metrics, EMA, etc.)
4. Complete Type Map

---

## 1. Quick-Start

```bash
# ❶ Training from scratch on COCO-format data
env WORLD_SIZE=1 \
python main.py \
    --train \
    --input-size 640 \
    --batch-size 32 

# ❷ Single-GPU evaluation of a trained model
python main.py --test

# ❸ Programmatic inference inside Python
from nets.nn import yolo_v8_n
import torch, cv2

model = yolo_v8_n(num_classes=80).eval().cuda().half()
weights = torch.load('weights/best.pt', map_location='cuda')
model.load_state_dict(weights['model'])

img = cv2.imread('demo.jpg')[:, :, ::-1]   # BGR→RGB
t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda() / 255
with torch.no_grad():
    preds = model(t)
print(preds.shape)   # (1, 6, N) => (x, y, w, h, conf, cls)
```

---

## 2. Command-Line Interface – `main.py`

### Synopsis

```text
python main.py [--train] [--test] [options]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input-size` | int | `640` | Square spatial size fed to the network. Images are padded/resized to this resolution. |
| `--batch-size` | int | `32` | Mini-batch size per GPU. |
| `--epochs` | int | `500` | Number of training epochs. |
| `--train` | flag | *False* | Enable the training loop. |
| `--test` | flag | *False* | Run evaluation on the validation split. |
| `--local_rank` | int | `0` | Automatically assigned by PyTorch DDP; **do not set manually**. |

### Environment Variables

* `WORLD_SIZE` – number of GPUs when running in Distributed-Data-Parallel (DDP) mode.
* `LOCAL_RANK` – the rank index automatically injected by `torchrun`.

### Examples

```bash
# Single-GPU training
python main.py --train

# 4-GPU distributed training
torchrun --nproc_per_node 4 main.py --train --batch-size 16

# Evaluation only
python main.py --test --input-size 640
```

---

## 3. Python Packages & Modules

### 3.1 `nets` – Model Architectures

#### `nets.nn.yolo_v8_n(num_classes: int = 80) -> torch.nn.Module`
Return the **YOLOv8-Nano** architecture pre-configured for `num_classes` classes.

Other factory helpers:

* `yolo_v8_s`, `yolo_v8_m`, `yolo_v8_l`, `yolo_v8_x` – Small, Medium, Large, and X-Large variants following the same signature.

All functions return a `YOLO` module (see below) with loaded architectural weights ( **randomly initialized – not pre-trained** ).

##### Example

```python
from nets.nn import yolo_v8_n
model = yolo_v8_n(num_classes=20)  # e.g. Pascal VOC
params = sum(p.numel() for p in model.parameters())
print(f"Model size: {params/1e6:.2f} M")
```

---

#### `class YOLO(width: List[int], depth: List[int], num_classes: int)`
High-level detection network composed of three sub-modules:

1. `DarkNet` backbone – feature extraction.
2. `DarkFPN` neck – feature pyramid aggregation.
3. `Head` – distribution focal loss head that predicts bounding boxes and class scores.

`forward(x: torch.Tensor) -> torch.Tensor`  
Input: tensor with shape `(N, 3, H, W)`  
Output during *eval*: `(N, 6, num_predictions)` where each vector is `(center_x, center_y, width, height, confidence, class)` in **pixel** units. During *training* the head returns raw multi-scale feature maps instead.

`fuse() -> YOLO` – Merge BatchNorm into Conv layers for faster inference.

---

#### `class Head`
Utility head internally used by `YOLO`.  Public attributes that might be useful when post-processing predictions:

* `stride` – tensor of strides (`[8, 16, 32]` by default). The absolute coordinates returned by `YOLO.forward` are already multiplied by this factor, so you rarely need it explicitly.

---

### 3.2 `utils.dataset` – Data-loading & Augmentation

#### `class Dataset(filenames: List[str], input_size: int, params: Dict[str, Any], augment: bool)`
`torch.utils.data.Dataset` implementation compatible with COCO-style YOLO labels.

| Attribute | Description |
|-----------|-------------|
| `labels` | List of `(n_objects, 5)` numpy arrays `[cls, x, y, w, h]` in normalized **xywh** format. |
| `shapes` | Original image shapes before resizing – used for metric scaling. |
| `mosaic` / `augment` | Run-time flags controlling status of advanced augmentations. |

`__getitem__(index)` returns a tuple `(sample, target, shape_info)` where

* `sample` – float tensor `(3, input_size, input_size)` 0-255 range (later normalized by caller).
* `target` – tensor `(n_objects, 6)` as `[img_idx, cls, x, y, w, h]` (_normalized xywh_).
* `shape_info` – tuple capturing original dim scaling, required for mAP computation.

##### Example

```python
from torch.utils.data import DataLoader
from utils.dataset import Dataset

with open('train.txt') as f:
    paths = [l.strip() for l in f]

params = yaml.safe_load(open('utils/args.yaml'))
trainset = Dataset(paths, input_size=640, params=params, augment=True)
loader = DataLoader(trainset, batch_size=8, shuffle=True,
                    collate_fn=Dataset.collate_fn, num_workers=4)
```

###### Key Augmentation Utilities

* `random_perspective` – 3-dof affine + shear warp.
* `mix_up` – MixUp augmentation.
* `augment_hsv` – Random HSV jitter.

These helpers live in the same module and can be reused independently.

---

### 3.3 `utils.util` – Generic Utilities & Losses

Below is a non-exhaustive list of the most frequently used public helpers.  All reside in `utils.util`.

| Function / Class | Purpose |
|------------------|---------|
| `setup_seed()` | Reproducible robot-like determinism across Python, NumPy, and PyTorch. |
| `setup_multi_processes()` | Safe multi-processing defaults: fork start-method, 1 × OMP/MKL threads, and OpenCV single-thread. |
| `make_anchors(features, strides, offset=0.5)` | Convert feature maps to grid anchor centers (returned as `(N, 2)` tensor). |
| `non_max_suppression(pred, conf_th=0.25, iou_th=0.45)` | Pure-PyTorch NMS producing `(n, 6)` `[x1,y1,x2,y2,conf,cls]`. |
| `box_iou(a, b)` | Vectorized IoU on `(x1,y1,x2,y2)` tensors. |
| `smooth(y, frac=0.05)` | Simple 1-D running-average smoother (useful for metric curves). |
| `compute_ap(...)` | COCO-style AP/mAP computation given TP/FN vectors. |
| `EMA(model, decay=0.9999)` _(class)_ | Exponential Moving Average wrapper; call `update(model)` after every optimizer step. |
| `AverageMeter()` _(class)_ | Streaming metric aggregator with `.update(value, n)` and `.avg` attribute. |
| `ComputeLoss(model, params)` _(class)_ | End-to-end loss combining **Task-Aligned** matching, IoU loss, class BCE, and Distribution Focal Loss. Callable – returns scalar PyTorch loss. |

##### Example – Manual EMA

```python
from utils.util import EMA
ema = EMA(model)
for epoch in range(E):
    for samples, targets in loader:
        loss = model(samples)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        ema.update(model)   # <- keep the moving average fresh
```

---

## 4. Complete Type Map

Below is a quick reference for tensor shapes encountered across the codebase ( *B = batch size* ).

| Symbol | Shape | When |
|--------|-------|------|
| `images` | `(B, 3, H, W)` | raw input to `YOLO.forward` |
| `raw_head_out` | list of 3 tensors: `(B, C_i, H_i, W_i)` | training mode – feature maps at strides 8/16/32 |
| `preds` | `(B, 6, N)` | eval mode – predictions after anchor decoding & sigmoid |
| `targets` | `(M, 6)` | loader target – `[batch_idx, cls, x, y, w, h]` normalized |
| `loss` | scalar | output of `ComputeLoss` |

---

_This documentation is auto-generated and can be regenerated at any time._