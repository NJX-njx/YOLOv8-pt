# YOLOv8 PyTorch Implementation - API Documentation

## Overview
This is a comprehensive API documentation for the YOLOv8 object detection model implementation in PyTorch. The implementation includes complete training and inference pipelines with support for distributed training, data augmentation, and various model sizes.

## Table of Contents
1. [Main Module](#main-module)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Utility Functions](#utility-functions)
4. [Dataset Management](#dataset-management)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

---

## Main Module

### `main.py`

The main entry point for training and testing the YOLOv8 model.

#### Functions

##### `learning_rate(args, params)`
Creates a learning rate scheduler function.

**Parameters:**
- `args`: Command line arguments object
- `params`: Dictionary of hyperparameters

**Returns:**
- `fn`: Learning rate function that takes epoch as input

**Example:**
```python
lr_fn = learning_rate(args, params)
current_lr = lr_fn(epoch)
```

##### `train(args, params)`
Main training function that handles the complete training pipeline.

**Parameters:**
- `args`: Command line arguments containing training configuration
- `params`: Dictionary of hyperparameters from YAML config

**Features:**
- Distributed training support (DDP)
- Exponential Moving Average (EMA)
- Mixed precision training
- Automatic gradient scaling
- Model checkpointing
- mAP evaluation during training

**Example:**
```python
train(args, params)
```

##### `test(args, params, model=None)`
Evaluation function that computes mAP metrics on validation set.

**Parameters:**
- `args`: Command line arguments
- `params`: Dictionary of hyperparameters
- `model`: Optional pre-loaded model (defaults to loading from weights/best.pt)

**Returns:**
- `tuple`: (mAP@50, mAP@50:95) scores

**Example:**
```python
map50, mean_ap = test(args, params)
print(f"mAP@50: {map50:.3f}, mAP@50:95: {mean_ap:.3f}")
```

##### `main()`
Entry point function that parses command line arguments and starts training or testing.

**Command Line Arguments:**
- `--input-size`: Input image size (default: 640)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 500)
- `--train`: Flag to enable training mode
- `--test`: Flag to enable testing mode

**Example:**
```bash
python main.py --train --batch-size 16 --epochs 300
python main.py --test
```

---

## Neural Network Architecture

### `nets/nn.py`

Contains all neural network components and model architectures.

#### Core Components

##### `Conv` Class
Basic convolutional block with BatchNorm and SiLU activation.

**Parameters:**
- `in_ch`: Input channels
- `out_ch`: Output channels
- `k`: Kernel size (default: 1)
- `s`: Stride (default: 1)
- `p`: Padding (default: None, auto-calculated)
- `d`: Dilation (default: 1)
- `g`: Groups (default: 1)

**Methods:**
- `forward(x)`: Standard forward pass
- `fuse_forward(x)`: Fused forward pass (after conv-bn fusion)

**Example:**
```python
conv = Conv(in_ch=64, out_ch=128, k=3, s=2)
output = conv(input_tensor)
```

##### `Residual` Class
Residual connection block with optional skip connection.

**Parameters:**
- `ch`: Number of channels
- `add`: Whether to add skip connection (default: True)

**Example:**
```python
residual = Residual(ch=256, add=True)
output = residual(input_tensor)
```

##### `CSP` Class
Cross Stage Partial connection block.

**Parameters:**
- `in_ch`: Input channels
- `out_ch`: Output channels
- `n`: Number of residual blocks (default: 1)
- `add`: Whether to use skip connections in residual blocks (default: True)

**Example:**
```python
csp = CSP(in_ch=128, out_ch=256, n=3, add=True)
output = csp(input_tensor)
```

##### `SPP` Class
Spatial Pyramid Pooling block.

**Parameters:**
- `in_ch`: Input channels
- `out_ch`: Output channels
- `k`: Kernel size for max pooling (default: 5)

**Example:**
```python
spp = SPP(in_ch=512, out_ch=512, k=5)
output = spp(input_tensor)
```

#### Model Architecture

##### `DarkNet` Class
Backbone network for feature extraction.

**Parameters:**
- `width`: List of channel widths for each stage
- `depth`: List of block depths for each stage

**Methods:**
- `forward(x)`: Returns features from P3, P4, P5 levels

**Example:**
```python
backbone = DarkNet(width=[3, 16, 32, 64, 128, 256], depth=[1, 2, 2])
p3, p4, p5 = backbone(input_tensor)
```

##### `DarkFPN` Class
Feature Pyramid Network for multi-scale feature fusion.

**Parameters:**
- `width`: List of channel widths
- `depth`: List of block depths

**Methods:**
- `forward(x)`: Takes (P3, P4, P5) features and returns fused features

**Example:**
```python
fpn = DarkFPN(width=[3, 16, 32, 64, 128, 256], depth=[1, 2, 2])
h2, h4, h6 = fpn((p3, p4, p5))
```

##### `Head` Class
Detection head for object detection and classification.

**Parameters:**
- `nc`: Number of classes (default: 80)
- `filters`: Tuple of filter sizes for each detection layer

**Methods:**
- `forward(x)`: Forward pass through detection head
- `initialize_biases()`: Initialize biases for better training convergence

**Example:**
```python
head = Head(nc=80, filters=(256, 512, 1024))
detections = head([feat1, feat2, feat3])
```

##### `YOLO` Class
Complete YOLOv8 model combining backbone, FPN, and head.

**Parameters:**
- `width`: Channel width configuration
- `depth`: Block depth configuration
- `num_classes`: Number of object classes

**Methods:**
- `forward(x)`: Complete forward pass
- `fuse()`: Fuse Conv-BN layers for inference optimization

**Example:**
```python
model = YOLO(width=[3, 16, 32, 64, 128, 256], depth=[1, 2, 2], num_classes=80)
predictions = model(input_tensor)
```

#### Model Variants

##### `yolo_v8_n(num_classes=80)`
YOLOv8 Nano model - smallest and fastest variant.

**Parameters:**
- `num_classes`: Number of object classes (default: 80)

**Returns:**
- `YOLO`: Configured YOLOv8n model

**Example:**
```python
model = yolo_v8_n(num_classes=80)
```

##### `yolo_v8_s(num_classes=80)`
YOLOv8 Small model - balanced speed and accuracy.

##### `yolo_v8_m(num_classes=80)`
YOLOv8 Medium model - higher accuracy than small.

##### `yolo_v8_l(num_classes=80)`
YOLOv8 Large model - high accuracy model.

##### `yolo_v8_x(num_classes=80)`
YOLOv8 Extra Large model - highest accuracy variant.

#### Utility Functions

##### `pad(k, p=None, d=1)`
Calculate padding for convolution operations.

**Parameters:**
- `k`: Kernel size
- `p`: Padding (if None, auto-calculated)
- `d`: Dilation

**Returns:**
- `int`: Calculated padding value

##### `fuse_conv(conv, norm)`
Fuse convolution and batch normalization layers for inference optimization.

**Parameters:**
- `conv`: Convolution layer
- `norm`: Batch normalization layer

**Returns:**
- `torch.nn.Conv2d`: Fused convolution layer

---

## Utility Functions

### `utils/util.py`

Contains essential utility functions for training, evaluation, and data processing.

#### Setup Functions

##### `setup_seed()`
Setup random seeds for reproducible training.

**Example:**
```python
setup_seed()  # Sets seeds for random, numpy, and torch
```

##### `setup_multi_processes()`
Configure multi-processing environment for optimal training performance.

**Example:**
```python
setup_multi_processes()  # Optimizes threading and multiprocessing
```

#### Data Processing Functions

##### `scale(coords, shape1, shape2, ratio_pad=None)`
Scale coordinates from one image size to another.

**Parameters:**
- `coords`: Bounding box coordinates tensor
- `shape1`: Source image shape
- `shape2`: Target image shape
- `ratio_pad`: Optional ratio and padding information

**Returns:**
- `torch.Tensor`: Scaled coordinates

**Example:**
```python
scaled_coords = scale(coords, (640, 640), (480, 480))
```

##### `make_anchors(x, strides, offset=0.5)`
Generate anchor points from feature maps.

**Parameters:**
- `x`: List of feature maps
- `strides`: List of stride values
- `offset`: Anchor offset (default: 0.5)

**Returns:**
- `tuple`: (anchor_points, stride_tensor)

**Example:**
```python
anchors, strides = make_anchors(feature_maps, [8, 16, 32])
```

##### `box_iou(box1, box2)`
Calculate Intersection over Union (IoU) between two sets of boxes.

**Parameters:**
- `box1`: First set of boxes (N, 4)
- `box2`: Second set of boxes (M, 4)

**Returns:**
- `torch.Tensor`: IoU matrix (N, M)

**Example:**
```python
iou_matrix = box_iou(pred_boxes, gt_boxes)
```

##### `non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45)`
Apply Non-Maximum Suppression to filter overlapping detections.

**Parameters:**
- `prediction`: Model predictions
- `conf_threshold`: Confidence threshold (default: 0.25)
- `iou_threshold`: IoU threshold for NMS (default: 0.45)

**Returns:**
- `list`: List of filtered detections for each image

**Example:**
```python
filtered_detections = non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4)
```

#### Evaluation Functions

##### `compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16)`
Compute Average Precision (AP) metrics.

**Parameters:**
- `tp`: True positives array
- `conf`: Confidence scores
- `pred_cls`: Predicted classes
- `target_cls`: Target classes
- `eps`: Small epsilon value for numerical stability

**Returns:**
- `tuple`: (tp, fp, precision, recall, map50, mean_ap)

**Example:**
```python
tp, fp, precision, recall, map50, mean_ap = compute_ap(tp, conf, pred_cls, target_cls)
```

##### `smooth(y, f=0.05)`
Apply smoothing filter to data.

**Parameters:**
- `y`: Input data array
- `f`: Smoothing factor (default: 0.05)

**Returns:**
- `numpy.ndarray`: Smoothed data

#### Training Utilities

##### `EMA` Class
Exponential Moving Average for model parameters.

**Parameters:**
- `model`: PyTorch model
- `decay`: Decay rate (default: 0.9999)
- `tau`: Tau parameter for decay ramp (default: 2000)
- `updates`: Number of updates (default: 0)

**Methods:**
- `update(model)`: Update EMA parameters

**Example:**
```python
ema = EMA(model, decay=0.9999)
ema.update(model)  # Call after each training step
```

##### `AverageMeter` Class
Track running averages of metrics.

**Methods:**
- `update(value, n)`: Update with new value
- `avg`: Current average
- `sum`: Current sum
- `num`: Number of samples

**Example:**
```python
loss_meter = AverageMeter()
loss_meter.update(loss_value, batch_size)
print(f"Average loss: {loss_meter.avg:.4f}")
```

##### `ComputeLoss` Class
Compute training loss including classification, box regression, and DFL losses.

**Parameters:**
- `model`: YOLO model
- `params`: Training parameters

**Methods:**
- `__call__(outputs, targets)`: Compute loss
- `assign(...)`: Task-aligned assignment of targets to anchors

**Example:**
```python
criterion = ComputeLoss(model, params)
loss = criterion(outputs, targets)
```

##### `clip_gradients(model, max_norm=10.0)`
Clip gradients to prevent gradient explosion.

**Parameters:**
- `model`: PyTorch model
- `max_norm`: Maximum gradient norm (default: 10.0)

**Example:**
```python
clip_gradients(model, max_norm=10.0)
```

##### `strip_optimizer(filename)`
Remove optimizer state from saved model to reduce file size.

**Parameters:**
- `filename`: Path to model file

**Example:**
```python
strip_optimizer('./weights/best.pt')
```

---

## Dataset Management

### `utils/dataset.py`

Handles dataset loading, preprocessing, and augmentation.

#### Main Dataset Class

##### `Dataset` Class
PyTorch Dataset class for YOLO training and validation.

**Parameters:**
- `filenames`: List of image file paths
- `input_size`: Target input size
- `params`: Training parameters
- `augment`: Whether to apply augmentations

**Methods:**
- `__getitem__(index)`: Get a single sample
- `__len__()`: Get dataset length
- `load_image(i)`: Load and resize image
- `load_mosaic(index, params)`: Load mosaic augmentation
- `collate_fn(batch)`: Static method for batch collation
- `load_label(filenames)`: Static method for label loading

**Example:**
```python
dataset = Dataset(filenames, input_size=640, params=params, augment=True)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=Dataset.collate_fn)
```

#### Augmentation Functions

##### `random_perspective(samples, targets, params, border=(0, 0))`
Apply random perspective transformation with rotation, scale, translation, and shear.

**Parameters:**
- `samples`: Input images
- `targets`: Bounding box targets
- `params`: Augmentation parameters
- `border`: Border values

**Returns:**
- `tuple`: (augmented_samples, augmented_targets)

**Example:**
```python
aug_samples, aug_targets = random_perspective(samples, targets, params)
```

##### `mix_up(image1, label1, image2, label2)`
Apply MixUp augmentation by blending two images.

**Parameters:**
- `image1`, `image2`: Input images
- `label1`, `label2`: Corresponding labels

**Returns:**
- `tuple`: (mixed_image, combined_labels)

**Example:**
```python
mixed_img, mixed_labels = mix_up(img1, labels1, img2, labels2)
```

##### `augment_hsv(image, params)`
Apply HSV color space augmentation.

**Parameters:**
- `image`: Input image (modified in-place)
- `params`: HSV augmentation parameters

**Example:**
```python
augment_hsv(image, params)  # Modifies image in-place
```

##### `resize(image, input_size, augment)`
Resize image while maintaining aspect ratio with padding.

**Parameters:**
- `image`: Input image
- `input_size`: Target size
- `augment`: Whether augmentation is enabled

**Returns:**
- `tuple`: (resized_image, (ratio_x, ratio_y), (pad_w, pad_h))

**Example:**
```python
resized_img, ratio, pad = resize(image, 640, augment=True)
```

#### Coordinate Conversion Functions

##### `wh2xy(x, w=640, h=640, pad_w=0, pad_h=0)`
Convert from width-height format to x1y1x2y2 format.

**Parameters:**
- `x`: Bounding boxes in [x_center, y_center, width, height] format
- `w`, `h`: Image dimensions
- `pad_w`, `pad_h`: Padding values

**Returns:**
- `numpy.ndarray`: Boxes in [x1, y1, x2, y2] format

##### `xy2wh(x, w=640, h=640)`
Convert from x1y1x2y2 format to normalized center-width-height format.

**Parameters:**
- `x`: Bounding boxes in [x1, y1, x2, y2] format
- `w`, `h`: Image dimensions

**Returns:**
- `numpy.ndarray`: Normalized boxes in [x_center, y_center, width, height] format

#### Albumentations Integration

##### `Albumentations` Class
Wrapper for Albumentations library integration.

**Methods:**
- `__call__(image, label)`: Apply Albumentations transforms

**Example:**
```python
albu = Albumentations()
aug_image, aug_label = albu(image, label)
```

---

## Configuration

### `utils/args.yaml`

Configuration file containing all hyperparameters and settings.

#### Training Parameters

```yaml
lr0: 0.010                    # Initial learning rate
lrf: 0.010                    # Final learning rate factor
momentum: 0.937               # SGD momentum
weight_decay: 0.0005          # Weight decay
warmup_epochs: 3.0            # Warmup epochs
warmup_momentum: 0.8          # Warmup momentum
warmup_bias_lr: 0.1           # Warmup bias learning rate
```

#### Loss Parameters

```yaml
box: 7.5                      # Box loss weight
cls: 0.5                      # Classification loss weight
dfl: 1.5                      # DFL loss weight
```

#### Augmentation Parameters

```yaml
hsv_h: 0.015                  # HSV hue augmentation
hsv_s: 0.7                    # HSV saturation augmentation
hsv_v: 0.4                    # HSV value augmentation
degrees: 0.0                  # Rotation degrees
translate: 0.1                # Translation fraction
scale: 0.5                    # Scale augmentation
shear: 0.0                    # Shear augmentation
flip_ud: 0.0                  # Flip up-down probability
flip_lr: 0.5                  # Flip left-right probability
mosaic: 1.0                   # Mosaic probability
mix_up: 0.0                   # MixUp probability
```

#### Class Names

The configuration includes all 80 COCO class names (person, bicycle, car, etc.).

---

## Usage Examples

### Basic Training

```python
import argparse
from main import train, test
import yaml

# Setup arguments
args = argparse.Namespace(
    input_size=640,
    batch_size=16,
    epochs=100,
    local_rank=0,
    world_size=1,
    train=True,
    test=False
)

# Load configuration
with open('utils/args.yaml') as f:
    params = yaml.safe_load(f)

# Start training
train(args, params)
```

### Inference with Custom Model

```python
import torch
from nets.nn import yolo_v8_n
from utils.util import non_max_suppression
import cv2

# Load model
model = yolo_v8_n(num_classes=80)
model.load_state_dict(torch.load('weights/best.pt')['model'])
model.eval()

# Load and preprocess image
image = cv2.imread('image.jpg')
image = cv2.resize(image, (640, 640))
image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# Inference
with torch.no_grad():
    predictions = model(image)
    detections = non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4)

# Process detections
for det in detections[0]:
    x1, y1, x2, y2, conf, cls = det
    print(f"Class: {int(cls)}, Confidence: {conf:.3f}, Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### Custom Dataset Training

```python
from utils.dataset import Dataset
from torch.utils.data import DataLoader

# Prepare file list
train_files = ['path/to/image1.jpg', 'path/to/image2.jpg', ...]

# Create dataset
dataset = Dataset(train_files, input_size=640, params=params, augment=True)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4,
    collate_fn=Dataset.collate_fn
)

# Training loop
for samples, targets, shapes in dataloader:
    # Your training code here
    pass
```

### Model Evaluation

```python
from main import test
import yaml

# Load configuration
with open('utils/args.yaml') as f:
    params = yaml.safe_load(f)

# Setup evaluation arguments
args = argparse.Namespace(
    input_size=640,
    batch_size=8,
    local_rank=0,
    world_size=1
)

# Run evaluation
map50, mean_ap = test(args, params)
print(f"Results: mAP@50={map50:.3f}, mAP@50:95={mean_ap:.3f}")
```

### Distributed Training

```bash
# Single node, multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 main.py --train --batch-size 64

# Multiple nodes
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12355 main.py --train --batch-size 64
```

This comprehensive documentation covers all the major components, functions, and usage patterns in the YOLOv8 implementation. Each section includes detailed parameter descriptions, return values, and practical examples to help users understand and utilize the API effectively.