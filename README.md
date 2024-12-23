# Deep Learning Applications

This repository contains a collection of deep learning applications for various computer vision tasks, including anomaly detection, image segmentation, object classification, OCR, object detection, and object counting.

# Environments

```
torch==1.13.1+cu116
pip install einops==0.6.0
pip install torchmetrics==0.10.3
pip install timm==0.6.12
pip install freia
pip install onnxruntime==1.14.1
pip install efficientnet-pytorch
pip install lmdb==1.4.0
pip install ipdb==0.13.11
pip install wandb==0.13.10
```

## Applications

- `anomaly`: detect anomalies in images using unsupervised, semi-supervised, or unsupervised learning methods.
- `segmentation`: an application for segmenting images using deep learning models
- `classification`: an application for classifying images using deep learning models
- `ocr`: an application for performing optical character recognition (OCR) on text in images using deep learning models
- `object_detection`: an application for detecting objects in images using deep learning models
- `object_counting`: an application for counting objects in images using deep learning models

## How to Train and Test

# Run
# Anomaly
Supervised:
- `BGAD`:
```bash
python main.py --task_type anomaly --model_type supervised --model_name bgad --yaml_config configs/anomaly/supervised/bgad/bottle.yaml
```
Semi-supervised
- `AESeg`:
```bash
python main.py --task_type anomaly --model_type semisupervised --model_name aeseg --yaml_config configs/anomaly/semisupervised/aeseg/bottle.yaml
```
- `CDO`:
```bash
python main.py --task_type anomaly --model_type semisupervised --model_name cdo --yaml_config configs/anomaly/semisupervised/cdo/bottle.yaml
```
- `DiffusionAD`:
```bash
python main.py --task_type anomaly --model_type semisupervised --model_name diffusionad --yaml_config configs/anomaly/semisupervised/diffusionad/bottle.yaml
```
- `DRAEM`:
```bash
python main.py --task_type anomaly --model_type semisupervised --model_name draem --yaml_config configs/anomaly/semisupervised/draem/bottle.yaml
```
- `MemSeg`:
```bash
python main.py --task_type anomaly --model_type semisupervised --model_name memseg --yaml_config configs/anomaly/semisupervised/memseg/bottle.yaml
```
- `PRNet`:
```bash
python main.py --task_type anomaly --model_type semisupervised --model_name prnet --yaml_config configs/anomaly/semisupervised/prnet/bottle.yaml
```
Unsupervised
- `ADFA`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name adfa --yaml_config configs/anomaly/unsupervised/adfa/bottle.yaml
```
- `CFA`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name cfa --yaml_config configs/anomaly/unsupervised/cfa/bottle.yaml
```
- `CFlow`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name cflow --yaml_config configs/anomaly/unsupervised/cflow/bottle.yaml
```
- `DFR`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name dfr --yaml_config configs/anomaly/unsupervised/dfr/bottle.yaml
```
- `DMAD`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name dmad --yaml_config configs/anomaly/unsupervised/dmad/bottle.yaml
```
- `EfficientAD`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name efficientad --yaml_config configs/anomaly/unsupervised/efficientad/bottle.yaml
```
- `FAPM`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name fapm --yaml_config configs/anomaly/unsupervised/fapm/bottle.yaml
```
- `FastFlow`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name fastflow --yaml_config configs/anomaly/unsupervised/fastflow/bottle.yaml
```
- `IKD`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name ikd --yaml_config configs/anomaly/unsupervised/ikd/bottle.yaml
```
- `MMR`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name mmr --yaml_config configs/anomaly/unsupervised/mmr/bottle.yaml
```
- `PatchCore`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name patchcore --yaml_config configs/anomaly/unsupervised/patchcore/bottle.yaml
```
- `PatchCoreRS`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name patchcorers --yaml_config configs/anomaly/unsupervised/patchcorers/bottle.yaml
```
- `ReContrast`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name recontrast --yaml_config configs/anomaly/unsupervised/recontrast/bottle.yaml
```
- `SAPatchCore`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name sapatchcore --yaml_config configs/anomaly/unsupervised/sapatchcore/bottle.yaml
```
- `STFPM`:
```bash
python main.py --task_type anomaly --model_type unsupervised --model_name stfpm --yaml_config configs/anomaly/unsupervised/stfpm/bottle.yaml
```
# Classification
Supervised
- `ConvMixer`: 
```bash
python main.py --task_type classify --model_type supervised --model_name convmixer --yaml_config configs/classify/supervised/convmixer/garbage.yaml
```
- `CSDarkNet`: 
```bash
python main.py --task_type classify --model_type supervised --model_name csdarknet --yaml_config configs/classify/supervised/csdarknet/garbage.yaml
```
- `Darknet19`: 
```bash
python main.py --task_type classify --model_type supervised --model_name csdarknet --yaml_config configs/classify/supervised/csdarknet/garbage.yaml
```
- `ElanCSPNet`: 
```bash
python main.py --task_type classify --model_type supervised --model_name elancspnet --yaml_config configs/classify/supervised/elancspnet/garbage.yaml
```
- `Elannet`: 
```bash
python main.py --task_type classify --model_type supervised --model_name elannet --yaml_config configs/classify/supervised/elannet/garbage.yaml
```
- `Elannetv2`: 
```bash
python main.py --task_type classify --model_type supervised --model_name elannetv2 --yaml_config configs/classify/supervised/elannetv2/garbage.yaml
```
- `MyModel`: 
```bash
python main.py --task_type classify --model_type supervised --model_name mymodel --yaml_config configs/classify/supervised/mymodel/garbage.yaml
```
- `VPT`: 
```bash
python main.py --task_type classify --model_type supervised --model_name vpt --yaml_config configs/classify/supervised/vpt/garbage.yaml
```
# Edge Detection
Supervised
- `CATS`:
```bash
python main.py --task_type edgedetection --model_type supervised --model_name cats --yaml_config configs/edgedetection/supervised/cats/bsds.yaml
```
- `CTFN`:
```bash
python main.py --task_type edgedetection --model_type supervised --model_name ctfn --yaml_config configs/edgedetection/supervised/ctfn/bsds.yaml
```
- `Dexined`:
```bash
python main.py --task_type edgedetection --model_type supervised --model_name dexined --yaml_config configs/edgedetection/supervised/dexined/bsds.yaml
```
- `LDC`:
```bash
python main.py --task_type edgedetection --model_type supervised --model_name ldc --yaml_config configs/edgedetection/supervised/ldc/bsds.yaml
```
# Object Counting
Fewshot
- `HLSafeCount`:
```bash
python main.py --task_type objectcounting --model_type fewshot --model_name hlsafecount --yaml_config configs/objectcounting/fewshot/hlsafecount/custom.yaml
```
# Object Detection
Supervised
- `Yolov1`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov1 --yaml_config configs/objectdetection/supervised/yolov1/pascal.yaml
```
- `Yolov2`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov2 --yaml_config configs/objectdetection/supervised/yolov2/pascal.yaml
```
- `Yolov3`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov3 --yaml_config configs/objectdetection/supervised/yolov3/pascal.yaml
```
- `Yolov4`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov4 --yaml_config configs/objectdetection/supervised/yolov4/pascal.yaml
```
- `Yolov5`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov5 --yaml_config configs/objectdetection/supervised/yolov5/pascal.yaml
```
- `Yolov7`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov7 --yaml_config configs/objectdetection/supervised/yolov7/pascal.yaml
```
- `Yolov8`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolov8 --yaml_config configs/objectdetection/supervised/yolov8/pascal.yaml
```
- `Yolox`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yolox --yaml_config configs/objectdetection/supervised/yolox/pascal.yaml
```
- `Yoloxv2`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name yoloxv2 --yaml_config configs/objectdetection/supervised/yoloxv2/pascal.yaml
```
- `DeTR`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name detr --yaml_config configs/objectdetection/supervised/detr/pascal.yaml
```
- `FreeYolo`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name freeyolo --yaml_config configs/objectdetection/supervised/freeyolo/pascal.yaml
```
- `FreeYolov2`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name freeyolov2 --yaml_config configs/objectdetection/supervised/freeyolov2/pascal.yaml
```
- `RTCDet`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name rtcdet --yaml_config configs/objectdetection/supervised/rtcdet/pascal.yaml
```
- `RTCDetv2`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name rtcdetv2 --yaml_config configs/objectdetection/supervised/rtcdetv2/pascal.yaml
```
- `DeTR`:
```bash
python main.py --task_type objectdetection --model_type supervised --model_name detr --yaml_config configs/objectdetection/supervised/detr/pascal.yaml
```
# Text Detection
Supervised
- `DCLNet`:
```bash
python main.py --task_type textdetection --model_type supervised --model_name dclnet --yaml_config configs/textdetection/supervised/dclnet/synth.yaml
```
- `EAST`:
```bash
python main.py --task_type textdetection --model_type supervised --model_name east --yaml_config configs/textdetection/supervised/east/synth.yaml
```
- `PAN`:
```bash
python main.py --task_type textdetection --model_type supervised --model_name pan --yaml_config configs/textdetection/supervised/pan/synth.yaml
```
- `PSENet`:
```bash
python main.py --task_type textdetection --model_type supervised --model_name psenet --yaml_config configs/textdetection/supervised/psenet/synth.yaml
```
Weakly-supervised
- `CRAFT`:
```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name craft --yaml_config configs/textdetection/weaklysupervised/craft/synth.yaml
```
- `RTTDet`:
```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name rttdet --yaml_config configs/textdetection/weaklysupervised/rttdet/synth.yaml
```
- `RTTDetv2`:
```bash
python main.py --task_type textdetection --model_type weaklysupervised --model_name rttdetv2 --yaml_config configs/textdetection/weaklysupervised/rttdetv2/synth.yaml
```
# Text Recognition
Supervised
- `RCNN`:
```bash
python main.py --task_type textrecognition --model_type supervised --model_name crnn --yaml_config configs/textrecognition/supervised/crnn/synth90k.yaml
```
## Demo

### Test .pth & .onnx Model
```
voila "test_model.ipynb" --port 8866 --Voila.ip 127.0.0.1 --show_tracebacks=True
```

<p align="left">
  <img src=assets/test_model.gif width="100%" />
</p>


## Installation

To use these applications, you need to have Python and several Python packages installed. To install the required packages, use the following command:
```
pip install -r requirements.txt"
```

## Usage

To use the applications, clone this repository to your local machine and follow the instructions in the `README.md` file in each application folder.

