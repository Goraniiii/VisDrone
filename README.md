# VisDrone Object Detection & Multi-Object Tracking

VisDrone2019 데이터셋을 기반으로 객체 검출(Object Detection)과 다중 객체 추적(Multi-Object Tracking, MOT)을 수행한다.

Ultralytics YOLOv8을 기반으로 Detection(Task A) → Tracking(Task B) → Evaluation → 결과 시각화(Video)의 파이프라인을 거친다.

### 데이터셋 및 클래스 선정 이유

VisDrone 데이터셋은 드론 촬영 환경에서 수집된 영상으로 구성되어, 고도 변화, 시점 이동등에 대해 다양한 케이스를 포함하고 있다. 클래스는 VisDrone에서 제공하는 객체 중 보행자, 차량 계열을 중심으로 사용하였다. 해당 클래스들은 프레임 간 이동이 명확하고 tracking 결과를 시각적으로 확인하기 용이하여 선택하였다.


### 모델 선정

객체 탐지 및 추적을 위해 YOLO 계열 모델 + Tracking 기능을 사용하였다.
YOLO는 single-stage detector로 실시간 처리에 적합하며, VisDrone과 같이 객체 수가 많고 프레임 크기가 큰 환경에서도 비교적 안정적으로 동작한다.

### 결과
#### 성능 변화
* epoch 수가 증가함에 따라 객체 탐지의 안정성은 개선되었으나, 작은 객체나 멀리 있는 객체에 대해서는 큰 성능 향상이 관찰되지 않았다.
* 이미지 해상도를 지나치게 낮출 경우, 작은 객체가 탐지되지 않거나 tracking ID가 자주 변경되는 현상이 발생하였다.

#### 실패 사례
* 작은 객체가 다수 밀집된 장면: 객체 크기가 매우 작아 detection 자체가 불안정
* Occlusion: 이전 ID를 유지하지 못하고 새로운 객체로 판단

---

## 1. Environment

* OS: Windows 10
* Python: 3.10.19
* PyTorch: 2.5.1 + CUDA 12.1
* Ultralytics: 8.4.3

---

## 2. Dataset Structure

VisDrone2019-DET / VisDrone2019-MOT 데이터셋을 사용한다.

```
VisDrone2019/
├─ VisDrone2019-DET-train/
│  ├─ images/
|  ├─ annotations/
│  └─ labels/       # YOLO format
├─ VisDrone2019-DET-val/
│  ├─ images/
|  ├─ annotations/
│  └─ labels/       # YOLO format
└─ VisDrone2019-MOT-val/
   └─ sequences/
      ├─ uav0000013_00000_v/
      │  ├─ 000001.jpg
      │  ├─ 000002.jpg
      │  └─ ...
```

Detection 학습을 위해 VisDrone annotation을 YOLO format으로 변환하였다.

---

## 3. Task A – Object Detection

[Detection Output](https://github.com/Goraniiii/VisDrone/tree/master/results)

### 3.1 Training

* Model: `YOLOv8n`
* Epochs: 30
* Image Size: 640
* Optimizer: SGD
* Seed: 42

```bash
python train.py --config configs/train.yaml
```

학습 결과는 `results/visdrone_det/`에 저장된다.

---

### 3.2 Evaluation

Validation set에 대해 mAP 기반 성능 평가를 수행한다.

```bash
python eval.py --config configs/eval.yaml
```

### Evaluation Output (`metrics.json`)

```json
{
  "dataset": "VisDrone-DET",
  "task": "object_detection",
  "model": "yolov8n",
  "epochs": 30,
  "imgsz": 640,
  "batch": 8,
  "mAP50": 0.3555103162097585,
  "mAP50_95": 0.1998574764972477,
  "seed": 42
}
```

---

## 4. Task B – Multi-Object Tracking

[Tracking Output](https://github.com/Goraniiii/VisDrone/tree/master/runs/detect/results/tracking)

### 4.1 Tracking Method

* Detector: YOLOv8 (Task A에서 학습한 best.pt)
* Tracker: ByteTrack (Ultralytics 내장)

### 4.2 Tracking Execution

```bash
python track.py --config configs/track.yaml
```

각 시퀀스에 대해:

* 프레임 단위 추적 결과 (`.txt`)
* 시각화된 결과

를 출력한다.

---

## 5. Video Generation

프레임 이미지(`0000001.jpg` 형식)를 기반으로 ffmpeg을 사용해 영상을 생성하였다.

* FPS: 30
* Codec: H.264
* Output: `.mp4`

---

## 6. Project Structure

```
VisDrone/
├─ configs/
│  ├─ train.yaml
│  ├─ eval.yaml
│  ├─ track.yaml
|  └─ ...
├─ train.py
├─ eval.py
├─ track.py
├─ convert_yolo_label.py
├─ image_to_video.py
├─ results/
│  ├─ visdrone_det/
│  └─ metrics.json
├─ run /
|  └─ detect/
|     └─ results/
|        └─ tracking/    # tracking results
└─ README.md
```

---

## 7. Notes

* Detection과 Tracking을 분리된 파이프라인으로 설계하여 확장성을 고려하였다.
* 모든 실험은 seed 고정을 통해 재현 가능하도록 구성하였다.
* 요구사항에 따라 정량 지표(JSON)와 정성 결과(Video)를 모두 산출하였다.

---

## 8. Conclusion

VisDrone 데이터셋을 대상으로 객체 검출 및 다중 객체 추적의 전 과정을 end-to-end로 구현하였다.
Detection 성능 평가(mAP)와 Tracking 시각화 결과를 통해 실제 환경에서의 적용 가능성을 확인하였다.



