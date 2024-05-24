# Adversarial Tracking

You can find the dataset here (KITTI and CARLA): https://github.com/wuhanstudio/adversarial-tracking/releases

![](docs/demo.png)


## 2D Object Tracking

    VIDEO=0 # 0 - 20

    python 2d-tracking-yolov3-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov3-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov3-strong-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov3-oc-sort.py --video ${VIDEO} --dataset carla

    python 2d-tracking-yolov4-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov4-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov4-strong-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov4-oc-sort.py --video ${VIDEO} --dataset carla

    python 2d-tracking-pcb-attack-yolov3-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-pcb-attack-yolov3-oc-sort.py --video ${VIDEO} --dataset carla

    python 2d-tracking-pcb-attack-yolov4-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-pcb-attack-yolov4-oc-sort.py --video ${VIDEO} --dataset carla

## 2D Adversarial Tracking

    VIDEO=0 # 0 - 20

    python 2d-tracking-yolov3-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov3-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov3-strong-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov3-oc-sort.py --video ${VIDEO} --dataset carla

    python 2d-tracking-yolov4-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov4-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov4-strong-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolov4-oc-sort.py --video ${VIDEO} --dataset carla

    python 2d-tracking-pcb-attack-yolov3-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-pcb-attack-yolov3-oc-sort.py --video ${VIDEO} --dataset carla

    python 2d-tracking-pcb-attack-yolov4-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-pcb-attack-yolov4-oc-sort.py --video ${VIDEO} --dataset carla

## 2d Adversarial Tracking (UAP)

    VIDEO=0 # 0 - 20

    python 2d-tracking-yolov3-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov3_noise_pcb_0003_99.npy
    python 2d-tracking-yolov3-deep-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov3_noise_pcb_0003_99.npy
    python 2d-tracking-yolov3-strong-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov3_noise_pcb_0003_99.npy
    python 2d-tracking-yolov3-oc-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov3_noise_pcb_0003_99.npy

    python 2d-tracking-yolov4-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov4_noise_pcb_0003_99.npy
    python 2d-tracking-yolov4-deep-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov4_noise_pcb_0003_99.npy
    python 2d-tracking-yolov4-strong-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov4_noise_pcb_0003_99.npy
    python 2d-tracking-yolov4-oc-sort.py --video ${VIDEO} --dataset carla --noise noise/yolov4_noise_pcb_0003_99.npy
