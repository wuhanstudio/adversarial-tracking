for VIDEO in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    python 2d-tracking-yolov3-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov3-deep-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov3-strong-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov3-oc-sort.py --video ${VIDEO} --dataset kitti

    python 2d-tracking-yolov4-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov4-deep-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov4-strong-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov4-oc-sort.py --video ${VIDEO} --dataset kitti

    python 2d-tracking-pcb-attack-yolov3-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-pcb-attack-yolov3-oc-sort.py --video ${VIDEO} --dataset kitti

    python 2d-tracking-pcb-attack-yolov4-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-pcb-attack-yolov4-oc-sort.py --video ${VIDEO} --dataset kitti
done

python eval_kitti.py