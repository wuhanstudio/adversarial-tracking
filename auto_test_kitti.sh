for VIDEO in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    python 2d-tracking-yolo-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-pcb-attack-oc-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-pcb-attack-sort.py --video ${VIDEO} --dataset kitti
done

python eval_kitti.py