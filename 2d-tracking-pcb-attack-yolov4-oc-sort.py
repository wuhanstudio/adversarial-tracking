import os
import argparse

import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

from oc_sort.ocsort import OCSort
from utils.box_utils import *

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.yolo.yolov4 import YOLOV4
from what.models.detection.yolo.yolov4_tiny import YOLOV4_TINY
from what.models.detection.yolo.utils.yolo_utils import yolo_process_output, yolov4_anchors, yolov4_tiny_anchors

from what.cli.model import *
from what.utils.file import get_file
from what.utils.resize import bilinear_resize

from what.attacks.detection.yolo.PCB import PCBAttack

SHOW_IMAGE = True

# Check what_model_list for all supported models
what_yolov4_model_list = what_model_list[4:6]

index = 0 # YOLOv4
# index = 1 # YOLOv4 Tiny

custom_objects = {
    'mish': lambda x: x * K.tanh(K.softplus(x)),
    'tf': tf
}

# Download the model first if not exists
WHAT_YOLO_MODEL_FILE = what_yolov4_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLO_MODEL_URL  = what_yolov4_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLO_MODEL_HASH = what_yolov4_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLO_MODEL_FILE)):
    get_file(WHAT_YOLO_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLO_MODEL_URL,
             WHAT_YOLO_MODEL_HASH)

# Darknet
model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLO_MODEL_FILE))
# model = YOLOV4_TINY(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))

attack = PCBAttack(os.path.join(WHAT_MODEL_PATH, WHAT_YOLO_MODEL_FILE), "multi_untargeted", COCO_CLASS_NAMES, decay=0.99, custom_objects=custom_objects)
attack.fixed = False

mot_tracker = OCSort(det_thresh=0.6, iou_threshold=0.3, use_byte=False)

def is_not_empty_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Detection (OC SORT)")
    parser.add_argument('--video', type=int, default=0,
                        help='MOT Video Index: 0-20')
    parser.add_argument('--dataset',
                        default='kitti',
                        const='kitti',
                        nargs='?',
                        choices=['kitti', 'carla'],
                        help='Evaluation Dataset (default: %(default)s)')

    args = parser.parse_args()

    DATASET = args.dataset
    GT_FOLDER = os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__))), f'data/gt/{DATASET}/{DATASET}_2d_box_train/')
    TRACKERS_FOLDER = os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__))), f'data/trackers/{DATASET}/{DATASET}_2d_box_train/')

    f_video = f'./data/video/{DATASET}/{args.video:04d}.mp4'
    print(f"Reading {DATASET} Video:", f_video)

    f_label = os.path.join(GT_FOLDER, 'label_02', f'{args.video:04d}.txt')
    print(f"Reading {DATASET} Label:", f_label)

    gt_labels = None
    if is_not_empty_file(f_label):
        gt_labels = pd.read_csv(f_label, header=None, sep=' ')
    else:
        print("Empty label file:", f_label)

    vid = cv2.VideoCapture(f_video)

    if (vid.isOpened()== False): 
        print("Error opening the video file")
        exit(1)

    OUT_FILE = os.path.join(TRACKERS_FOLDER, 'YOLOv4-OC-SORT-PCB',
                            'data', f'{args.video:04d}.txt')
    if not os.path.exists(os.path.dirname(OUT_FILE)):
        # Create a new directory if it does not exist
        os.makedirs(os.path.dirname(OUT_FILE))
    try:
        f_tracker = open(OUT_FILE, "w+")
    except OSError:
        print("Could not open file:", OUT_FILE)
        exit(1)

    # Read until video is completed
    i_frame = 0
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()

        if frame is None:
            break;

        # Labels for the current frame
        if gt_labels is not None:
            c_labels = gt_labels[gt_labels[0] == i_frame]
            c_labels = c_labels[c_labels[1] != -1]
            c_labels = c_labels[(c_labels[2] == 'Van') |
                                (c_labels[2] == 'Car')]
        else:
            c_labels = pd.DataFrame([])

        if ret == True:
            origin = frame.copy()
            height, width, _ = frame.shape

            # Draw bounding boxes onto the original image
            labels = []
            ids = []
            boxes = []
            for _, c_label in c_labels.iterrows():
                x1, y1, x2, y2 = c_label[6], c_label[7], c_label[8], c_label[9]
                boxes.append(np.array([x1, y1, x2, y2]))
                labels.append(c_label[2])
                ids.append(c_label[1])

            draw_bounding_boxes(origin, np.array(boxes), labels, ids)

            # Image preprocessing
            input_cv_image = cv2.resize(frame, (416, 416))
            input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0
            input_cv_image = cv2.cvtColor(input_cv_image, cv2.COLOR_BGR2RGB)

            # Adversarial Attack
            image, outs = attack.attack(input_cv_image)
            boxes, labels, probs = yolo_process_output(outs, yolov4_anchors, len(COCO_CLASS_NAMES))

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Resize the noise to the same shape as the input image
            noise = attack.noise
            noise_r = bilinear_resize(noise[:, :, 0], height, width)
            noise_g = bilinear_resize(noise[:, :, 1], height, width)
            noise_b = bilinear_resize(noise[:, :, 2], height, width)
            noise = np.dstack((noise_r, noise_g, noise_b))

            # Apply adversarial perturbations
            out_img = np.array(frame).astype(np.float32) / 255.0 + noise
            out_img = np.clip(out_img, 0, 1)
            out_img = (out_img * 255.0).astype(np.uint8)

            # Only draw 2: car, 5: bus, 7: truck
            boxes = np.array([box for box, label in zip(boxes, labels) if label in [2, 5, 7]])
            probs = np.array([prob for prob, label in zip(probs, labels) if label in [2, 5, 7]])
            labels = np.array([2 for label in labels if label in [2, 5, 7]])

            # convert [x1, y1, w, h] to [x1, y1, x2, y2]
            if len(boxes) > 0:
                sort_boxes = boxes.copy()

                # (xc, yc, w, h) --> (x1, y1, x2, y2)
                height, width, _ = frame.shape

                for box in sort_boxes:
                    box[0] *= width
                    box[1] *= height
                    box[2] *= width 
                    box[3] *= height

                    # From center to top left
                    box[0] -= box[2] / 2
                    box[1] -= box[3] / 2

                    # From width and height to x2 and y2
                    box[2] += box[0]
                    box[3] += box[1]

                labels = ['Car'] * len(boxes)
                dets = np.concatenate((np.array(sort_boxes), np.array(probs).reshape((len(probs), -1))), axis=1)

                # Update tracker
                trackers = mot_tracker.update(dets, [height, width], (height, width))

                for track in trackers:
                    f_tracker.write(f'{i_frame} {int(track[4])} Car -1.000000 -1 -1 {track[0]} {track[1]} {track[2]} {track[3]} -1 -1 -1 -1 -1 -1 -1 -1 1 \n')
                    f_tracker.flush()

                # Draw bounding boxes onto the predicted image
                draw_bounding_boxes(out_img, trackers[:, 0:4], labels, trackers[:, 4])

            i_frame = i_frame + 1

            if SHOW_IMAGE:
                # Display the resulting frame
                cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)

                if args.dataset == "kitti":
                    cv2.imshow('Frame', draw_gt_pred_image(origin, out_img, orientation="vertical"))
                else:
                    cv2.imshow('Frame', draw_gt_pred_image(origin, out_img, orientation="horizontal"))

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else: 
            break

    f_tracker.close()
    vid.release()

    if SHOW_IMAGE:
        cv2.destroyAllWindows()
