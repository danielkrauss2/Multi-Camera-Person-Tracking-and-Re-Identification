# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import warnings
import argparse
from pathlib import Path

import sys
import cv2
import numpy as np
import base64
import requests
import urllib
from urllib import parse
import json
import random
import time
from PIL import Image
from collections import Counter
import operator

from yolo_v3 import YOLO3
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from reid import REID
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
parser.add_argument('-all', help='Combine all videos into one', default=True)
args = parser.parse_args()  # vars(parser.parse_args())


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        print('Length of {}: {:d} frames'.format(path, self.vn))

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh


import json


def tracking_phase(yolo, args):
    print("Starting Tracking Phase...")
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.4
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=300)

    temp_dir = 'temp_crops'
    os.makedirs(temp_dir, exist_ok=True)
    tracking_results = []

    for video in args.videos:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        frame_counter = 0
        bbox_cache = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_counter % 10 == 0:
                image = Image.fromarray(frame[..., ::-1])  # Convert BGR to RGB
                boxs = yolo.detect_image(image)
                features = encoder(frame, boxs)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                bbox_cache = detections
            else:
                detections = bbox_cache

            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                crop_path = os.path.join(temp_dir, f"id_{track.track_id}_frame_{frame_counter}.jpg")
                cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imwrite(crop_path, cropped_image)

                tracking_results.append({
                    "video": video,
                    "frame": frame_counter,
                    "track_id": track.track_id,
                    "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "crop_path": crop_path
                })

            frame_counter += 1
            print(f"Processed frame {frame_counter} of video {video}")

        video_capture.release()

    # Save tracking results to file
    with open("tracking_results.json", "w") as f:
        json.dump(tracking_results, f)
    print("Tracking Phase Completed. Results saved to 'tracking_results.json'")


def reid_and_selection_phase(args):
    print("Starting ReID and Selection Phase...")
    reid = REID()
    threshold = 375

    # Load tracking results
    tracking_results_file = "tracking_results.json"
    if not os.path.exists(tracking_results_file):
        print("Error: Tracking results file not found. Please run the tracking phase first.")
        return

    with open(tracking_results_file, "r") as f:
        tracking_results = json.load(f)

    # Group crops by track_id
    images_by_id = {}
    for result in tracking_results:
        track_id = result["track_id"]
        if track_id not in images_by_id:
            images_by_id[track_id] = []
        images_by_id[track_id].append(result["crop_path"])

    # Perform ReID
    feats = {}
    for track_id, crop_paths in images_by_id.items():
        print(f"Processing ID {track_id} with {len(crop_paths)} crops.")
        batch_images = [cv2.imread(path) for path in crop_paths]
        feats[track_id] = reid._features(batch_images)

    # User Selection
    selected_ids = set()
    for track_id in feats.keys():
        user_input = input(f"Do you want to track person with ID {track_id}? (y/n): ")
        if user_input.lower() == 'y':
            selected_ids.add(track_id)

    # Generate output video based on selected IDs
    print("Generating output video for selected IDs...")
    output_video_path = "selected_persons.avi"
    frame_size = (tracking_results[0]["bbox"][2], tracking_results[0]["bbox"][3])
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, frame_size)

    for result in tracking_results:
        if result["track_id"] in selected_ids:
            frame = cv2.imread(result["crop_path"])
            out.write(frame)

    out.release()
    print(f"Output video saved to '{output_video_path}'")

    # Clean up temporary files
    for result in tracking_results:
        if os.path.exists(result["crop_path"]):
            os.remove(result["crop_path"])
    print("ReID and Selection Phase Completed. Temporary files cleaned.")


import os
import json


def main(yolo, args):
    tracking_results_file = "tracking_results.json"

    # Check if tracking has already been completed
    if os.path.exists(tracking_results_file):
        print("Tracking results already exist. Skipping tracking phase.")
    else:
        # Run the tracking phase
        tracking_phase(yolo, args)

    # Proceed to ReID and selection phase
    reid_and_selection_phase(args)


def create_video_writer(out_dir, segment_index, filename, frame_rate, w, h, codec='MJPG'):
    complete_path = os.path.join(out_dir, filename + "_" + str(segment_index) + ".avi")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
    return out, complete_path


# Function to display frame with all detected persons and prompt for user input
def display_and_select_ids(frame, final_fuse_id, track_cnt, current_frame, new_ids):
    displayed_frame = frame.copy()
    for idx in new_ids:
        for i in final_fuse_id[idx]:
            for f in track_cnt[i]:
                if f[0] == current_frame:
                    x1, y1, x2, y2 = f[1], f[2], f[3], f[4]
                    cv2.rectangle(displayed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(displayed_frame, f"ID: {idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

    # Instructions for selecting IDs
    instructions = "Detected new person IDs. Enter 'y' to track or 'n' to ignore each ID."
    cv2.putText(displayed_frame, instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("New Person Detected", displayed_frame)
    cv2.waitKey(1)

    # Collect user decisions for each new ID
    selected_ids = set()
    for idx in new_ids:
        while True:
            user_input = input(f"Do you want to track person with ID {idx}? (y/n): ")
            if user_input.lower() in ['y', 'n']:
                if user_input.lower() == 'y':
                    selected_ids.add(idx)
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    cv2.destroyAllWindows()
    return selected_ids


def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness


def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)


def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)
    # print('save results to {}'.format(filename))


warnings.filterwarnings('ignore')


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color



if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    main(yolo=YOLO3() if args.version == 'v3' else YOLO4())
