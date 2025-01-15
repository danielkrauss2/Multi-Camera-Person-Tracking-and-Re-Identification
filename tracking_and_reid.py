# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from reid import REID
from yolo_v3 import YOLO3
from yolo_v4 import YOLO4

class LoadVideo:
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file {path} not found.")

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = img_size[0]
        self.height = img_size[1]
        print(f'Length of {path}: {self.vn} frames')

    def get_VideoLabels(self):
        return self.cap, self.frame_rate, self.vw, self.vh


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

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []

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

    for result in tracking_results:
        if os.path.exists(result["crop_path"]):
            os.remove(result["crop_path"])
    print("ReID and Selection Phase Completed. Temporary files cleaned.")


def main(yolo):
    tracking_results_file = "tracking_results.json"

    if os.path.exists(tracking_results_file):
        print("Tracking results already exist. Skipping tracking phase.")
    else:
        tracking_phase(yolo, args)

    reid_and_selection_phase(args)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', help='Model(yolo_v3 or yolo_v4)', default='yolo_v4')
    parser.add_argument('--videos', nargs='+', help='List of videos', required=True)
    args = parser.parse_args()

    yolo = YOLO3() if args.version == 'yolo_v3' else YOLO4()
    main(yolo)
