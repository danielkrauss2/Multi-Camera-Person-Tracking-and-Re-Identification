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
    track_cnt = {}  # Dictionary to store tracking information
    images_by_id = {}  # Dictionary to store image paths by track ID

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
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))

                # Ensure bounding box is valid
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 > x1 and y2 > y1:  # Valid bounding box
                    frame_path = os.path.join(temp_dir, f"frame_{frame_counter}.jpg")
                    if not os.path.exists(frame_path):
                        cv2.imwrite(frame_path, frame)

                    if track.track_id not in track_cnt:
                        # Initialize tracking information and store the frame path
                        track_cnt[track.track_id] = [
                            [frame_counter, x1, y1, x2, y2, area]
                        ]
                        images_by_id[track.track_id] = [frame_path]
                    else:
                        # Update tracking information
                        track_cnt[track.track_id].append([frame_counter, x1, y1, x2, y2, area])

                    tracking_results.append({
                        "video": video,
                        "frame": frame_counter,
                        "track_id": track.track_id,
                        "bbox": [x1, y1, x2, y2],
                        "frame_path": frame_path
                    })
                else:
                    print(f"Skipped invalid bounding box {bbox} for frame {frame_counter}")

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

    # Group frames and bounding boxes by track_id
    images_by_id = {}
    track_cnt = {}
    frames_by_id = {}
    for result in tracking_results:
        track_id = result["track_id"]
        if track_id not in images_by_id:
            images_by_id[track_id] = []
            frames_by_id[track_id] = set()
        images_by_id[track_id].append(result)
        frames_by_id[track_id].add(result["frame"])

        if track_id not in track_cnt:
            track_cnt[track_id] = []
        track_cnt[track_id].append([result["frame"], *result["bbox"]])

    # Perform ReID
    feats = {}
    for track_id, entries in images_by_id.items():
        print(f"Processing ID {track_id} with {len(entries)} entries.")
        batch_images = [cv2.imread(result["frame_path"]) for result in entries]
        feats[track_id] = reid._features(batch_images)

    final_fuse_id = {}
    exist_ids = set()

    for frame_set in track_cnt.values():
        for entry in frame_set:
            frame_id, x1, y1, x2, y2 = entry
            for new_id in images_by_id.keys():
                if new_id not in exist_ids:
                    dis = []
                    unpickable = []

                    for key, value in final_fuse_id.items():
                        if key in exist_ids:
                            unpickable += value

                    for old_id in (exist_ids - set(unpickable)):
                        if old_id in feats:
                            dist = np.mean(reid.compute_distance(feats[new_id], feats[old_id]))
                            dis.append([old_id, dist])

                    dis.sort(key=lambda x: x[1])
                    exist_ids.add(new_id)

                    if not dis or dis[0][1] >= threshold:
                        final_fuse_id[new_id] = [new_id]
                    else:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[new_id]
                        final_fuse_id[combined_id].append(new_id)

    print(f"Final fuse IDs: {final_fuse_id}")

    # User Selection for New Persons
    selected_ids = set()
    for new_id in final_fuse_id.keys():
        print(f"Displaying selections for ID: {new_id}")
        first_frame_with_id = min(frames_by_id[new_id])
        frame_path = images_by_id[new_id][0]["frame_path"]
        first_frame = cv2.imread(frame_path)
        user_selected_ids = display_and_select_ids(first_frame, {new_id: track_cnt[new_id]})
        selected_ids.update(user_selected_ids)

    # Generate output video based on selected IDs
    print("Generating output video for selected IDs...")
    output_video_path = "selected_persons.avi"
    loadvideo = LoadVideo(args.videos[0])
    video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (w, h))

    frame_counter = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Create a mask with all areas blacked out
        mask = np.zeros_like(frame)

        for track_id in selected_ids:
            if track_id in track_cnt:
                for bbox in track_cnt[track_id]:
                    if bbox[0] == frame_counter:
                        x1, y1, x2, y2 = bbox[1:]
                        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        out.write(mask)
        frame_counter += 1

    out.release()
    video_capture.release()
    print(f"Output video saved to '{output_video_path}'")

    # Clean up temporary files
    for result in tracking_results:
        if os.path.exists(result["crop_path"]):
            os.remove(result["crop_path"])
    print("ReID and Selection Phase Completed. Temporary files cleaned.")

def create_video_writer(out_dir, segment_index, filename, frame_rate, w, h, codec='MJPG'):
    complete_path = os.path.join(out_dir, filename + "_" + str(segment_index) + ".avi")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
    return out, complete_path

def display_and_select_ids(frame, final_fuse_id):
    displayed_frame = frame.copy()
    for idx, bboxes in final_fuse_id.items():
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[1:]
            cv2.rectangle(displayed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(displayed_frame, f"ID: {idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    instructions = "Detected new person IDs. Enter 'y' to track or 'n' to ignore each ID."
    cv2.putText(displayed_frame, instructions, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("New Person Detected", displayed_frame)
    cv2.waitKey(1)

    selected_ids = set()
    for idx in final_fuse_id.keys():
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

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color




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
