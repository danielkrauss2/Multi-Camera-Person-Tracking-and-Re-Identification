# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from timeit import time
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


def main(yolo):
    print(f'Using {yolo} model')
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.4

    # deep_sort
    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=300)

    output_frames = []
    output_rectanger = []
    output_areas = []
    output_wh_ratio = []

    is_vis = True
    out_dir = 'videos/output/'
    print('The output folder is', out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    video_frame_length = {}

    print("Actual frame length: ", video_frame_length)

    tracking_path = out_dir + 'tracking' + '.avi'
    combined_path = out_dir + 'allVideos' + '.avi'

    # Initialize tracking file
    filename = out_dir + '/tracking.txt'
    open(filename, 'w')

    fps = 0.0
    frame_cnt = 0
    t1 = time.time()

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    bbox_cache = []  # Cache to store bounding boxes for the 9 frames

    for video in args.videos:
        loadvideo = LoadVideo(video)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()

        # Output video writer
        if is_vis:
            tracking_path = out_dir + f'tracking_{Path(video).stem}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))

        frame_counter = 0
        bbox_cache = []  # Cache to store bounding boxes for the last 9 frames

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Only process every 10th frame
            if frame_counter % 10 == 0:
                image = Image.fromarray(frame[..., ::-1])  # Convert BGR to RGB
                boxs = yolo.detect_image(image)
                features = encoder(frame, boxs)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                # Non-maxima suppression
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                bbox_cache = detections
            else:
                detections = bbox_cache

            # Tracker update
            tracker.predict()
            tracker.update(detections)

            tmp_ids = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()  # Convert the bounding box to top-left-bottom-right format
                area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))

                # Ensure the bounding box is within frame dimensions
                if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < h and bbox[2] < w:
                    tmp_ids.append(track.track_id)

                    # Update track_cnt and images_by_id for the current track ID
                    temp_dir = 'temp_crops'
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)

                    if track.track_id not in track_cnt:
                        track_cnt[track.track_id] = [
                            [frame_counter, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                        ]
                        # Save cropped frame to disk
                        crop_path = os.path.join(temp_dir, f"id_{track.track_id}_frame_{frame_counter}.jpg")
                        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        cv2.imwrite(crop_path, cropped_image)
                        images_by_id[track.track_id] = [crop_path]
                    else:
                        track_cnt[track.track_id].append([
                            frame_counter,
                            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                            area
                        ])
                        # Save cropped frame to disk
                        crop_path = os.path.join(temp_dir, f"id_{track.track_id}_frame_{frame_counter}.jpg")
                        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        cv2.imwrite(crop_path, cropped_image)
                        images_by_id[track.track_id].append(crop_path)

                    # Draw the bounding box and add text to the frame
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(
                        track.track_id,
                        frame,
                        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                        line_thickness,
                        text_thickness,
                        text_scale
                    )

                    # Write results to the tracking output file
                    write_results(
                        filename,
                        'mot',
                        frame_counter + 1,
                        str(track.track_id),
                        int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                        w, h
                    )

            # Append the set of IDs detected in the current frame to ids_per_frame
            ids_per_frame.append(set(tmp_ids))

            if is_vis:
                out.write(frame)
            frame_counter += 1
            print(f"Processed frame {frame_counter}")

        if is_vis:
            out.release()
        video_capture.release()


    if is_vis:
        out.release()
    print('Tracking finished in {} seconds'.format(int(time.time() - t1)))
    print('Tracked video : {}'.format(tracking_path))
    print('Combined video : {}'.format(combined_path))

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    reid = REID()
    threshold = 375
    exist_ids = set()
    final_fuse_id = dict()

    print(f'Total IDs = {len(images_by_id)}')
    feats = dict()
    for i in images_by_id:
        print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
        # Load images from disk in batches
        batch_images = [cv2.imread(img_path) for img_path in images_by_id[i][:min(len(images_by_id[i]), 100)]]
        feats[i] = reid._features(batch_images)

    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        print(dis[0][1])
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]
    print('Final ids and their sub-ids:', final_fuse_id)
    print('MOT took {} seconds'.format(int(time.time() - t1)))
    t2 = time.time()

    # To generate MOT for each person, declare 'is_vis' to True
    is_vis = False
    if is_vis:
        print('Writing videos for each ID...')
        output_dir = 'videos/output/tracklets/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        loadvideo = LoadVideo(combined_path)
        video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        for idx in final_fuse_id:
            tracking_path = os.path.join(output_dir, str(idx) + '.avi')
            out = cv2.VideoWriter(tracking_path, fourcc, frame_rate, (w, h))
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, f[0])
                    _, frame = video_capture.read()
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(idx, frame, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
                    out.write(frame)
            out.release()
        video_capture.release()

    # Clean up temporary files
    for track_id, crop_paths in images_by_id.items():
        for crop_path in crop_paths:
            if os.path.exists(crop_path):
                os.remove(crop_path)
    os.rmdir(temp_dir)
    print("Temporary files cleaned up.")

    # Generate a single video with complete MOT/ReID
    if args.all:
        selected_ids = set()  # Set to store selected IDs
        seen_ids = set()  # Set to store seen IDs

        segment_index = 0
        segment_paths = []

        for video in args.videos:
            loadvideo = LoadVideo(video)
            video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()

            filename = Path(video).stem
            out, complete_path = create_video_writer(out_dir, segment_index, filename, frame_rate, w, h)
            segment_paths.append(complete_path)

            frame_counter = 0
            print(f"Processing video: {video}")

            while True:
                ret, frame2 = video_capture.read()
                if not ret:
                    break  # End of video

                # Identify new IDs in the current frame
                new_ids = set()
                for idx in final_fuse_id:
                    if idx not in seen_ids:
                        for i in final_fuse_id[idx]:
                            for f in track_cnt[i]:
                                if frame_counter == f[0]:
                                    new_ids.add(idx)
                                    seen_ids.add(idx)
                                    break

                # Display frame and get user input for new IDs
                if new_ids:
                    new_selected_ids = display_and_select_ids(frame2, final_fuse_id, track_cnt, frame_counter, new_ids)
                    selected_ids.update(new_selected_ids)

                # Create a black mask of the same size as frame2
                mask = np.zeros_like(frame2)
                frame_height, frame_width = frame2.shape[:2]

                for idx in selected_ids:
                    if idx in final_fuse_id:
                        for i in final_fuse_id[idx]:
                            for f in track_cnt[i]:
                                if frame_counter == f[0]:
                                    x1, y1, x2, y2 = f[1], f[2], f[3], f[4]

                                    # Calculate and apply 65% margin
                                    margin_x = int(0.95 * (x2 - x1) / 2)
                                    margin_y = int(0.70 * (y2 - y1) / 2)

                                    # Adjust coordinates
                                    x1 = max(x1 - margin_x, 0)
                                    y1 = max(y1 - margin_y, 0)
                                    x2 = min(x2 + margin_x, frame_width)
                                    y2 = min(y2 + margin_y, frame_height)

                                    # Copy the detected person area from frame2 to the mask
                                    mask[y1:y2, x1:x2] = frame2[y1:y2, x1:x2]

                                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                                    cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness,
                                               text_scale)

                out.write(mask)
                print(f"Writing frame {frame_counter} to {complete_path}")
                frame_counter += 1

            out.release()
            video_capture.release()
            segment_index += 1

        for path in segment_paths:
            print(f'Segment video at {path}')


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
