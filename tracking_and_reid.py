# ! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations      # ← add this line right at the top

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
from collections import defaultdict
import random, itertools


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

# ────────────────────────────────────────────────────────────────────────────────
# Helper: union-find based clustering of track IDs
# ────────────────────────────────────────────────────────────────────────────────
def fuse_by_reid(representatives: dict[int, np.ndarray],
                 threshold: float, frames_by_id: dict[int, set[int]]) -> tuple[dict[int, list[int]], list[tuple[int, int, float]]]:
    """
    representatives : dict {track_id: 1×d vector (ℓ2-normalised)}
    Returns         : dict {root_id : [member_id, ...]}
    """
    keys   = list(representatives.keys())
    feats  = np.stack([representatives[k] for k in keys])
    # cosine distance matrix
    dists  = 1.0 - feats @ feats.T

    parent = {k: k for k in keys}
    merges = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y, dist: float) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            # keep the smaller track id as root (arbitrary but stable)
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry
            merges.append((x, y, dist))
            print(f"[ReID] fused {x} ← {y}  (dist={dist:.3f})")

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if dists[i, j] >= threshold:
                continue  # not similar enough by appearance
            # NEW: don’t merge if they coexist in any frame
            if frames_by_id[keys[i]] & frames_by_id[keys[j]]:
                continue  # non-empty intersection → veto
            union(keys[i], keys[j], dists[i, j])

    clusters = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)
    return clusters, merges

# ────────────────────────────────────────────────────────────────────────────────
# Helper: merge bookkeeping dicts according to clusters
# ────────────────────────────────────────────────────────────────────────────────
def merge_clusters_into_dicts(clusters, images_by_id, track_cnt):
    """
    Modifies images_by_id and track_cnt *in place* so that all member IDs
    are moved under the root ID defined in clusters.
    """
    for root, members in clusters.items():
        for mid in members:
            if mid == root:
                continue
            # extend & delete
            images_by_id[root].extend(images_by_id.pop(mid, []))
            track_cnt[root].extend(track_cnt.pop(mid, []))

# ────────────────────────────────────────────────────────────────────────────────
# Re-ID and selection phase (reworked)
# ────────────────────────────────────────────────────────────────────────────────
def reid_and_selection_phase(args):
    print("Starting Re-ID and Selection Phase...")
    reid = REID()
    #THRESHOLD = args.reid_thresh               # cosine distance threshold

    print("Treshold is set to", args.reid_thresh)

    tracking_file = Path("tracking_results.json")
    if not tracking_file.exists():
        print("Error: tracking_results.json not found – run tracking first.")
        return

    tracking_results = json.loads(tracking_file.read_text())

    # Group frames and boxes by track_id
    images_by_id = defaultdict(list)
    track_cnt    = defaultdict(list)
    frames_by_id = defaultdict(set)

    for res in tracking_results:
        tid = res["track_id"]
        images_by_id[tid].append(res)
        frames_by_id[tid].add(res["frame"])
        track_cnt[tid].append([res["frame"], *res["bbox"]])

    # ── Extract features & build representative vector per track ────────────
    feats = {}
    BATCH = 10000                     # adjust if you want smaller mini-batches

    total_tracks = len(images_by_id)
    print("Total Persons found: ", total_tracks)

    for tid, entries in images_by_id.items():
        print(f"[ReID] extracting features for person {tid} "
              f"({len(entries)} crops)")
        all_feats = []

        # walk through this track’s frames in mini-batches
        for i in range(0, len(entries), BATCH):
            crops = []
            for e in entries[i:i + BATCH]:
                frame = cv2.imread(e["frame_path"])
                if frame is None:
                    continue

                # ---------- person crop ----------
                x1, y1, x2, y2 = e["bbox"]
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, 0.20, w, h)  # ← +20 %

                if x2 <= x1 or y2 <= y1:
                    continue          # skip empty / invalid boxes

                crop = frame[y1:y2, x1:x2]               # BGR crop
                crop = cv2.resize(crop, (128, 256),      # many Re-ID nets expect 128×256
                                  interpolation=cv2.INTER_LINEAR)
                crops.append(crop)
                # -----------------------------------

            if not crops:
                continue

            # Re-ID model expects a list/array of images
            batch_feats = reid._features(crops)
            all_feats.append(batch_feats)

        if not all_feats:
            continue

        all_feats = np.vstack(all_feats)   # shape: (N_frames, feat_dim)
        feats[tid] = all_feats

    # build ℓ2-normalised representative vector
    representatives = {}
    for tid, f in feats.items():
        rep = f.mean(axis=0)
        rep /= (np.linalg.norm(rep) + 1e-12)
        representatives[tid] = rep

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # DEBUG BLOCK

    if len(representatives) > 1:  # nothing to sample if just 1 track
        pairs = list(itertools.combinations(representatives.keys(), 2))
        random.shuffle(pairs)
        pairs = pairs[:200]  # sample at most 200 pairs
        dists = [1.0 - representatives[a] @ representatives[b] for a, b in pairs]
        print(f"[ReID debug] sample pairwise distances:"
              f" min={min(dists):.3f}, median={np.median(dists):.3f},"
              f" max={max(dists):.3f}")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    print(f"Computed representatives for {len(representatives)} tracks.")
    clusters, merge_log = fuse_by_reid(representatives, args.reid_thresh, frames_by_id)

    #clusters = fuse_by_reid(representatives, THRESHOLD)
    print(f"→ {len(clusters)} unique IDs after fusion.")

    print("\n[ReID] fusion summary:")
    for root, members in clusters.items():
        if len(members) == 1:
            print(f"  ID {root}: singleton")
        else:
            others = [m for m in members if m != root]
            print(f"  ID {root}: merged ← {others}")
    print(f"  total clusters: {len(clusters)}  "
          f"(from {len(representatives)} original IDs)\n")

    # merge bookkeeping dicts so that only root IDs remain
    merge_clusters_into_dicts(clusters, images_by_id, track_cnt)

    # ── User selection phase ───────────────────────────────────────
    selected_ids = set()
    for root_id in clusters.keys():
        first_frame_idx = min(frames_by_id[root_id])
        frame_path = images_by_id[root_id][0]["frame_path"]
        frame      = cv2.imread(frame_path)
        chosen = display_and_select_ids(frame,
                                        {root_id: track_cnt[root_id]},
                                        first_frame_idx)
        selected_ids.update(chosen)

    # ── Generate output video with masked IDs  ──────────────────────
    print("Generating output video for selected IDs...")
    video_input = Path(args.videos[0])
    out_dir     = video_input.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = out_dir / f"{video_input.stem}_output.avi"

    loadvideo = LoadVideo(args.videos[0])
    video_capture, frame_rate, w, h = loadvideo.get_VideoLabels()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out    = cv2.VideoWriter(str(output_video_path), fourcc, frame_rate, (w, h))

    frame_counter = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        mask = np.zeros_like(frame)
        for tid in selected_ids:
            for bbox in track_cnt.get(tid, []):
                if bbox[0] == frame_counter:
                    _, x1, y1, x2, y2 = bbox
                    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        out.write(mask)
        frame_counter += 1

    out.release()
    video_capture.release()
    print(f"Output video saved to {output_video_path}")

    # ── Clean-up temp crops and JSON ───────────────────────────────────────────
    for res in tracking_results:
        fp = res.get("frame_path")
        if fp and os.path.exists(fp):
            os.remove(fp)

    if tracking_file.exists():  # Path.unlink has no 'missing_ok' in 3.7
        tracking_file.unlink()

    print("Temporary crops & tracking_results.json removed.")

# ────────────────────────────────────────────────────────────────────────────────
# Misc helper functions
# ────────────────────────────────────────────────────────────────────────────────

def create_video_writer(out_dir, segment_index, filename, frame_rate, w, h, codec='MJPG'):
    complete_path = os.path.join(out_dir, filename + "_" + str(segment_index) + ".avi")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(complete_path, fourcc, frame_rate, (w, h))
    return out, complete_path

def display_and_select_ids(frame, final_fuse_id, current_frame):
    displayed_frame = frame.copy()
    for idx, bboxes in final_fuse_id.items():
        for bbox in bboxes:
            if bbox[0] == current_frame:  # Show only the bounding box for the current frame
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

def pad_box(x1, y1, x2, y2, pad_ratio, img_w, img_h):
    """Expand (x1,y1,x2,y2) by pad_ratio (e.g. 0.20 = +20 %) in both axes."""
    w = x2 - x1
    h = y2 - y1
    px = int(w * pad_ratio / 2)
    py = int(h * pad_ratio / 2)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(img_w - 1, x2 + px)
    y2 = min(img_h - 1, y2 + py)
    return x1, y1, x2, y2



def main(yolo, args):
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
    parser.add_argument('--reid_thresh', type=float, default=0.10,
                        help='Cosine-distance threshold for ReID fusion '
                             '(lower keeps IDs separate)')

    args = parser.parse_args()

    yolo = YOLO3() if args.version == 'yolo_v3' else YOLO4()
    main(yolo, args)
