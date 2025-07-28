import cv2, sys, subprocess, shlex, json, re, os, numpy as np

video = sys.argv[1]

# -- 1. what does ffprobe say about raster size?
cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "{video}"'
meta = json.loads(subprocess.check_output(shlex.split(cmd), text=True))
print("[ffprobe] reported raster   :",
      meta["streams"][0]["width"], "x", meta["streams"][0]["height"])

# -- 2. any rotation side-data?
cmd = ('ffprobe -v error -select_streams v:0 -show_entries stream_side_data '
       '-of json "{}"'.format(video))
sd   = json.loads(subprocess.check_output(shlex.split(cmd), text=True))
for side in sd["streams"][0].get("side_data_list", []):
    if side.get("side_data_type") == "Display Matrix":
        m = re.search(r'rotation of ([\-0-9\.]+)', side.get("displaymatrix",""))
        if m:
            print("[ffprobe] display-matrix rot :", m.group(1), "deg")

# -- 3. what does OpenCV deliver?
cap = cv2.VideoCapture(video)
ok, fr = cap.read()
cap.release()
if ok:
    h, w = fr.shape[:2]
    print("[OpenCV] first frame shape :", w, "x", h)
    # save for visual confirmation
    cv2.imwrite("frame_raw.jpg", fr)
    print("saved raw frame → frame_raw.jpg")
else:
    print("OpenCV failed to read the first frame")
