import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0', flip_input=False)

landmark_id_ranges = [(0, 17), (17, 22), (22, 27), (27, 31), (31, 36), (36, 42), (42, 48), (48, 68)]
def get_3d_points(preds):
    segments = list()
    for begin, end in landmark_id_ranges:
        cur_segment = list()
        for i in range(begin, end):
            x, y, z = preds[i, :]
            cur_segment.append((x, y, z))
        segments.append(cur_segment)
    return segments

def draw_segment(img, segment):
    x0, y0, z0 = segment[0]
    for i in range(1, len(segment)):
        x, y, z = segment[i]
        cv2.line(img, (int(x0), int(y0)), (int(x), int(y)), (0, 255, 0), 2)
        x0, y0, z0 = x, y, z
        
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    img = cv2.resize(img, (320, 240))

    img_show = img.copy()

    all_preds = fa.get_landmarks(img)

    if all_preds is not None:
        preds = all_preds[-1]
        segments = get_3d_points(preds)
        for seg in segments:
            draw_segment(img_show, seg)
    
    cv2.imshow('test',img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
