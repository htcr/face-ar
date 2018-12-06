import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
from cal_affine import generate_glass_img

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

def draw_segment(img, segment, color=(0, 255, 0)):
    x0, y0, z0 = segment[0]
    for i in range(1, len(segment)):
        x, y, z = segment[i]
        cv2.line(img, (int(x0), int(y0)), (int(x), int(y)), color, 2)
        x0, y0, z0 = x, y, z

def get_face_frame(preds):
    nose_tip = preds[28, :]
    l_par = preds[0:4, :]
    r_par = preds[13:17, :]
    bottom_tip = preds[8, :]
    
    # get z direction
    z_dir = ((nose_tip - l_par) + (nose_tip - r_par)) / 2.0
    z_dir = -np.mean(z_dir, axis=0) # (3, )
    z_dir = z_dir / np.linalg.norm(z_dir)

    # get temp y direction
    y0_dir = ((bottom_tip - l_par) + (bottom_tip - r_par)) / 2.0
    y0_dir = np.mean(y0_dir, axis=0) # (3, )

    # get x direction
    x_dir = np.cross(y0_dir, z_dir)
    x_dir = x_dir / np.linalg.norm(x_dir)

    # get y direction
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)

    # from face frame to screen frame
    T = np.stack((x_dir, y_dir, z_dir), axis=1)
    
    # get origin
    origin = preds[27, :] # (3,)
    
    # get scale
    l_tip = preds[0, :]
    r_tip = preds[16, :]
    face_width = np.sum((l_tip - r_tip)**2)**0.5
    face_scale = face_width / 2.0

    return origin, T, face_scale
    
def draw_face_frame(img, origin, axes, size):
    '''
    print(axes)
    print(size)
    print(origin)
    '''
    
    vecs = axes*size + origin.reshape(-1, 1)
    endpoints2d = vecs[:2, :]

    '''
    print(vecs)
    print(endpoints2d)

    print(origin)
    '''

    ox, oy = origin[:2]

    '''
    print(ox, oy)
    '''

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(3):
        ex, ey = endpoints2d[:, i]
        cv2.line(img, (int(ox), int(oy)), (int(ex), int(ey)), colors[i], 2)

def get_9points(face_origin, face_axes, face_scale):
    x = face_axes[:, 0]
    y = face_axes[:, 1]

    kps = list()

    for i in range(-1, 2):
        for j in range(-1, 2):
            kp = face_origin + i * x * face_scale + j * y * face_scale
            kp = kp[0:2]
            kps.append(kp)

    kps = np.array(kps)
    return kps

def draw_9points(img, kps):
    for kp in kps:
        x, y = kp
        cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255))


glass_points = []

for x in [0, 400, 800]:
    for y in [-100, 300, 700]:
        glass_points.append([x,y])
glass_points = np.array(glass_points)

ar_obj_path = 'glass.jpg'
ar_obj_kps = glass_points
ar_obj = cv2.imread('glass.jpg', cv2.IMREAD_COLOR)

cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    img = cv2.resize(img, (320, 240))

    img_show = img.copy()
    img_ar = img

    all_preds = fa.get_landmarks(img)

    if all_preds is not None:
        preds = all_preds[-1]
        segments = get_3d_points(preds)
        for seg in segments:
            draw_segment(img_show, seg, color=(128, 128, 128))
    
        face_origin, face_axes, face_scale = get_face_frame(preds)
        
        draw_face_frame(img_show, face_origin, face_axes, face_scale)

        kps = get_9points(face_origin, face_axes, face_scale)
        draw_9points(img_show, kps)

        img_ar = generate_glass_img(img, kps, ar_obj, ar_obj_kps)

    cv2.imshow('test',img_show)
    cv2.imshow('ar', img_ar)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
