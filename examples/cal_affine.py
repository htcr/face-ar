import numpy as np
import cv2
from matplotlib import pyplot as plt

def cal_affine(src, dst):
    ''' input: n*2, m*2
        output: 2*3
    '''
    b = dst.flatten()
    A = np.zeros((b.shape[0], 6))
    A[::2, 0:2] = src
    A[1::2, 3:5] = src
    A[::2, 2] = 1.0
    A[1::2, 5] = 1.0
    m = np.linalg.lstsq(A, b)[0].reshape((2,3))
    return m

def warp_glass(glass_name, dst_name, M):
    glass = cv2.imread(glass_name, cv2.IMREAD_COLOR)
    background = cv2.imread(dst_name, cv2.IMREAD_COLOR)
    size = background.shape[0:2][::-1]
    glass_warp = cv2.warpAffine(glass, M, size, borderValue=(255,255,255))
    plt.imshow(glass)
    plt.grid()
    plt.show()
    glass_warp = cv2.cvtColor(glass_warp, cv2.COLOR_BGR2GRAY)
    glass_warp = glass_warp > 128.0
    output = background * glass_warp[:,:,np.newaxis]
    return output

def warp_glass_from_img(glass, background, M):
    size = background.shape[0:2][::-1]
    glass_warp = cv2.warpAffine(glass, M, size, borderValue=(0,255,0))
    '''
    plt.imshow(glass)
    plt.grid()
    plt.show()
    '''
    
    #mask = cv2.cvtColor(glass_warp, cv2.COLOR_BGR2GRAY)
    mask = (np.logical_and(np.logical_and(glass_warp[:,:,1] > 200,glass_warp[:,:,2] < 50),
                           glass_warp[:,:,0] < 50)).astype(np.uint8)[:,:,np.newaxis]
    output = (background * mask + glass_warp*(1-mask))
    return output


def generate_glass_img(img, dst, glass_img, src):
    M = cal_affine(src, dst)
    img = warp_glass_from_img(glass_img, img, M)
    return img

    
if __name__=='__main__':
    glass_points = []
    for y in [-100, 300, 700]:
        for x in [0, 400, 800]:
            glass_points.append([x,y])
    glass_points = np.array(glass_points)
    src = np.array([[1.0,2.0],[4.0,5.0],[2.0,1.0]])
    src = glass_points
    dst = []
    for y in [80, 280, 480]:
        for x in [750, 950, 1150]:
            dst.append([x,y])
    dst = np.array(dst)
    
    img = glass = cv2.imread('face.jpg', cv2.IMREAD_COLOR)
    glass_img = cv2.imread('thug.jpg', cv2.IMREAD_COLOR)

    img = generate_glass_img(img, dst, glass_img, src)
    plt.imshow(img[:,:,[2,1,0]])
    plt.grid()
    plt.show()
    