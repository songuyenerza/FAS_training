import cv2 
import numpy as np 
from skimage import transform as trans 

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32) 
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32) 

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32) 

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32) 

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32) 

src = np.array([src1, src2, src3, src4, src5]) 
src_map = {112: src, 224: src * 2} 

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32) 

arcface_src = np.expand_dims(arcface_src, axis=0) 

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'): 
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def get_rotate_crop_image(img, points, pad):
    height, width = img.shape[:2]

    img_padded = img
    points_padded = np.array(points)
    
    # Use Green's theory to judge clockwise or counterclockwise
    d = 0.0
    for index in range(-1, 3):
        d += -0.5 * (points_padded[index + 1][1] + points_padded[index][1]) * (
            points_padded[index + 1][0] - points_padded[index][0])
    if d < 0:  # counterclockwise
        tmp = points_padded.copy()
        points_padded[1], points_padded[3] = tmp[3], tmp[1]

    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points_padded[0] - points_padded[1]),
                np.linalg.norm(points_padded[2] - points_padded[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points_padded[0] - points_padded[3]),
                np.linalg.norm(points_padded[1] - points_padded[2])))
        
        # Calculate padding, ensure it doesn't exceed image boundaries
        max_padding_x = min(width - max(points_padded[:, 0]), min(points_padded[:, 0]))
        max_padding_y = min(height - max(points_padded[:, 1]), min(points_padded[:, 1]))

        padding_x = int(min(img_crop_width * pad, max_padding_x))
        padding_y = int(min(img_crop_height * pad, max_padding_y))

        pts_std = np.float32([[padding_x, padding_y], [img_crop_width + padding_x, padding_y],
                              [img_crop_width + padding_x, img_crop_height + padding_y],
                              [padding_x, img_crop_height + padding_y]])
        M = cv2.getPerspectiveTransform(np.float32(points_padded), pts_std)
        dst_img = cv2.warpPerspective(
            img_padded,
            M, (img_crop_width + 2 * padding_x, img_crop_height + 2 * padding_y),
            borderMode=0,
            flags=cv2.INTER_CUBIC)
        
        return dst_img
    except Exception as e:
        print(f"Error in get_rotate_crop_image: {e}")
        return None

def norm_crop(img, landmark, image_size=224, padding = 0 ): 
    mode='arcface'
    M, pose_index = estimate_norm(landmark, image_size, mode)
    # warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    warped_corners = np.array([[0, 0], [image_size, 0], [image_size, image_size], [0, image_size]])
    warped_corners_homogeneous = np.hstack((warped_corners, np.ones((4, 1))))
    M_extended = np.vstack((M, [0, 0, 1]))
    M_extended = np.linalg.pinv(M_extended)
    original_corners_homogeneous = np.dot(M_extended, warped_corners_homogeneous.T)
    original_corners = (original_corners_homogeneous / original_corners_homogeneous[2, :])[:2, :].T

    warped =  get_rotate_crop_image(img, original_corners, pad = padding)

    warped = cv2.resize(warped, (image_size, image_size))


    return warped



def square_crop(im, S): 
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation): 
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M): 
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M): 
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts

def trans_points(pts, M): 
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)

def find_x(a,b):
    if a==b:
        return -b/a
    
def find_x_pragma(a,b): 
    if a==b:
        return -b/a

