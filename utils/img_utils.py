import torch
import numpy as np
from PIL import Image
import random
import torchvision
def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1

def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet
def get_affine_transform(center, scale, optical_center, out_res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [1])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = - optical_center[0]
    t_mat[1, 2] = - optical_center[1]
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = (
        t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [1])
    )
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, out_res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(
        transformed_center[:2], scale, out_res
    )
    return (
        total_trans.astype(np.float32),
        affinetrans_post_rot.astype(np.float32),
    )
def get_affine_transform_test(center, scale, res, rot=0):
    rot_mat = np.zeros((3, 3))
    sn, cs = np.sin(rot), np.cos(rot)
    rot_mat[0, :2] = [cs, -sn]
    rot_mat[1, :2] = [sn, cs]
    rot_mat[2, 2] = 1
    # Rotate center to obtain coordinate of center in rotated image
    origin_rot_center = rot_mat.dot(center.tolist() + [
        1,
    ])[:2]
    # Get center for transform with verts rotated around optical axis
    # (through pixel center, smthg like 128, 128 in pixels and 0,0 in 3d world)
    # For this, rotate the center but around center of image (vs 0,0 in pixel space)
    t_mat = np.eye(3)
    t_mat[0, 2] = -res[1] / 2
    t_mat[1, 2] = -res[0] / 2
    t_inv = t_mat.copy()
    t_inv[:2, 2] *= -1
    transformed_center = t_inv.dot(rot_mat).dot(t_mat).dot(center.tolist() + [
        1,
    ])
    post_rot_trans = get_affine_trans_no_rot(origin_rot_center, scale, res)
    total_trans = post_rot_trans.dot(rot_mat)
    # check_t = get_affine_transform_bak(center, scale, res, rot)
    # print(total_trans, check_t)
    affinetrans_post_rot = get_affine_trans_no_rot(transformed_center[:2],
                                                   scale, res)
    return total_trans.astype(np.float32), affinetrans_post_rot.astype(
        np.float32)
def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows.astype(int)
def transform_img(img, affine_trans, res):
    """
    Args:
    center (tuple): crop center coordinates
    scale (int): size in pixels of the final crop
    res (tuple): final image size
    """
    trans = np.linalg.inv(affine_trans)

    img = img.transform(
        tuple(res), Image.AFFINE, (trans[0, 0], trans[0, 1], trans[0, 2],
                                   trans[1, 0], trans[1, 1], trans[1, 2])
    )
    return img

def global_align(gtj0, prj0):
    gtj = gtj0.copy()
    prj = prj0.copy()

    # gtj :B*21*3
    # prj :B*21*3
    root_idx = 9  # root
    ref_bone_link = [0, 9]  # mid mcp
    pred_align = prj.copy()
    for i in range(prj.shape[0]):

        pred_ref_bone_len = np.linalg.norm(prj[i][ref_bone_link[0]] - prj[i][ref_bone_link[1]])
        gt_ref_bone_len = np.linalg.norm(gtj[i][ref_bone_link[0]] - gtj[i][ref_bone_link[1]])
        scale = gt_ref_bone_len / pred_ref_bone_len

        for j in range(21):
            pred_align[i][j] = gtj[i][root_idx] + scale * (prj[i][j] - prj[i][root_idx])

    return gtj, pred_align


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(
            max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(
            max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor

def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue)

    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img