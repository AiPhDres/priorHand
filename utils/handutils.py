import torch
from PIL import Image
import torch.nn as nn
import numpy as np

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
def get_affine_trans_no_rot(center, scale, res):
    affinet = np.zeros((3, 3))
    affinet[0, 0] = float(res[1]) / scale
    affinet[1, 1] = float(res[0]) / scale
    affinet[0, 2] = res[1] * (-float(center[0]) / scale + .5)
    affinet[1, 2] = res[0] * (-float(center[1]) / scale + .5)
    affinet[2, 2] = 1
    return affinet
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
def parse_input(pred_joint, infos):
    JOINTS_PARENT = [
                0,  # root parent
                0,  # 1's parent
                1,
                2,
                3,
                0,  # 5's parent
                5,
                6,
                7,
                0,  # 9's parent
                9,
                10,
                11,
                0,  # 13's parent
                13,
                14,
                15,
                0,  # 17's parent
                17,
                18,
                19,
    ]
    ref_bone_link = (0, 9)
    root = infos['joint_root'].unsqueeze(1)  # (B, 1, 3)
    bone = get_joint_bone(pred_joint, ref_bone_link)  # (B, 1)
    bone = bone.unsqueeze(1)  # (B,1,1)

    pred_joint_noroot = pred_joint - root  # (B,1,3)
    pred_joint_mean = pred_joint_noroot / bone
    pred_chain = [
        pred_joint_mean[:, i, :] - pred_joint_mean[:, JOINTS_PARENT[i], :]
        for i in range(21)
    ]
    pred_chain = pred_chain[1:]  # id 0's parent is itself
    pred_chain = torch.stack(pred_chain, dim=1)  # (B, 20, 3)
    len = torch.norm(pred_chain, dim=-1, keepdim=True)  # (B, 20, 1)
    pred_chain = pred_chain / (len + 1e-5)
    return pred_joint_mean, pred_chain
def get_annot_scale(annots, visibility=None, scale_factor=2.0):
    """
    Retreives the size of the square we want to crop by taking the
    maximum of vertical and horizontal span of the hand and multiplying
    it by the scale_factor to add some padding around the hand
    """
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    max_delta = max(delta_x, delta_y)
    s = max_delta * scale_factor
    return s

def get_joint_bone(joint, ref_bone_link=None):
    if ref_bone_link is None:
        ref_bone_link = (0, 9)

    if (
            not torch.is_tensor(joint)
            and not isinstance(joint, np.ndarray)
    ):
        raise TypeError('joint should be ndarray or torch tensor. Got {}'.format(type(joint)))
    if (
            len(joint.shape) != 3
            or joint.shape[1] != 21
            or joint.shape[2] != 3
    ):
        raise TypeError('joint should have shape (B, njoint, 3), Got {}'.format(joint.shape))

    batch_size = joint.shape[0]
    bone = 0
    if torch.is_tensor(joint):
        bone = torch.zeros((batch_size, 1)).to(joint.device)
        for jid, nextjid in zip(ref_bone_link[:-1], ref_bone_link[1:]):
            bone += torch.norm(joint[:, jid, :] - joint[:, nextjid, :],dim=1, keepdim=True)  # (B, 1)
    elif isinstance(joint, np.ndarray):
        bone = np.zeros((batch_size, 1))
        for jid, nextjid in zip(
                ref_bone_link[:-1], ref_bone_link[1:]
        ):
            bone += np.linalg.norm(
                (joint[:, jid, :] - joint[:, nextjid, :]),
                ord=2, axis=1, keepdims=True
            )  # (B, 1)
    return bone

class HM3D2UVD(nn.Module):
    def __init__(self):
        super(HM3D2UVD, self).__init__()

    def forward(self, hm3d):
        """
        hm3d: B, 21, D, H, W

        """
        d_acc = torch.sum(hm3d,dim=[3,4])
        v_acc = torch.sum(hm3d,dim=[2,4])
        u_acc = torch.sum(hm3d,dim=[2,3])

        w_d = torch.arange(d_acc.shape[-1],dtype=d_acc.dtype,device=d_acc.device)/d_acc.shape[-1]
        w_v = torch.arange(v_acc.shape[-1],dtype=v_acc.dtype,device=v_acc.device)/v_acc.shape[-1]
        w_u = torch.arange(u_acc.shape[-1],dtype=u_acc.dtype,device=u_acc.device)/u_acc.shape[-1]

        d = d_acc.mul(w_d)
        v = v_acc.mul(w_v)
        u = u_acc.mul(w_u)

        d = torch.sum(d,dim=-1,keepdim=True)
        v = torch.sum(v,dim=-1,keepdim=True)
        u = torch.sum(u,dim=-1,keepdim=True)

        uvd = torch.cat([u,v,d],dim=-1)
        return uvd
def uvd2xyz(
        uvd,
        joint_root,
        joint_bone,
        intr=None,
        trans=None,
        scale=None,
        inp_res=256,
        mode='persp'
):
    bs = uvd.shape[0]
    if mode in ['persp', 'perspective']:
        if intr is None:
            raise Exception("No intr found in perspective")
        '''1. denormalized uvd'''
        uv = uvd[:, :, :2] * inp_res  # 0~256
        depth = (uvd[:, :, 2] * 3.0) - 1.5
        root_depth = joint_root[:, -1].unsqueeze(-1)  # (B, 1)
        z = depth * joint_bone.expand_as(uvd[:, :, 2]) + \
            root_depth.expand_as(uvd[:, :, 2])  # B x M

        '''2. uvd->xyz'''
        camparam = torch.zeros((bs, 4)).float().to(intr.device)  # (B, 4)
        camparam[:, 0] = intr[:, 0, 0]  # fx
        camparam[:, 1] = intr[:, 1, 1]  # fx
        camparam[:, 2] = intr[:, 0, 2]  # cx
        camparam[:, 3] = intr[:, 1, 2]  # cy
        camparam = camparam.unsqueeze(1).expand(-1, uvd.size(1), -1)  # B x M x 4
        xy = ((uv - camparam[:, :, 2:4]) / camparam[:, :, :2]) * \
             z.unsqueeze(-1).expand_as(uv)  # B x M x 2
        return torch.cat((xy, z.unsqueeze(-1)), -1)  # B x M x 3
    elif mode in ['ortho', 'orthogonal']:
        if trans is None or scale is None:
            raise Exception("No trans or scale found in orthorgnal")
        raise Exception("orth Unimplement !")
    else:
        raise Exception("Unkonwn mode type. should in ['persp', 'ortho']")
    

def get_annot_center(annots, visibility=None):
    # Get scale
    if visibility is not None:
        annots = annots[visibility]
    min_x, min_y = annots.min(0)
    max_x, max_y = annots.max(0)
    c_x = int((max_x + min_x) / 2)
    c_y = int((max_y + min_y) / 2)
    return np.asarray([c_x, c_y])