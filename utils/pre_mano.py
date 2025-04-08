from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle

class MANOHandJoints:
    n_joints = 21

    labels = [
        'W', #0
        'I0', 'I1', 'I2', #3
        'M0', 'M1', 'M2', #6
        'L0', 'L1', 'L2', #9
        'R0', 'R1', 'R2', #12
        'T0', 'T1', 'T2', #15
        'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
    ]

    # finger tips are not joints in MANO, we label them on the mesh manually
    mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

    parents = [
        None,
        0, 1, 2,
        0, 4, 5,
        0, 7, 8,
        0, 10, 11,
        0, 13, 14,
        3, 6, 9, 12, 15
    ]


class MPIIHandJoints:
    n_joints = 21

    labels = [
        'W', #0
        'T0', 'T1', 'T2', 'T3', #4
        'I0', 'I1', 'I2', 'I3', #8
        'M0', 'M1', 'M2', 'M3', #12
        'R0', 'R1', 'R2', 'R3', #16
        'L0', 'L1', 'L2', 'L3', #20
    ]

    parents = [
        None,
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]


def mpii_to_mano(mpii):
    mano = []
    for j in range(MANOHandJoints.n_joints):
        mano.append(
        mpii[MPIIHandJoints.labels.index(MANOHandJoints.labels[j])]
        )
    mano = np.stack(mano, 0)
    return mano


def mano_to_mpii(mano):
    mpii = []
    for j in range(MPIIHandJoints.n_joints):
        mpii.append(
        mano[MANOHandJoints.labels.index(MPIIHandJoints.labels[j])]
        )
    mpii = np.stack(mpii, 0)
    return mpii


def xyz_to_delta(xyz, joints_def):

    delta = []
    for j in range(joints_def.n_joints):
        p = joints_def.parents[j]
        if p is None:
            delta.append(np.zeros(3))
        else:
            delta.append(xyz[j] - xyz[p])
    delta = np.stack(delta, 0)
    lengths = np.linalg.norm(delta, axis=-1, keepdims=True)
    delta /= np.maximum(lengths, np.finfo(xyz.dtype).eps)
    return delta, lengths
def xyz_to_delta_batch(xyz_batch, joints_def):
    """
    Batch version of xyz_to_delta using PyTorch.

    Parameters
    ----------
    xyz_batch : torch.Tensor, shape [N, J, 3]
        Batch of joint coordinates, where N is the batch size and J is the number of joints.
    joints_def : object
        Joint definition object that contains the parents of each joint.

    Returns
    -------
    delta_batch : torch.Tensor, shape [N, J, 3]
        Batch of bone orientations.
    lengths_batch : torch.Tensor, shape [N, J, 1]
        Batch of bone lengths.
    """
    # Get the number of joints
    n_joints = joints_def.n_joints

    # Initialize delta and lengths tensors
    device = xyz_batch.device
    delta_batch = torch.zeros((xyz_batch.shape[0], n_joints, 3), dtype=xyz_batch.dtype, device=device)
    lengths_batch = torch.zeros((xyz_batch.shape[0], n_joints, 1), dtype=xyz_batch.dtype, device=device)

    # Iterate over each joint
    for j in range(n_joints):
        p = joints_def.parents[j]
        if p is None:
            # If no parent, set delta to zero and length to zero
            delta_batch[:, j, :] = 0.0
            lengths_batch[:, j, :] = 0.0
        else:
            # Compute delta as child minus parent
            delta_batch[:, j, :] = xyz_batch[:, j, :] - xyz_batch[:, p, :]
            lengths_batch[:, j, :] = torch.norm(delta_batch[:, j, :], dim=-1, keepdim=True)

    # Normalize delta by dividing by the length
    eps = torch.tensor(1e-6, dtype=xyz_batch.dtype, device=xyz_batch.device)  # 定义 eps 为一个 PyTorch 张量
    delta_batch = delta_batch / torch.maximum(lengths_batch, eps)

    return delta_batch, lengths_batch

def prepare_mano():
    with open('D:/DeepLearning/minimal_hand/mano/models/MANO_LEFT.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    output = {}
    output['verts'] = np.array(data['v_template'])
    output['faces'] = np.array(data['f'])
    output['mesh_basis'] = np.transpose(data['shapedirs'], (2, 0, 1))

    j_regressor = np.zeros([21, 778])
    j_regressor[:16] = data['J_regressor'].toarray()
    for k, v in MANOHandJoints.mesh_mapping.items():
        j_regressor[k, v] = 1
    output['j_regressor'] = j_regressor
    output['joints'] = np.matmul(output['j_regressor'], output['verts'])

    raw_weights = data['weights']
    weights = [None] * 21
    weights[0] = raw_weights[:, 0]
    for j in 'IMLRT':
        weights[MANOHandJoints.labels.index(j + '0')] = np.zeros(778)
        for k in [1, 2, 3]:
            src_idx = MANOHandJoints.labels.index(j + str(k - 1))
            tar_idx = MANOHandJoints.labels.index(j + str(k))
            weights[tar_idx] = raw_weights[:, src_idx]
    output['weights'] = np.expand_dims(np.stack(weights, -1), -1)
    with open('D:/DeepLearning/minimal_hand/utils/mano_ref.pkl', 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    prepare_mano()
    # with open('D:/DeepLearning/minimal_hand/utils/mano_ref.pkl', 'rb') as f:
    #     data = pickle.load(f, encoding='latin1')
    # print(data.keys())
    # print(data['verts'].shape)
    # print(data['faces'].shape)