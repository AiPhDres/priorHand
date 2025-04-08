import torch
import torch.nn as nn
import utils.quatutils as quatutils
import utils.handutils as handutils
from manopth.manopth.manolayer import ManoLayer



class mano_head(nn.Module):
    def __init__(
            self,
            num_joints=21,
            dropout=0,
    ):
        super(mano_head, self).__init__()

        ''' quat '''
        hidden_neurons_quat = [256, 512, 1024, 1024, 512, 256]
        in_neurons_quat = 3 * num_joints + 3 * (num_joints - 1)
        out_neurons_quat = 16 * 4  # joints quats representation
        neurons_quat = [in_neurons_quat] + hidden_neurons_quat

        quat_fc_layers = []
        for layer_idx, (inps, outs) in enumerate(
                zip(neurons_quat[:-1], neurons_quat[1:])
        ):
            if dropout:
                quat_fc_layers.append(nn.Dropout(p=dropout))
            quat_fc_layers.append(nn.Linear(inps, outs))
            quat_fc_layers.append(nn.ReLU())

        quat_fc_layers.append(nn.Linear(neurons_quat[-1], out_neurons_quat))

        self.quat_fc_layers = nn.Sequential(*quat_fc_layers)

        ''' shape '''
        hidden_neurons_shape = [128, 256, 512, 256, 128]
        in_neurons_shape = num_joints * 3
        out_neurons_shape = 10
        neurons_shape = [in_neurons_shape] + hidden_neurons_shape

        shape_fc_layers = []
        for layer_idx, (inps, outs) in enumerate(
                zip(neurons_shape[:-1], neurons_shape[1:])
        ):
            if dropout:
                shape_fc_layers.append(nn.Dropout(p=dropout))
            shape_fc_layers.append(nn.Linear(inps, outs))
            shape_fc_layers.append(nn.ReLU())

        shape_fc_layers.append(nn.Linear(neurons_shape[-1], out_neurons_shape))
        self.shape_fc_layers = nn.Sequential(*shape_fc_layers)

        self.mano_layer = ManoLayer(
            center_idx=9,
            side="right",
            mano_root="D:/DeepLearning/minimal_hand/mano/models",
            use_pca=False,
            flat_hand_mean=True,
        )

        self.ref_bone_link = (0, 9)  # mid mcp
        self.joint_root_idx = 9  # root

    def forward(self, pred_joints_r, pred_chain):
        batch_size = pred_joints_r.shape[0]
        x = torch.cat((pred_joints_r, pred_chain), dim=1)
        x = x.reshape(batch_size, -1)
        quat = self.quat_fc_layers(x)
        quat = quat.reshape(batch_size, 16, 4)

        y = pred_joints_r.reshape(batch_size, -1)
        beta = self.shape_fc_layers(y)
        quat_norm = quatutils.normalize_quaternion(quat)
        theta_pose_coef = quatutils.quaternion_to_angle_axis(quat_norm)
        theta_pose_coef = theta_pose_coef.reshape(batch_size, -1)

        verts_mano, joint_mano, _ = self.mano_layer(
            th_pose_coeffs=theta_pose_coef,
            th_betas=beta
        )

        bone_pred = handutils.get_joint_bone(joint_mano, self.ref_bone_link)  # (B, 1)
        bone_pred = bone_pred.unsqueeze(1)  # (B,1,1)
        joint_mean = joint_mano / bone_pred
        verts_mean = verts_mano / bone_pred

        results = {
            'verts_mean': verts_mean,
            'joint_mean': joint_mean,
            'quat': quat,
            'beta': beta,
            'theta_pose_coef': theta_pose_coef,
            'verts_mano': verts_mano,
            'joint_mano': joint_mano,
        }
        
        return results

if __name__ =="__main__":
    infos = {
        "joint_root":torch.randn(1,3)
    }
    pred_j = torch.randn(1,21,3)
    manohead = mano_head()
    pred_joint_mean, pred_chain = handutils.parse_input(pred_j, infos)
    print(pred_joint_mean.shape)
    print(pred_chain.shape)
    mano_res = manohead(pred_joint_mean, pred_chain)
    print(mano_res['verts_mean'].shape)
    print(mano_res['joint_mean'].shape)
    print(mano_res['quat'].shape)
    print(mano_res['beta'].shape)
    print(mano_res['theta_pose_coef'].shape)