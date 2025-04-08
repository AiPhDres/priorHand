import torch
import torch.nn as nn
from termcolor import cprint
from networks.net.priorHand.backbone import FPN
from networks.net.priorHand.hand_2dhead import hand_2dhead
from networks.net.priorHand.hand_3dhead import hand_3dhead
from utils.handutils import uvd2xyz,HM3D2UVD
from utils.misc import param_count

class hand_encoder(nn.Module):
    def __init__(
            self,
            net_parts,
            stacks=2,
            blocks=1,
            num_joints=21,
            in_chs = 256,
            h=64,
            w=64,

    ):
        super(hand_encoder, self).__init__()
        self.in_chs = in_chs
        self.stacks = stacks
        self.blocks = blocks
        self.num_joints = num_joints
        self.h = h
        self.w = w
        self.HM3D2UVD = HM3D2UVD()
        self.net_parts =["FPN","hand_2dhead"] + [
            "hand_3dhead" if ("hand_3dhead" in net_parts) else " "
        ]
        
        self.backbone = FPN()
        cprint('params FPN: {:.3f}M'.format(param_count(self.backbone)), 'red')
        
        self.hand_2dhead = hand_2dhead(
            joint_nb=self.num_joints,
            stacks=self.stacks,
            blocks=self.blocks,
            in_channels=self.in_chs,
        )
        cprint('params hand_2dhead: {:.3f}M'.format(
            param_count(self.hand_2dhead)), 'blue')

        if "hand_3dhead" in net_parts:
            self.hand_3dhead = hand_3dhead(
                joint_nb=self.num_joints,
                stacks=self.stacks,
                blocks=self.blocks,
            )
            cprint('params hand_3dhead: {:.3f}M'.format(
                param_count(self.hand_3dhead)), 'green')

    def forward(self, x, info):
        joint_root = info["joint_root"]
        joint_bone = info["joint_bone"]

        feat = self.backbone(x)

        pred_hm, pred_seg, encoding = self.hand_2dhead(feat)

        pred_uvd,pred_dep, pred_joint,temp = [],[],[],0

        if "hand_3dhead" in self.net_parts:
            pred_hm3d, pred_dep,temp = self.hand_3dhead(pred_hm[-1],
                                                        pred_seg[-1],
                                                        encoding[-1])
            for i in range(len(pred_hm3d)):
                hm3d = pred_hm3d[i]
                uvd = self.HM3D2UVD(hm3d)
                pred_uvd.append(uvd)
            for i in range(len(pred_uvd)):
                joint = uvd2xyz(
                    pred_uvd[i],joint_root,joint_bone,
                    intr=info["intr"],mode="persp"
                )
                pred_joint.append(joint)

            pred_results = {
                "pred_hm2d": pred_hm,
                'pred_hm3d': pred_hm3d,
                "pred_seg": pred_seg,
                "pred_dep": pred_dep,
                "pred_uvd": pred_uvd,
                "pred_joint":pred_joint
            }
        else:
            pred_results = {
                "pred_hm2d": pred_hm,
                "pred_seg": pred_seg,
            }

        temp_results = {
            "encoding": encoding[-1],
            "temp": temp,
            'feat':feat
        }
        return pred_results, temp_results

if __name__ == "__main__":
    from backbone import FPN
    img = torch.rand(1, 3, 256, 256).float()
    joint_root = torch.randn(1,3).float()
    joint_bone = torch.randn(1,1).float()
    intr = torch.randn(1,3,3).float()
    metas = {
        'joint_root': joint_root,
        'joint_bone': joint_bone,
        "intr": intr,
    }
    net_parts = ["FPN","hand_2dhead", "hand_3dhead","mano_head"]
    net = hand_encoder(net_parts=net_parts)
    pred_results, temp_results = net(img, metas)
    print(pred_results["pred_hm2d"][-1].shape)
    print(pred_results["pred_seg"][-1].shape)
    print(pred_results["pred_dep"][-1].shape)
    print(pred_results["pred_uvd"][-1].shape)
    print(pred_results["pred_joint"][-1].shape)

    print(temp_results["encoding"][-1].shape)
    print(temp_results["temp"][-1].shape)