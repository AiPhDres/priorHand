import os

from networks.net.priorHand.hand_encoder import hand_encoder
from networks.net.priorHand.mano_head import mano_head
import  torch.nn as nn
import utils.handutils as handutils
from utils import misc
from utils.misc import param_count
from termcolor import cprint
class EasyHandModel(nn.Module):
    def __init__(
            self,
            subnet=None,
            num_joints=21,
            in_channels=256,
            out_hm_HW=64,
            out_depth_HW=64,
            stacks = 2,
            blocks=1
    ):
        super(EasyHandModel, self).__init__()
        if subnet is None:
            subnet = ["FPN","hand_2dhead", "hand_3dhead", "mano_head"]
        self.inchannels = in_channels
        self.subnet = subnet
        self.num_joints = num_joints
        self.hand_encoder = hand_encoder(
            num_joints=self.num_joints,
            in_chs=self.inchannels,
            h=out_hm_HW,
            w=out_depth_HW,
            blocks=blocks,
            stacks=stacks,
            net_parts=self.subnet
        )
         
        cprint("params hand encoder: {:.3f}M".format(param_count(self.hand_encoder)),
            "green", attrs=["bold"])
        if "mano_head" in self.subnet:
            self.mano_head = mano_head()
            cprint("params mano_head: {:.3f}M".format(param_count(self.mano_head)),
                "blue", attrs=["bold"])

    def load_checkpoints(self,
                         ckp_backbone=None,
                         ckp_hand2dhead=None,
                         ckp_hand3dhead=None,
                         ckp_manohead=None
                         ):
        # NOTE: File exist check! (refer to issue #1)
        if ckp_backbone and not os.path.isfile(ckp_backbone):
            cprint(f"{ckp_backbone} is not exist", "red")
        if ckp_hand2dhead and not os.path.isfile(ckp_hand2dhead):
            cprint(f"{ckp_hand2dhead} is not exist", "red")
        if ckp_hand3dhead and not os.path.isfile(ckp_hand3dhead):
            cprint(f"{ckp_hand3dhead} is not exist", "red")
        if ckp_manohead and not os.path.isfile(ckp_manohead):
            cprint(f"{ckp_manohead} is not exist", "red")
        
        if ckp_backbone and os.path.isfile(ckp_backbone)and "FPN" in self.subnet:
            misc.load_checkpoint_ik(self.hand_encoder.backbone, ckp_backbone)

        if ckp_hand2dhead and os.path.isfile(ckp_hand2dhead) and "hand_2dhead" in self.subnet:
            misc.load_checkpoint_ik(self.hand_encoder.hand_2dhead, ckp_hand2dhead)

        if ckp_hand3dhead and os.path.isfile(ckp_hand3dhead) and "hand_3dhead" in self.subnet:
            misc.load_checkpoint_ik(self.hand_encoder.hand_3dhead, ckp_hand3dhead)

        if ckp_manohead and os.path.isfile(ckp_manohead) and "mano_head" in self.subnet:
            misc.load_checkpoint_ik(self.mano_head, ckp_manohead)

    def forward(self, x, ref_infos):
        """
        hand 2d & 3d infos
        """
        pred_results, temp_results = self.hand_encoder(x, ref_infos)
        """
        encoding mano infos
        """
        mano_results = {}
        if "mano_head" in self.subnet:
            pred_joint = pred_results["pred_joint"]
            pred_joint_norm, ref_chain = handutils.parse_input(pred_joint[-1], ref_infos)
            mano_results = self.mano_head(pred_joint_norm,ref_chain)

        results = {
            **pred_results,
            **mano_results
        }
        return results
    