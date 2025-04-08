import torch
from manopth.manopth.manolayer import ManoLayer
from manopth.manopth import demo

batch_size = 10
# Select number of principal components for pose space

# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=False, ncomps=45, flat_hand_mean=True)

# Generate random shape parameters
random_shape = torch.rand(batch_size, 10)
# Generate random pose parameters, including 3 values for global axis-angle rotation
random_pose = torch.rand(batch_size, 45 + 3)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
},
                  mano_faces=mano_layer.th_faces)
