import torch
# render gsplat output from output_path
import torch.nn.functional as F
from gaussiansplatting.scene.colmap_loader import (
    read_intrinsics_binary, 
    read_extrinsics_binary,
    qvec2rotmat
)
from gsplat.rendering import rasterization
import numpy as np


cameras_path = "./dataset/llff_data/fern/vggt/sparse/0/cameras.bin"
images_path = "./dataset/llff_data/fern/vggt/sparse/0/images.bin"

# Load gsplat checkpoint
ckpt_path = "tmp_tmp.pt"
ckpt = torch.load(ckpt_path, map_location='cuda')['splats']

# Prepare Gaussian parameters (apply activations)
means = ckpt["means"]
quats = F.normalize(ckpt["quats"], p=2, dim=-1)
scales = torch.exp(ckpt["scales"])
opacities = torch.sigmoid(ckpt["opacities"])
sh0 = ckpt["sh0"]
shN = ckpt["shN"]
colors = torch.cat([sh0, shN], dim=-2)

print(f"Number of Gaussians: {len(means)}")

# Load COLMAP camera data
cameras = read_intrinsics_binary(cameras_path)
images = read_extrinsics_binary(images_path)

print(f"Number of cameras: {len(cameras)}")
print(f"Number of images: {len(images)}")

# Select camera to render (change cam_idx to render different views)
cam_idx = 0
image_id = sorted(images.keys())[cam_idx]
img = images[image_id]
cam = cameras[img.camera_id]

print(f"\nRendering camera {cam_idx}: {img.name}")
print(f"  Image size: {cam.width}x{cam.height}")

# Build viewmat (world-to-camera, 4x4)
R = qvec2rotmat(img.qvec)
t = img.tvec
w2c = np.eye(4)
w2c[:3, :3] = R
w2c[:3, 3] = t
viewmat = torch.from_numpy(w2c).float().cuda()

# Build intrinsic matrix K
fx, fy, cx, cy = cam.params
K = torch.tensor([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=torch.float32).cuda()

width, height = cam.width, cam.height

# Compute sh_degree from colors shape
sh_degree = int(np.sqrt(colors.shape[1]) - 1)
print(f"  SH degree: {sh_degree}")

# Render RGB
render_colors, render_alphas, meta = rasterization(
    means,          # [N, 3]
    quats,          # [N, 4]
    scales,         # [N, 3]
    opacities,      # [N]
    colors,         # [N, S, 3]
    viewmat[None],  # [1, 4, 4]
    K[None],        # [1, 3, 3]
    width,
    height,
    sh_degree=sh_degree,
    render_mode="RGB",
)

rendered_img = render_colors[0]  # [H, W, 3]
print(f"\nRendered image shape: {rendered_img.shape}")
print(f"Rendered image range: [{rendered_img.min():.3f}, {rendered_img.max():.3f}]")