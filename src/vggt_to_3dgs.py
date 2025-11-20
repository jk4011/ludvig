"""
VGGT output to 3DGS format converter.

Converts VGGT predictions to 3D Gaussian Splatting format:
- point_cloud.ply: Gaussian attributes (xyz, sh, opacity, scale, rotation)
- cameras.bin: COLMAP binary format for camera intrinsics
- images.bin: COLMAP binary format for camera extrinsics
"""

import os
import struct
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from einops import rearrange

from gaussiansplatting.scene.colmap_loader import rotmat2qvec


def inverse_sigmoid(x):
    """Inverse of sigmoid function."""
    return np.log(x / (1 - x))


def rgb_to_sh0(rgb):
    """Convert RGB values to 0th order spherical harmonics coefficient."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def vggt_to_3dgs(
    pred_dict,
    output_dir,
    sh_degree=0,
    init_opacity=0.1,
    extrinsic_is_c2w=False,
    pointmap_indices=None,
    pixel_scale_factor=0.35,
):
    """
    Convert VGGT output to 3DGS format.

    Args:
        pred_dict: VGGT prediction dictionary containing:
            - world_points_from_depth: (B, H, W, 3) 3D points
            - images: (B, 3, H, W) RGB images
            - depth: (B, H, W, 1) depth maps
            - extrinsic: (B, 3, 4) camera extrinsics [R|t]
            - intrinsic: (B, 3, 3) camera intrinsics
        output_dir: Output directory path
        sh_degree: Spherical harmonics degree (default: 0)
        init_opacity: Initial opacity value before inverse sigmoid (default: 0.1)
        extrinsic_is_c2w: If True, extrinsic is camera-to-world; else world-to-camera (default: False)
        pointmap_indices: Optional indices to select specific point maps (default: None)
        pixel_scale_factor: Scale factor for Gaussian size relative to 1 pixel (default: 0.35)
                           Smaller values = tighter Gaussians. 0.35 gives ~1 pixel coverage (3σ radius)

    Returns:
        dict: Paths to saved files
    """
    # Extract data from pred_dict
    world_points = pred_dict["world_points_from_depth"].float()  # (B, H, W, 3)
    images = pred_dict["images"]  # (B, 3, H, W)
    depth = pred_dict["depth"]  # (B, H, W, 1)
    if pointmap_indices is not None:
        world_points = world_points[pointmap_indices]
        images = images[pointmap_indices]
        depth = depth[pointmap_indices]

    extrinsics = pred_dict["extrinsic"]  # (B, 3, 4)
    intrinsics = pred_dict["intrinsic"]  # (B, 3, 3)
    image_names = pred_dict["image_names"]
    image_names = [name.split("/")[-1] for name in image_names]

    # Convert to numpy if torch tensors
    if isinstance(world_points, torch.Tensor):
        world_points = world_points.cpu().numpy()
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.cpu().numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()

    b, h, w, _ = world_points.shape
    n = b * h * w

    # Create output directories
    ply_dir = os.path.join(output_dir, "point_cloud", "iteration_0")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # === 1. Save point_cloud.ply ===
    ply_path = os.path.join(ply_dir, "point_cloud.ply")
    save_gaussian_ply(
        world_points=world_points,
        images=images,
        depth=depth,
        intrinsics=intrinsics,
        output_path=ply_path,
        sh_degree=sh_degree,
        init_opacity=init_opacity,
        pixel_scale_factor=pixel_scale_factor,
    )

    # === 2. Save cameras.bin ===
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    write_cameras_binary(
        intrinsics=intrinsics,
        height=h,
        width=w,
        output_path=cameras_path,
    )

    # === 3. Save images.bin ===
    images_path = os.path.join(sparse_dir, "images.bin")
    write_images_binary(
        extrinsics=extrinsics,
        image_names=image_names,
        output_path=images_path,
        extrinsic_is_c2w=extrinsic_is_c2w,
    )

    # === 4. Save points3D.bin (0 points) ===
    points_path = os.path.join(sparse_dir, "points3D.bin")
    with open(points_path, "wb") as f:
        # Write number of points (0) as unsigned long long (8 bytes)
        f.write(struct.pack("<Q", 0))

    print(f"Saved point_cloud.ply to {ply_path}")
    print(f"Saved cameras.bin to {cameras_path}")
    print(f"Saved images.bin to {images_path}")

    return {
        "ply_path": ply_path,
        "cameras_path": cameras_path,
        "images_path": images_path,
    }


def save_gaussian_ply(
    world_points,
    images,
    depth,
    intrinsics,
    output_path,
    sh_degree=0,
    init_opacity=0.1,
    pixel_scale_factor=0.35,
):
    """
    Save Gaussian attributes to PLY file.

    Args:
        world_points: (B, H, W, 3) 3D point positions
        images: (B, 3, H, W) RGB images
        depth: (B, H, W, 1) depth maps
        intrinsics: (B, 3, 3) camera intrinsics
        output_path: Output PLY file path
        sh_degree: Spherical harmonics degree
        init_opacity: Initial opacity value
        pixel_scale_factor: Scale factor for Gaussian size relative to 1 pixel projection
    """
    b, h, w, _ = world_points.shape
    n = b * h * w

    # Flatten points: (B, H, W, 3) -> (N, 3)
    xyz = world_points.reshape(-1, 3).astype(np.float32)

    # Normals (zeros)
    normals = np.zeros_like(xyz)

    # Convert RGB to SH0: (B, 3, H, W) -> (N, 3)
    rgb = np.transpose(images, (0, 2, 3, 1))  # (B, H, W, 3)
    rgb = rgb.reshape(-1, 3)  # (N, 3)
    sh0 = rgb_to_sh0(rgb).astype(np.float32)  # (N, 3)

    # SH rest coefficients (zeros for sh_degree=0)
    num_sh_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    sh_rest = np.zeros((n, num_sh_rest), dtype=np.float32)

    # Opacity: inverse sigmoid of init_opacity
    opacities = np.full((n, 1), inverse_sigmoid(init_opacity), dtype=np.float32)

    # Scale: Gaussian size in world space
    # Goal: When projected to 2D, each Gaussian should cover ~1 pixel
    #
    # Projection formula: projected_scale = focal_length * world_scale / depth
    # For 1 pixel: world_scale = depth / focal_length
    # But we want σ (standard deviation), not full width
    #
    # Gaussian contribution: exp(-0.5 * (r/σ)²)
    # Most contribution (99.7%) is within 3σ radius
    # To have ~1 pixel coverage (3σ ≈ 1 pixel): σ ≈ 0.33 pixel
    # Therefore: pixel_scale_factor ≈ 0.33

    depth_flat = depth.reshape(b, -1)  # (B, H*W)
    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]  # (B,)
    f_avg = (fx + fy) / 2  # (B,)

    # Compute scale for each point
    scales_per_view = []
    for i in range(b):
        view_depth = depth_flat[i]  # (H*W,)
        # Apply pixel_scale_factor to reduce Gaussian size
        view_scale = (view_depth / f_avg[i]) * pixel_scale_factor
        scales_per_view.append(view_scale)

    scales_flat = np.concatenate(scales_per_view)  # (N,)

    # 3DGS stores log(scale)
    log_scales = np.log(np.maximum(scales_flat, 1e-7))  # (N,)
    scales = np.stack([log_scales, log_scales, log_scales], axis=-1).astype(np.float32)  # (N, 3)

    # Rotation: identity quaternion [w, x, y, z] = [1, 0, 0, 0]
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    # Construct PLY attributes
    attr_names = ["x", "y", "z", "nx", "ny", "nz"]

    # SH DC coefficients
    for i in range(3):
        attr_names.append(f"f_dc_{i}")

    # SH rest coefficients
    for i in range(num_sh_rest):
        attr_names.append(f"f_rest_{i}")

    attr_names.append("opacity")

    for i in range(3):
        attr_names.append(f"scale_{i}")

    for i in range(4):
        attr_names.append(f"rot_{i}")

    dtype_full = [(attr, "f4") for attr in attr_names]

    # Concatenate all attributes
    elements = np.empty(n, dtype=dtype_full)
    attributes = np.concatenate(
        [xyz, normals, sh0, sh_rest, opacities, scales, rotations],
        axis=1
    )
    elements[:] = list(map(tuple, attributes))

    # Save PLY
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)


def write_cameras_binary(intrinsics, height, width, output_path):
    """
    Write cameras.bin in COLMAP binary format.

    Args:
        intrinsics: (B, 3, 3) camera intrinsic matrices
        height: Image height
        width: Image width
        output_path: Output file path
    """
    b = intrinsics.shape[0]

    with open(output_path, "wb") as f:
        # Write number of cameras
        f.write(struct.pack("<Q", b))

        for i in range(b):
            K = intrinsics[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            camera_id = i + 1
            model_id = 1  # PINHOLE

            # Write camera properties: camera_id, model_id, width, height
            f.write(struct.pack("<iiQQ", camera_id, model_id, width, height))

            # Write PINHOLE parameters: fx, fy, cx, cy
            f.write(struct.pack("<dddd", fx, fy, cx, cy))


def write_images_binary(extrinsics, image_names, output_path, extrinsic_is_c2w=True):
    """
    Write images.bin in COLMAP binary format.

    Args:
        extrinsics: (B, 3, 4) camera extrinsic matrices [R|t]
        image_names: List of image file names
        output_path: Output file path
        extrinsic_is_c2w: If True, convert c2w to w2c for COLMAP format
    """
    b = extrinsics.shape[0]

    with open(output_path, "wb") as f:
        # Write number of images
        f.write(struct.pack("<Q", b))

        for i in range(b):
            image_id = i + 1
            camera_id = i + 1

            # Extract R and t from extrinsic matrix
            R = extrinsics[i, :3, :3]  # (3, 3)
            t = extrinsics[i, :3, 3]   # (3,)

            # Convert c2w to w2c if needed (COLMAP uses w2c)
            if extrinsic_is_c2w:
                R_w2c = R.T
                t_w2c = -R.T @ t
                R = R_w2c
                t = t_w2c

            # Convert rotation matrix to quaternion
            qvec = rotmat2qvec(R)  # (4,) [w, x, y, z]

            # Write image properties: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id
            f.write(struct.pack(
                "<idddddddi",
                image_id,
                qvec[0], qvec[1], qvec[2], qvec[3],
                t[0], t[1], t[2],
                camera_id
            ))

            # Write image name (null-terminated string)
            image_name = image_names[i]
            f.write(image_name.encode("utf-8"))
            f.write(b"\x00")

            # Write number of 2D points (0 for VGGT output)
            f.write(struct.pack("<Q", 0))


if __name__ == "__main__":
    # Example usage
    print("VGGT to 3DGS converter")
    print("Usage: from src.vggt_to_3dgs import vggt_to_3dgs")
    print("       vggt_to_3dgs(pred_dict, output_dir)")
