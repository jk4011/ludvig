import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from einops import rearrange
from jhutil import rgb_to_sh0
from vggt import vggt_inference
from src.vggt_to_3dgs import vggt_to_3dgs


if __name__ == "__main__":

    scene_list = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    for scene in scene_list:
        print(f"preprocess {scene}...")
        image_folder = f"./dataset/llff_data/{scene}/images"
        vggt_out = vggt_inference(image_folder=image_folder)
        output_dir = f"./dataset/llff_data/{scene}/vggt"
        vggt_to_3dgs(vggt_out, output_dir=output_dir, pointmap_indices=[0, 2, 5, 7, 10])
        
        try:
            os.symlink(f"../images", f"{output_dir}/images", )
        except:
            pass