
import os
import json
from typing import Dict, List, Optional
from glob import glob

from base.viewpoint import FrameData, ViewpointData


class Stanford2D3DDataLoader:
    """Organizes the Stanford2D3D dataset, containing multiple viewpoints."""
    
    def __init__(self, root_dir: str):
        """Initialize the data loader with the root directory."""
        self.root_dir = root_dir
        self.viewpoints = []
        

    def get_viewpoints(self):
        """Return all viewpoints."""
        return self.viewpoints

    def load_data(self):
        """Load and organize all dataset files."""
        # Load all files at once into lists

        rgb_files = os.listdir(os.path.join(self.root_dir, "data", "rgb"))
        depth_files = os.listdir(os.path.join(self.root_dir, "data", "depth"))
        normal_files = os.listdir(os.path.join(self.root_dir, "data", "normal"))
        semantic_files = os.listdir(os.path.join(self.root_dir, "data", "semantic"))
        semantic_pretty_files = os.listdir(os.path.join(self.root_dir, "data", "semantic_pretty"))
        pose_files = os.listdir(os.path.join(self.root_dir, "data", "pose"))


        pano_files = os.listdir(os.path.join(self.root_dir, "pano", "rgb"))

        # pano_files[0].split("_")[1]     
        view_ids = ['_'.join(f.split("_")[1:4]) for f in pano_files if "camera" in f]


        for view_id in view_ids:
            viewpoint = self._generate_viewpoint_data(view_id)
            if not viewpoint is None:
                self.viewpoints.append(viewpoint)


    def num_viewpoints(self):
        return len(self.viewpoints)

    def _generate_viewpoint_data(self, view_id) -> Optional[ViewpointData]:
        frames = []

        has_all_frames = True
        for i in range(0, 64):
            frame_id = f"{view_id}_frame_{str(i)}"

            frame_data = self._generate_frame_data(frame_id)
            frames.append(frame_data)
            if frame_data is None:
                has_all_frames = False

        if not has_all_frames:
            return None

        return ViewpointData(view_id=view_id, frames=frames)


    def _generate_frame_data(self, frame_id):
        """Generate frame data for all viewpoints."""

        rgb_name = "camera_" + frame_id + "_domain_rgb.png"
        pose_name = "camera_" + frame_id + "_domain_pose.json"

        rgb_path = os.path.join(self.root_dir, "data", "rgb", rgb_name)
        pose_path = os.path.join(self.root_dir, "data", "pose", pose_name)

        if not (os.path.exists(rgb_path) and os.path.exists(pose_path)):
            return None  

        return FrameData(rgb_path=rgb_path, pose_path=pose_path)

        
        pose_data = json.load(open(pose_path))

        import ipdb
        ipdb.set_trace()




