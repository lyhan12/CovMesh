

from dataclasses import dataclass, field

from typing import Dict, List, Optional
import cv2
import json

import torch

@dataclass
class FrameData:
    """Holds data for a single frame (depth, normal, pose, rgb, semantic)."""
    depth_path: Optional[str] = None
    normal_path: Optional[str] = None
    pose_path: Optional[str] = None
    rgb_path: Optional[str] = None
    semantic_path: Optional[str] = None
    semantic_pretty_path: Optional[str] = None

    @property
    def rgb(self):
        return cv2.imread(self.rgb_path)

    @property
    def pose_data(self):
        return json.load(open(self.pose_path))


@dataclass
class ViewpointData:
    view_id: str
    frames: List[FrameData]

    rgb_path: Optional[str] = None
    pose_path: Optional[str] = None

    desc: Optional[torch.Tensor] = None

    @property
    def rgb(self):
        return cv2.imread(self.rgb_path)

    @property
    def pose_data(self):
        return json.load(open(self.pose_path))

    def __getitem__(self, i):
        return self.frames[i]

