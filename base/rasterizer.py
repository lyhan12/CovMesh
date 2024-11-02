import torch
import json
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardFlatShader,
    RasterizationSettings, PerspectiveCameras
)

from utils.helper import parse_pose_data

import cv2


class Rasterizer():

    def __init__(self):
        pass

    def calculate_face_counts(self, mesh, frame_data, device="cuda"):
        fragments = self.rasterize(mesh, frame_data, device=device)
        counts = fragments.pix_to_face[0].unique(return_counts=True)
        return counts


    def rasterize(self, mesh, frame_data, device="cuda"):
            # Parse the pose data
        R, T, K = parse_pose_data(frame_data.pose_data)

        R = R.T
        T = -R @ T

        # R_xyz_90 = torch.tensor(\
        #         [[0, 0, 1], \
        #         [0, 1, 0], \
        #         [-1, 0, 0]], dtype=torch.float32, device=device)



        R_xyz_90 = torch.tensor(\
                [[1, 0, 0], \
                [0, 0, 1], \
                [0, -1, 0]], dtype=torch.float32, device=device)


        #         [[0, 0, 1], \
        #         [1, 0, 0], \
        #         [0, 1, 0]], dtype=torch.float32, device=device)

        R = R_xyz_90 @ R
        T = R_xyz_90 @ T

        R[:, 0] = -R[:, 0]
        R[:, 1] = -R[:, 1]



        T = -R.T @ T
        # R = R.T

        # R = R.t()

        # R = R.t()


        
        # Adjust extrinsics if necessary

        # Normalize focal lengths
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # fx /= frame_data.rgb.shape[1]
        # fy /= frame_data.rgb.shape[0]

        # Configure the camera with intrinsics and extrinsics


        image_size = frame_data.rgb.shape[:2]

        # Configure raster settings
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        cameras = PerspectiveCameras(
            focal_length=[(fx, fy)],
            principal_point=[(cx, cy)],
            device=device,
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            # K=K.unsqueeze(0),
            image_size=[image_size],
            in_ndc=False,
        )

        # Set up the rasterizer
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # Render the image
        fragments = rasterizer(mesh)

        # rgb = mesh.sample_textures(fragments)
        # cv2.imwrite("test.png", cv2.cvtColor(rgb.squeeze().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # save rgb->bgr

        counts = fragments.pix_to_face[0].unique(return_counts=True)

        return fragments


