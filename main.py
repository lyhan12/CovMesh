
import argparse

from base.data_loader_standford2d3d import Stanford2D3DDataLoader

import os

from utils.helper import load_mesh, parse_pose_data


import numpy as np

from tqdm import tqdm
import time

from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardFlatShader,
    RasterizationSettings, PerspectiveCameras
)
from pytorch3d.structures import Meshes
import torch

import cv2


import scipy.sparse as sp


def main(args):
    # Predefined list of supported data types

    num_nns_str = args.num_nn.split(",")
    num_nns = [int(num_nn) for num_nn in num_nns_str]

    data_dir = args.data_dir
    device = args.device

    print(f"Data Directory: {data_dir}")

    # Initialize and load the data using Stanford2D3DDataLoader
    data_loader = Stanford2D3DDataLoader(root_dir=data_dir)
    data_loader.load_data()  # Load the data

    pano_files = os.listdir(os.path.join(data_dir, "pano", "rgb"))
    pano_files = [f for f in pano_files if "camera" in f]
    print("Number of Pano Files:", len(pano_files))
    print("Number of loaded viewpoints:", data_loader.num_viewpoints())

    
    # # initialize to numeric max
    # min_x = min_y = min_z = np.inf
    # max_x = max_y = max_z = -np.inf

    # with open("xyz.txt", "w") as f:
    #     for viewpoint in data_loader.viewpoints:
    #         print(viewpoint.view_id)
    #         frame = viewpoint[0]
    #         R,T,K = parse_pose_data(frame.pose_data)
    #         R = R.t()
    #         T = -R @ T
    #         x, y, z = T[0].item(), T[1].item(), T[2].item()
    #         f.write(f"{x} {y} {z}\n")


    # for viewpoint in data_loader.viewpoints:
    #     print(viewpoint.view_id)

    #     frame = viewpoint[0]

    #     R,T,K = parse_pose_data(frame.pose_data)

    #     R = R.t()
    #     T = -R @ T

    #     x, y, z = T[0].item(), T[1].item(), T[2].item()



    #     if T[0] < min_x:
    #         min_x = T[0]
    #     if T[1] < min_y:
    #         min_y = T[1]
    #     if T[2] < min_z:
    #         min_z = T[2]
    #     if T[0] > max_x:
    #         max_x = T[0]
    #     if T[1] > max_y:
    #         max_y = T[1]
    #     if T[2] > max_z:
    #         max_z = T[2]
    #     # print(R,T)

    # #extract value
    # min_x = min_x.item()
    # min_y = min_y.item()
    # min_z = min_z.item()
    # max_x = max_x.item()
    # max_y = max_y.item()
    # max_z = max_z.item()

    # print("Ranges:" , min_x, max_x, min_y, max_y, min_z, max_z)



    print("Loading mesh...")
    start_time = time.time()
    mesh_path = os.path.join(data_dir, args.mesh_path)


    if mesh_path[-3:] == "obj":
        mesh = load_objs_as_meshes([mesh_path], device=device, create_texture_atlas=True)
    elif mesh_path[-3:] == "ply":
        verts, faces = load_ply(mesh_path)
        mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])

    end_time = time.time()
    print("Finished loading mesh")
    print("Time taken to load mesh:", end_time - start_time, "s")

    # for viewpoint_data in data_loader.viewpoints:
    #     for frame_data in viewpoint_data.frames:
    #         fragments = rasterizer.rasterize(mesh, frame_data)



    half_size = 320
    fs = [(half_size, half_size)] * 6
    cs = [(half_size, half_size)] * 6
    image_sizes = [(2*half_size, 2*half_size)] * 6
    image_size = image_sizes[0]
    Rs = torch.zeros(6, 3, 3, device=device)
    Ts = torch.zeros(6, 3, device=device)

    R_front = torch.eye(3, device=device)
    R_back = torch.tensor(\
            [[-1, 0, 0], \
            [0, 1, 0], \
            [0, 0, -1]], dtype=torch.float32, device=device)
    R_left = torch.tensor(\
            [[0, 0, 1], \
            [0, 1, 0], \
            [-1, 0, 0]], dtype=torch.float32, device=device)
    R_right = torch.tensor(\
            [[0, 0, -1], \
            [0, 1, 0], \
            [1, 0, 0]], dtype=torch.float32, device=device)
    R_up = torch.tensor(\
            [[1, 0, 0], \
            [0, 0, 1], \
            [0, -1, 0]], dtype=torch.float32, device=device)
    R_down = torch.tensor(\
            [[1, 0, 0], \
            [0, 0, -1], \
            [0, 1, 0]], dtype=torch.float32, device=device)
    # concatenate these in one line code
    R_cubes = torch.stack((R_front, R_left, R_right, R_up, R_down, R_back))


    # use tqdm
    for i_view, viewpoint_data in enumerate(tqdm(data_loader.viewpoints, desc="Processing viewpoints for face counts")):


        start_time = time.time()
        for i, R_cube in enumerate(R_cubes):

            R, T, K = parse_pose_data(viewpoint_data.pose_data)

            R = R.T
            T = -R @ T

            R_xyz_90 = torch.tensor(\
                    [[1, 0, 0], \
                    [0, 0, 1], \
                    [0, -1, 0]], dtype=torch.float32, device=device)
            R = R_xyz_90 @ R
            T = R_xyz_90 @ T

            # R[:, 0] = -R[:, 0]
            # R[:, 1] = -R[:, 1]

            R = R @ R_cube
            T = -R.T @ T

            Rs[i] = R 
            Ts[i] = T


        end_time = time.time()
        print("time taken to parse pose data:", end_time - start_time, "s")

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        cameras = PerspectiveCameras(
            focal_length=fs,
            principal_point=cs,
            device=device,
            R=Rs,
            T=Ts,
            image_size=image_sizes,
            in_ndc=False,
        )


        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        N = len(Rs)
        batched_mesh = mesh.extend(N)

        # verts = mesh.verts_list()[0]  # (V, 3)
        # faces = mesh.faces_list()[0]  # (F, 3)

        # # start_time = time.time()
        # # Repeat the vertices and faces N times along the batch dimension
        # batched_verts = verts.unsqueeze(0).repeat(N, 1, 1)  # (N, V, 3)
        # batched_faces = faces.unsqueeze(0).repeat(N, 1, 1)  # (N, F, 3)

        # # # Create a batched Meshes object
        # batched_mesh = Meshes(verts=batched_verts, faces=batched_faces)
        end_time = time.time()
        print("Time taken to batch mesh:", end_time - start_time, "s")

        start_time = time.time()
        fragments = rasterizer(batched_mesh)
        end_time = time.time()
        print("Time taken to rasterize mesh:", end_time - start_time, "s")

        # rgbs = batched_mesh.sample_textures(fragments)
        # cv2.imwrite(f"view_{i_view}_test_0.png", cv2.cvtColor(rgbs[0].squeeze().detach().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"view_{i_view}_test_1.png", cv2.cvtColor(rgbs[1].squeeze().detach().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"view_{i_view}_test_2.png", cv2.cvtColor(rgbs[2].squeeze().detach().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"view_{i_view}_test_3.png", cv2.cvtColor(rgbs[3].squeeze().detach().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"view_{i_view}_test_4.png", cv2.cvtColor(rgbs[4].squeeze().detach().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"view_{i_view}_test_5.png", cv2.cvtColor(rgbs[5].squeeze().detach().cpu().numpy()*255.0, cv2.COLOR_RGB2BGR))

        face_num = mesh.faces_list()[0].shape[0]

        indices = fragments.pix_to_face
        mask = indices != -1

        start_time = time.time()
        indices[mask] = indices[mask] % face_num

        end_time = time.time()
        print("Time taken to mask and calculates indices:", end_time - start_time, "s")

        start_time = time.time()
        indices = indices.reshape(-1).squeeze()

        counts = torch.bincount(indices + 1)
        counts = counts[1:]
        counts = counts.float() / N

        end_time = time.time()
        print("Time taken to calculate counts:", end_time - start_time, "s")


        start_time = time.time()

        non_zero_indices = torch.nonzero(counts).squeeze().detach().cpu().numpy()
        non_zero_values = counts[non_zero_indices].detach().cpu().numpy()

        count_desc = sp.csr_matrix((non_zero_values, (non_zero_indices, np.zeros_like(non_zero_indices))), shape=(face_num, 1))
        count_desc = np.sqrt(count_desc) / np.sqrt(image_size[0] * image_size[1])

        end_time = time.time()
        print("Time taken to create sparse matrix:", end_time - start_time, "s")


        viewpoint_data.desc = count_desc


    # make place holder csr matrix with size len(count_desc) X 500

    num_views = len(data_loader.viewpoints)
    num_feats = data_loader.viewpoints[0].desc.shape[0]
    desc_data = np.zeros((num_views, num_feats), dtype=np.float32)

    for i_view, viewpoint_data in enumerate(data_loader.viewpoints):
        if viewpoint_data.desc is None:
            continue
        desc_data[i_view, :] = viewpoint_data.desc.toarray().squeeze()

    sp_desc_data = sp.csr_matrix(desc_data)
    corr_mat = (sp_desc_data @ sp_desc_data.T).toarray()

    # for i_view, viewpoint_data in enumerate(data_loader.viewpoints):
    #     print(viewpoint_data.view_id)
    #     for j in range(num_nn):
    #         print("\t", data_loader.viewpoints[nn_indices[i_view, j]].view_id)


    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_nn_max = max(num_nns)
    corr_thres = args.corr_thres

    nn_indices = np.argsort(-corr_mat, axis=1)[:, 1:1+num_nn_max]
    fn_indices = np.argsort(-corr_mat, axis=1)[:, -num_nn_max:]

    pairs_info = []
    unique_pairs = set()  # To track unique pairs in sorted order

    for i_view, viewpoint_data in enumerate(data_loader.viewpoints):
        ref_view_id = viewpoint_data.view_id
        for j in range(num_nn_max):
            nn_index = nn_indices[i_view, j]
            nn_score = corr_mat[i_view, nn_index]
            nn_view_id = data_loader.viewpoints[nn_index].view_id
            nn_k = j+1

            # Create a sorted tuple to ensure (ref_view_id, nn_view_id) == (nn_view_id, ref_view_id)
            sorted_pair = tuple(sorted((ref_view_id, nn_view_id)))

            # Add pair only if it is not in unique_pairs
            if sorted_pair not in unique_pairs:
                unique_pairs.add(sorted_pair)
                pairs_info.append((ref_view_id, nn_view_id, nn_score, nn_k))

    pair_info_file = os.path.join(output_dir, "pair_info.txt")
    with open(pair_info_file, "w") as file:
        for ref_view_id, nn_view_id, nn_score, nn_k in pairs_info:
            file.write(f"{ref_view_id} {nn_view_id} {nn_score:.4f} {nn_k}\n")

    for num_nn in num_nns:
        pairs_file = os.path.join(output_dir, f"pairs_{num_nn}.txt")
        with open(pairs_file, "w") as file:
            for ref_view_id, nn_view_id, nn_score, nn_k in pairs_info:
                ref_rgb_path = os.path.join(data_dir, "pano", "rgb", f"camera_{ref_view_id}_frame_equirectangular_domain_rgb.png")
                qry_rgb_path = os.path.join(data_dir, "pano", "rgb", f"camera_{nn_view_id}_frame_equirectangular_domain_rgb.png")
                if nn_k <= num_nn and nn_score > corr_thres:
                    file.write(f"{ref_rgb_path} {qry_rgb_path}\n")    

    skipped_view_ids_file = os.path.join(output_dir, "skipped_view_ids.txt")
    for skipped_view_id in data_loader.skipped_view_ids:
        with open(skipped_view_ids_file, "w") as file:
            file.write(f"{skipped_view_id}\n")

    debug_output_dir = args.debug_output_dir


    if debug_output_dir != "":
        num_nn_debug = 4
        nn_indices_debug = np.argsort(-corr_mat, axis=1)[:, 1:1+num_nn_debug]
        fn_indices_debug = np.argsort(-corr_mat, axis=1)[:, -num_nn_debug:]

        if not os.path.exists(debug_output_dir):
            os.makedirs(debug_output_dir, exist_ok=True)

        if not os.path.exists(os.path.join(debug_output_dir, "pairs")):
            os.makedirs(os.path.join(debug_output_dir, "pairs"), exist_ok=True)

        import matplotlib.pyplot as plt
            # Plot the heatmap
        plt.imshow(corr_mat, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(label='Intensity (0-1)')  # Optional color bar for reference

        # Save the heatmap as an image
        plt.savefig(os.path.join(debug_output_dir, "heatmap.png"), bbox_inches='tight')
        plt.close()  # Close the figure to free up memory


        # Write covisibility results to "covisibility_result.txt"
        with open(os.path.join(debug_output_dir, "covisibility_result.txt"), "w") as file:
            for i_view, viewpoint_data in enumerate(data_loader.viewpoints):
                ref_view_id = viewpoint_data.view_id
                file.write(f"{i_view} {ref_view_id}\n")
                
                # Write nearest neighbors
                for j in range(num_nn_debug):
                    nn_index = nn_indices_debug[i_view, j]
                    nn_score = corr_mat[i_view, nn_index]
                    nn_view_id = data_loader.viewpoints[nn_index].view_id
                    file.write(f"\t{nn_score:.4f} {nn_index} {nn_view_id}\n")

                # Write farthest neighbors
                for j in range(num_nn_debug):
                    fn_index = fn_indices_debug[i_view, j]
                    fn_score = corr_mat[i_view, fn_index]
                    fn_view_id = data_loader.viewpoints[fn_index].view_id
                    file.write(f"\t{fn_score:.4f} {fn_index} {fn_view_id}\n")


        # save nearest images and farthest images for visual inspection
        for i_view, viewpoint_data in enumerate(data_loader.viewpoints):

            # random value between 0 and 1
            rand_ratio = np.random.rand()
            if rand_ratio > 0.1:
                continue
            cv2.imwrite(os.path.join(debug_output_dir, "pairs", f"{viewpoint_data.view_id}.png"), viewpoint_data.rgb)
            for j in range(num_nn_debug):
                cv2.imwrite(os.path.join(debug_output_dir, "pairs", f"{viewpoint_data.view_id}_nn_{j}_{corr_mat[i_view, nn_indices_debug[i_view, j]]:.4f}.png"), data_loader.viewpoints[nn_indices_debug[i_view, j]].rgb)
            for j in range(num_nn_debug):
                cv2.imwrite(os.path.join(debug_output_dir, "pairs", f"{viewpoint_data.view_id}_fn_{j}_{corr_mat[i_view, fn_indices_debug[i_view, j]]:.4f}.png"), data_loader.viewpoints[fn_indices_debug[i_view, j]].rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stanford2D3D Data Loader CLI")

    # Define command-line arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--corr_thres", type=float, default=0.05, help="Correlation threshold for covisibility [0, 1]")
    parser.add_argument("--num_nn", type=str, default="2", help="Number of nearest neighbors to consider to be saved in the output directory")
    parser.add_argument("--debug_output_dir", type=str, default="", help="Path to the debug output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load the mesh on (cuda/cpu)")
    parser.add_argument("--mesh_path", type=str, default="3d/rgb.obj", help="Path to the mesh file, if not provided, a default mesh path 3d/rgb.obj is used")

    # Parse the arguments
    args = parser.parse_args()
    main(args)

