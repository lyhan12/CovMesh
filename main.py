
import argparse

from base.data_loader_standford2d3d import Stanford2D3DDataLoader
from base.rasterizer import Rasterizer

import os



def main(data_dir):
    # Predefined list of supported data types

    print(f"Data Directory: {data_dir}")

    # Initialize and load the data using Stanford2D3DDataLoader
    data_loader = Stanford2D3DDataLoader(root_dir=data_dir)
    data_loader.load_data()  # Load the data

    pano_files = os.listdir(os.path.join(data_dir, "pano", "rgb"))
    print("Number of Pano Files:", len(pano_files))
    print("Number of loaded viewpoints:", data_loader.num_viewpoints())

    import ipdb
    ipdb.set_trace()

    rasterizer = Rasterizer()
    frame_data = data_loader.viewpoints[0][0]

    mesh = ""

    rasterizer.rasterize(mesh, frame_data)

    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stanford2D3D Data Loader CLI")

    # Define command-line arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")

    # Parse the arguments
    args = parser.parse_args()
    main(args.data_dir)

