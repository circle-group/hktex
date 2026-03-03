import sys
from pathlib import Path
import os

try:
    script_dir = Path(__file__).resolve().parent.parent
except NameError:
    script_dir = Path.cwd().parent
sys.path.append(str(script_dir))

import trimesh
import mitsuba as mi
from tqdm import tqdm

from heatsplats.data.objaverse_downloader import ObjaverseDownloader, find_filenames
from heatsplats.utils import load_mesh, show_video
from heatsplats.rendering.uv_texture_renderer import UVTextureRenderer

if __name__ == "__main__":
    root = "/data2/objaverse"

    downloader = ObjaverseDownloader(
        root=root,
        filter_only_files_with=[
            "has_lvis",
            "downloadable",
            "a_texture",
            "less_60k_verts",
            "faces",
            "static",
            "more_0_likes",
            "more_0_views",
            "less_20_intersections",
            "is_manifold",
            "no_disconnected_components",
            "check_lbo_eig_256",
        ],
        num_workers=max(1, (os.cpu_count() or 1) - 1),
    )
    # downloader.download()
    # downloader.post_download_cleanup()

    all_filenames = find_filenames(root)

    renderings = []
    frame_labels = []
    for i in tqdm(range(600, 900)):
        m = load_mesh(os.path.join(root, all_filenames[i]), merge_tex=False)
        renderer = UVTextureRenderer({})
        m_mi = renderer.mesh_to_mitsuba(m)
        r = mi.Bitmap(renderer.render(m_mi, True)).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )
        renderings.append(r)

        components = len(trimesh.graph.connected_components(m.edges))
        frame_labels.append(f"{i}: n_v={m.vertices.shape[0]}, disc={components > 1}")

    print("show_video(renderings, frame_texts=frame_labels)")
