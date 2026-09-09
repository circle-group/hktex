import os
import trimesh
import objaverse
import multiprocessing
import glob
import urllib.error
import contextlib
import warnings

import numpy as np
from typing import Protocol
from tqdm import tqdm

from hktex.utils.typing import *
from hktex.utils import load_mesh, compute_mesh_laplacian, compute_eig_laplacian


def find_filenames(root: str, file_ext: str | Tuple[str] | None = None) -> List[str]:
    if file_ext is None:
        file_ext = (".ply", ".obj", ".glb")
    root_l = len(root)
    files = []
    for dirpath, _, fnames in os.walk(root):
        for f in fnames:
            if f.endswith(file_ext):
                absolute_path = os.path.join(dirpath, f)
                f = absolute_path[dirpath.index(root) + root_l + 1 :]
                files.append(f)
    return files


class LockLike(Protocol):
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, exc_type, exc, tb) -> None: ...


class ObjaverseDownloader:
    def __init__(
        self,
        root: str,
        processed_dir_name: str = "processed",
        filter_only_files_with: str | None = None,
        num_workers: int = 8,
    ):
        # MONKEY PATCHING
        # The original objaverse download function prints to the console, which is not
        # ideal when downloading multiple files in parallel. This monkey patch
        # silences the output of the download function.
        # Keep a copy of the original download function
        __original_download_object = objaverse._download_object

        def __silenced_download_object(*args, **kwargs):
            # Create a file-like object to redirect stdout
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                return __original_download_object(*args, **kwargs)

        # Replace the original download function with the silenced one
        objaverse._download_object = __silenced_download_object

        # Monkey patching to redefine default behaviour of objaverse library
        objaverse.BASE_PATH = root
        objaverse._VERSIONED_PATH = os.path.join(  # pylint: disable=protected-access
            objaverse.BASE_PATH, "hf-objaverse-v1"
        )

        self.root = root

        processed_dir = os.path.join(root, processed_dir_name)
        os.makedirs(processed_dir, exist_ok=True)

        # Define path where the log of deleted files is stored
        self._deleted_fpath = os.path.join(processed_dir, "deleted_files_log.txt")
        if not os.path.exists(self._deleted_fpath):
            open(self._deleted_fpath, "a+").close()

        # Define path for the cleanup completion flag
        self._cleanup_done_fpath = os.path.join(
            processed_dir, "post_download_cleanup_done.flag"
        )

        # Get dataset annotations and filter out the objects that do not satisfy
        # the required properties based on the available annotations
        self._filter_only_files_with = filter_only_files_with

        self._num_workers = num_workers

    def _filter_annotations(self, annotations: Dict[str, Any]) -> List[str]:
        conditions = {}
        for attribute_condition in self._filter_only_files_with:
            lvis_uids = None
            if attribute_condition == "has_lvis":
                # Use only meshes with LVIS annotations
                lvis_annotations = objaverse.load_lvis_annotations()
                lvis_uids = sum([ids for ids in lvis_annotations.values()], [])
                annotations = objaverse.load_annotations(lvis_uids)
            elif attribute_condition == "faces":
                # Ignore point clouds
                conditions["faceCount"] = lambda x: x > 0
            elif "verts" in attribute_condition:
                # Ignore meshes with too many or not enough vertices
                mv = int(attribute_condition.split("_")[1][:-1]) * 1000
                if "less" in attribute_condition:
                    conditions["vertexCount"] = lambda x: x < mv
                else:
                    conditions["vertexCount"] = lambda x: x > mv
            elif attribute_condition == "downloadable":
                # Ignore if not downloadable
                conditions["isDownloadable"] = lambda x: x
            elif attribute_condition == "static":
                # Ignore animated meshes
                conditions["animationCount"] = lambda x: x == 0
            elif "likes" in attribute_condition:
                # Ignore if no one liked them
                assert "more" in attribute_condition
                min_likes = int(attribute_condition.split("_")[1])
                conditions["likeCount"] = lambda x: x > min_likes
            elif "views" in attribute_condition:
                # Ignore if no one even viewed them
                assert "more" in attribute_condition
                min_views = int(attribute_condition.split("_")[1])
                conditions["viewCount"] = lambda x: x > min_views
            elif attribute_condition == "a_texture":
                pass  # performed regardless and maybe checked in download
            elif "intersections" in attribute_condition:
                pass  # performed during download
            elif attribute_condition == "is_manifold":
                pass  # performed during download
            elif attribute_condition == "no_disconnected_components":
                pass  # performed during download
            elif "check_lbo_eig" in attribute_condition:
                pass  # performed during download
            else:
                raise NotImplementedError(
                    f"Filtering with {attribute_condition} not implemented yet."
                )

        uids_filtered = []
        for uid, annotation in annotations.items():
            if all(con(annotation[key]) for key, con in conditions.items()):
                # Ignore if no textures, this happens regardless
                for v in annotation["archives"].values():
                    if "textureCount" in v and v["textureCount"]:
                        uids_filtered.append(uid)
                        break

        uids = objaverse.load_uids()
        print(
            f"{len(uids_filtered)}/{len(uids)} objects passed the",
            "annotation prefiltering stage",
        )
        return uids_filtered

    def download(self):
        # Download objaverse https://huggingface.co/datasets/allenai/objaverse
        annotations = objaverse.load_annotations()
        uids = self._filter_annotations(annotations)

        objects_uid_and_paths = self.load_objects(
            uids=uids, download_processes=self._num_workers
        )
        total_files_count = len(
            glob.glob(os.path.join(objaverse._VERSIONED_PATH, "glbs", "*", "*.glb"))
        )
        n_additional_deleted_files = self.post_download_cleanup()
        print(
            f"{len(objects_uid_and_paths)} objects were downloaded and",
            f"{len(uids) - total_files_count}/{len(uids)} were immediately deleted.",
            f"{n_additional_deleted_files} additional files were deleted post download",
            f". Using {total_files_count - n_additional_deleted_files} files in total.",
        )
        final_files_count = len(
            glob.glob(os.path.join(objaverse._VERSIONED_PATH, "glbs", "*", "*.glb"))
        )
        assert final_files_count == total_files_count - n_additional_deleted_files

    def load_objects(
        self, uids: List[str], download_processes: int = 1
    ) -> Dict[str, str]:
        object_paths = objaverse._load_object_paths()
        out = {}

        # Create a manager and lock for process-safe file writing
        lock = multiprocessing.Manager().Lock()

        # Load the log of deleted files
        with open(self._deleted_fpath, "r") as f:
            deleted_files = f.read().splitlines()

        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                print(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(objaverse._VERSIONED_PATH, object_path)
            if not os.path.exists(local_path) and uid not in deleted_files:
                args.append((uid, object_paths[uid]))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(
            f"starting download of {len(args)} objects",
            f"with {download_processes} processes",
        )
        start_file_count = len(
            glob.glob(os.path.join(objaverse._VERSIONED_PATH, "glbs", "*", "*.glb"))
        ) - len(deleted_files)

        args_lst = [(*a, len(args), start_file_count, lock) for a in args]

        pbar = tqdm(total=len(args_lst))

        def _on_result(res):
            # callback runs in parent process thread -> safe to update out and tqdm
            if res is None:
                pbar.update(1)
                return
            uid, local_path = res
            out[uid] = local_path
            pbar.update(1)

        def _on_error(err):
            # ensure progress still advances if a worker raises
            pbar.update(1)

        with multiprocessing.Pool(download_processes) as pool:
            for a in args_lst:
                pool.apply_async(
                    self._download_object,
                    args=a,
                    callback=_on_result,
                    error_callback=_on_error,
                )
            pool.close()
            pool.join()
        pbar.close()
        return out

    def _download_object(
        self,
        uid: str,
        object_path: str,
        total_downloads: float,
        start_file_count: int,
        lock: Optional[LockLike] = None,
    ) -> Tuple[str, str] | None:
        # Attempt to download the object.
        try:
            uid, local_path = objaverse._download_object(
                uid, object_path, total_downloads, start_file_count
            )
        except urllib.error.URLError:
            # Download failed, log and return. No file to clean up.
            local_path = "DELETED"
            if lock is not None:
                with lock:
                    with open(self._deleted_fpath, "a") as f:
                        f.write(uid + "\n")
            return uid, local_path

        # Validate the downloaded object and delete it if does not meet criteria.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                mesh = load_mesh(local_path)
                material = mesh.visual.material
                if isinstance(material, trimesh.visual.material.SimpleMaterial):
                    texture = material.image
                else:
                    texture = material.baseColorTexture
                if texture is None:
                    raise AttributeError

                self._mesh_quality_checks(mesh)

                # find max number of grid intersections coming from configs
                mi = [c for c in self._filter_only_files_with if "intersect" in c]
                mi = int(mi[0].split("_")[1]) if len(mi) > 0 else 20
                # check if the mesh has more intersections than allowed
                if self._max_grid_intersections(mesh) > mi:
                    raise ValueError

        except (ValueError, RuntimeError, AttributeError, RuntimeWarning):
            if os.path.exists(local_path):
                os.remove(local_path)
                directory = os.path.dirname(local_path)
                try:
                    # Attempt to remove the directory if it's empty
                    if not os.listdir(directory):
                        os.rmdir(directory)
                except (OSError, FileNotFoundError):
                    pass  # Ignore if dir not empty or another process removed it
            local_path = "DELETED"

        if local_path == "DELETED" and lock is not None:
            with lock:
                with open(self._deleted_fpath, "a") as f:
                    f.write(uid + "\n")
        return uid, local_path

    def _mesh_quality_checks(self, mesh: trimesh.Trimesh, check_laplacian: bool = True):
        check_conn = any(
            c == "no_disconnected_components" for c in self._filter_only_files_with
        )
        if check_conn and len(trimesh.graph.connected_components(mesh.edges)) > 1:
            raise ValueError

        check_man = any(c == "is_manifold" for c in self._filter_only_files_with)
        if check_man and not ObjaverseDownloader._is_mesh_manifold(mesh):
            raise ValueError

        check_lapl = any("lbo_eig" in c for c in self._filter_only_files_with)
        if check_lapl and check_laplacian:
            k_eig = [
                c.split("_")[-1] for c in self._filter_only_files_with if "lbo" in c
            ]
            k_eig = int(k_eig[0])
            if mesh.vertices.shape[0] < k_eig:
                k_eig = mesh.vertices.shape[0] - 1

            lapl, mass = compute_mesh_laplacian(
                np.array(mesh.vertices), np.array(mesh.faces)
            )
            _, _ = compute_eig_laplacian(lapl, mass, k_eig=k_eig)  # ValueError if fails

    def post_download_cleanup(self):
        if os.path.exists(self._cleanup_done_fpath):
            print("Post-download cleanup has already been performed. Skipping.")
            return 0

        # Clean up any partially downloaded files
        with open(self._deleted_fpath, "r") as f:
            deleted_files = f.read().splitlines()

        all_filenames = find_filenames(self.root)

        n_deleted_files = 0
        for f in tqdm(all_filenames, desc="Post-download cleanup", leave=False):
            m = load_mesh(os.path.join(self.root, f))
            try:
                self._mesh_quality_checks(m, check_laplacian=False)
            except (ValueError, RuntimeError):
                os.remove(os.path.join(self.root, f))
                n_deleted_files += 1
                directory = os.path.dirname(os.path.join(self.root, f))
                try:
                    # Attempt to remove the directory if it's empty
                    if not os.listdir(directory):
                        os.rmdir(directory)
                except (OSError, FileNotFoundError):
                    pass  # Ignore if dir not empty or another process removed it
                if f not in deleted_files:
                    with open(self._deleted_fpath, "a") as f_log:
                        f_log.write(f + "\n")

        # Create the flag file to indicate completion
        with open(self._cleanup_done_fpath, "w") as f:
            f.write("done")
        return n_deleted_files

    def _max_grid_intersections(self, mesh: trimesh.Trimesh):
        rays_o, rays_d, _ = self._get_bounding_box_diagonal_and_grid_rays(
            mesh.bounds, face_grid_step=5, return_segments=False
        )
        _, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=rays_o, ray_directions=rays_d
        )
        n_per_ray_intersections = np.bincount(index_ray)
        return max(n_per_ray_intersections)

    @staticmethod
    def _get_bounding_box_diagonal_and_grid_rays(
        bounding_box_bounds: np.ndarray,
        face_grid_step: int = 5,
        return_segments: bool = False,
    ):
        p, q = bounding_box_bounds
        diagonals = np.array(
            [
                [[p[0], p[1], p[2]], [q[0], q[1], q[2]]],
                [[p[0], q[1], p[2]], [q[0], p[1], q[2]]],
                [[q[0], q[1], p[2]], [p[0], p[1], q[2]]],
                [[q[0], p[1], p[2]], [p[0], q[1], q[2]]],
            ]
        )

        step = face_grid_step
        # Create a NxN 2D grid on one face of the bounding box where points are
        gf0 = np.meshgrid(np.linspace(p[0], q[0], step), np.linspace(p[1], q[1], step))
        gf0 = np.stack([gf0[0].ravel(), gf0[1].ravel()], -1)

        # Add 3rd coordinate to define start and end of the rays
        gf0_start = np.concatenate([gf0, p[2] * np.ones([gf0.shape[0], 1])], -1)
        gf0_end = np.concatenate([gf0, q[2] * np.ones([gf0.shape[0], 1])], -1)
        g0_segments = np.stack([gf0_start, gf0_end], axis=1)

        # Repeat on a perpendicular face
        gf1 = np.meshgrid(np.linspace(p[0], q[0], step), np.linspace(p[2], q[2], step))
        gf1 = np.stack([gf1[0].ravel(), gf1[1].ravel()], -1)
        gf1_start = np.concatenate(
            [
                np.expand_dims(gf1[:, 0], axis=-1),
                p[1] * np.ones([gf1.shape[0], 1]),
                np.expand_dims(gf1[:, 1], axis=-1),
            ],
            axis=-1,
        )
        gf1_end = np.concatenate(
            [
                np.expand_dims(gf1[:, 0], axis=-1),
                q[1] * np.ones([gf1.shape[0], 1]),
                np.expand_dims(gf1[:, 1], axis=-1),
            ],
            axis=-1,
        )
        g1_segments = np.stack([gf1_start, gf1_end], axis=1)

        # Repeat on third perpendicular face
        gf2 = np.meshgrid(np.linspace(p[1], q[1], step), np.linspace(p[2], q[2], step))
        gf2 = np.stack([gf2[0].ravel(), gf2[1].ravel()], -1)
        gf2_start = np.concatenate(
            [p[0] * np.ones([gf2.shape[0], 1]), gf2],
            axis=-1,
        )
        gf2_end = np.concatenate(
            [q[0] * np.ones([gf2.shape[0], 1]), gf2],
            axis=-1,
        )
        g2_segments = np.stack([gf2_start, gf2_end], axis=1)

        ray_segments = np.concatenate(
            [diagonals, g0_segments, g1_segments, g2_segments], axis=0
        )

        # Compute ray origins and directions
        ray_origins = ray_segments[:, 0, :]
        ray_directions = ray_segments[:, 1, :] - ray_segments[:, 0, :]

        if not return_segments:
            ray_segments = None

        return ray_origins, ray_directions, ray_segments

    @staticmethod
    def _is_mesh_manifold(mesh: trimesh.Trimesh, allow_boundary: bool = True) -> bool:
        """
        Return True if mesh is manifold:
        - every unique edge has 1 or 2 incident faces (if allow_boundary=True)
            otherwise requires exactly 2 faces per edge (closed manifold)
        - for every vertex, the incident faces form a single connected component (a fan)
        """
        # An empty mesh is not considered manifold in this context
        if mesh.is_empty:
            return False

        # 1) edges: check counts per unique edge
        faces = mesh.faces.astype(np.int64)
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges = np.sort(edges, axis=1)  # undirected edges canonical form
        _, counts = np.unique(edges, axis=0, return_counts=True)

        if allow_boundary:
            if np.any(counts > 2):
                return False
        else:
            # closed manifold / no boundary
            if np.any(counts != 2):
                return False

        # 2) vertex fans: use face_adjacency to build adjacency among faces
        # incident to each vertex
        adj = mesh.face_adjacency
        if adj.size == 0:
            return True

        for v in range(len(mesh.vertices)):
            faces = mesh.vertex_faces[v]
            faces = faces[faces != -1]
            n = faces.size
            if n <= 1:
                continue  # trivial fan

            # select adjacency edges where both faces are in 'faces'
            mask = np.isin(adj[:, 0], faces) & np.isin(adj[:, 1], faces)
            if not np.any(mask):
                # more than one face at vertex but no adjacency between them
                return False

            # build small adjacency dict for BFS
            sel = adj[mask]
            neighbors = {int(f): set() for f in faces}
            for a, b in sel:
                neighbors[int(a)].add(int(b))
                neighbors[int(b)].add(int(a))

            # BFS from first face
            start = int(faces[0])
            seen = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for nb in neighbors.get(cur, ()):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

            if len(seen) != n:
                return False

        return True
