import os
import open3d as o3d
import mcubes
import struct
import numpy as np
import glob
from multiprocessing import Pool
from numpy.core.fromnumeric import _argsort_dispatcher
import tqdm
from triangle_hash import TriangleHash as _TriangleHash
import trimesh

GEN_WATERTIGHT_MESH_AND_SDF_PATH = "./generate-watertight-meshes-and-sdf-grids/build"
DATASET_PATH = "../data/ShapeNetCore.v1"
CACHE_PATH = "../data/ShapeNetCore.v1"

NUM_SAMPLE_POINTS = 16000

# CATEGORIES_IDS = {
#     'airplane': '02691156',
#     'car': '02958343',
#     'chair': '03001627',
#     'lamp': '03636649',
#     'table': '04379243',
#     'sofa':'04256520',
#     'telephone': '04401088',
#     'vessel':'04530566',
#     'loudspeaker':'03691459',
#     'cabinet': '02933112',
#     'display':'03211117',
#     'bench':'02828884',
#     'rifle':'04090263'
#     }

CATEGORIES_IDS = {
    'airplane': '02691156'
    }

IDS_CATEGORIES = {index: category for category, index in CATEGORIES_IDS.items()}

NUM_POINTS_UNIFORM = 100000

class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = np.zeros(points.shape[0], dtype=bool)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        
        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains

class MeshIntersector:
    def __init__(self, mesh, resolution=512):
        triangles = mesh.vertices[mesh.faces].astype(np.float64)
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles)
        # assert(np.allclose(triangles.reshape(-1, 3).min(0), 0.5))
        # assert(np.allclose(triangles.reshape(-1, 3).max(0), resolution - 0.5))

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        contains = np.zeros(len(points), dtype=bool)

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])
        
        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        if (contains1 != contains2).any():
            print('Warning: contains1 != contains2 for some points.')
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)
    
        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array

def check_mesh_contains(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    return contains

def generate_occupancy(path):
    if os.path.exists(path.replace("model.obj", "model_occupancy.npy")):
        return
    mesh = trimesh.load(path.replace("model.obj", "model_watertight.ply"))
    assert mesh.is_watertight, 'Warning: mesh %s is not watertight! Cannot sample points.' % path

    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1])/2
    scale = (bbox[1] - bbox[0])
    points_uniform = (np.random.rand(NUM_POINTS_UNIFORM, 3) - 0.5) * scale
    occupancies = check_mesh_contains(mesh, points_uniform).astype(np.uint8)
    np.save(path.replace("model.obj", "model_occupancy.npy"), np.concatenate([points_uniform, np.expand_dims(occupancies, 1)], axis=1))
    return

def get_all_obj_path(use_cache=True):
    # Using cached obj file paths
    if use_cache:
        assert os.path.exists(CACHE_PATH), "Please make sure the cache file %s is avaliable..." % CACHE_PATH
        print("Using cached paths to retrive all obj files...")
        id_count = {}
        with open(CACHE_PATH, "r") as f:
            lines = f.readlines()
        paths = [path.strip() for path in lines]
        for path in paths:
            file_cat = path.split("/")[-3]
            if file_cat in id_count:
                id_count[file_cat] += 1
            else:
                id_count[file_cat] = 1
        
        for key in id_count.keys():
            print("CATEGORY: %s, NUMBER OF FILES: %d" % (IDS_CATEGORIES[key].upper(), id_count[key]))
        return paths
    # Scan directory for all obj files
    else:
        os.chdir(DATASET_PATH)
        files = []
        print("Gathering all obj files...")
        for ID in tqdm.tqdm(IDS_CATEGORIES):
            cat_files = glob.glob("%s/**/model.obj" % os.path.join(DATASET_PATH, ID), recursive=True)
            files.append(cat_files)
        for index, ID in enumerate(IDS_CATEGORIES):
            print("CATEGORY: %s, NUMBER OF FILES: %d" % (IDS_CATEGORIES[ID].upper(), len(files[index])))
        files = [item for sublist in files for item in sublist]
        files = [os.path.join(DATASET_PATH, i) for i in files]
        with open(CACHE_PATH, "w") as f:
            for path in files:
                f.write(path + "\n")
        return files

def parallel_run(f, args):
    pool = Pool(16)
    for _ in tqdm.tqdm(pool.imap_unordered(f, args), total=len(args)):
        pass
    pool.close()
    pool.join()

def read_obj_as_o3d(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    # print(lines[:])
    
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            line = line.strip()
            vertices.append(list(map(float, line.split(" ")[1:])) )
        if line.startswith("f "):
            if "/" not in line:
                faces.append([int(line.split(" ")[1])-1, int(line.split(" ")[2])-1 , int(line.split(" ")[3])-1])
            else:
                faces.append([int(line.split(" ")[1].split("/")[0])-1, int(line.split(" ")[2].split("/")[0])-1 , int(line.split(" ")[3].split("/")[0])-1])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

class Vox:
    def __init__(self, dims=None, res=None, grid2world=None, sdf=None):
        self.dims = dims
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf

def load_vox(filename):
    assert os.path.isfile(filename), "file not found: %s" % filename

    fin = open(filename, 'rb')

    s = Vox()
    s.dims = [0,0,0]
    s.dims[0] = struct.unpack('I', fin.read(4))[0]
    s.dims[1] = struct.unpack('I', fin.read(4))[0]
    s.dims[2] = struct.unpack('I', fin.read(4))[0]
    s.res = struct.unpack('f', fin.read(4))[0]
    n_elems = s.dims[0]*s.dims[1]*s.dims[2]

    s.grid2world = struct.unpack('f'*16, fin.read(16*4))
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order="F")
    fin.close()

    # -> sdf 1-channel
    offset = 4*(3 + 1 + 16)
    s.sdf = np.fromfile(filename, count=n_elems, dtype=np.float32, offset=offset).reshape([s.dims[2], s.dims[1], s.dims[0]])
    # <-
    return s

def generate_watertight_mesh_and_sdf(path):
    if os.path.exists(path.replace("model.obj", "model_watertight.ply")):
        return
    os.system("%s/build/watertight --in %s --out %s" % (GEN_WATERTIGHT_MESH_AND_SDF_PATH, path.replace("model.obj", "model_centered.obj"), path.replace("model.obj", "model_sdf.vox")))
    grid = load_vox(path.replace("model.obj", "model_sdf.vox"))
    sdf = (grid.sdf<0).astype(np.float32)
    verts, faces = mcubes.marching_cubes(sdf, 0.5)
    verts = np.fliplr(verts) # <-- go from zyx to xyz
    rot = grid.grid2world[0:3,0:3]
    trans = grid.grid2world[0:3,3]
    verts = np.matmul(verts, rot.transpose()) + trans
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(path.replace("model.obj", "model_watertight.ply"), mesh)
    return

def transform_v1_to_BSP(obj_path):
    shapenet_v1 = read_obj_as_o3d(obj_path)
    shapenet_v1 = shapenet_v1.rotate(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

    min_bound = shapenet_v1.get_min_bound()
    max_bound = shapenet_v1.get_max_bound()
    loc = (min_bound + max_bound)/2
    scale = np.linalg.norm(max_bound - min_bound)
    shapenet_v1 = shapenet_v1.translate(-loc)
    shapenet_v1 = shapenet_v1.scale(scale, np.zeros([3,1]))

    # write .mtl file
    with open(obj_path.replace("model.obj", "model_centered.mtl"), "w") as f:
        f.write("# Created by Open3D\n")
        f.write("# object name: model_centered\n")
        f.write("newmtl model_centered\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")



    vertices = np.asarray(shapenet_v1.vertices)
    faces = np.asarray(shapenet_v1.triangles)

    with open(obj_path.replace("model.obj", "model_centered.obj"), "w") as f:
        f.write("# Created by Open3D\n")
        f.write("# object name: model_centered\n")
        f.write("# number of vertices: %d\n" % vertices.shape[0])
        f.write("# number of triangles: %d\n" % faces.shape[0])
        f.write("mtllib model_centered.mtl\n")
        f.write("usemtl model_centered\n")
        for i in range(vertices.shape[0]):
            f.write("v %f %f %f\n" % (vertices[i][0], vertices[i][1], vertices[i][2]))
        for i in range(faces.shape[0]):
            # Note: obj file vertice index starts with 1
            f.write("f %d %d %d\n" % (faces[i][0]+1, faces[i][1]+1, faces[i][2]+1))

def sample_surface_points(path):
    if os.path.exists(path.replace("model.obj", "model_surface_point_cloud.ply")):
        return
    mesh = o3d.io.read_triangle_mesh(path.replace("model.obj", "model_watertight.ply"))
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.compute_vertex_normals()
    cloud = mesh.sample_points_uniformly(number_of_points=NUM_SAMPLE_POINTS)
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path.replace("model.obj", "model_surface_point_cloud.ply"), pcd)


if __name__ == "__main__":
    files = get_all_obj_path(use_cache=False)
    parallel_run(transform_v1_to_BSP, files)
    parallel_run(generate_watertight_mesh_and_sdf, files)
    parallel_run(sample_surface_points, files)
    parallel_run(generate_occupancy, files)
