import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d


def rotation_matrix_to_quaternion(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def depth_to_xyz_map(depth_np, fx, fy, cx, cy, R_obj_to_cam, t_obj_to_cam):
    H, W = depth_np.shape
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))

    assert depth_np.dtype == np.float32, f"Expected depth_np to be float32"
    Z = depth_np
    valid = Z > 1e-8

    Xc = (uu - cx) * Z / fx
    Yc = (vv - cy) * Z / fy

    xyz_cam = np.stack([Xc, Yc, Z], axis=-1).astype(np.float32)   # [H,W,3]

    R = np.asarray(R_obj_to_cam, dtype=np.float32)
    t = np.asarray(t_obj_to_cam, dtype=np.float32)

    xyz_obj = np.zeros_like(xyz_cam, dtype=np.float32)

    pts = xyz_cam.reshape(-1, 3)
    valid_flat = valid.reshape(-1)

    pts_valid = pts[valid_flat]
    # Xc = R Xo + t  ->  Xo = R^T (Xc - t)
    pts_obj_valid = (pts_valid - t[None, :]) @ R

    xyz_obj = xyz_obj.reshape(-1, 3)
    xyz_obj[valid_flat] = pts_obj_valid
    xyz_obj = xyz_obj.reshape(H, W, 3)

    return xyz_obj


def depth_tensor_to_xyz_map(depth_hw: torch.Tensor, fx, fy, cx, cy, R: torch.Tensor, t: torch.Tensor):
    device = depth_hw.device
    H, W = depth_hw.shape
    if not hasattr(depth_tensor_to_xyz_map, "UU") or depth_tensor_to_xyz_map.UU.shape != (H, W):
        uu, vv = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
        depth_tensor_to_xyz_map.UU = uu - cx
        depth_tensor_to_xyz_map.VV = vv - cy
         
    assert depth_hw.dtype == torch.float32, f"Expected depth_hw to be float32"
    Z = depth_hw
    valid = Z > 1e-8

    Xc = depth_tensor_to_xyz_map.UU * Z / fx
    Yc = depth_tensor_to_xyz_map.VV * Z / fy

    # xyz_cam: chw
    xyz_cam = torch.stack([Xc, Yc, Z], dim=0)   # [H,W,3]

    xyz_obj = torch.zeros_like(xyz_cam, dtype=torch.float32)

    pts_valid = xyz_cam[:, valid]
    # Xc = R Xo + t  ->  Xo = R^T (Xc - t)
    pts_obj_valid = R.T @ (pts_valid - t.unsqueeze(1))

    xyz_obj[:, valid] = pts_obj_valid
    return xyz_obj, valid


"""
[가정 - gsplat 기본 convention]
  - viewmat (w2c): world-to-camera, OpenCV convention (x:right, y:down, z:forward)
  - depth: camera space의 z-depth. render_mode="RGB+ED"의 expected depth 사용.
           (render_mode="D"의 누적 depth라면 alpha로 나눠 정규화한 뒤 넣을 것)
  - K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]
"""

# ---------------------------------------------------------------------------
# 0. Depth unprojection : depth map(+mask) -> world space point cloud
# ---------------------------------------------------------------------------
def unproject(depth, K, c2w, mask, rgb, depth_max=None):
    """
    depth   : (H, W)      z-depth (world unit)
    K       : (3, 3)      intrinsics
    c2w     : (4, 4)      camera-to-world ( = inv(viewmat) )
    mask    : (H, W) bool 물체 영역만 True
    rgb     : (H, W, 3)   color ( [0,1] 또는 [0,255] )
    return  : points(N,3) world, colors(N,3) in [0,1] (rgb 없으면 None)
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float64)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    valid = (z > 0).reshape(-1)
    if depth_max is not None:
        valid &= (z <= depth_max).reshape(-1)
    valid &= mask.astype(bool).reshape(-1)

    pts_cam = pts_cam[valid]
    R, t = c2w[:3, :3], c2w[:3, 3]
    pts_world = pts_cam @ R.T + t

    cols = None    
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    cols = rgb[valid]
    return pts_world, cols


def mesh_from_depth_grid(pts_world, valid, rgb, z, edge_thresh=0.05):
    H, W = pts_world.shape[:2]
    
    # 유효 픽셀에만 vertex index 부여
    idx_map = -np.ones((H, W), dtype=np.int64)
    idx_map[valid] = np.arange(int(valid.sum()))
    verts = pts_world[valid]

    # 2x2 패치마다 삼각형 2개
    v00, v01 = idx_map[:-1, :-1], idx_map[:-1, 1:]
    v10, v11 = idx_map[1:, :-1], idx_map[1:, 1:]
    z00, z01 = z[:-1, :-1], z[:-1, 1:]
    z10, z11 = z[1:, :-1], z[1:, 1:]

    def good(a, b, c, za, zb, zc):
        m = (a >= 0) & (b >= 0) & (c >= 0)
        m &= np.abs(za - zb) < edge_thresh
        m &= np.abs(zb - zc) < edge_thresh
        m &= np.abs(za - zc) < edge_thresh
        return m

    m1 = good(v00, v11, v10, z00, z11, z10)
    m2 = good(v00, v01, v11, z00, z01, z11)
    tris = np.concatenate([
        np.stack([v00[m1], v11[m1], v10[m1]], -1),
        np.stack([v00[m2], v01[m2], v11[m2]], -1),
    ], 0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    
    rgb_f = rgb.astype(np.float64)
    if rgb_f.max() > 1.0:
        rgb_f = rgb_f / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_f[valid])
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# 4. 후처리 (선택)
# ---------------------------------------------------------------------------
def clean_mesh(mesh, min_component_ratio=0.05, smooth_iter=0):
    """작은 조각 제거 + 정리 + (선택) Taubin smoothing."""
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    clusters, n_tris, _ = mesh.cluster_connected_triangles()
    clusters, n_tris = np.asarray(clusters), np.asarray(n_tris)
    if len(n_tris) > 0:
        keep = n_tris >= n_tris.max() * min_component_ratio
        mesh.remove_triangles_by_mask(~keep[clusters])
        mesh.remove_unreferenced_vertices()

    if smooth_iter > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
        mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# 사용 예시
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- (A) gsplat 렌더링 (참고용 의사코드) ---------------------------------
    # from gsplat import rasterization
    # render, alphas, meta = rasterization(
    #     means, quats, scales, opacities, colors,
    #     viewmats, Ks, width, height, render_mode="RGB+ED")
    # rgb_all   = render[..., :3].cpu().numpy()   # (C, H, W, 3)  [0,1]
    # depth_all = render[..., 3].cpu().numpy()    # (C, H, W)     z-depth
    # Ks_np     = Ks.cpu().numpy()                # (C, 3, 3)
    # w2c_all   = viewmats.cpu().numpy()          # (C, 4, 4)  world-to-camera
    # masks     = ...                             # (C, H, W) bool  관심 물체 mask
    # -------------------------------------------------------------------------

    # 아래는 본인 데이터 로딩으로 교체:
    #   rgb   : (H,W,3),  depth : (H,W),  mask : (H,W) bool,
    #   K     : (3,3),    w2c   : (4,4)
    #
    # 예) 단일 view (.npy로 저장해 둔 경우)
    # rgb   = np.load("rgb.npy")
    # depth = np.load("depth.npy")
    # mask  = np.load("mask.npy").astype(bool)
    # K     = np.load("K.npy")
    # w2c   = np.load("w2c.npy")

    # views = [{"rgb": rgb, "depth": depth, "mask": mask, "K": K, "w2c": w2c}]

    # ---- (B) multi-view면 TSDF, single-view면 depth-grid 추천 ----------------
    # # multi-view
    # mesh = mesh_from_tsdf(views, voxel_length=0.008, sdf_trunc=0.03)
    # mesh = clean_mesh(mesh, smooth_iter=5)
    # o3d.io.write_triangle_mesh("object_tsdf.ply", mesh)

    # # single-view
    # c2w = np.linalg.inv(views[0]["w2c"])
    # mesh = mesh_from_depth_grid(views[0]["depth"], views[0]["K"], c2w,
    #                             mask=views[0]["mask"], rgb=views[0]["rgb"],
    #                             edge_thresh=0.05)
    # o3d.io.write_triangle_mesh("object_grid.ply", mesh)

    print("위 예시 블록의 주석을 풀고 데이터 경로를 채워서 실행하세요.")
