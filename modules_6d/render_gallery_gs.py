import json
import subprocess
import sys
import tempfile
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_pose_json_for_render(pose, K_list, width, height, out_path):
    """render_single_pose.py가 요구하는 pose JSON 포맷으로 저장"""
    data = {
        "width": width,
        "height": height,
        "K": K_list,
        "R_obj_to_cam": pose["R_obj_to_cam"],
        "t_obj_to_cam": pose["t_obj_to_cam"],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_intrinsics_as_list(path):
    """intrinsics.txt → [[fx,0,cx],[0,fy,cy],[0,0,1]] 리스트"""
    import numpy as np
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.extend([float(x) for x in line.split()])
    if len(vals) == 9:
        K = np.array(vals).reshape(3, 3)
        return K.tolist()
    elif len(vals) == 4:
        fx, fy, cx, cy = vals
        return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    else:
        raise ValueError(f"Unsupported intrinsics: {path}")


def _run_per_pose(args, gs_repo, gs_python, gs_model_dir,
                  gallery_pose_json, intrinsics_path, gs_output_dir):
    """
    render_single_pose.py를 pose별로 호출.
    2DGS처럼 render_gallery.py가 없거나 인자가 맞지 않을 때 사용.
    """
    render_script = gs_repo / "scripts" / "render_single_pose.py"
    if not render_script.exists():
        raise FileNotFoundError(
            f"render_single_pose.py not found: {render_script}\n"
            f"배포 방법: cp render_single_pose_2dgs.py {render_script}"
        )

    with open(gallery_pose_json, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    K_list  = _load_intrinsics_as_list(str(intrinsics_path))
    width   = int(args.render_width)
    height  = int(args.render_height)
    gs_iter = int(args.gs_iter) if args.gs_iter is not None else 30000

    save_xyz = getattr(args, "save_xyz", False)
    xyz_dir  = Path(args.xyz_dir) if getattr(args, "xyz_dir", None) else None

    ensure_dir(gs_output_dir)
    if xyz_dir:
        ensure_dir(xyz_dir)

    render_meta = {
        "stage":     "step4",
        "gs_mode":   args.gs_mode,
        "model_dir": str(gs_model_dir),
        "num_poses": len(pose_data["poses"]),
        "renders":   [],
    }

    poses = pose_data["poses"]
    total = len(poses)

    with tempfile.TemporaryDirectory(prefix="step4_poses_") as td:
        td = Path(td)
        for i, pose in enumerate(poses):
            idx            = int(pose["index"])
            pose_json_path = td / f"pose_{idx:04d}.json"
            output_png     = gs_output_dir / f"{idx:04d}.png"

            _save_pose_json_for_render(pose, K_list, width, height, pose_json_path)

            cmd = [
                str(gs_python),
                str(render_script.resolve()),
                "--model_dir",       str(gs_model_dir.resolve()),
                "--iteration",       str(gs_iter),
                "--intrinsics_path", str(intrinsics_path.resolve()),
                "--pose_json",       str(pose_json_path.resolve()),
                "--output_png",      str(output_png.resolve()),
                "--width",           str(width),
                "--height",          str(height),
                "--bg_color",        str(args.bg_color),
            ]

            if save_xyz:
                cmd.append("--save_xyz")
                if xyz_dir:
                    cmd.extend(["--xyz_output",
                                 str((xyz_dir / f"{idx:04d}.npy").resolve())])

            print(f"  [Step 4] pose {i+1}/{total} (idx={idx}) ...")
            subprocess.run(cmd, check=True, cwd=str(gs_repo))

            render_meta["renders"].append({
                "index":         idx,
                "file":          f"{idx:04d}.png",
                "azimuth_deg":   pose.get("azimuth_deg"),
                "elevation_deg": pose.get("elevation_deg"),
                "radius":        pose.get("radius"),
                "roll_deg":      pose.get("roll_deg", 0.0),
            })

    # step45에서 참조하는 gallery_render_meta.json
    save_json(gs_output_dir / "render_meta.json", render_meta)
    print(f"  [step4] complete: {total} poses → {gs_output_dir}")


def run_step4_render_gallery_gs(args):
    if args.gs_repo is None:
        raise ValueError("--gs_repo is required for step4")

    if args.gs_model_dir is None:
        raise ValueError("--gs_model_dir is required for step4")
    if args.gallery_pose_json is None:
        raise ValueError("--gallery_pose_json is required for step4")
    if args.intrinsics_path is None:
        raise ValueError("--intrinsics_path is required for step4")
    if args.gs_output_dir is None:
        raise ValueError("--gs_output_dir is required for step4")

    gs_repo           = Path(args.gs_repo)
    gs_python         = Path(args.gs_python or sys.executable)
    gs_model_dir      = Path(args.gs_model_dir)
    gallery_pose_json = Path(args.gallery_pose_json)
    intrinsics_path   = Path(args.intrinsics_path)
    gs_output_dir     = Path(args.gs_output_dir)

    ensure_dir(gs_output_dir)

    gs_mode = getattr(args, "gs_mode", "3dgs")
    render_gallery_script = gs_repo / "render_gallery.py"

    # 2DGS이거나 render_gallery.py가 없으면 per-pose 방식 사용
    use_per_pose = (gs_mode == "2dgs") or (not render_gallery_script.exists())

    if use_per_pose:
        print("=" * 60)
        print(f"[Step 4] {gs_mode.upper()} → per-pose render_single_pose.py")
        print("=" * 60)
        _run_per_pose(
            args, gs_repo, gs_python, gs_model_dir,
            gallery_pose_json, intrinsics_path, gs_output_dir,
        )
    else:
        # 3DGS: 기존 render_gallery.py 일괄 호출
        cmd = [
            str(gs_python),
            str(render_gallery_script.resolve()),
            "--model_dir",         str(gs_model_dir.resolve()),
            "--gallery_pose_json", str(gallery_pose_json.resolve()),
            "--intrinsics_path",   str(intrinsics_path.resolve()),
            "--output_dir",        str(gs_output_dir.resolve()),
            "--width",             str(args.render_width),
            "--height",            str(args.render_height),
            "--background",        str(args.bg_color),
            "--iteration",         str(args.gs_iter if args.gs_iter is not None else -1),
            "--sh_degree",         "3",
            "--gs_mode",           str(gs_mode),
        ]

        if getattr(args, "save_depth", False):
            cmd.append("--save_depth")
        if getattr(args, "save_xyz", False):
            cmd.append("--save_xyz")
        if getattr(args, "depth_dir", None):
            cmd.extend(["--depth_dir",     str(Path(args.depth_dir).resolve())])
        if getattr(args, "depth_vis_dir", None):
            cmd.extend(["--depth_vis_dir", str(Path(args.depth_vis_dir).resolve())])
        if getattr(args, "xyz_dir", None):
            cmd.extend(["--xyz_dir",       str(Path(args.xyz_dir).resolve())])

        print("=" * 60)
        print("[Step 4] Running GS gallery render")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 60)
        subprocess.run(cmd, check=True, cwd=str(gs_repo))

    summary = {
        "stage":             "step4",
        "gs_mode":           gs_mode,
        "gs_repo":           str(gs_repo),
        "gs_model_dir":      str(gs_model_dir),
        "gallery_pose_json": str(gallery_pose_json),
        "output_dir":        str(gs_output_dir),
        "render_width":      args.render_width,
        "render_height":     args.render_height,
    }
    save_json(Path(args.out_dir) / "step4_summary.json", summary)

    print("=" * 60)
    print("[Step 4] GS gallery render complete")
    print(f"  output_dir : {gs_output_dir}")
    print("=" * 60)