import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy import io

from modules_6d.retrieval_dino import compute_nonblack_bbox, expand_bbox


def load_file(path):
    """Load .npy or .mat file → numpy array."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(str(path))

    if path.suffix.lower() == ".mat":
        # try scipy first (MATLAB < v7.3), then h5py (v7.3+)
        try:
            import scipy.io
            mat = scipy.io.loadmat(str(path))
            # filter out meta keys (start with '__')
            data_keys = [k for k in mat if not k.startswith("__")]
            if len(data_keys) == 1:
                return np.array(mat[data_keys[0]])
            # multiple variables: print and pick first numeric array
            print(f"[mat] variables: {data_keys}")
            for k in data_keys:
                v = mat[k]
                if isinstance(v, np.ndarray) and v.dtype.kind in "fiu":
                    print(f"[mat] using '{k}' shape={v.shape}")
                    return np.array(v)
        except Exception as e:
            # fall back to h5py for v7.3 HDF5-based .mat
            try:
                import h5py
                with h5py.File(str(path), "r") as f:
                    keys = list(f.keys())
                    print(f"[mat/hdf5] keys: {keys}")
                    for k in keys:
                        v = np.array(f[k])
                        if v.dtype.kind in "fiu":
                            print(f"[mat/hdf5] using '{k}' shape={v.shape}")
                            return v
            except Exception as e2:
                raise RuntimeError(f"Failed to load {path}: scipy={e}, h5py={e2}")

    raise ValueError(f"Unsupported file type: {path.suffix}")


def to_uint8(arr):
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.uint8)
    vmin, vmax = arr[finite].min(), arr[finite].max()
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(arr, 128, dtype=np.uint8)
    out = np.zeros_like(arr, dtype=np.float32)
    out[finite] = (arr[finite] - vmin) / (vmax - vmin) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_colormap(arr_2d):
    return cv2.applyColorMap(to_uint8(arr_2d), cv2.COLORMAP_JET)


def crop_xyz_map(arr, margin=12):
    h, w = arr.shape[:2]
    norm_map = np.linalg.norm(arr.astype(np.float32), axis=2)
    gray = (norm_map > 1e-6).astype(np.uint8) * 255
    pseudo_bgr = np.stack([gray, gray, gray], axis=2)
    bbox = compute_nonblack_bbox(pseudo_bgr, thresh=127)
    bbox = expand_bbox(bbox, margin, w, h)
    x1, y1, x2, y2 = bbox
    return arr[y1:y2, x1:x2].copy(), bbox


def save_2d(arr, save_path):
    img = apply_colormap(arr)
    cv2.imwrite(str(save_path), img)
    print(f"[OK] {save_path}")


def save_rgb(arr, save_path):
    img = cv2.cvtColor(to_uint8(arr), cv2.COLOR_RGB2BGR) if arr.shape[2] == 3 else to_uint8(arr)
    cv2.imwrite(str(save_path), img)
    print(f"[OK] {save_path}")


def save_xyz(arr, save_path):
    x, y, z = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    norm = np.linalg.norm(arr, axis=2)

    def labeled(channel, name):
        img = apply_colormap(channel)
        cv2.putText(img, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    row0 = np.hstack([labeled(x, "X"),    labeled(y, "Y")])
    row1 = np.hstack([labeled(z, "Z"),    labeled(norm, "norm")])
    grid = np.vstack([row0, row1])
    cv2.imwrite(str(save_path), grid)
    print(f"[OK] {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy_path",    required=True)
    p.add_argument("--mode",        default="auto", choices=["auto", "2d", "rgb", "xyz"])
    p.add_argument("--save_path",   default=None)
    p.add_argument("--crop",        action="store_true")
    p.add_argument("--crop_margin", type=int, default=12)
    p.add_argument("--save_npy",    default=None)
    args = p.parse_args()

    arr = np.load(args.npy_path).astype(np.float32)
    stem = Path(args.npy_path).stem

    print(f"file  : {args.npy_path}")
    print(f"shape : {arr.shape}  dtype: {arr.dtype}")
    print(f"min   : {np.nanmin(arr):.6f}  max: {np.nanmax(arr):.6f}")

    mode = args.mode
    if mode == "auto":
        if arr.ndim == 2:
            mode = "2d"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = "xyz"
        elif arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
            mode = "rgb"
        else:
            mode = "2d"
        print(f"auto → {mode}")

    bbox = None
    if args.crop and mode == "xyz" and arr.ndim == 3 and arr.shape[2] == 3:
        arr, bbox = crop_xyz_map(arr, margin=args.crop_margin)
        print(f"cropped : {arr.shape}  bbox={bbox}")

    if args.save_npy:
        out = Path(args.save_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), arr.astype(np.float16))
        print(f"[OK] npy: {out}")

        io.savemat("xyz_map_1942.mat", {'data' : arr})
        print(f"[OK] mat: xyz_map_1942.mat")


    save_path = args.save_path or str(
        Path(args.npy_path).parent / f"{stem}{'_crop' if bbox else ''}_{mode}.png"
    )

    if mode == "2d":
        save_2d(arr, save_path)
    elif mode == "rgb":
        save_rgb(arr, save_path)
    elif mode == "xyz":
        save_xyz(arr, save_path)


if __name__ == "__main__":
    main()
