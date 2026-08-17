"""
Export full EDM pair matcher ONNX.

Input:
  input  float32 [1, 2, H, W]
         channel 0 = query grayscale image in [0, 1]
         channel 1 = gallery grayscale image in [0, 1]

Output:
  output float32 [K, 11]
         [mkpts0_c, mkpts1_c, offset01, offset10, score01, score10, mconf]
"""

from pathlib import Path
import argparse
import sys

import onnx
import torch
from onnxsim import simplify


def build_edm(edm_repo: Path, ckpt_path: Path, height: int, width: int):
    sys.path.insert(0, str(edm_repo))
    from src.edm.edm import EDM
    from src.config.default import get_cfg_defaults
    from src.utils.misc import lower_config

    topk = int(height / 8 * width / 8 * 0.35)

    config = get_cfg_defaults()
    config.merge_from_file(str(edm_repo / "configs/edm/outdoor/edm_base.py"))
    config.merge_from_file(str(edm_repo / "configs/data/megadepth_test_1500.py"))
    config.EDM.TEST_RES_H = height
    config.EDM.TEST_RES_W = width
    config.EDM.COARSE.TOPK = topk
    config.EDM.NECK.NPE = [
        config.EDM.TRAIN_RES_H,
        config.EDM.TRAIN_RES_W,
        height,
        width,
    ]
    config.EDM.DEPLOY = True
    config.EDM.COARSE.DS_OPT = False

    edm = EDM(config=lower_config(config)["edm"]).eval()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    edm.load_state_dict(ckpt["state_dict"])
    return edm, topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edm_repo", default=str(Path.home() / "EDM_repo"))
    parser.add_argument("--ckpt", default=str(Path.home() / "EDM_repo/weights/edm_weights/edm_outdoor.ckpt"))
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_simplify", action="store_true")
    args = parser.parse_args()

    edm_repo = Path(args.edm_repo)
    ckpt_path = Path(args.ckpt)
    topk = int(args.height / 8 * args.width / 8 * 0.35)
    out_path = Path(args.out) if args.out else (
        edm_repo / "deploy" / f"edm_outdoor_w{args.width}_h{args.height}_topk{topk}.onnx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.width}x{args.height} TOPK={topk}")
    print(f"Output: {out_path}")

    matcher, topk = build_edm(edm_repo, ckpt_path, args.height, args.width)
    print("Weights loaded OK")

    dummy = torch.zeros(1, 2, args.height, args.width)
    with torch.no_grad():
        out = matcher(dummy)
    print(f"PyTorch forward OK: output={tuple(out.shape)}")

    print("Exporting ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            matcher,
            dummy,
            str(out_path),
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
        )

    if not args.no_simplify:
        print("Simplifying ...")
        model = onnx.load(str(out_path))
        model_simp, check = simplify(model)
        assert check, "ONNX simplification failed"
        onnx.save(model_simp, str(out_path))

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
