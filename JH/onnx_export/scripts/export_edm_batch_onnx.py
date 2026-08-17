"""
Export EDM for either a fixed number of pairs or a dynamic pair batch.

Input:
  input  float32 [B, 2, H, W]
         input[:, 0] = query grayscale images in [0, 1]
         input[:, 1] = gallery grayscale images in [0, 1]

Output:
  output float32 [B, K, 11]
         [mkpts0_c, mkpts1_c, offset01, offset10, score01, score10, mconf]

Modes:
  fixed   B is fixed at export time (use --batch_size 2 for exactly two pairs)
  dynamic B is a symbolic ONNX dimension and may vary at runtime
"""

from pathlib import Path
import argparse

import onnx
import torch
import torch.nn as nn
from onnxsim import simplify

from export_edm_pair_onnx import build_edm


class EDMBatchDeploy(nn.Module):
    """Preserve the pair-batch dimension flattened by EDM deploy forward."""

    def __init__(self, matcher: nn.Module, topk: int):
        super().__init__()
        self.matcher = matcher
        self.topk = int(topk)

    def forward(self, pair_batch):
        flat_output = self.matcher(pair_batch)
        return flat_output.reshape(pair_batch.shape[0], self.topk, 11)


def _default_output_path(
    edm_repo: Path,
    width: int,
    height: int,
    topk: int,
    mode: str,
    batch_size: int,
):
    batch_tag = f"b{batch_size}" if mode == "fixed" else "dynamic_batch"
    return (
        edm_repo
        / "deploy"
        / f"edm_outdoor_w{width}_h{height}_topk{topk}_{batch_tag}.onnx"
    )


def _set_metadata(
    model,
    *,
    mode: str,
    trace_batch_size: int,
    height: int,
    width: int,
    topk: int,
):
    onnx.helper.set_model_props(
        model,
        {
            "edm.batch_mode": mode,
            "edm.trace_batch_size": str(trace_batch_size),
            "edm.height": str(height),
            "edm.width": str(width),
            "edm.topk": str(topk),
            "edm.output_layout": "B,K,11",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edm_repo", default="EDM_repo")
    parser.add_argument(
        "--ckpt",
        default="EDM_repo/weights/edm_weights/edm_outdoor.ckpt",
    )
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--mode", choices=("fixed", "dynamic"), required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Fixed B, or the example B used while tracing a dynamic model.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_simplify", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")

    edm_repo = Path(args.edm_repo)
    ckpt_path = Path(args.ckpt)
    topk = int(args.height / 8 * args.width / 8 * 0.35)
    out_path = (
        Path(args.out)
        if args.out
        else _default_output_path(
            edm_repo,
            args.width,
            args.height,
            topk,
            args.mode,
            args.batch_size,
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Target: mode={args.mode} trace_batch={args.batch_size} "
        f"size={args.width}x{args.height} TOPK={topk}"
    )
    print(f"Output: {out_path}")

    matcher, topk = build_edm(
        edm_repo,
        ckpt_path,
        args.height,
        args.width,
    )
    model = EDMBatchDeploy(matcher, topk).eval()
    dummy = torch.zeros(args.batch_size, 2, args.height, args.width)

    with torch.no_grad():
        output = model(dummy)
    expected_shape = (args.batch_size, topk, 11)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected PyTorch output: {tuple(output.shape)} != {expected_shape}"
        )
    print(f"PyTorch forward OK: output={tuple(output.shape)}")

    dynamic_axes = None
    if args.mode == "dynamic":
        dynamic_axes = {
            "input": {0: "batch"},
            "output": {0: "batch"},
        }

    print("Exporting ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=16,
        )

    exported = onnx.load(str(out_path))
    onnx.checker.check_model(exported)

    if not args.no_simplify:
        print("Simplifying ...")
        exported, check = simplify(exported)
        if not check:
            raise RuntimeError("ONNX simplification failed")
        onnx.checker.check_model(exported)

    _set_metadata(
        exported,
        mode=args.mode,
        trace_batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        topk=topk,
    )
    onnx.save(exported, str(out_path))

    input_dims = [
        dim.dim_param or dim.dim_value
        for dim in exported.graph.input[0].type.tensor_type.shape.dim
    ]
    output_dims = [
        dim.dim_param or dim.dim_value
        for dim in exported.graph.output[0].type.tensor_type.shape.dim
    ]
    print(f"ONNX input : {input_dims}")
    print(f"ONNX output: {output_dims}")
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
