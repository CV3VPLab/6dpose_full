"""Export the retrieval DINOv2 model with a dynamic batch axis.

The exported graph intentionally preserves the two outputs of the existing
``weights/dino_vits14_224.onnx`` model:

* ``last_hidden_state``: [B, 257, 384]
* ``pooler_output``: [B, 384]

Image resizing and normalization remain outside the ONNX graph and are handled
by ``AutoImageProcessor`` in ``modules_6d/retrieval_dino.py``. L2 feature
normalization also remains in the runtime code for baseline compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoModel


DEFAULT_MODEL = "facebook/dinov2-small"
DEFAULT_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


class DinoV2Deploy(nn.Module):
    """Expose the same two tensors returned by the existing DINO ONNX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        output = self.model(pixel_values=pixel_values, return_dict=False)
        last_hidden_state = output[0]
        pooler_output = last_hidden_state[:, 0, :]
        return last_hidden_state, pooler_output


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _compare(
    label: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{label}: shape mismatch {actual.shape} != {expected.shape}"
        )
    abs_diff = np.abs(actual - expected)
    max_abs = float(abs_diff.max(initial=0.0))
    mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
    print(f"  {label}: max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}")
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"{label}: numerical comparison failed "
            f"(atol={atol}, rtol={rtol})"
        )


def _parse_batches(value: str) -> list[int]:
    batches = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError(
            "validation batches must be positive comma-separated integers"
        )
    return batches


def _validate(
    model: nn.Module,
    onnx_path: Path,
    baseline_path: Path | None,
    batches: list[int],
    height: int,
    width: int,
    atol: float,
    rtol: float,
) -> None:
    print("Validating with ONNXRuntime CPU ...")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    baseline_session = None
    if baseline_path is not None:
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline ONNX not found: {baseline_path}")
        baseline_session = ort.InferenceSession(
            str(baseline_path),
            providers=["CPUExecutionProvider"],
        )

    generator = torch.Generator(device="cpu").manual_seed(20260817)
    for batch_size in batches:
        sample = torch.randn(
            batch_size,
            3,
            height,
            width,
            generator=generator,
            dtype=torch.float32,
        )
        with torch.no_grad():
            expected = model(sample)
        expected_np = [tensor.detach().cpu().numpy() for tensor in expected]
        actual = session.run(None, {input_name: sample.numpy()})

        print(f"B={batch_size}")
        _compare(
            "last_hidden_state vs PyTorch",
            actual[0],
            expected_np[0],
            atol=atol,
            rtol=rtol,
        )
        _compare(
            "pooler_output vs PyTorch",
            actual[1],
            expected_np[1],
            atol=atol,
            rtol=rtol,
        )

        if baseline_session is not None and batch_size == 1:
            baseline_input = baseline_session.get_inputs()[0].name
            baseline = baseline_session.run(
                None,
                {baseline_input: sample.numpy()},
            )
            _compare(
                "last_hidden_state vs baseline ONNX",
                actual[0],
                baseline[0],
                atol=atol,
                rtol=rtol,
            )
            _compare(
                "pooler_output vs baseline ONNX",
                actual[1],
                baseline[1],
                atol=atol,
                rtol=rtol,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--trace_batch_size", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--out",
        default="weights/dino_vits14_224_dynamic_batch.onnx",
    )
    parser.add_argument(
        "--baseline",
        default="weights/dino_vits14_224.onnx",
        help="Existing fixed-B=1 ONNX used for compatibility validation.",
    )
    parser.add_argument(
        "--validate_batches",
        type=_parse_batches,
        default=_parse_batches("1,4"),
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    if args.trace_batch_size <= 0:
        parser.error("--trace_batch_size must be positive")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {args.model}@{args.revision} "
        f"(local_files_only={args.local_files_only})"
    )
    backbone = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    ).eval()
    model = DinoV2Deploy(backbone).eval()

    hidden_size = int(backbone.config.hidden_size)
    patch_size = int(backbone.config.patch_size)
    if args.height % patch_size or args.width % patch_size:
        parser.error(
            f"Input size must be divisible by patch size {patch_size}: "
            f"got {args.height}x{args.width}"
        )
    token_count = (args.height // patch_size) * (args.width // patch_size) + 1

    dummy = torch.zeros(
        args.trace_batch_size,
        3,
        args.height,
        args.width,
        dtype=torch.float32,
    )
    with torch.no_grad():
        reference = model(dummy)
    expected_shapes = (
        (args.trace_batch_size, token_count, hidden_size),
        (args.trace_batch_size, hidden_size),
    )
    if tuple(reference[0].shape) != expected_shapes[0]:
        raise RuntimeError(
            f"Unexpected hidden-state shape: {tuple(reference[0].shape)}"
        )
    if tuple(reference[1].shape) != expected_shapes[1]:
        raise RuntimeError(
            f"Unexpected pooler shape: {tuple(reference[1].shape)}"
        )

    print(
        f"Exporting trace B={args.trace_batch_size}, "
        f"input={args.height}x{args.width}, tokens={token_count}, "
        f"hidden={hidden_size}, opset={args.opset}"
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "last_hidden_state": {0: "batch"},
                "pooler_output": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)

    if args.simplify:
        from onnxsim import simplify

        print("Simplifying ONNX ...")
        exported, check = simplify(exported)
        if not check:
            raise RuntimeError("ONNX simplification failed")
        onnx.checker.check_model(exported)

    onnx.helper.set_model_props(
        exported,
        {
            "dinov2.model": args.model,
            "dinov2.revision": args.revision,
            "dinov2.batch_mode": "dynamic",
            "dinov2.trace_batch_size": str(args.trace_batch_size),
            "dinov2.height": str(args.height),
            "dinov2.width": str(args.width),
            "dinov2.hidden_size": str(hidden_size),
            "dinov2.outputs": "last_hidden_state,pooler_output",
            "dinov2.preprocessing": "external AutoImageProcessor",
            "dinov2.l2_normalization": "external runtime",
        },
    )
    onnx.save(exported, str(output_path))

    print(f"ONNX input : {_shape(exported.graph.input[0])}")
    for output in exported.graph.output:
        print(f"ONNX output: {output.name} {_shape(output)}")

    if not args.skip_validation:
        baseline_path = Path(args.baseline) if args.baseline else None
        _validate(
            model,
            output_path,
            baseline_path,
            args.validate_batches,
            args.height,
            args.width,
            args.atol,
            args.rtol,
        )

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
