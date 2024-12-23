from pathlib import Path
from typing import List
import torch
import torch.nn as nn


def export_to_onnx(model: nn.Module, dummy_input, onnx_path: Path, opset_version,
                   input_names: List[str], output_names: List[str]):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
    )
