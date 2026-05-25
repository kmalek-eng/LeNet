import argparse
from pathlib import Path

import numpy as np
import torch

from model import UNET_MODEL


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/fold_1/best_model.pth",
        help="Path to the trained checkpoint file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./csharp_export",
        help="Directory where UNetWeights.cs will be saved.",
    )

    parser.add_argument("--channel-reduction", type=int, default=2)
    parser.add_argument("--digits", type=int, default=4)

    return parser.parse_args()


def format_weights(tensor, digits):
    return np.round(tensor.detach().cpu().numpy(), digits)


def make_csharp_array(array_name, values):
    flat_values = values.flatten()
    array_text = ", ".join(map(str, flat_values))

    return f"""
    public static readonly float[] {array_name} = new float[]
    {{
        {array_text}
    }};
"""


def export_weights_to_csharp(model, output_path, digits):
    csharp_code = """
using System;

public static class UNetWeights
{
"""

    for name, param in model.named_parameters():
        clean_name = name.replace(".", "_")
        values = format_weights(param, digits)

        if "weight" in name and values.ndim >= 2:
            for i, sub_values in enumerate(values):
                csharp_code += make_csharp_array(f"{clean_name}_{i}", sub_values)
        else:
            csharp_code += make_csharp_array(clean_name, values)

    csharp_code += """
}
"""

    output_path.write_text(csharp_code)


def load_trained_model(checkpoint_path, channel_reduction):
    model = UNET_MODEL(
        args=None,
        channel_reduction=channel_reduction,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(
        checkpoint_path=args.checkpoint_path,
        channel_reduction=args.channel_reduction,
    )

    export_weights_to_csharp(
        model=model,
        output_path=output_dir / "UNetWeights.cs",
        digits=args.digits,
    )

    print(f"Saved: {output_dir / 'UNetWeights.cs'}")


if __name__ == "__main__":
    main()