#!/usr/bin/env python3
"""Convert fp32 ONNX models to fp16 for 2x faster inference."""
import sys
import os

def convert(input_path, output_path):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    print(f"Converting {input_path} → {output_path}...")
    model = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)

    size_in = os.path.getsize(input_path) / 1024 / 1024
    size_out = os.path.getsize(output_path) / 1024 / 1024
    print(f"  {size_in:.1f} MB → {size_out:.1f} MB ({size_out/size_in*100:.0f}%)")


if __name__ == "__main__":
    onnx_dir = sys.argv[1] if len(sys.argv) > 1 else "onnx"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "onnx_fp16"
    os.makedirs(out_dir, exist_ok=True)

    for name in ["clip_text_encoder.onnx", "unet.onnx", "vae_decoder.onnx"]:
        convert(os.path.join(onnx_dir, name), os.path.join(out_dir, name))

    print("\nDone! fp16 models in:", out_dir)
