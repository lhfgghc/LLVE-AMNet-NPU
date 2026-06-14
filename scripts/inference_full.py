import os
import sys
import glob
import argparse

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

current_path = os.path.dirname(os.path.abspath(__file__))
amnet_path = os.path.dirname(current_path)
sys.path.insert(0, amnet_path)

from models.amnet import AMNet

MODEL_CONFIG = {
    "in_ch": 3,
    "encoder_channels": [64, 128, 256, 256],
    "latent_dim": 256,
    "convlstm_hidden_ch": 256,
    "convlstm_layers": 1,
    "encoder_num_blocks": [1, 1, 2, 2],
    "decoder_num_blocks": [2, 2, 1, 2],
    "use_multimodal_illumination": True,
    "snr_factor": 1.0,
    "snr_threshold": 0.5,
    "snr_fusion_depth": 1,
    "use_snr_guided_fusion": True,
    "use_checkpoint": True,
    "checkpoint_encoder": True,
    "checkpoint_decoder": True,
    "checkpoint_aux_encoder": True,
    "checkpoint_fusion": True,
}

DEFAULT_CKPT = os.path.join(amnet_path, "checkpoints", "amnet_pretrained.pth")
DEFAULT_INPUT = os.path.join(amnet_path, "demo_videos", "inputs")
DEFAULT_OUTPUT = os.path.join(amnet_path, "demo_videos", "outputs")


def get_device(pref=None):
    if pref:
        return torch.device(pref)
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            return torch.device("npu")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path, device):
    print(f"[Model] Loading checkpoint: {ckpt_path}")
    model = AMNet(**MODEL_CONFIG)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if state_dict and list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Model] Unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    print(f"[Model] Loaded on {device}")
    return model


def infer_clip(model, clip_dir, output_dir, device):
    frame_paths = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
    if not frame_paths:
        print(f"  [Skip] No .jpg frames found in {clip_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in frame_paths]
    x = torch.stack(frames, dim=0).to(device)
    print(f"  Input: {len(frames)} frames, shape {list(x.shape)}")

    out = model.inference_video(x)

    for i, t in enumerate(out):
        arr = (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(output_dir, f"{i+1:03d}.png"))
    print(f"  Saved {len(out)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(args.checkpoint, device)

    clip_names = sorted(
        d for d in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, d))
    )
    if not clip_names:
        print(f"[Error] No clip subdirectories found in {args.input}")
        return

    print(f"\n[Demo] Found {len(clip_names)} clip(s): {clip_names}")
    for clip_name in clip_names:
        print(f"\n--- {clip_name} ---")
        infer_clip(
            model,
            os.path.join(args.input, clip_name),
            os.path.join(args.output, clip_name),
            device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
