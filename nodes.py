import os
import sys
import threading
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.hub import download_url_to_file

import folder_paths

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import get_default_device
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


MODEL_TYPE = "matanyone2"
MODEL_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
DEFAULT_MODEL_NAME = "matanyone2.pth"
MODEL_DIR = os.path.join(folder_paths.models_dir, MODEL_TYPE)

_MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
_MODEL_LOCK = threading.Lock()


def _register_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        folder_paths.add_model_folder_path(MODEL_TYPE, MODEL_DIR, False)
    except TypeError:
        folder_paths.add_model_folder_path(MODEL_TYPE, MODEL_DIR)
    except Exception:
        # If already registered by another node/plugin, continue.
        pass


def _checkpoint_choices():
    choices = set()
    try:
        choices.update(folder_paths.get_filename_list(MODEL_TYPE))
    except Exception:
        pass
    choices.add(DEFAULT_MODEL_NAME)
    return sorted(choices)


def _resolve_checkpoint_path(model_name: str) -> str:
    full_path = None
    try:
        full_path = folder_paths.get_full_path(MODEL_TYPE, model_name)
    except Exception:
        full_path = None

    if full_path:
        return full_path
    return os.path.join(MODEL_DIR, model_name)


def _download_checkpoint_if_needed(target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        print(f"[ComfyUI-MatAnyone2] Downloading checkpoint to: {target_path}")
        download_url_to_file(MODEL_URL, target_path, progress=True)
    return target_path


def _to_first_mask(mask: torch.Tensor, height: int, width: int, threshold: float) -> np.ndarray:
    if mask.ndim == 2:
        first_mask = mask
    elif mask.ndim == 3:
        first_mask = mask[0]
    else:
        raise ValueError(f"`first_mask` shape not supported: {tuple(mask.shape)}")

    if first_mask.shape[-2:] != (height, width):
        first_mask = F.interpolate(
            first_mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )[0, 0]

    first_mask = (first_mask.clamp(0, 1) >= threshold).float() * 255.0
    return first_mask.detach().cpu().numpy().astype(np.uint8)


_register_model_dir()


class MatAnyone2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_checkpoint_choices(),),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MATANYONE2_MODEL",)
    RETURN_NAMES = ("matanyone2_model",)
    FUNCTION = "load_model"
    CATEGORY = "MatAnyone2"

    def load_model(self, model_name: str, auto_download: bool):
        ckpt_path = _resolve_checkpoint_path(model_name)
        if auto_download:
            ckpt_path = _download_checkpoint_if_needed(ckpt_path)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Enable `auto_download` or place `{DEFAULT_MODEL_NAME}` under `{MODEL_DIR}`."
            )

        device = get_default_device()
        cache_key = (os.path.abspath(ckpt_path), str(device))

        with _MODEL_LOCK:
            if cache_key not in _MODEL_CACHE:
                print(f"[ComfyUI-MatAnyone2] Loading model from: {ckpt_path}")
                _MODEL_CACHE[cache_key] = get_matanyone2_model(ckpt_path, device=device)
            model = _MODEL_CACHE[cache_key]

        return ({"network": model, "device": device, "ckpt_path": ckpt_path},)


class MatAnyone2VideoMatting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "matanyone2_model": ("MATANYONE2_MODEL",),
                "images": ("IMAGE",),
                "first_mask": ("MASK",),
                "warmup": ("INT", {"default": 10, "min": 0, "max": 64, "step": 1}),
                "erode_kernel": ("INT", {"default": 10, "min": 0, "max": 128, "step": 1}),
                "dilate_kernel": ("INT", {"default": 10, "min": 0, "max": 128, "step": 1}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_internal_size": ("INT", {"default": -1, "min": -1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("foreground", "composite", "alpha")
    FUNCTION = "run"
    CATEGORY = "MatAnyone2"

    def run(
        self,
        matanyone2_model,
        images: torch.Tensor,
        first_mask: torch.Tensor,
        warmup: int,
        erode_kernel: int,
        dilate_kernel: int,
        mask_threshold: float,
        max_internal_size: int,
    ):
        if not isinstance(matanyone2_model, dict) or "network" not in matanyone2_model:
            raise ValueError("Invalid `matanyone2_model`. Please connect `MatAnyone2ModelLoader` output.")

        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"`images` must be [T,H,W,3], got: {tuple(images.shape)}")

        network = matanyone2_model["network"]
        device = matanyone2_model.get("device", get_default_device())

        network.to(device)
        network.eval()

        processor = InferenceCore(network, cfg=network.cfg, device=device)
        processor.max_internal_size = int(max_internal_size)

        frames = images.detach().float().cpu().clamp(0.0, 1.0)
        frame_count, height, width, _ = frames.shape
        if frame_count == 0:
            raise ValueError("`images` batch is empty.")

        mask_np = _to_first_mask(first_mask, height, width, threshold=float(mask_threshold))
        if dilate_kernel > 0:
            mask_np = gen_dilate(mask_np, int(dilate_kernel), int(dilate_kernel))
        if erode_kernel > 0:
            mask_np = gen_erosion(mask_np, int(erode_kernel), int(erode_kernel))

        mask_tensor = torch.from_numpy(mask_np).float().to(device)

        warmup = int(warmup)
        total_steps = warmup + int(frame_count)
        progress = ProgressBar(total_steps) if ProgressBar is not None else None

        green_bg = torch.tensor([120.0, 255.0, 155.0], dtype=torch.float32).view(1, 1, 3) / 255.0

        foreground_list = []
        composite_list = []
        alpha_list = []

        objects = [1]
        with torch.inference_mode():
            for ti in range(total_steps):
                src_idx = 0 if ti < warmup else (ti - warmup)
                frame = frames[src_idx]
                image = frame.permute(2, 0, 1).to(device)

                if ti == 0:
                    output_prob = processor.step(image, mask_tensor, objects=objects)
                    output_prob = processor.step(image, first_frame_pred=True)
                elif ti <= warmup:
                    output_prob = processor.step(image, first_frame_pred=True)
                else:
                    output_prob = processor.step(image)

                pred_alpha = processor.output_prob_to_mask(output_prob).float().detach().cpu().clamp(0.0, 1.0)

                if ti >= warmup:
                    foreground = frame * pred_alpha.unsqueeze(-1)
                    composite = foreground + green_bg * (1.0 - pred_alpha.unsqueeze(-1))

                    foreground_list.append(foreground)
                    composite_list.append(composite)
                    alpha_list.append(pred_alpha)

                if progress is not None:
                    progress.update(1)

        foreground_batch = torch.stack(foreground_list, dim=0).clamp(0.0, 1.0)
        composite_batch = torch.stack(composite_list, dim=0).clamp(0.0, 1.0)
        alpha_batch = torch.stack(alpha_list, dim=0).clamp(0.0, 1.0)

        return foreground_batch, composite_batch, alpha_batch


NODE_CLASS_MAPPINGS = {
    "MatAnyone2ModelLoader": MatAnyone2ModelLoader,
    "MatAnyone2VideoMatting": MatAnyone2VideoMatting,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyone2ModelLoader": "MatAnyone2 Model Loader",
    "MatAnyone2VideoMatting": "MatAnyone2 Video Matting",
}
