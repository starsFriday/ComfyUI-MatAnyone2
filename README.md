# ComfyUI-MatAnyone2

A standalone ComfyUI custom node project that wraps [MatAnyone2](https://github.com/pq-yang/MatAnyone2) for video matting.

This repository is independent:
- It includes the `matanyone2` inference code (no dependency on your local `MatAnyone2` folder).
- It provides ComfyUI nodes for video matting (frame sequence + first-frame mask).
- It supports automatic first-time checkpoint download (`matanyone2.pth`).

## Demo Videos

| Original | Result |
| --- | --- |
| <video src="https://github.com/user-attachments/assets/14df1772-eb9f-46b5-b4b2-e069332b5d5c" controls muted></video> | <video src="https://github.com/user-attachments/assets/c2517188-2d51-466c-8a95-3ea293312a54" controls muted></video> |

## Nodes

1. `MatAnyone2 Model Loader`
- Loads MatAnyone2 checkpoint weights.
- Can auto-download the checkpoint if it does not exist locally.

2. `MatAnyone2 Video Matting`
- Inputs: `IMAGE` frame batch, `MASK` (first-frame target mask)
- Outputs:
  - `foreground`: extracted foreground (black background)
  - `composite`: green-screen composite preview (same style as official inference visualization)
  - `alpha`: per-frame alpha matte (`MASK`)

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/starsFriday/ComfyUI-MatAnyone2.git
```

Install dependencies:

```bash
cd ComfyUI-MatAnyone2
pip install -r requirements.txt
```

Restart ComfyUI.

## Checkpoint

Default checkpoint filename: `matanyone2.pth`

Download URL:
- https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth

Recommended local path:
- `ComfyUI/models/matanyone2/matanyone2.pth`

You can also enable `auto_download=true` in `MatAnyone2 Model Loader` to download automatically on first run.

## Workflow Usage

1. Prepare a video frame sequence as an `IMAGE` batch.
- You can use a video loader node to produce `IMAGE` frames.

2. Prepare the first-frame target mask as `MASK`.
- White area = foreground person, black area = background.
- If multiple mask frames are provided, only the first one is used.

3. Connect nodes:
- `MatAnyone2 Model Loader` -> `MatAnyone2 Video Matting.matanyone2_model`
- video frames -> `MatAnyone2 Video Matting.images`
- first-frame mask -> `MatAnyone2 Video Matting.first_mask`

4. Key parameters:
- `warmup`: first-frame warmup iterations (default: `10`)
- `erode_kernel` / `dilate_kernel`: first-frame mask morphology
- `mask_threshold`: mask binarization threshold
- `max_internal_size`: internal shortest-side limit (`-1` = disabled)

## Project Structure

```text
ComfyUI-MatAnyone2/
├── __init__.py
├── nodes.py
├── requirements.txt
├── README.md
├── LICENSE_MatAnyone2.txt
├── pretrained_models/
│   └── .gitkeep
└── matanyone2/
    ├── config/
    ├── inference/
    ├── model/
    └── utils/
```

## Notes

- The current node follows the official MatAnyone2 single-target pipeline (`objects=[1]`).
- This is a human video matting model; result quality strongly depends on input quality and the first-frame mask.
- First model load may be slow; subsequent runs use in-process model cache.

## License and Credits

- This project contains and adapts code from MatAnyone2.
- See the original MatAnyone2 license in [`LICENSE_MatAnyone2.txt`](./LICENSE_MatAnyone2.txt).
- Please follow the original license terms for use and redistribution.
