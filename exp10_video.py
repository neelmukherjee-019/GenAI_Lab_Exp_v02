"""Experiment 10: Text-to-video generation using pretrained LTX-Video."""
from __future__ import annotations
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "Lightricks/LTX-Video"


def load_video_pipeline(model_id: str = DEFAULT_MODEL_ID, cpu_offload: bool = True):
    import torch
    from diffusers import LTXPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("LTX-Video generation needs a compatible GPU for practical use. In Colab, select Runtime → Change runtime type → GPU.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = LTXPipeline.from_pretrained(model_id, dtype=dtype)
    if cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def generate_video(prompt: str, pipe, output_path: str | Path = "outputs/exp10/generated_video.mp4", width: int = 512, height: int = 320, num_frames: int = 25, steps: int = 30, fps: int = 24, negative_prompt: str = "blurry, worst quality, inconsistent motion, jittery, distorted") -> dict[str, Any]:
    import torch
    from diffusers.utils import export_to_video
    if not prompt or not prompt.strip():
        raise ValueError("Enter a text-to-video prompt first.")
    if int(num_frames) % 8 != 1:
        raise ValueError("For LTX-Video, select a frame count in the form 8n+1, such as 17, 25, 33, or 49.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        result = pipe(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt or None,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            decode_timestep=0.05,
            decode_noise_scale=0.025,
        )
    frames = result.frames[0]
    export_to_video(frames, str(out), fps=int(fps))
    return {"video_path": str(out), "num_frames": int(num_frames), "fps": int(fps), "duration_seconds": int(num_frames) / int(fps), "prompt": prompt.strip()}


if __name__ == "__main__":
    pipe = load_video_pipeline()
    print(generate_video("A small wooden boat sailing across a calm lake at sunrise, cinematic wide shot" , pipe))
