"""Experiment 4: Text-to-image generation using Stable Diffusion XL."""
from __future__ import annotations
from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def load_image_pipeline(model_id: str = DEFAULT_MODEL_ID, use_cpu_offload: bool = True):
    """Load SDXL. A CUDA GPU is strongly recommended for usable latency."""
    import torch
    from diffusers import StableDiffusionXLPipeline
    if torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    )
    if torch.cuda.is_available() and use_cpu_offload:
        # Offloading reduces VRAM pressure at the cost of some speed.
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def generate_image(prompt: str, pipe, output_path: str | Path = "outputs/exp04/generated_image.png", steps: int = 25, guidance_scale: float = 7.0, negative_prompt: str = "blurry, low quality, distorted") -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Enter an image prompt first.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image = pipe(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=1024,
        height=1024,
    ).images[0]
    image.save(out)
    return str(out)


if __name__ == "__main__":
    pipe = load_image_pipeline()
    path = generate_image("A peaceful mountain lake at sunrise, cinematic landscape photography", pipe)
    print("Saved:", path)
