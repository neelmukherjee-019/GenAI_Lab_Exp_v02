"""Experiment 8: Text-to-music generation using the pretrained MusicGen model."""
from __future__ import annotations
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "facebook/musicgen-small"


def load_music_pipeline(model_id: str = DEFAULT_MODEL_ID):
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-to-audio", model=model_id, device=device)


def generate_music(prompt: str, music_pipe, duration_seconds: int = 8, output_path: str | Path = "outputs/exp08/generated_music.wav", temperature: float = 0.9) -> dict[str, Any]:
    """Generate and save a WAV file. MusicGen uses roughly 50 generated tokens/second."""
    import numpy as np
    from scipy.io import wavfile
    if not prompt or not prompt.strip():
        raise ValueError("Enter a music prompt first.")
    duration_seconds = max(1, min(int(duration_seconds), 30))
    output = music_pipe(
        prompt.strip(),
        generate_kwargs={"do_sample": True, "temperature": float(temperature), "max_new_tokens": int(duration_seconds * 50)},
    )
    # Pipeline output is typically {audio: ndarray, sampling_rate: int}.
    audio = np.asarray(output["audio"], dtype=np.float32)
    sample_rate = int(output["sampling_rate"])
    if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
        audio = audio.T
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, sample_rate, pcm)
    return {"audio_path": str(out), "sample_rate": sample_rate, "duration_seconds": float(audio.shape[0] / sample_rate), "prompt": prompt.strip()}


if __name__ == "__main__":
    pipe = load_music_pipeline()
    print(generate_music("Peaceful instrumental piano with soft flute and a warm atmosphere", pipe))
