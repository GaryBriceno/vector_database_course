# image_tool.py
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline, LCMScheduler
from crewai.tools import tool

# ---- Config ----
MODEL_ID = os.getenv("SD_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
OUT_DIR = Path(os.getenv("IMG_OUT_DIR", "./generated_images"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps"  # Apple Silicon
DTYPE = torch.float16

# ---- Lazy-loaded pipeline (para no cargar el modelo en import) ----
_PIPE: Optional[DiffusionPipeline] = None


def _slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_len] if len(text) > max_len else text


def get_pipe(use_lcm: bool = True) -> DiffusionPipeline:
    """
    Carga el pipeline una sola vez y lo reutiliza.
    use_lcm=True => más velocidad con pocos steps (LCM scheduler).
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal) no disponible. Verifica tu instalación de PyTorch en macOS.")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )

    pipe.to(DEVICE)

    # Velocidad: usa LCM scheduler si deseas pocos steps
    if use_lcm:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Opcional: reduce memoria en algunos casos
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _PIPE = pipe
    return _PIPE


def generate_image(
    prompt: str,
    negative_prompt: str = "blurry, low quality, watermark, text, logo",
    width: int = 1024,
    height: int = 1024,
    steps: int = 6,              # 4-8 (rápido). Sube a 20 si quieres más calidad
    guidance_scale: float = 1.2, # bajo para LCM (1.0–2.0)
    seed: Optional[int] = None,
) -> str:
    """
    Genera una imagen offline y devuelve el path del PNG.
    """
    pipe = get_pipe(use_lcm=True)

    if seed is None:
        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    filename = f"{_slugify(prompt)}__{int(time.time())}__{seed}.png"
    out_path = OUT_DIR / filename

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    image = result.images[0]
    image.save(out_path)

    return str(out_path)


@tool
def generate_visual(prompt: str) -> str:
    """
    Tool para CrewAI: genera una imagen local y retorna la ruta.
    """
    path = generate_image(prompt)
    return f"Imagen generada en {path}"