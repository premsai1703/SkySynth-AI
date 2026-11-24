# ============================================================
#  SKYSYNTH AI — Synthetic Aerial Imagery Generation Pipeline
#  Source for the standalone Google Colab notebook
# ============================================================
#
#  USAGE:
#  1. Open Google Colab → Runtime → Change runtime type → T4 GPU
#  2. Upload skysynth_notebook.ipynb and choose Run all
#  3. Enable the optional Gradio demo in the final cell
# ============================================================


# ─────────────────────────────────────────────────────────────
# CELL 1 — Install Dependencies
# ─────────────────────────────────────────────────────────────
# Locally: python -m pip install -r requirements.txt
# The notebook builder installs dependencies without replacing Colab's PyTorch.


# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports & GPU Verification
# ─────────────────────────────────────────────────────────────

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
import gradio as gr

# ── Device & dtype selection ──────────────────────────────────
# float16 halves VRAM usage on GPU; fallback to float32 on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"Device  : {device}")
print(f"Dtype   : {dtype}")
if device == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM    : {vram_gb:.1f} GB")
else:
    print("WARNING: No GPU detected. Generation will be very slow on CPU.")


# ─────────────────────────────────────────────────────────────
# CELL 3 — Load Stable Diffusion Models
# ─────────────────────────────────────────────────────────────
# This downloads ~4 GB of model weights. Takes 5-10 minutes
# on first run; cached on disk for subsequent runs.

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
print(f"Loading model: {MODEL_ID}")
print("Please wait — first run downloads ~4 GB of weights...\n")

# ── Load Text-to-Image pipeline ───────────────────────────────
txt2img_pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    use_safetensors=True,
    **({"variant": "fp16"} if device == "cuda" else {}),
).to(device)

# ── Swap default PNDM scheduler → DPM-Solver++ ───────────────
# DPM-Solver++ lets us use a shorter denoising schedule for the demo.
txt2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    txt2img_pipe.scheduler.config
)

# ── Reuse weights for img2img pipeline ───────────────────────
# Passes shared model components (VAE, UNet, tokenizer, text encoder)
# so we do NOT download or store a second copy of the weights
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(txt2img_pipe)
img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(txt2img_pipe.scheduler.config)

print("\nModels loaded successfully!")


# ─────────────────────────────────────────────────────────────
# CELL 4 — Terrain, Weather & Altitude Configuration
# ─────────────────────────────────────────────────────────────

# Each terrain entry is a descriptive phrase appended to the base prompt.
# More specific = better structural coherence in generated images.
TERRAIN_PROMPTS = {
    "Urban":        "dense urban city grid with buildings, roads, rooftops, and city blocks",
    "Forest":       "dense green forest canopy, treetops, woodland, lush tropical vegetation",
    "Desert":       "arid desert terrain with sand dunes, rocky outcrops, and minimal vegetation",
    "Coastal":      "coastal terrain with sandy beach, ocean shoreline, shallow turquoise water",
    "Agricultural": "agricultural farmland with crop field patterns, irrigation channels, rural patchwork",
    "Mountain":     "rugged mountain terrain with rocky peaks, steep slopes, and alpine valleys",
}

# Tuple format: (weather_prompt_fragment, img2img_strength)
# strength=0.0 → no img2img pass needed (clear weather = base image)
# Higher strengths add more noise before denoising.
# Higher strength = more weather effect, but risks losing terrain structure
WEATHER_PROMPTS = {
    "Clear":    ("clear sky, bright sunlight, sharp visibility, vivid colors",   0.00),
    "Foggy":    ("dense ground fog, pale gray haze fading distant terrain, low contrast", 0.63),
    "Rainy":    ("heavy rainfall, reflective wet ground, puddles, muted gray lighting", 0.67),
    "Night":    ("dark nighttime, deep shadows across the ground, isolated artificial lights", 0.75),
    "Overcast": ("heavily overcast, flat grey lighting, full cloud cover",        0.44),
    "Snowy":    ("snow covered landscape, white blanket of snow, winter scene",   0.57),
}

WEATHER_NEGATIVES = {
    "Foggy": "crystal-clear visibility, crisp distant detail, direct sunlight",
    "Rainy": "dry ground, bright sunshine, hard sun shadows",
    "Night": "daylight, sunlight, evenly lit terrain",
}

# Altitude phrases guide the model on perceived scale and detail level
ALTITUDE_PROMPTS = {
    "Low (100m)":    "very close low altitude drone view, fine ground detail visible",
    "Medium (400m)": "standard medium altitude drone view, moderate level of detail",
    "High (1000m)":  "high altitude aerial overview, large geographic area visible",
}

# Negative prompt: tokens the model steers AWAY from during generation
# These ensure aerial top-down perspective and photorealism
NEGATIVE_PROMPT = (
    "blurry, distorted, unrealistic, cartoon, painting, illustration, "
    "low quality, jpeg artifacts, watermark, text, logo, "
    "tilted view, sideways view, street level, ground level, "
    "fisheye, lens distortion, person, human, face"
)


# ─────────────────────────────────────────────────────────────
# CELL 5 — Prompt Builder
# ─────────────────────────────────────────────────────────────

def build_base_prompt(terrain: str, altitude: str) -> str:
    """
    Constructs the full text prompt for Stage 1 (txt2img).
    Always generates clear-weather base to preserve terrain structure.
    """
    terrain_desc  = TERRAIN_PROMPTS[terrain]
    altitude_desc = ALTITUDE_PROMPTS[altitude]

    return (
        f"photorealistic aerial drone photograph of {terrain_desc}, "
        f"clear weather, bright daylight, {altitude_desc}, "
        f"top-down bird's eye view, satellite imagery style, "
        f"8k resolution, ultra detailed, professional drone photography, "
        f"sharp focus, high dynamic range"
    )


def build_weather_prompt(terrain: str, weather: str, altitude: str) -> str:
    """
    Constructs the full text prompt for Stage 2 (img2img weather conditioning).
    Retains terrain context while adding weather semantics.
    """
    terrain_desc  = TERRAIN_PROMPTS[terrain]
    weather_desc, _ = WEATHER_PROMPTS[weather]
    altitude_desc = ALTITUDE_PROMPTS[altitude]

    return (
        f"{weather_desc}, top-down aerial drone photograph of {terrain_desc}, "
        f"{altitude_desc}"
    )


# ─────────────────────────────────────────────────────────────
# CELL 6 — Core Two-Stage Generation Pipeline
# ─────────────────────────────────────────────────────────────

def generate_aerial_image(
    terrain       : str   = "Urban",
    weather       : str   = "Clear",
    altitude      : str   = "Medium (400m)",
    seed          : int   = -1,
    num_steps     : int   = 25,
    guidance_scale: float = 7.5,
) -> tuple[Image.Image, int]:
    """
    Two-stage SkySynth generation pipeline.

    Stage 1 — Text-to-Image (txt2img):
        A clean base aerial image is generated from terrain + altitude prompt.
        Uses the full reverse diffusion process (T → 0 timesteps).

    Stage 2 — SDEdit Weather Conditioning (img2img):
        The base image is encoded into latent space, partially noised to
        timestep t = strength × T, then denoised with a weather-conditioned
        prompt. This transfers weather semantics while preserving terrain
        structure from Stage 1.

        strength controls the trade-off:
            low  strength → subtle weather, strong terrain preservation
            high strength → dramatic weather, risk of terrain drift

    Args:
        terrain        : terrain category key
        weather        : weather category key
        altitude       : altitude category key
        seed           : RNG seed (-1 for random); same seed = same base image
        num_steps      : number of denoising steps (20-25 sufficient with DPM-Solver++)
        guidance_scale : CFG scale — higher = more prompt-adherent, less diverse

    Returns:
        (PIL.Image, seed_used)
    """
    # ── Seed handling ─────────────────────────────────────────
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    base = generate_clear_base(terrain, altitude, seed, num_steps, guidance_scale)
    return apply_weather(base, terrain, weather, altitude, seed, num_steps, guidance_scale), seed


def generate_clear_base(terrain, altitude, seed, num_steps, guidance_scale):
    generator = torch.Generator(device=device).manual_seed(seed)

    # ── Stage 1: Base terrain generation ─────────────────────
    base_prompt = build_base_prompt(terrain, altitude)

    return txt2img_pipe(
        prompt           = base_prompt,
        negative_prompt  = NEGATIVE_PROMPT,
        num_inference_steps = num_steps,
        guidance_scale   = guidance_scale,
        generator        = generator,
        height           = 512,
        width            = 512,
    ).images[0]


def apply_weather(base_image, terrain, weather, altitude, seed, num_steps, guidance_scale):
    """Apply SDEdit to an existing clear image so all variants share that exact input."""
    if weather == "Clear":
        return base_image

    # ── Stage 2: Weather conditioning via SDEdit ──────────────
    _, strength = WEATHER_PROMPTS[weather]
    weather_prompt = build_weather_prompt(terrain, weather, altitude)
    negative_prompt = NEGATIVE_PROMPT
    if weather in WEATHER_NEGATIVES:
        negative_prompt += ", " + WEATHER_NEGATIVES[weather]

    # Reinitialise generator with same seed for determinism
    generator = torch.Generator(device=device).manual_seed(seed)

    return img2img_pipe(
        prompt           = weather_prompt,
        negative_prompt  = negative_prompt,
        image            = base_image,
        strength         = strength,
        num_inference_steps = num_steps,
        guidance_scale   = guidance_scale,
        generator        = generator,
    ).images[0]

# ─────────────────────────────────────────────────────────────
# CELL 7 — Batch Results Grid (3 Terrains × 3 Weathers = 9 Images)
# ─────────────────────────────────────────────────────────────

def generate_results_grid(
    terrains : list = ["Urban", "Forest", "Desert"],
    weathers : list = ["Clear", "Foggy", "Night"],
    altitude : str  = "Medium (400m)",
    seed     : int  = 42,          # Fixed seed → reproducible grid
    num_steps: int  = 25,
):
    """
    Generates a grid of images across terrain × weather combinations.
    Reuses one exact base image for each terrain. Edits can still change geometry.
    """
    rows, cols = len(terrains), len(weathers)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    fig.suptitle(
        "SkySynth AI — Synthetic Aerial Imagery Results\n"
        f"({rows} Terrains × {cols} Weather Conditions | Seed: {seed})",
        fontsize=16, fontweight="bold", y=1.02
    )

    for i, terrain in enumerate(terrains):
        base = generate_clear_base(terrain, altitude, seed, num_steps, 7.5)
        for j, weather in enumerate(weathers):
            print(f"[{i*cols + j + 1}/{rows*cols}] Generating: {terrain} × {weather}...")

            img = apply_weather(base, terrain, weather, altitude, seed, num_steps, 7.5)

            ax = axes[i][j]
            ax.imshow(np.array(img))
            ax.set_title(f"{terrain}  |  {weather}", fontsize=12, fontweight="bold", pad=8)
            ax.axis("off")

    plt.tight_layout()
    output_path = "skysynth_results_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrid saved → {output_path}")
    return output_path

# ─────────────────────────────────────────────────────────────
# CELL 8 — Gradio Interactive Demo
# ─────────────────────────────────────────────────────────────

def gradio_inference(terrain, weather, altitude, seed_text, steps, guidance_scale):
    """Generate one base and show the requested weather edit alongside it."""
    raw_seed = (seed_text or "").strip()
    try:
        seed = int(raw_seed) if raw_seed else -1
    except ValueError:
        raise gr.Error("Enter a whole-number seed or leave the field blank.") from None

    if not -1 <= seed <= 2**32 - 1:
        raise gr.Error("Use -1 for a random seed or an integer from 0 to 4,294,967,295.")
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    steps = int(steps)
    base = generate_clear_base(terrain, altitude, seed, steps, guidance_scale)
    variant = apply_weather(base, terrain, weather, altitude, seed, steps, guidance_scale)
    settings = (
        f"**Seed** {seed} | **Terrain** {terrain} | **Weather** {weather} | "
        f"**Altitude preset** {altitude} | **Steps** {steps} | "
        f"**Guidance** {guidance_scale:g}"
    )
    return base, variant, settings, str(seed)


with gr.Blocks(
    title="SkySynth AI",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        "# SkySynth AI\n"
        "Generate a synthetic aerial scene and compare it with a weather edit of the same image."
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=240):
            terrain_dd = gr.Dropdown(list(TERRAIN_PROMPTS), value="Urban", label="Terrain")
            weather_dd = gr.Dropdown(list(WEATHER_PROMPTS), value="Foggy", label="Weather")
            altitude_dd = gr.Dropdown(
                list(ALTITUDE_PROMPTS), value="Medium (400m)", label="Altitude preset"
            )
            seed_input = gr.Textbox(
                label="Seed",
                placeholder="Leave blank for random",
                info="Keep the generated seed to try more weather edits on the same base.",
            )
            with gr.Accordion("Generation settings", open=False):
                steps_input = gr.Slider(10, 50, value=25, step=5, label="Denoising steps")
                guidance_input = gr.Slider(3, 15, value=7.5, step=0.5, label="Prompt guidance")
            generate_button = gr.Button("Generate comparison", variant="primary")

        with gr.Column(scale=3, min_width=320):
            with gr.Row():
                clear_image = gr.Image(label="Clear base", format="png", interactive=False)
                weather_image = gr.Image(label="Selected weather", format="png", interactive=False)
            generation_settings = gr.Markdown()

    generate_button.click(
        fn=gradio_inference,
        inputs=[terrain_dd, weather_dd, altitude_dd, seed_input, steps_input, guidance_input],
        outputs=[clear_image, weather_image, generation_settings, seed_input],
        concurrency_limit=1,
        show_progress="full",
    )

# Launch locally when running the script directly.
if __name__ == "__main__":
    demo.launch(share=False, quiet=True)
