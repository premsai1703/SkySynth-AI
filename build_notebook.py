"""Regenerate the standalone Colab notebook from the checked-in Python sources."""
import json
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = (ROOT / "skysynth_colab.py").read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^# CELL \d+ .+\n# [─]+\n", source)[1:]
    if len(blocks) != 8:
        raise ValueError("Expected eight source cells.")
    blocks = [re.sub(r"\n# [─]+\s*$", "", code.strip()) for code in blocks]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    blocks[0] = (
        "# Keep Colab's preinstalled torch and torchvision matched to CUDA.\n"
        "%pip install -q " + " ".join(
            shlex.quote(req) for req in requirements if req and not req.startswith(("#", "torch"))
        )
    )
    blocks[1] += (
        '\n\nif not torch.cuda.is_available():\n'
        '    raise RuntimeError("In Colab, choose Runtime > Change runtime type > T4 GPU.")\n'
        'if tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 6):\n'
        '    raise RuntimeError("Use a current Colab runtime with PyTorch 2.6 or newer.")'
    )
    blocks[6] += "\n\ngrid_path = generate_results_grid(num_steps=20)"
    blocks[7] = blocks[7].split("# Launch locally")[0].rstrip() + (
        '\n\n# Opt in to a public Gradio share link by running this cell with True.\n'
        'LAUNCH_GRADIO = False\n'
        'if LAUNCH_GRADIO:\n'
        '    demo.launch(share=True, prevent_thread_lock=True)\n'
    )
    benchmark = (ROOT / "benchmark_yolo.py").read_text(encoding="utf-8").split("\ndef main():")[0]
    clip = (ROOT / "evaluate_clip.py").read_text(encoding="utf-8").replace("MODEL_ID", "CLIP_MODEL_ID")
    experiment = '''# Reuse one vehicle-oriented clear scene across all four conditions.
import shutil
from IPython.display import display

TERRAIN = "Urban"
ALTITUDE = "Low (100m)"
SEED = 42
STEPS = 25
CFG = 7.5
ADVERSE = "Foggy"  # Change to "Rainy" to switch the YOLO comparison.
OUTPUT = Path("skysynth_report") / f"seed_{SEED}_{ADVERSE.lower()}"

# Add a parking lot to the urban terrain for this experiment to encourage detectable cars.
previous_urban = TERRAIN_PROMPTS[TERRAIN]
TERRAIN_PROMPTS[TERRAIN] = "a city block with roads and a parking lot with parked cars"
try:
    base = generate_clear_base(TERRAIN, ALTITUDE, SEED, STEPS, CFG)
    variants = {
        weather: apply_weather(base, TERRAIN, weather, ALTITUDE, SEED, STEPS, CFG)
        for weather in WEATHERS
    }
    eval_prompts = {
        "Clear": "a top-down aerial photograph in clear daylight",
        "Foggy": "a top-down aerial photograph in dense fog",
        "Rainy": "a top-down aerial photograph during heavy rain",
        "Night": "a top-down aerial photograph at night",
    }
    metadata = dict(seed=SEED, sd_model=MODEL_ID, terrain=TERRAIN,
                    terrain_description=TERRAIN_PROMPTS[TERRAIN], altitude=ALTITUDE,
                    steps=STEPS, guidance_scale=CFG,
                    base_prompt=build_base_prompt(TERRAIN, ALTITUDE),
                    edit_prompts={w: build_weather_prompt(TERRAIN, w, ALTITUDE)
                                  for w in WEATHERS if w != "Clear"},
                    weather_strengths={w: WEATHER_PROMPTS[w][1] for w in WEATHERS})
finally:
    TERRAIN_PROMPTS[TERRAIN] = previous_urban

scene_dir = OUTPUT / "scenes"
scene_dir.mkdir(parents=True, exist_ok=True)
for weather, image in variants.items():
    image.save(scene_dir / f"{weather.lower()}.png")
(fig, axes) = plt.subplots(1, len(WEATHERS), figsize=(14, 4), layout="constrained")
for ax, (weather, image) in zip(axes, variants.items()):
    ax.imshow(image)
    ax.set_title(f"Requested: {weather}")
    ax.axis("off")
fig.savefig(OUTPUT / "weather_comparison.png", dpi=150)
plt.show()
plt.close(fig)
(OUTPUT / "generation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

yolo_rows = benchmark_pair(variants["Clear"], variants[ADVERSE],
                           weather=ADVERSE, output_dir=OUTPUT / "yolo", seed=SEED)
display(Image.open(OUTPUT / "yolo" / "comparison.png"))
clip_rows = evaluate_weather_alignment(variants, eval_prompts, OUTPUT / "clip")
archive = shutil.make_archive(str(OUTPUT), "zip", root_dir=OUTPUT)
print(f"Download the report from the Colab Files panel: {archive}")
'''
    cells = []

    def add(kind, content):
        cell = dict(cell_type=kind, id=f"sky-{len(cells):02}", metadata={},
                    source=content.strip().splitlines(keepends=True))
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# SkySynth AI | SEC 302 Generative AI
Upload this notebook to Colab, choose **Runtime → Change runtime type → T4 GPU**, then choose **Runtime → Run all**. No companion files are needed. The first run downloads the three pretrained models. The Gradio demo is optional in the final cell.

Stable Diffusion 1.5 generates a clear base; SDEdit adds weather by partially noising and denoising that base. Matched variants share the same clear input, though editing can change objects or geometry.""")
    for i, block in enumerate(blocks[:7], 1):
        add("code", f"# Cell {i}\n" + block)
    add("markdown", """## Weather evaluation
YOLOv8n compares Clear with Foggy (or Rainy) at the same detector settings. Counts and confidence can go up or down. The black tick is the mean of retained detections; no detections gives an undefined mean. The exported boxes are pseudo-labels, and an empty label file only means the detector returned no boxes.

The image captions give the requested conditions. CLIP ranks four short prompts that differ in weather wording. CLIPScore uses 100 × max(0, cosine similarity). A positive target margin means the requested prompt scored above the strongest alternative. Gain vs clear compares the edit and the original clear image against the same target prompt. Small margins or gains indicate weak separation even when the top prompt matches. Inspect the four unannotated images together to judge weather visibility and scene preservation.""")
    add("code", benchmark.rstrip() + "\n\n" + clip.rstrip() + "\n\n" + experiment)
    add("code", "# Cell 8 | Optional Gradio demo\n" + blocks[7])
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata={
        "kernelspec": dict(display_name="Python 3", language="python", name="python3"),
        "language_info": dict(name="python"), "colab": {}, "accelerator": "GPU",
    })
    (ROOT / "skysynth_notebook.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
