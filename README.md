# SkySynth AI: Synthetic Aerial Imagery Generation & Multi-Weather Evaluation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.35.2-orange.svg)](https://github.com/huggingface/diffusers)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange.svg)](https://gradio.app/)

**SkySynth AI** is an end-to-end generative pipeline and evaluation framework for drone aerial perception. Collecting real-world drone imagery across diverse environmental conditions (fog, heavy rain, low-light night) is cost-prohibitive and operationally constrained. SkySynth synthesizes high-fidelity aerial imagery and generates paired adverse-weather variants from identical base scenes to systematically benchmark how computer vision perception systems degrade under low-visibility conditions.

---

## Key Features

- **Dual-Stage Generative Pipeline**:
  - **Stage 1 (Base Generation - Text-to-Image)**: Generates high-resolution clear aerial base imagery using **Stable Diffusion v1.5** with **DPM-Solver++ Multistep Scheduler** across 6 terrain presets (*Urban*, *Forest*, *Desert*, *Coastal*, *Agricultural*, *Mountain*) and 3 altitude guidance levels (*Low ~100m*, *Medium ~400m*, *High ~1000m*).
  - **Stage 2 (Weather Synthesis - SDEdit / img2img)**: Applies stochastic differential editing (SDEdit) to inject controlled Gaussian noise ($t = \text{strength} \times T$) into the clear base latent space and denoises with adverse weather conditioning (*Foggy*, *Rainy*, *Night*, *Snowy*, *Overcast*).
- **Structure-Preserving Controlled Haze**: Image-space contrast attenuation overlay module for evaluating vehicle detection under visibility reduction without altering underlying pixel coordinates or vehicle geometry.
- **Automated YOLOv8 Perception Benchmark**:
  - Tests off-the-shelf and fine-tuned object detectors (**YOLOv8n**) on paired clear vs. weather variants.
  - Automatically exports normalized YOLO format pseudo-labels (`class_id x_center y_center width height`) and outputs comparative detection confidence distributions.
- **CLIPScore Multi-Prompt Evaluation**:
  - Evaluates cross-modal image-text alignment using **OpenAI CLIP ViT-B/32**.
  - Computes cosine similarity ($100 \times \max(0, \cos(\theta))$), target prompt margin, and score gain vs. clear reference scenes.
- **Interactive Gradio Interface**: Real-time side-by-side visual comparison with seed retention, interactive prompt tuning, and direct PNG asset downloads.

---

## Repository Structure

```
SkySynth-AI/
├── skysynth_colab.py          # Standalone pipeline module & Gradio interactive application
├── skysynth_notebook.ipynb    # Complete end-to-end Google Colab notebook with outputs
├── benchmark_yolo.py          # YOLOv8 object detection benchmark & pseudo-label exporter
├── evaluate_clip.py           # CLIP ViT-B/32 semantic alignment and prompt margin evaluation
├── build_notebook.py          # Script to compile standalone notebook from modular source files
├── SkySynth_Report_Points.txt # Experimental observations, findings, and college project notes
├── requirements.txt           # Python dependencies
├── .gitignore                 # Standard Python/PyTorch/Jupyter ignore rules
└── README.md                  # Project documentation
```

---

## Quickstart

### 1. Google Colab (Recommended)
1. Upload [`skysynth_notebook.ipynb`](skysynth_notebook.ipynb) to [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime under **Runtime → Change runtime type → T4 GPU**.
3. Run all cells (**Runtime → Run all**). The notebook will automatically download model checkpoints, render sample grids, and run evaluation benchmarks.
4. To launch a public Gradio demo directly from Colab, set `LAUNCH_GRADIO = True` in the final cell.

### 2. Local Setup & Interactive Gradio Web UI
Ensure you have Python 3.10+ and PyTorch 2.6+ with CUDA support installed:

```bash
# Clone the repository
git clone https://github.com/premsai1703/SkySynth-AI.git
cd SkySynth-AI

# Install dependencies
pip install -r requirements.txt

# Run Gradio application
python skysynth_colab.py
```
Open the local URL (`http://127.0.0.1:7860`) printed in the console.

### 3. YOLOv8 Perception Benchmarking CLI
To benchmark an existing paired set of clear and adverse weather images:

```bash
python benchmark_yolo.py clear.png foggy.png --weather Foggy --seed 42 --conf 0.25 --out yolo_benchmark
```

Outputs generated in `--out`:
- `comparison.png`: Side-by-side detection visualization and confidence scatter plot.
- `summary.csv`: Detection counts and mean confidence metrics.
- `settings.json`: Reproducible benchmark configuration.
- `labels/`: Normalized YOLO-format bounding box annotations (`.txt`).
- `images/`: Paired image exports.

### 4. Regenerating the Notebook
If you modify `skysynth_colab.py`, `benchmark_yolo.py`, or `evaluate_clip.py`, regenerate the Colab notebook by running:

```bash
python build_notebook.py
```

---

## Methodological Analysis & Evaluation Notes

1. **Perception Degradation under Weather**:
   - In benchmark trials with synthetic parking lots (seed 2024, YOLOv8n @ conf=0.25), clear imagery yielded car detections at mean confidence ~0.527, while controlled fog overlay dropped detections to 0, demonstrating the sensitivity of standard pretrained detectors to contrast loss in aerial viewpoints.
2. **SDEdit vs. Controlled Overlay**:
   - **SDEdit** provides photorealistic weather textures and ambient lighting shifts, but higher diffusion strengths ($\ge 0.60$) introduce subtle geometry drift in small ground objects (e.g. parked vehicles).
   - **Controlled Fog Overlay** guarantees zero structural drift, isolating pure visibility/contrast degradation for precise object recall benchmarking.
3. **Pseudo-Labels & Ground Truth**:
   - YOLO exports represent detector pseudo-labels. In absence of human ground-truth annotations, changes in detection counts reflect both true missed detections and false positives.
4. **CLIPScore Scope**:
   - CLIPScore evaluates overall semantic prompt alignment; it does not measure fine-grained physical realism or pixel-level geometric fidelity.

---

## License

Academic and research use under the MIT License. Developed for SEC 302 Generative AI.
