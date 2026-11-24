"""Measure weather prompt alignment using CLIP ViT-B/32."""
import csv
from pathlib import Path

MODEL_ID = "openai/clip-vit-base-patch32"
WEATHERS = ("Clear", "Foggy", "Rainy", "Night")


def evaluate_weather_alignment(images, prompts, output_dir="clip_evaluation", device="cpu"):
    """Print and save a four-weather report; score = 100 × max(0, cosine)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if any(w not in images or w not in prompts for w in WEATHERS):
        raise ValueError(f"Provide an image and target prompt for each of {WEATHERS}.")
    processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=False).to(device).eval()
    inputs = processor(
        images=[images[w].convert("RGB") for w in WEATHERS],
        text=[prompts[w] for w in WEATHERS],
        return_tensors="pt", padding=True, truncation=True,
        max_length=model.config.text_config.max_position_embeddings,
    ).to(device)
    with torch.inference_mode():
        output = model(**inputs)
        image_embeddings = torch.nn.functional.normalize(output.image_embeds.float(), dim=-1)
        text_embeddings = torch.nn.functional.normalize(output.text_embeds.float(), dim=-1)
        similarities = (image_embeddings @ text_embeddings.T).cpu().tolist()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f'{"Requested":<10} {"CLIPScore":>10} {"Top prompt":>12} '
          f'{"Target margin":>14} {"Gain vs clear":>14}')
    for i, weather in enumerate(WEATHERS):
        scores = similarities[i]
        cosine = scores[i]
        top = WEATHERS[max(range(len(WEATHERS)), key=scores.__getitem__)]
        margin = cosine - max(score for j, score in enumerate(scores) if j != i)
        gain_vs_clear = cosine - similarities[0][i]
        row = dict(weather=weather, cosine_similarity=cosine,
                   clip_score=100 * max(0, cosine), top_weather_prompt=top,
                   cosine_margin=margin, cosine_gain_vs_clear=gain_vs_clear,
                   target_prompt=prompts[weather], model=MODEL_ID)
        rows.append(row)
        gain_text = f"{gain_vs_clear:+.4f}" if i else "reference"
        print(f'{weather:<10} {row["clip_score"]:>10.2f} {top:>12} '
              f'{margin:>+14.4f} {gain_text:>14}')

    with (output_dir / "alignment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "cosine_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image / target prompt", *WEATHERS])
        writer.writerows([w, *scores] for w, scores in zip(WEATHERS, similarities))
    return rows
