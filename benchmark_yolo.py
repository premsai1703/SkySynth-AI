"""Compare detections on paired synthetic images and export YOLO pseudo-labels."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def export_yolo_dataset(images, results, output_dir):
    """Export paired mappings keyed by filename stem; no confidence in label files."""
    if not images or images.keys() != results.keys():
        raise ValueError("Provide matching, nonempty image and detection mappings.")
    if any(not stem or not stem.replace("-", "").replace("_", "").isalnum()
           for stem in images):
        raise ValueError("Image stems must use letters, numbers, hyphens or underscores.")
    output_dir = Path(output_dir)
    for folder in ("images", "labels"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    for stem, image in images.items():
        result = results[stem]
        image.convert("RGB").save(output_dir / "images" / f"{stem}.png")
        lines = []
        if result.boxes is not None:
            for class_id, coords in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist()):
                x1, y1, x2, y2 = (min(1.0, max(0.0, x)) for x in coords)
                if x2 <= x1 or y2 <= y1:
                    continue
                xywh = ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
                lines.append(f"{int(class_id)} " + " ".join(f"{x:.8f}" for x in xywh))
        (output_dir / "labels" / f"{stem}.txt").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
    names = next(iter(results.values())).names
    (output_dir / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    return output_dir


def benchmark_pair(clear_image, adverse_image, weather="Foggy", output_dir="yolo_benchmark",
                   weights="yolov8n.pt", conf=0.25, imgsz=640, device="cpu", seed=None):
    """Run one detector at identical settings on both inputs; return per-image summaries."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    weather = weather.title()
    if weather not in {"Foggy", "Rainy"}:
        raise ValueError("Choose Foggy or Rainy for the adverse image.")
    if clear_image.size != adverse_image.size:
        raise ValueError("Paired images must have the same dimensions.")
    output_dir = Path(output_dir)
    prefix = f"seed_{seed}_" if seed is not None else ""
    images = {f"{prefix}clear": clear_image, f"{prefix}{weather.lower()}": adverse_image}
    model = YOLO(weights)
    predictions = model.predict(
        [image.convert("RGB") for image in images.values()],
        conf=conf, imgsz=imgsz, device=device, verbose=False,
    )
    results = dict(zip(images, predictions, strict=True))
    export_yolo_dataset(images, results, output_dir)
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(3, 1))
    confidence_ax = fig.add_subplot(grid[1, :])
    rows = []
    for i, (label, prediction) in enumerate(zip(("Clear", weather), predictions)):
        scores = prediction.boxes.conf.tolist() if prediction.boxes is not None else []
        avg = mean(scores) if scores else None
        rows.append({"weather": label, "count": len(scores), "mean_confidence": avg})
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(prediction.plot(conf=True, pil=True))
        mean_text = f"{avg:.3f}" if avg is not None else "N/A"
        ax.set_title(f"{label} | {len(scores)} detections | mean confidence {mean_text}")
        ax.axis("off")
        if scores:
            confidence_ax.scatter([i]*len(scores), sorted(scores), alpha=0.6)
            confidence_ax.scatter(i, avg, marker="_", s=300, color="black", zorder=3)
        else:
            confidence_ax.text(i, 0.5, "No detections", ha="center")
    confidence_ax.set_xticks([0, 1], ["Clear", weather])
    confidence_ax.set(xlim=(-0.5, 1.5), ylim=(0, 1.04), ylabel="YOLO confidence")
    confidence_ax.axhline(conf, color="gray", linestyle=":", label="Detection threshold")
    confidence_ax.legend(loc="upper right")
    fig.suptitle(f"{Path(weights).name} | image size {imgsz}" +
                 (f" | generation seed {seed}" if seed is not None else ""))
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "settings.json").write_text(
        json.dumps(dict(weights=str(weights), conf=conf, imgsz=imgsz, seed=seed,
                        adverse_weather=weather, labels="detector pseudo-labels"), indent=2),
        encoding="utf-8",
    )
    return rows


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clear")
    parser.add_argument("adverse")
    parser.add_argument("--weather", choices=["Foggy", "Rainy"], default="Foggy")
    parser.add_argument("--seed", type=int, help="Generation seed (metadata only)")
    parser.add_argument("--out", default="yolo_benchmark")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with Image.open(args.clear) as clear, Image.open(args.adverse) as adverse:
        rows = benchmark_pair(clear, adverse, args.weather, args.out, args.weights,
                              args.conf, args.imgsz, args.device, args.seed)
    for row in rows:
        score = row["mean_confidence"]
        print(f'{row["weather"]:5} | {row["count"]:3} detections | '
              f'mean confidence: {f"{score:.3f}" if score is not None else "N/A"}')
    print(f"Saved comparison, settings and pseudo-labels to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
