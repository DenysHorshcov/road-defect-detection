"""Fine-tune a YOLO detector on the road defect dataset, tracked in MLflow.

Ultralytics ships MLflow logging enabled by default and reads its configuration
from environment variables, so this script sets those up and then stays out of
the way -- params, per-epoch metrics and end-of-run artifacts are logged by the
integration rather than by hand.

Usage:
    uv run python -m road_defect.train --model yolo26n.pt --epochs 50
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DATA = Path("data/processed/rdd_japan/dataset.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    # 4 GB of VRAM: batch 8 fits yolo26n at 640, yolo26s needs 4 or less.
    parser.add_argument("--batch", type=int, default=8)
    # 0 = load in the main process. Spawn-based DataLoader workers deadlock on
    # Windows before the first epoch; the 600x600 JPEGs decode fast enough that
    # single-process loading keeps the GPU fed anyway.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default=None, help="run name; defaults to the model stem")
    parser.add_argument("--experiment", default=None, help="overrides MLFLOW_EXPERIMENT_NAME")
    parser.add_argument(
        "--cache",
        default="",
        help="Ultralytics image cache: '' (default), 'disk' or 'ram'. "
        "Disk caching writes ~1 MB of .npy per image, so check free space first.",
    )
    args = parser.parse_args()

    load_dotenv()
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set. Copy .env.example to .env.")
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run road_defect.data.split first.")

    # Imported after the environment is prepared: the MLflow callback reads
    # these variables when the ultralytics module initialises.
    from ultralytics import YOLO, settings

    if not settings.get("mlflow"):
        settings.update({"mlflow": True})

    run_name = args.name or Path(args.model).stem
    print(f"Tracking to {tracking_uri} (experiment {os.environ.get('MLFLOW_EXPERIMENT_NAME')})")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        # project is deliberately unset: Ultralytics defaults it to
        # <runs_dir>/<task>, i.e. runs/detect. Passing any relative path here is
        # resolved against that same default and nests the directory twice.
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    print("\nFinal validation metrics")
    for key, value in results.results_dict.items():
        print(f"  {key:<28}{value:.4f}")
    print(f"\nWeights: {results.save_dir}")


if __name__ == "__main__":
    main()
