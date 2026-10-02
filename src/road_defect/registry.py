"""Promote a trained run into the MLflow Model Registry, and read it back.

Ultralytics logs `weights/best.pt` as a plain run artifact, so a registered
version points at that directory rather than at a packaged model flavour. That
keeps the registry the single place that answers "which weights are live",
which is what the serving layer asks at startup.

MLflow's named stages (Staging/Production) are deprecated, so promotion is
expressed with aliases: @challenger for a candidate, @champion for what serves.

Usage:
    uv run python -m road_defect.registry promote --experiment road-defect-v1
    uv run python -m road_defect.registry show
"""

import argparse
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient

MODEL_NAME = "road-defect-detector"
SELECTION_METRIC = "metrics/mAP50-95B"


def _client() -> MlflowClient:
    load_dotenv()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set. Copy .env.example to .env.")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def best_run(client: MlflowClient, experiment_name: str, metric: str = SELECTION_METRIC):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"No experiment named {experiment_name!r}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.`{metric}` DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No finished runs in {experiment_name!r}")
    return runs[0]


def promote(experiment_name: str, alias: str, metric: str = SELECTION_METRIC) -> None:
    client = _client()
    run = best_run(client, experiment_name, metric)

    print(f"Best run: {run.info.run_name} ({run.info.run_id})")
    for key in ("metrics/mAP50B", "metrics/mAP50-95B", "metrics/precisionB", "metrics/recallB"):
        if key in run.data.metrics:
            print(f"  {key:<24}{run.data.metrics[key]:.4f}")

    # mlflow.register_model() expects an MLmodel manifest, which Ultralytics doesn't
    # write -- weights/best.pt is a plain run artifact. create_model_version() against
    # the artifact's underlying URI registers it without that check.
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass
    runs_uri = f"runs:/{run.info.run_id}/weights"
    source = RunsArtifactRepository.get_underlying_uri(runs_uri)
    version = client.create_model_version(MODEL_NAME, source, run.info.run_id)
    client.set_registered_model_alias(MODEL_NAME, alias, version.version)
    print(f"\nRegistered {MODEL_NAME} v{version.version} as @{alias}")


def show() -> None:
    client = _client()
    try:
        versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    except Exception:
        raise SystemExit(f"No registered model named {MODEL_NAME!r}") from None

    print(f"{'version':<10}{'aliases':<26}{'run':<36}{SELECTION_METRIC}")
    for version in sorted(versions, key=lambda v: int(v.version)):
        run = client.get_run(version.run_id)
        score = run.data.metrics.get(SELECTION_METRIC, float("nan"))
        aliases = ",".join(f"@{a}" for a in version.aliases) or "-"
        print(f"{version.version:<10}{aliases:<26}{run.info.run_name:<36}{score:.4f}")


def download_weights(alias: str = "champion", dest: Path = Path("artifacts")) -> Path:
    """Fetch the aliased model's best.pt, returning the local path."""
    client = _client()
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    dest.mkdir(parents=True, exist_ok=True)

    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{alias}", dst_path=str(dest)
    )
    weights = Path(local) / "best.pt"
    if not weights.exists():
        raise SystemExit(f"best.pt not found under {local}")
    print(f"{MODEL_NAME}@{alias} -> v{version.version} -> {weights}")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_promote = sub.add_parser("promote", help="register the best run under an alias")
    p_promote.add_argument("--experiment", required=True)
    p_promote.add_argument("--alias", default="champion")
    p_promote.add_argument("--metric", default=SELECTION_METRIC)

    sub.add_parser("show", help="list registered versions")

    p_fetch = sub.add_parser("fetch", help="download the aliased weights")
    p_fetch.add_argument("--alias", default="champion")
    p_fetch.add_argument("--dest", type=Path, default=Path("artifacts"))

    args = parser.parse_args()
    if args.command == "promote":
        promote(args.experiment, args.alias, args.metric)
    elif args.command == "show":
        show()
    else:
        download_weights(args.alias, args.dest)


if __name__ == "__main__":
    main()
