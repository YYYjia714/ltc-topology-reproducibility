from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.models import MotionForecastModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize audited seed42 reuse records")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    output_root = args.output
    reviewed_files: list[dict[str, object]] = []
    for path_text, expected_sha256 in plan["source_sha256"].items():
        path = Path(path_text)
        if not path.is_file():
            raise RuntimeError(f"Reviewed source is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"Reviewed source changed before inventory freeze: {path}")
        reviewed_files.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    records: list[dict[str, object]] = []
    for item in plan["reused_artifacts"]:
        source_dir = Path(item["source_dir"])
        dataset = item["dataset"]
        role = item["role"]
        source_files = []
        for name in item["required_outputs"]:
            path = source_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            source_files.append(
                {"name": name, "path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
        record: dict[str, object] = {
            "id": item["id"],
            "dataset": dataset,
            "role": role,
            "seed": 42,
            "source_dir": str(source_dir),
            "source_files": source_files,
        }
        if role == "skeleton_no_rollout":
            metadata_path = Path(plan["data_root"]) / dataset / "common18" / "metadata.json"
            metadata = read_json(metadata_path)
            edges = [[int(a), int(b)] for a, b in metadata["skeleton_edges"]]
            guidance_index = int(metadata["guidance_joint_index"])
            model = MotionForecastModel(
                model_type="ltc_topology",
                joints=18,
                history=25,
                future=25,
                hidden_size=256,
                num_layers=2,
                dropout=0.0,
                use_topology_encoder=True,
                use_topology_decoder=True,
                use_root_guidance=True,
                skeleton_edges=edges,
                guidance_joint_index=guidance_index,
            )
            source_checkpoint = source_dir / "best.pt"
            checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            architecture_signature = {
                "model": "ltc_topology",
                "adjacency_mode": "skeleton",
                "use_anchor_guidance": True,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.0,
            }
            compatible_dir = output_root / dataset / "seed42_skeleton_no_rollout"
            compatible_dir.mkdir(parents=True, exist_ok=True)
            compatible_path = compatible_dir / "best_compatible.pt"
            torch.save(
                {
                    "epoch": int(checkpoint.get("epoch", 0)),
                    "model_state": checkpoint["model_state"],
                    "architecture_signature": architecture_signature,
                    "legacy_source_path": str(source_checkpoint),
                    "legacy_source_sha256": sha256_file(source_checkpoint),
                    "conversion": "metadata_only_state_dict_strict_load_verified",
                    "manifest_sha256": plan["split_manifest_sha256"],
                    "seed": 42,
                },
                compatible_path,
            )
            record["compatible_checkpoint"] = {
                "path": str(compatible_path),
                "size": compatible_path.stat().st_size,
                "sha256": sha256_file(compatible_path),
                "architecture_signature": architecture_signature,
                "state_dict_strict_load_verified": True,
            }
        records.append(record)
    inventory = {
        "status": "PASS",
        "plan_path": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "files": reviewed_files,
        "records": records,
    }
    inventory_path = output_root / "reused_inventory.json"
    atomic_json(inventory_path, inventory)
    print(json.dumps({"path": str(inventory_path), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
