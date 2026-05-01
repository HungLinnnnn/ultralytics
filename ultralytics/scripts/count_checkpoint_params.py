import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn


def format_count(count: int) -> str:
    return f"{count:,} ({count / 1e6:.3f}M)"


def count_module_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def count_tensor_dict_params(state: dict) -> int:
    total = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += value.numel()
    return total


def select_payload(payload):
    if isinstance(payload, nn.Module):
        return "module", payload

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(payload)!r}")

    for key in ("ema", "model"):
        value = payload.get(key)
        if isinstance(value, nn.Module):
            return key, value

    for key in ("state_dict", "model_state_dict", "ema_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return key, value

    tensor_like_items = {k: v for k, v in payload.items() if torch.is_tensor(v)}
    if tensor_like_items:
        return "checkpoint_tensor_dict", tensor_like_items

    raise KeyError("No module or state dict found in checkpoint.")


def discover_repo_roots(checkpoint_path: Path, explicit_root: Path | None) -> list[Path]:
    roots = []

    if explicit_root is not None:
        roots.append(explicit_root.expanduser().resolve())

    for parent in checkpoint_path.parents:
        if (parent / "models").is_dir() and (parent / "utils").is_dir():
            roots.append(parent)
        elif (parent / "ultralytics").is_dir():
            roots.append(parent)

    seen = set()
    unique_roots = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique_roots.append(root)
    return unique_roots


def prepend_sys_path(paths: list[Path]) -> None:
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_checkpoint(checkpoint_path: Path, repo_root: Path | None):
    repo_roots = discover_repo_roots(checkpoint_path, repo_root)
    prepend_sys_path(repo_roots)

    try:
        return torch.load(checkpoint_path, map_location="cpu"), repo_roots
    except ModuleNotFoundError as exc:
        searched = ", ".join(str(p) for p in repo_roots) if repo_roots else "(none)"
        raise ModuleNotFoundError(
            f"{exc}. Tried adding repo roots: {searched}. "
            "If the checkpoint was saved from another repo, pass --repo-root /path/to/repo."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Print parameter counts from a PyTorch checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt/.pth checkpoint")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional project root to prepend to sys.path before loading checkpoints with custom modules.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload, repo_roots = load_checkpoint(checkpoint_path, args.repo_root)
    source, obj = select_payload(payload)

    print(f"checkpoint: {checkpoint_path}")
    print(f"source: {source}")
    if repo_roots:
        print("repo_roots:")
        for root in repo_roots:
            print(f"  - {root}")

    if isinstance(obj, nn.Module):
        total, trainable = count_module_params(obj)
        print(f"total_params: {format_count(total)}")
        print(f"trainable_params: {format_count(trainable)}")
    else:
        total = count_tensor_dict_params(obj)
        print(f"total_params: {format_count(total)}")
        print("trainable_params: unavailable (state_dict only)")


if __name__ == "__main__":
    main()
