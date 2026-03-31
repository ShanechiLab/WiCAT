import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from wicat.model_setup import build_model, load_model_from_checkpoint, resolve_metadata


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, project_root: Path, must_exist: bool = True) -> str:
    candidate = Path(path)

    if candidate.is_absolute():
        resolved = candidate
    else:
        cwd_resolved = (Path.cwd() / candidate).resolve()
        if cwd_resolved.exists():
            resolved = cwd_resolved
        else:
            resolved = (project_root / candidate).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(
            f"Could not find path: {path}\n"
            f"Checked absolute/CWD/project-root relative under: {project_root}"
        )

    return str(resolved)


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="wicat/config/train.yaml")
    args = parser.parse_args()

    train_config_path = resolve_path(args.config, project_root=project_root, must_exist=True)
    train_cfg = load_yaml(train_config_path)

    model_config_path = resolve_path(
        train_cfg["model_config"], project_root=project_root, must_exist=True
    )
    model_cfg = load_yaml(model_config_path)

    decoder_config_path = train_cfg.get("decoder_config", "")
    if decoder_config_path:
        decoder_config_path = resolve_path(
            decoder_config_path, project_root=project_root, must_exist=True
        )
        decoder_cfg = load_yaml(decoder_config_path)
        model_cfg.setdefault("model", {})["decoder"] = decoder_cfg["decoder"]

    metadata_csv = train_cfg.get("metadata_csv", "")
    if metadata_csv:
        metadata_csv = resolve_path(metadata_csv, project_root=project_root, must_exist=True)

    checkpoint_path = train_cfg.get("checkpoint_path", "")
    if checkpoint_path:
        checkpoint_path = resolve_path(
            checkpoint_path, project_root=project_root, must_exist=True
        )

    metadata = resolve_metadata(
        metadata_csv=metadata_csv,
        checkpoint_path=checkpoint_path,
    )
    model = build_model(model_config=model_cfg, metadata=metadata)

    dtype_name = str(train_cfg.get("dtype", "bfloat16")).lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}. Use one of {list(dtype_map.keys())}.")
    compute_dtype = dtype_map[dtype_name]

    ckpt_path = checkpoint_path
    if ckpt_path:
        missing, unexpected = load_model_from_checkpoint(
            model=model,
            checkpoint_path=ckpt_path,
            strict=train_cfg.get("strict_load", False),
            map_location=train_cfg.get("device", "cpu"),
        )
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

    device = train_cfg.get("device", "cpu")
    model.to(device=device, dtype=compute_dtype)
    model.eval()
    print("WiCAT model initialized and ready.")

    metadata_base_dir = Path(metadata_csv).resolve().parent

    def resolve_segment_path(path_value: str) -> Path:
        path_obj = Path(str(path_value))
        if path_obj.is_absolute():
            return path_obj
        return (metadata_base_dir / path_obj).resolve()

    train_cfg_block = train_cfg.get("training", {})
    if train_cfg_block.get("train_decoder", True):
        for _, param in model.get_upstream_params():
            param.requires_grad = False
        for _, param in model.get_task_params():
            param.requires_grad = True

        model.train()
        optimizer = torch.optim.AdamW(
            [param for _, param in model.get_task_params() if param.requires_grad],
            lr=float(train_cfg_block.get("lr", 1e-3)),
            weight_decay=float(train_cfg_block.get("weight_decay", 1e-4)),
        )

        epochs = int(train_cfg_block.get("epochs", 20))
        train_rows = metadata._metadata_df.iloc[:10].copy().reset_index(drop=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_count = 0

            for row in train_rows.itertuples(index=False):
                segment_path = resolve_segment_path(row.path)
                segment = torch.load(segment_path, map_location="cpu", weights_only=False)

                if "imaging" not in segment or "kinem" not in segment:
                    continue

                x = segment["imaging"].to(dtype=compute_dtype).unsqueeze(0).to(device)
                target = segment["kinem"].to(dtype=compute_dtype).to(device)

                if target.dim() == 1:
                    target = target.unsqueeze(-1)

                preds = model(
                    x,
                    [str(row.subject_session)],
                    [str(row.subject)],
                ).squeeze(0)

                time_dim = min(preds.shape[0], target.shape[0])
                feat_dim = min(preds.shape[1], target.shape[1])
                preds_aligned = preds[:time_dim, :feat_dim]
                target_aligned = target[:time_dim, :feat_dim]

                loss = F.mse_loss(preds_aligned.float(), target_aligned.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                valid_count += 1

            avg_loss = epoch_loss / max(valid_count, 1)
            print(f"[Decoder Train] Epoch {epoch+1}/{epochs}, samples={valid_count}, mse={avg_loss:.6f}")

        save_decoder_path = train_cfg_block.get("save_decoder_path", "pretrained_models/decoder_head.pt")
        save_decoder_path = resolve_path(
            save_decoder_path, project_root=project_root, must_exist=False
        )
        Path(save_decoder_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.decoder_head.state_dict(), save_decoder_path)
        print(f"Saved decoder head to: {save_decoder_path}")

        model.eval()

    row = metadata._metadata_df.sample(n=1, random_state=42).iloc[0]
    sample_path = resolve_segment_path(row.path)
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)["imaging"]
    sample = sample.to(dtype=compute_dtype).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(sample, [str(row.subject_session)], [str(row.subject)])
    print(f"Sample output shape: {tuple(outputs.shape)}, dtype: {outputs.dtype}")

if __name__ == "__main__":
    main()
