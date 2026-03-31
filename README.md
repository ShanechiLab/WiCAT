# WiCAT (Anonymous Release)

This repository is an anonymized minimal WiCAT codebase for loading a pretrained model and running a small decoder finetuning task.

## Included assets

- Pretrained checkpoint: `pretrained_models/best-mae.ckpt`
- Decoder checkpoint output (after training): `pretrained_models/decoder_head.pt`
- Small local metadata subset: `pretrained_models/metadata_5257c_local10.csv`
- Small local segment subset: `pretrained_models/dataset/` (10 processed segment files)



## Environment

Build a Python environment with the required dependencies in `requirements.txt`

## Run

From repository root:

```bash
python -m wicat.train --config wicat/config/train.yaml
```

This command:

1. Loads metadata from `metadata_csv` in `wicat/config/train.yaml`
2. Loads pretrained backbone/tokenizer weights from `checkpoint_path`
3. Trains only the decoder head on the small local 10-sample subset
4. Saves decoder weights to `training.save_decoder_path`
5. Runs one sample forward pass and prints output shape/dtype

## Key configs

- Main training config: `wicat/config/train.yaml`
- Model config: `wicat/config/model.yaml`
- Decoder config: `wicat/config/decoder.yaml`
