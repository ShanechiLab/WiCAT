# WiCAT

## Publication

This repository provides the implementation of WiCAT, a framework for cross-subject modeling of widefield imaging neural data. Mohammad Hosseini, Eray Erturk, Saba Hashemi, and Maryam M. Shanechi. [_WiCAT: Cross-Subject Modeling of Widefield Imaging Neural Data_](https://openreview.net/forum?id=pZq2RMptsQ).

WiCAT is a compact public release for loading pretrained cross-subject widefield-imaging encoders and fitting a lightweight behavioral regression head on top of the frozen representation.

In this repo we include the model code, a tiny processed Musall-style example dataset, metadata for that example subset, pretrained encoder weights, and a training entry point that fits only the downstream decoder head.

## What The Model Does

WiCAT maps widefield imaging movies to token-level latent embeddings and then decodes behavior from those embeddings.

The public model path is:

1. `ImagePatchTokenizer` splits imaging tensors into temporal/spatial patches.
2. Each patch is projected into a shared hidden dimension.
3. Learnable spatial patch embeddings are added.
4. A rotary-position Transformer backbone contextualizes the patch tokens across time.
5. For behavioral regression, tokens are reshaped as `[batch, time, spatial_patch, hidden]`.
6. Spatial patch tokens are pooled per time bin.
7. A small MLP decoder predicts behavior for every time bin.

For the default config, a sample has 205 time bins, 16 spatial patches per time bin, and 512-dimensional latent tokens. The decoder therefore receives a `[batch, 205, 512]` sequence after spatial pooling and predicts `[batch, 205, d_kinem]`.

## Repository Map

- `wicat/train.py`: command-line training script. It loads configs, builds the model, restores pretrained weights, freezes the upstream encoder, trains the decoder head, saves the decoder, and runs a sample forward pass.
- `wicat/model_setup.py`: model construction and checkpoint loading helpers. Public checkpoints are stored as plain PyTorch state dicts with `tokenizer.*` and `backbone.*` keys.
- `wicat/models/model.py`: `WiCAT` wrapper combining tokenizer, Transformer backbone, and downstream decoder.
- `wicat/models/tokenizer.py`: image-to-token patching and patch embedding logic.
- `wicat/models/transformer.py`: rotary self-attention Transformer encoder.
- `wicat/models/MLP.py`: decoder MLP used for regression.
- `wicat/data/metadata.py`: metadata table wrapper used to infer session input shapes and behavioral output dimensions.
- `wicat/data/*_dataset.py`: dataset processing utilities retained for users who want to prepare full datasets.
- `wicat/config/model.yaml`: encoder architecture.
- `wicat/config/decoder.yaml`: downstream regression head.
- `wicat/config/train.yaml`: runnable example using `pretrained_models/best-model-musall.ckpt`.
- `wicat/config/train_all.yaml`: runnable example using `pretrained_models/best-model-all.ckpt`.

## Included Assets

- `pretrained_models/best-model-musall.ckpt`: Musall pretrained WiCAT encoder checkpoint.
- `pretrained_models/best-model-all.ckpt`: pretrained encoder checkpoint converted from the all-dataset pretraining run `pret_all2_mask0.9_sps32_d512_nl8_1`.
- `pretrained_models/pret_weights/`: split compressed archives for the pretrained checkpoints.
- `pretrained_models/metadata_5257c_local10.csv`: metadata for the tiny local example subset.
- `pretrained_models/dataset/`: 10 processed example segments.
- `pretrained_models/decoder_head.pt`: example decoder-head output file produced by training.

The raw `.ckpt` and `.pt` files are ignored by git, so use the split archives under `pretrained_models/pret_weights` for repository upload or release assets.

## Environment

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

The model uses PyTorch, xFormers attention, einops, pandas/numpy, and PyYAML. Use a CUDA-enabled PyTorch/xFormers build when running on GPU.

## Restore Pretrained Weights From Split Archives

The raw checkpoint file is not present; reconstruct it from the split archive parts.


All-dataset checkpoint:

```bash
cat pretrained_models/pret_weights/best-model-all.z01 \
    pretrained_models/pret_weights/best-model-all.z02 \
    pretrained_models/pret_weights/best-model-all.z03 \
    pretrained_models/pret_weights/best-model-all.z04 \
    pretrained_models/pret_weights/best-model-all.z05 \
    pretrained_models/pret_weights/best-model-all.zip \
    > pretrained_models/pret_weights/best-model-all.full.zip

unzip pretrained_models/pret_weights/best-model-all.full.zip -d pretrained_models/
```
Existing Musall dataset checkpoint:

```bash
cat pretrained_models/pret_weights/best-model-musall.z01 \
    pretrained_models/pret_weights/best-model-musall.z02 \
    pretrained_models/pret_weights/best-model-musall.z03 \
    pretrained_models/pret_weights/best-model-musall.z04 \
    pretrained_models/pret_weights/best-model-musall.z05 \
    pretrained_models/pret_weights/best-model-musall.zip \
    > pretrained_models/pret_weights/best-model-musall.full.zip

unzip pretrained_models/pret_weights/best-model-musall.full.zip -d pretrained_models/
```

## Fit A Regression Decoder On Top Of A Pretrained Encoder

From the repository root:

```bash
python -m wicat.train --config wicat/config/train.yaml
```

This fits only the regression decoder head:

- Loads metadata from `metadata_csv`.
- Builds the tokenizer/backbone/decoder from YAML configs.
- Loads the pretrained encoder checkpoint from `checkpoint_path`.
- Freezes tokenizer and Transformer backbone parameters.
- Optimizes the decoder MLP with MSE loss against `kinem`.
- Saves the decoder head to `training.save_decoder_path`.
- Runs one sample forward pass and prints output shape/dtype.

To use the all-dataset pretrained encoder:

```bash
python -m wicat.train --config wicat/config/train_all.yaml
```

To fit your own regression target, prepare a metadata CSV with at least:

- `subject`
- `session`
- `subject_session`
- `d_imaging`
- `d_kinem`
- `path`

Each segment file referenced by `path` should be a PyTorch object containing:

- `imaging`: widefield tensor shaped like `[time, channel, height, width]`.
- `kinem`: regression target shaped like `[time, d_kinem]`.

Then update `metadata_csv`, `checkpoint_path`, and `training.save_decoder_path` in a train config.

## Checkpoint Format

The public checkpoints are not full pretraining checkpoints. They are plain PyTorch state dicts containing:

- `tokenizer.*`
- `backbone.*`

Pretraining codes, MAE reconstruction heads, optimizer state, callbacks, trainer state, and internal experiment metadata are removed for now.

## Citation

Please cite the paper linked on the OpenReview page above when using this code or pretrained models.

## License

Copyright (c) 2026 University of Southern California

See full notice in [LICENSE.md](LICENSE.md).

Mohammad Hosseini and Maryam M. Shanechi  
Shanechi Lab, University of Southern California
