# Pretrained Models

This directory stores local checkpoint files and tiny example data for WiCAT.

## Checkpoints

- `best-model-musall.ckpt`: Musall pretrained WiCAT encoder checkpoint.
- `best-model-all.ckpt`: encoder checkpoint converted from the all-dataset pretraining run `/data/hoseini/DATA/njepa/analysis/pret_all2_mask0.9_sps32_d512_nl8_1`.

Both checkpoints are plain PyTorch state dicts with only `tokenizer.*` and `backbone.*` keys. The full Lightning trainer state, optimizer state, MAE predictor/reconstruction heads, callbacks, and run config are intentionally removed.

## Split Archives

Large raw checkpoints are ignored by git. The uploadable checkpoint archives live in `pret_weights/`.

To restore `best-model-all.ckpt`:

```bash
cat pret_weights/best-model-all.z01 \
    pret_weights/best-model-all.z02 \
    pret_weights/best-model-all.z03 \
    pret_weights/best-model-all.z04 \
    pret_weights/best-model-all.z05 \
    pret_weights/best-model-all.zip \
    > pret_weights/best-model-all.full.zip

unzip pret_weights/best-model-all.full.zip -d .
```

To restore `best-model-musall.ckpt`:

```bash
cat pret_weights/best-model-musall.z01 \
    pret_weights/best-model-musall.z02 \
    pret_weights/best-model-musall.z03 \
    pret_weights/best-model-musall.z04 \
    pret_weights/best-model-musall.z05 \
    pret_weights/best-model-musall.zip \
    > pret_weights/best-model-musall.full.zip

unzip pret_weights/best-model-musall.full.zip -d .
```

## Example Data

- `metadata_5257c_local10.csv`: metadata for the included 10 processed Musall-style segments.
- `dataset/`: the local segment files referenced by the metadata.
- `decoder_head.pt`: an example downstream decoder head output.
