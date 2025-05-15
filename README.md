# Beyond Shadows: A Large-Scale Benchmark for High-Fidelity Facial Shadow Removal

# 🔮 Dataset

The benchmark datasets are available at [Kaggle](https://www.kaggle.com/datasets/xuhangc/facialshadowremoval).

# ⚙️ Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python test.py
```
