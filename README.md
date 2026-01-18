# Know2Seg

This repository presents a novel approach for generating highly realistic synthetic liver tumors using layer decomposition and advanced texture synthesis. AI models trained on our synthetic tumor dataset achieve comparable or superior performance on real tumor segmentation tasks compared to models trained exclusively on real tumor data.

```
Know2Seg
│  main.py
│  monai_trainer.py // Training script using MONAI framework
│  transfer_label.py
│  tumor_saver.py  // Saves tumor data
│  validation.py // Validation script
│
├─datafolds
│
├─external
│  └─surface-distance // External library for surface distance calculations
│
├─networks // Contains various neural network architectures
│
├─networks2 // Alternative implementations of network architectures
│
├─optimizers
│      lr_scheduler.py // Learning rate scheduler
│      __init__.py
│
└─TumorGenerated
        TumorGenerated.py
        utils.py // Utility functions for tumor generation
        __init__.py
```

## Model

| Organ    | Tumor | Model             | Pre-trained? | Download                                                     |
| -------- | ----- | ----------------- | ------------ | ------------------------------------------------------------ |
| liver    | real  | unet              | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/liver/real/real.liver.no_pretrain.unet) |
| liver    | real  | swin_unetrv2_base | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/liver/real/real.liver.no_pretrain.swin_unetrv2_base) |
| liver    | synt  | unet              | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/liver/synt/synt.liver.no_pretrain.unet) |
| liver    | synt  | swin_unetrv2_base | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/liver/synt/synt.liver.no_pretrain.swin_unetrv2_base) |
| pancreas | real  | unet              | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/pancreas/real/real.pancreas.no_pretrain.unet) |
| pancreas | real  | swin_unetrv2_base | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/pancreas/real/real.pancreas.no_pretrain.swin_unetrv2_base) |
| pancreas | synt  | unet              | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/pancreas/synt/synt.pancreas.no_pretrain.unet) |
| pancreas | synt  | swin_unetrv2_base | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/pancreas/synt/synt.pancreas.no_pretrain.swin_unetrv2_base) |
| kidney   | real  | unet              | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/kidney/real/real.kidney.no_pretrain.unet) |
| kidney   | real  | swin_unetrv2_base | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/kidney/real/real.kidney.no_pretrain.swin_unetrv2_base) |
| kidney   | synt  | unet              | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/kidney/synt/synt.kidney.no_pretrain.unet) |
| kidney   | synt  | swin_unetrv2_base | no           | [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main/kidney/synt/synt.kidney.no_pretrain.swin_unetrv2_base) |

**You can download other materials from these links:**

All other checkpoints: [link](https://huggingface.co/MrGiovanni/Pixel2Cancer/tree/main)

Data: Liver ([link](https://www.dropbox.com/scl/fi/ulok1xpk5e6nzicfipqxd/04_LiTS.tar.gz?rlkey=amo7x516if5m85x13q2iddgpj&dl=0)), Kidney ([link](https://www.dropbox.com/scl/fi/i7gzoocjnxyrqiavwuwp1/05_KiTS.tar.gz?rlkey=02mxa8f9sabcpe1858ww9580o&dl=0)), Pancreas ([link](https://www.dropbox.com/scl/fi/p35mz72vnvc01epdhr95r/Task07_Pancreas.tar.gz?rlkey=9z6grnqt6dpmh5yzz299g3wqx&dl=0))

## Installation

#### Dataset

Please download these datasets and save to `<data-path>` (user-defined).

- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094)

```bash
wget https://www.modelscope.cn/datasets/koukihk/Know2Seg/resolve/master/CT.zip # from ModelScope
wget https://www.modelscope.cn/datasets/koukihk/Know2Seg/resolve/master/Task03_Liver.zip # from ModelScope
```

#### Data Setting

```bash
# Task03_Liver training data list
--json_dir /datafolds/fold_0.json
--json_dir /datafolds/fold_1.json
--json_dir /datafolds/fold_2.json
--json_dir /datafolds/fold_3.json
--json_dir /datafolds/fold_4.json
```

#### Dependency

The code is tested on `python 3.8, Pytorch 1.11`.

```bash
conda create -n know2seg python=3.8
source activate know2seg (or conda activate know2seg)
cd Know2Seg
pip install external/surface-distance
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

#### Label

Our synthetic algorithm requires label as `0: background, 1: liver`, you need to transfer the label before training AI model.

```python
python transfer_label.py --data_path <data-path>  # <data-path> is user-defined data path to save datasets
```

or you can just download the label

```bash
wget https://www.modelscope.cn/datasets/koukihk/Know2Seg/resolve/master/label.zip # from ModelScope
```

## Train segmentation models

```bash
conda activate know2seg
cd Know2Seg
train_path=datafolds/healthy_ct
val_path=datafolds/10_Decathlon/Task03_Liver
fold=0
dist=$((RANDOM % 99999 + 10000))

# UNET (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=4000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --ellipsoid --layer_decomposition --logdir="runs/synt.unet$fold" --train_dir $train_path --val_dir $val_path --json_dir datafolds/fold_$fold.json
```

## Evaluation


```bash
conda activate know2seg
cd Know2Seg
val_path=datafolds/10_Decathlon/Task03_Liver
fold=0

# UNET (no.pretrain)
python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $val_path --json_dir datafolds/fold_$fold.json --log_dir runs/synt.unet$fold --save_dir outs
```

