import csv
import os
import time
import warnings
import math

from functools import partial

import nibabel as nib
import numpy as np
import torch
from monai import transforms, data
from TumorGenerated import AddMissingKeysd
from monai.data import load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted, Compose, Invertd
from scipy import ndimage
from scipy.ndimage import label
from surface_distance import (compute_surface_distances, compute_surface_dice_at_tolerance,
                              compute_average_surface_distance, compute_robust_hausdorff,
                              compute_surface_overlap_at_tolerance)

from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description='liver tumor validation')

# file dir
parser.add_argument('--val_dir', default=None, type=str)
parser.add_argument('--json_dir', default=None, type=str)
parser.add_argument('--save_dir', default='out', type=str)
parser.add_argument('--checkpoint', action='store_true')

parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--feature_size', default=16, type=int)
parser.add_argument('--val_overlap', default=0.5, type=float)
parser.add_argument('--num_classes', default=3, type=int)

parser.add_argument('--model', default='unet', type=str)
parser.add_argument('--swin_type', default='tiny', type=str)
parser.add_argument('--analyze_tumor_size', action='store_true', help='Analyze tumor by size')
parser.add_argument('--layer_decomposition', action='store_true', help='Enable layer decomposition (5 output channels)')


def to_percentage(value):
    return f"{value * 100:.2f}"


def to_decimal(value):
    return f"{value:.2f}"


def voxel2R(A):
    return (np.array(A)/4*3/np.pi)**(1/3)


def pixel2voxel(A, res):
    return np.array(A)*(res[0]*res[1]*res[2])


def denoise_pred(pred: np.ndarray):
    """
    # 0: background, 1: liver, 2: tumor.
    pred.shape: (3, H, W, D)
    """
    denoise_pred = np.zeros_like(pred)

    live_channel = pred[1, ...]
    labels, nb = label(live_channel)
    max_sum = -1
    choice_idx = -1
    for idx in range(1, nb + 1):
        component = (labels == idx)
        if np.sum(component) > max_sum:
            choice_idx = idx
            max_sum = np.sum(component)
    component = (labels == choice_idx)
    denoise_pred[1, ...] = component

    # 膨胀然后覆盖掉liver以外的tumor
    liver_dilation = ndimage.binary_dilation(denoise_pred[1, ...], iterations=30).astype(bool)
    denoise_pred[2, ...] = pred[2, ...].astype(bool) * liver_dilation

    denoise_pred[0, ...] = 1 - np.logical_or(denoise_pred[1, ...], denoise_pred[2, ...])

    return denoise_pred


def cal_dice(pred, true):
    if pred.shape != true.shape:
        common_shape = tuple(min(p, t) for p, t in zip(pred.shape, true.shape))
        pred = pred[tuple(slice(0, s) for s in common_shape)]
        true = true[tuple(slice(0, s) for s in common_shape)]
    intersection = np.sum(pred[true == 1]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def cal_dice_nsd(pred, truth, spacing_mm=(1, 1, 1), tolerance=2, percent=95):
    dice = cal_dice(pred, truth)
    # cal nsd
    surface_distances = compute_surface_distances(truth.astype(bool), pred.astype(bool), spacing_mm=spacing_mm)
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)
    rhd = compute_robust_hausdorff(surface_distances, percent)
    sd = max(compute_average_surface_distance(surface_distances))
    return (dice, nsd, sd, rhd)


def analyze_tumor_by_size(pred_tumor, label_tumor, spacing_mm):
    size_bins = {'0-5mm': {'tp': 0, 'fn': 0, 'fp': 0}, 
                 '5-10mm': {'tp': 0, 'fn': 0, 'fp': 0}, 
                 '>10mm': {'tp': 0, 'fn': 0, 'fp': 0}}
    
    pred_tumor = pred_tumor.astype(bool)
    label_tumor = label_tumor.astype(bool)
    
    if np.sum(label_tumor) > 0:
        label_cc, label_num = ndimage.label(label_tumor)
        for i in range(1, label_num + 1):
            tumor_region = (label_cc == i)
            tumor_size = np.sum(tumor_region)
            if tumor_size < 8:
                continue
                
            tumor_volume_mm = pixel2voxel(tumor_size, spacing_mm)
            tumor_radius_mm = voxel2R(tumor_volume_mm)
            
            overlap = np.sum(np.logical_and(tumor_region, pred_tumor)) / tumor_size
            detected = overlap > 0.1 
            
            if tumor_radius_mm <= 5:
                size_bins['0-5mm']['tp' if detected else 'fn'] += 1
            elif tumor_radius_mm <= 10:
                size_bins['5-10mm']['tp' if detected else 'fn'] += 1
            else:
                size_bins['>10mm']['tp' if detected else 'fn'] += 1
    
    if np.sum(pred_tumor) > 0:
        pred_cc, pred_num = ndimage.label(pred_tumor)
        for i in range(1, pred_num + 1):
            pred_region = (pred_cc == i)
            pred_size = np.sum(pred_region)
            if pred_size < 8:
                continue
                
            overlap = np.sum(np.logical_and(pred_region, label_tumor)) / pred_size
            if overlap <= 0.1:  # 假阳
                pred_volume_mm = pixel2voxel(pred_size, spacing_mm)
                pred_radius_mm = voxel2R(pred_volume_mm)
                
                if pred_radius_mm <= 5:
                    size_bins['0-5mm']['fp'] += 1
                elif pred_radius_mm <= 10:
                    size_bins['5-10mm']['fp'] += 1
                else:
                    size_bins['>10mm']['fp'] += 1
                
    return size_bins


def _get_model(args):
    inf_size = [96, 96, 96]
    print(args.model)
    if args.model == 'swin_unetrv2':
        if args.swin_type == 'tiny':
            feature_size = 12
        elif args.swin_type == 'small':
            feature_size = 24
        elif args.swin_type == 'base':
            feature_size = 48

        model = SwinUNETR_v2(in_channels=1,
                             out_channels=3,
                             img_size=(96, 96, 96),
                             feature_size=feature_size,
                             patch_size=2,
                             depths=[2, 2, 2, 2],
                             num_heads=[3, 6, 12, 24],
                             window_size=[7, 7, 7])

    elif args.model == 'unet':
        from monai.networks.nets import UNet
        out_channels = 5 if args.layer_decomposition else 3
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    else:
        raise ValueError('Unsupported model ' + str(args.model))

    if args.checkpoint:
        checkpoint = torch.load(os.path.join(args.log_dir, 'model.pt'), map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.', '')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)
        print('Use logdir weights')
    else:
        model_dict = torch.load(os.path.join(args.log_dir, 'model.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use logdir weights')

    model = model.cuda()
    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,
                            overlap=args.val_overlap, mode='gaussian')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    return model, model_inferer


def _get_loader(args):
    val_data_dir = args.val_dir
    datalist_json = args.json_dir
    val_org_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
            transforms.SpatialPadd(
                keys=["image", "label"],
                mode=["minimum", "constant"],
                spatial_size=[96, 96, 96]
            ),
            AddMissingKeysd(keys=["tumor_texture_layer", "tumor_mask_layer", "alpha"]),
            transforms.ToTensord(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"]),
        ]
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=val_data_dir)
    val_org_ds = data.Dataset(val_files, transform=val_org_transform)
    val_org_loader = data.DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4, sampler=None,
                                     pin_memory=True)

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        Invertd(
            keys="label",
            transform=val_org_transform,
            orig_keys="image",
            meta_keys="label_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        ),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        AsDiscreted(keys="pred", argmax=True, to_onehot=args.num_classes),
        AsDiscreted(keys="label", to_onehot=args.num_classes),
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg",
        # resample=False,output_dtype=np.uint8,separate_folder=False),
    ])

    return val_org_loader, post_transforms


def main():
    args = parser.parse_args()
    model_name = args.log_dir.split('/')[-1]
    args.model_name = model_name
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    torch.cuda.set_device(0)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    ## loader and post_transform
    val_loader, post_transforms = _get_loader(args)

    ## NETWORK
    model, model_inferer = _get_model(args)

    liver_dice = []
    liver_nsd = []
    liver_sd = []
    liver_rhd = []
    tumor_dice = []
    tumor_nsd = []
    tumor_sd = []
    tumor_rhd = []
    header = ['name', 'organ_dice', 'organ_nsd', 'organ_sd', 'organ_rhd', 'tumor_dice', 'tumor_nsd', 'tumor_sd',
              'tumor_rhd']

    rows = []
    
    tumor_size_stats = {'0-5mm': {'tp': 0, 'fn': 0, 'fp': 0}, 
                         '5-10mm': {'tp': 0, 'fn': 0, 'fp': 0}, 
                         '>10mm': {'tp': 0, 'fn': 0, 'fp': 0}}

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            val_inputs = val_data["image"].cuda()
            name = val_data['label_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
            original_affine = val_data["label_meta_dict"]["affine"][0].numpy()
            pixdim = val_data['label_meta_dict']['pixdim'].cpu().numpy()
            spacing_mm = tuple(pixdim[0][1:4])

            if args.layer_decomposition:
                val_prediction = model_inferer(val_inputs)[:, 2:5, ...]
            else:
                val_prediction = model_inferer(val_inputs)
            val_data["pred"] = val_prediction
            val_data = [post_transforms(i) for i in data.decollate_batch(val_data)]
            # val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            val_outputs, val_labels = val_data[0]['pred'], val_data[0]['label']

            # val_outpus.shape == val_labels.shape  (3, H, W, Z)
            val_outputs, val_labels = val_outputs.detach().cpu().numpy(), val_labels.detach().cpu().numpy()

            # denoise the ouputs
            val_outputs = denoise_pred(val_outputs)

            # align shapes defensively before metrics
            if val_outputs.shape[1:] != val_labels.shape[1:]:
                min_shape = tuple(min(a, b) for a, b in zip(val_outputs.shape[1:], val_labels.shape[1:]))
                val_outputs = val_outputs[:, :min_shape[0], :min_shape[1], :min_shape[2]]
                val_labels = val_labels[:, :min_shape[0], :min_shape[1], :min_shape[2]]

            current_liver_dice, current_liver_nsd, current_liver_sd, current_liver_rhd = cal_dice_nsd(
                val_outputs[1, ...], val_labels[1, ...], spacing_mm=spacing_mm)

            current_tumor_dice, current_tumor_nsd, current_tumor_sd, current_tumor_rhd = cal_dice_nsd(
                val_outputs[2, ...], val_labels[2, ...], spacing_mm=spacing_mm)

            if math.isinf(current_tumor_sd):
                current_tumor_sd = 100
                print('inf')

            if math.isinf(current_tumor_rhd):
                current_tumor_rhd = 200
                print('inf')

            liver_dice.append(current_liver_dice)
            liver_nsd.append(current_liver_nsd)
            liver_sd.append(current_liver_sd)
            liver_rhd.append(current_liver_rhd)
            tumor_dice.append(current_tumor_dice)
            tumor_nsd.append(current_tumor_nsd)
            tumor_sd.append(current_tumor_sd)
            tumor_rhd.append(current_tumor_rhd)

            row = [name, current_liver_dice, current_liver_nsd, current_liver_sd, current_liver_rhd, current_tumor_dice,
                   current_tumor_nsd, current_tumor_sd, current_tumor_rhd]
            rows.append(row)

            print(name, val_outputs[0].shape,
                  'dice: [{:.3f}  {:.3f}]; nsd: [{:.3f}  {:.3f}]'.format(current_liver_dice, current_tumor_dice,
                                                                         current_liver_nsd, current_tumor_nsd),
                  'sd: [{:.3f}  {:.3f}]; rhd: [{:.3f}  {:.3f}]'.format(current_liver_sd, current_tumor_sd,
                                                                       current_liver_rhd, current_tumor_rhd),
                  'time {:.2f}s'.format(time.time() - start_time))

            if args.analyze_tumor_size:
                size_stats = analyze_tumor_by_size(val_outputs[2, ...], val_labels[2, ...], spacing_mm)
                for size_bin in size_stats:
                    for metric in size_stats[size_bin]:
                        tumor_size_stats[size_bin][metric] += size_stats[size_bin][metric]
                
                print(f"Tumor size stats for {name}:")
                for size_bin in size_stats:
                    tp = size_stats[size_bin]['tp']
                    fn = size_stats[size_bin]['fn']
                    fp = size_stats[size_bin]['fp']
                    print(f"  {size_bin}: TP={tp}, FN={fn}, FP={fp}")

            # save the prediction
            output_dir = os.path.join(args.save_dir, args.model_name, str(args.val_overlap), 'pred')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            val_outputs = np.argmax(val_outputs, axis=0)

            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                os.path.join(output_dir, f'{name}.nii.gz')
            )

        print("organ dice:", np.mean(liver_dice))
        print("organ nsd:", np.mean(liver_nsd))
        print("organ sd:", np.mean(liver_sd))
        print("organ rhd:", np.mean(liver_rhd))
        print("tumor dice:", np.mean(tumor_dice))
        print("tumor nsd", np.mean(tumor_nsd))
        print("tumor sd:", np.mean(tumor_sd))
        print("tumor rhd:", np.mean(tumor_rhd))

        print("organ dice:", to_percentage(np.mean(liver_dice)))
        print("organ nsd:", to_percentage(np.mean(liver_nsd)))
        print("organ sd:", to_decimal(np.mean(liver_sd)))
        print("organ rhd:", to_decimal(np.mean(liver_rhd)))
        print("tumor dice:", to_percentage(np.mean(tumor_dice)))
        print("tumor nsd:", to_percentage(np.mean(tumor_nsd)))
        print("tumor sd:", to_decimal(np.mean(tumor_sd)))
        print("tumor rhd:", to_decimal(np.mean(tumor_rhd)))

        results = [
                ["organ dice", np.mean(liver_dice)],
                ["organ nsd", np.mean(liver_nsd)],
                ["organ sd", np.mean(liver_sd)],
                ["organ rhd", np.mean(liver_rhd)],
                ["tumor dice", np.mean(tumor_dice)],
                ["tumor nsd", np.mean(tumor_nsd)],
                ["tumor sd", np.mean(tumor_sd)],
                ["tumor rhd", np.mean(tumor_rhd)]
            ]
            
        if args.analyze_tumor_size:
            print("\n=== Tumor Size Analysis ===")
            for size_bin in tumor_size_stats:
                tp = tumor_size_stats[size_bin]['tp']
                fn = tumor_size_stats[size_bin]['fn']
                fp = tumor_size_stats[size_bin]['fp']
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * sensitivity * precision / (sensitivity + precision) if (sensitivity + precision) > 0 else 0
                
                print(f"\nTumor size {size_bin}:")
                print(f"  Total: TP={tp}, FN={fn}, FP={fp}")
                print(f"  Sensitivity: {to_percentage(sensitivity)}%")
                print(f"  Precision: {to_percentage(precision)}%")
                print(f"  F1 Score: {to_decimal(f1)}")
                
                results.append([f"tumor_{size_bin}_tp", tp])
                results.append([f"tumor_{size_bin}_fn", fn])
                results.append([f"tumor_{size_bin}_fp", fp])
                results.append([f"tumor_{size_bin}_sensitivity", sensitivity])
                results.append([f"tumor_{size_bin}_precision", precision])
                results.append([f"tumor_{size_bin}_f1", f1])
                
            header.extend([
                'tumor_0-5mm_tp', 'tumor_0-5mm_fn', 'tumor_0-5mm_fp', 
                'tumor_0-5mm_sensitivity', 'tumor_0-5mm_precision', 'tumor_0-5mm_f1',
                'tumor_5-10mm_tp', 'tumor_5-10mm_fn', 'tumor_5-10mm_fp', 
                'tumor_5-10mm_sensitivity', 'tumor_5-10mm_precision', 'tumor_5-10mm_f1',
                'tumor_>10mm_tp', 'tumor_>10mm_fn', 'tumor_>10mm_fp', 
                'tumor_>10mm_sensitivity', 'tumor_>10mm_precision', 'tumor_>10mm_f1'
            ])

        # save metrics to cvs file
        csv_save = os.path.join(args.save_dir, args.model_name, str(args.val_overlap))
        if not os.path.exists(csv_save):
            os.makedirs(csv_save)
        csv_name = os.path.join(csv_save, 'metrics.csv')
        with open(csv_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            writer.writerows(results)


# save path: save_dir/log_dir_name/str(args.val_overlap)/pred/
if __name__ == "__main__":
    main()