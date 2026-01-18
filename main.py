import os
import warnings
from functools import partial

import nibabel as nb
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from monai import transforms, data
from monai.data import load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
# from monai.transforms import AsDiscrete

from monai_trainer import AMDistributedSampler, run_training, LayerDecompositionLoss
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tumor_analyzer import EllipsoidFitter

warnings.filterwarnings("ignore")

## Online Tumor Generation
from TumorGenerated import TumorGenerated, TumorFilter, AddValidKeyd, AddMissingKeysd
from tumor_saver import TumorSaver

import argparse

class SaveSyntheticallyGeneratedData(transforms.Transform):
    """
    Custom transform to save synthetically generated data using TumorSaver.
    """
    def __init__(self, folder='generated_data'):
        self.folder = folder

    def __call__(self, data):
        d = dict(data)
        TumorSaver.save_data(d, folder=self.folder)
        return data

parser = argparse.ArgumentParser(description='brats21 segmentation testing')

parser.add_argument('--syn', action='store_true')  # use synthetic tumors for training
parser.add_argument('--filter', action='store_true', help='Enable tumor quality filtering')
parser.add_argument('--filter_dir', default='runs/standard_all.unet', type=str)
parser.add_argument('--filter_name', default='unet', type=str)
parser.add_argument('--cutmix', action='store_true', help='Enable cutmix augmentation')
parser.add_argument('--cutmix_beta', type=float, default=1.0, help='Beta parameter for cutmix')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability of applying cutmix')
parser.add_argument('--simple_mixup', action='store_true', help='Enable simple mixup augmentation')
parser.add_argument('--mixup', action='store_true', help='Enable mixup augmentation')
parser.add_argument('--mixup_alpha', type=float, default=1.0, help='Alpha parameter for mixup')
parser.add_argument('--mixup_prob', type=float, default=0.5, help='Probability of applying mixup')
parser.add_argument('--ellipsoid', action='store_true')
parser.add_argument('--save_syn_data', action='store_true', help='Save synthetically generated data.')
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--logdir', default=None)
parser.add_argument('--save_checkpoint', action='store_true')
parser.add_argument('--max_epochs', default=5000, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optim_lr', default=1e-4, type=float)

parser.add_argument('--optim_name', default='adamw', type=str)

parser.add_argument('--reg_weight', default=1e-5, type=float)
parser.add_argument('--task', default='brats18')

parser.add_argument('--quick', action='store_true')  # distributed multi gpu
parser.add_argument('--noamp', action='store_true')  # experimental
parser.add_argument('--val_every', default=1, type=int)
parser.add_argument('--dropout_prob', default=0, type=float)
parser.add_argument('--val_overlap', default=0.5, type=float)

parser.add_argument('--distributed', action='store_true')  # distributed multi gpu
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=4, type=int)

parser.add_argument('--model_name', default='unet', type=str)
parser.add_argument('--swin_type', default='tiny', type=str)

# segmentation flex params
parser.add_argument('--seg_block', default='basic_pre', type=str)
parser.add_argument('--seg_num_blocks', default='1,2,2,4', type=str)
parser.add_argument('--seg_base_filters', default=16, type=int)
parser.add_argument('--seg_relu', default='relu', type=str)
parser.add_argument('--seg_lastnorm_init_zero', action='store_true')

parser.add_argument('--seg_mode', default=1, type=int)

parser.add_argument('--seg_use_se', action='store_true')
parser.add_argument('--seg_norm_name', default='instancenorm', type=str)
parser.add_argument('--seg_noskip', action='store_true')
parser.add_argument('--seg_aug_mode', default=0, type=int)
parser.add_argument('--seg_aug_noflip', action='store_true')

parser.add_argument('--seg_norm_mode', default=0, type=int)
parser.add_argument('--seg_crop_mode', default=0, type=int)

parser.add_argument('--optuna', action='store_true')
parser.add_argument('--optuna_study_name', default='optuna_study', type=str)
parser.add_argument('--optuna_sampler', default=None, type=str)
parser.add_argument('--optuna_allfolds', action='store_true')

# unetr params
parser.add_argument('--pos_embedd', default='conv', type=str)
parser.add_argument('--norm_name', default='instance', type=str)
parser.add_argument('--num_steps', default=40000, type=int)
parser.add_argument('--eval_num', default=500, type=int)
parser.add_argument('--warmup_steps', default=500, type=int)
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--mlp_dim', default=3072, type=int)
parser.add_argument('--hidden_size', default=768, type=int)
# parser.add_argument('--feature_size', default=12, type=int)
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--out_channels', default=5, type=int)
parser.add_argument('--num_classes', default=3, type=int)
parser.add_argument('--res_block', action='store_true')
parser.add_argument('--conv_block', action='store_true')
parser.add_argument('--roi_x', default=96, type=int)
parser.add_argument('--roi_y', default=96, type=int)
parser.add_argument('--roi_z', default=96, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--decay', default=1e-5, type=float)
parser.add_argument('--lrdecay', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--amp_scale', action='store_true')
parser.add_argument('--opt_level', default='O2', type=str)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lrschedule', default='warmup_cosine', type=str)
parser.add_argument('--randaugment_n', default=0, type=int)
parser.add_argument('--warmup_epochs', default=100, type=int)
parser.add_argument('--resume_ckpt', action='store_true')
parser.add_argument('--pretrained_dir', default=None, type=str)

parser.add_argument('--dataset_flag', default='d', type=str)
parser.add_argument('--train_dir', default=None, type=str)
parser.add_argument('--val_dir', default=None, type=str)
parser.add_argument('--json_dir', default=None, type=str)
parser.add_argument('--extra_train_dir', default=None, type=str)
parser.add_argument('--extra_json', default=None, type=str)
parser.add_argument('--cache_num', default=500, type=int)

parser.add_argument('--use_pretrained', action='store_true')
parser.add_argument('--layer_decomposition', action='store_true', help='Enable layer decomposition training (5 output channels)')
parser.add_argument('--ablation_mode', default=None, type=str, help='Ablation mode: no_l0, no_l1, no_l2, no_l3, only_l2, no_recon')


def optuna_objective(trial, args):
    if args.optuna_study_name == 'feta21_randaugment':
        args.seg_aug_mode = 5
        args.randaugment_n = trial.suggest_categorical("randaugment_n", [1, 2, 3, 4, 5, 6])
        args.randaugment_p = trial.suggest_categorical("randaugment_p", [0.1, 0.3, 0.5, 0.7, 0.9])

    else:
        args.seg_block = trial.suggest_categorical("seg_block", ["basic_pre", "basic"])
        args.seg_norm_name = trial.suggest_categorical("seg_norm_name", ["groupnorm", "instancenorm"])
        args.seg_relu = trial.suggest_categorical("seg_relu", ["relu", "leaky_relu"])
        args.seg_use_se = trial.suggest_categorical("seg_use_se", [True, False])
        args.reg_weight = trial.suggest_categorical("reg_weight", [0, 1e-5])

    # create the formatted name of log directory
    if args.logdir_init is not None:
        sall = []
        for s in trial.params.values():
            if isinstance(s, float):
                sall.append('{:.1e}'.format(s) if s < 0.001 else "{:.3f}".format(s))
            else:
                sall.append(str(s))
        args.logdir = args.logdir_init + '/' + str(trial.number) + '_' + '_'.join(sall)  # unique logdir name
        trial.set_user_attr('logdir', args.logdir)

    print("Optuna updated argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    if not args.optuna_allfolds:
        accuracy = main_worker(gpu=0, args=args)
    else:
        accuracy = 0
        for i in range(5):
            print('Running fold', i)
            args.fold = i
            accuracy += main_worker(gpu=0, args=args)
        accuracy = accuracy / 5.0

    return accuracy


def optuna_run(args):
    import optuna
    from optuna.trial import TrialState

    args.logdir_init = args.logdir

    study_name = args.optuna_study_name
    sampler = None

    if args.optuna_sampler is not None:
        if args.optuna_sampler == 'tpe':
            sampler = optuna.samplers.TPESampler()
        elif args.optuna_sampler == 'random':
            sampler = optuna.samplers.RandomSampler()
        elif args.optuna_sampler == 'grid':

            if args.optuna_study_name == 'feta21_randaugment':

                search_space = {"randaugment_n": [1, 2, 3, 4, 5, 6],
                                "randaugment_p": [0.1, 0.3, 0.5, 0.7, 0.9]
                                }

            else:
                search_space = {"seg_block": ["basic_pre", "basic"],
                                "seg_norm_name": ["groupnorm", "instancenorm"],
                                "seg_relu": ["relu", "leaky_relu"],
                                "seg_use_se": [False, True],
                                "reg_weight": [0, 1e-5],
                                }
            sampler = optuna.samplers.GridSampler(search_space=search_space)

    print('Using optuna sampler', sampler, study_name)

    objective = partial(optuna_objective, args=args)
    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        sampler=sampler,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True)
    #
    callbacks = []

    study.optimize(objective, callbacks=callbacks, gc_after_trial=True)
    # study.optimize(objective, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def _get_transform(args, ellipsoid_model=None, filter_model=None, filter_inferer=None):
    if args.syn:
        train_transform_list = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                                mode=("bilinear", "nearest")),
            TumorGenerated(keys=["image", "label"], prob=0.98, ellipsoid_model=ellipsoid_model),
            AddMissingKeysd(keys=["tumor_texture_layer", "tumor_mask_layer", "alpha"]),
        ]
        if args.save_syn_data:
            train_transform_list.append(SaveSyntheticallyGeneratedData(folder='syn_run'))

        train_transform_list.extend([
            # AddValidKeyd(keys=["image", "label"]),
            # TumorFilter(keys=["image", "label"], prob=0.8, rank=args.rank, filter=filter_model, filter_inferer=filter_inferer,
            #              use_inferer=True, threshold=0.5),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-21, a_max=189,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            transforms.SpatialPadd(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"], mode=["minimum", "constant", "constant", "constant", "constant"],
                                   spatial_size=[96, 96, 96]),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"],
                label_key="label",

                spatial_size=(96, 96, 96),
                pos=2,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
            transforms.ToTensord(keys=["image", "label", "tumor_texture_layer", "tumor_mask_layer", "alpha"]),
        ])
        train_transform = transforms.Compose(train_transform_list)

    else:
        train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-21, a_max=189,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
            transforms.ToTensord(keys=["image", "label"]),
        ]
        )

    val_transform = transforms.Compose(
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

    return train_transform, val_transform


def main():
    args = parser.parse_args()
    # set dynamic output channels according to layer decomposition flag
    if args.layer_decomposition:
        args.out_channels = 5
    else:
        args.out_channels = 3
    args.amp = not args.noamp

    if args.randaugment_n > 0:
        args.seg_aug_mode = 5

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    if args.optuna:
        optuna_run(args)
    else:
        if args.distributed:
            args.ngpus_per_node = torch.cuda.device_count()
            print('Found total gpus', args.ngpus_per_node)

            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))

        else:
            main_worker(gpu=0, args=args)

def load_ellipsoid_model():
    center = [190, 140, 90]
    axes = [[-0.8, 0.6, 0.2], 
        [0.6, 0.8, -0.2], 
        [-0.3, 0.0, -1.0]]
    radii = [230, 130, 80]
    normalized_axes = np.zeros_like(axes)
    for i in range(3):
        normalized_axes[i] = axes[i] / np.linalg.norm(axes[i])
    ellipsoid_model = EllipsoidFitter.from_precomputed_parameters(center, normalized_axes, radii)
    return ellipsoid_model

def load_filter(args):
    inf_size = [96, 96, 96]
    if args.filter_name == 'swin_unetrv2':
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
    elif args.filter_name == 'unet':
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    else:
        raise ValueError('Unsupported model ' + str(args.filter_name))

    checkpoint = torch.load(os.path.join(args.filter_dir, 'model.pt'), map_location='cpu')

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace('backbone.', '')] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.cuda()
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=1,
        predictor=model,
        overlap=args.val_overlap,
        mode='gaussian'
    )
    return model, model_inferer

def main_worker(gpu, args):
    ellipsoid_model = None
    if args.ellipsoid:
        ellipsoid_model = load_ellipsoid_model()

    filter_model = None
    filter_inferer = None
    if args.filter:
        filter_model, filter_inferer = load_filter(args)

    if args.distributed:
        # in new Pytorch/python lambda functions fail to pickle with spawn
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    torch.cuda.set_device(args.gpu)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    roi_size = [args.roi_x, args.roi_y, args.roi_x]
    inf_size = [args.roi_x, args.roi_y, args.roi_x]

    data_dir = args.train_dir
    val_data_dir = args.val_dir

    datalist_json = args.json_dir

    train_transform, val_transform = _get_transform(args, ellipsoid_model, filter_model, filter_inferer)

    ## NETWORK
    if (args.model_name is None) or args.model_name == 'unet':
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    elif args.model_name == 'swin_unetrv2':

        if args.swin_type == 'tiny':
            feature_size = 12
        elif args.swin_type == 'small':
            feature_size = 24
        elif args.swin_type == 'base':
            feature_size = 48

        model = SwinUNETR_v2(in_channels=1,
                             out_channels=args.out_channels,
                             img_size=(96, 96, 96),
                             feature_size=feature_size,
                             patch_size=2,
                             depths=[2, 2, 2, 2],
                             num_heads=[3, 6, 12, 24],
                             window_size=[7, 7, 7])

        if args.use_pretrained:
            pretrained_add = 'model_swinvit.pt'
            model.load_from(weights=torch.load(pretrained_add))
            print('Use pretrained ViT weights from: {}'.format(pretrained_add))

    elif args.model_name == 'nnunet':
        from monai.networks.nets import DynUNet
        from dynunet_pipeline.create_network import get_kernels_strides
        from dynunet_pipeline.task_params import deep_supr_num
        task_id = 'custom'
        kernels, strides = get_kernels_strides(task_id)
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
        )

    else:
        raise ValueError('Unsupported model ' + str(args.model_name))

    if args.resume_ckpt:
        model_dict = torch.load(args.pretrained_dir)
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')

    # from monai.losses import DiceLoss
    # import torch.nn.functional as F
    # def soft_dice_ce_loss(pred, target, smooth=1e-6):
    #     pred_softmax = F.softmax(pred, dim=1)
    #     dice_loss = DiceLoss(to_onehot_y=False, softmax=False, squared_pred=True, smooth_nr=0, smooth_dr=smooth)
    #     dice = dice_loss(pred_softmax, target)
    #     pred_log_softmax = F.log_softmax(pred, dim=1)
    #     kl_div = F.kl_div(pred_log_softmax, target, reduction='batchmean')
    #     return dice + kl_div

    # dice_loss = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=1e-6)
    #dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    # dice_loss = soft_dice_ce_loss
    if args.layer_decomposition:
        dice_loss = LayerDecompositionLoss(
            lambda_recon_normal=0.3, 
            lambda_recon_tumor=0.5, 
            lambda_seg=2.0,
            lambda_blend=0.8,      # L3 blended reconstruction weight
            use_l3_blend=True,     # Enable L3 blend loss
            stage_ii_mode=False,   # Disable Stage II initially
            confidence_threshold=0.9,
            ablation_mode=args.ablation_mode  # Pass ablation mode
        )
    else:
        dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)

    # post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
    # post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)

    val_channel_names = ['val_liver_dice', 'val_tumor_dice']

    print('Crop size', roi_size)

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=val_data_dir)

    if args.extra_json and args.extra_train_dir:
        extra_datalist = load_decathlon_datalist(args.extra_json, True, "training", base_dir=args.extra_train_dir)
        datalist.extend(extra_datalist)

    # train_datalist = [data for data in datalist if data.get('valid', True)]
    train_datalist = datalist

    new_datalist = []
    for item in datalist:
        new_item = {}
        new_item['image'] = item['image'].replace('.npy', '')
        new_item['label'] = item['label'].replace('.npy', '')
        new_datalist.append(new_item)

    new_val_files = []
    for item in val_files:
        new_item = {}
        new_item['image'] = item['image'].replace('.npy', '.gz')
        new_item['label'] = item['label'].replace('.npy', '.gz')
        new_val_files.append(new_item)

    val_shape_dict = {}

    for d in new_val_files:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]
    print('Totoal number of validation: {}'.format(len(val_shape_dict)))

    print('train_files files', len(new_datalist), 'validation files', len(new_val_files))

    train_ds = data.SmartCacheDataset(
        data=train_datalist,
        transform=train_transform,
        cache_num=min(args.cache_num, len(new_datalist)),
        cache_rate=1.0,
        num_init_workers=max(4, args.workers//2),
        num_replace_workers=2,
        progress=False
    )

    train_sampler = AMDistributedSampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=8,
                                   sampler=train_sampler, pin_memory=True, persistent_workers=True)

    val_ds = data.Dataset(data=new_val_files, transform=val_transform)
    val_sampler = AMDistributedSampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, sampler=val_sampler,
                                 pin_memory=True)

    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,
                            overlap=args.val_overlap, mode='gaussian')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.', '')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)  # ??

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=False)

    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=0.99, nesterov=True,
                                    weight_decay=args.reg_weight)  # momentum 0.99, nestorov=True, following nnUnet
    else:
        raise ValueError('Unsupported optim_name' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )


    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)

    else:
        scheduler = None

    accuracy = run_training(model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_func=dice_loss,
                            args=args,
                            model_inferer=model_inferer,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                            val_channel_names=val_channel_names,
                            val_shape_dict=val_shape_dict)

    return accuracy


if __name__ == '__main__':
    main()
