import shutil
import sys

import math
import scipy.ndimage as ndimage
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler  # native AMP
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../pipextra/lib/python3.6/site-packages')  # add missing packages

# template copied from torch.utils.data.distributed.DistributedSampler
class AMDistributedSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.shuffle = shuffle
        self.make_even = make_even

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # to track of smaller batches
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        # deterministically shuffle based on epoch

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible (otherwise will return last batch smaller)

        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(
                        indices))  # this ensures we get valid ids (if dataset is much smaller then world_size
                    indices += [indices[ids] for ids in extra_ids]

            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def distributed_all_gather(tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False,
                           is_valid=None):
    if world_size is None:
        world_size = torch.distributed.get_world_size()

    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)  # it can't be more then world_size
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)

    if not no_barrier:
        torch.distributed.barrier()  # synch processess, do we need it??

    tensor_list_out = []
    with torch.no_grad():  # ? do we need it

        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]  # list of bools
            # print('is_valid list', is_valid)

        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)

            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]  # keep only valid elements
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
                # print('updated gather list', gather_list)

            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]  # convert to numpy

            tensor_list_out.append(gather_list)

    return tensor_list_out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n

        self.count += n
        # self.avg = self.sum / self.count if self.count > 0 else self.sum
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import monai
from torch.distributions import Beta
from torch.cuda.amp import autocast

import os
from mixup import mixup

class MixDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.data_list = list(loader)

    def get_random_batch(self):
        return random.choice(self.data_list)


def tumor_weighted_mixup_3d(data, target, mixup_loader, alpha=1.0, mixup_prob=0.5, num_classes=3):
    # Convert target to soft labels regardless of mixup application
    target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    # print("target_one_hot.shape", target_one_hot.shape)
    if random.random() > mixup_prob or alpha <= 0:
        return data, target_one_hot  # Return soft labels even if no mixup

    tumor_mask = (target.squeeze(1) == 2).float().sum() > 0
    if not tumor_mask:
        return data, target_one_hot

    beta_dist = Beta(alpha, alpha)
    lam = beta_dist.sample().item()
    random_batch = mixup_loader.get_random_batch()
    other_data = random_batch["image"].to(data.device)
    other_target = random_batch["label"].to(target.device)
    mixed_data = lam * data + (1 - lam) * other_data

    other_target_one_hot = F.one_hot(other_target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2,
                                                                                                      3).float()
    mixed_target = lam * target_one_hot + (1 - lam) * other_target_one_hot
    return mixed_data, mixed_target


def tumor_aware_cutmix_3d(data, target, mixup_loader, beta=1.0, cutmix_prob=0.5, num_classes=3):
    # Convert target to soft labels regardless of cutmix application
    target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    # print("target_one_hot.shape", target_one_hot.shape)
    if np.random.rand() > cutmix_prob:
        return data, target_one_hot  # Return soft labels even if no cutmix

    random_batch = mixup_loader.get_random_batch()
    other_data = random_batch["image"].to(data.device)
    other_target = random_batch["label"].to(target.device)
    lam = np.random.beta(beta, beta) if beta > 0 else 1
    D, H, W = data.shape[2:]
    cut_d = int(D * (lam ** (1 / 3)))
    cut_h = int(H * (lam ** (1 / 3)))
    cut_w = int(W * (lam ** (1 / 3)))
    tumor_mask = (target.squeeze(1) == 2).float()
    tumor_coords = torch.nonzero(tumor_mask)

    if tumor_coords.size(0) == 0:
        d1 = np.random.randint(0, D - cut_d + 1)
        h1 = np.random.randint(0, H - cut_h + 1)
        w1 = np.random.randint(0, W - cut_w + 1)
    else:
        center = tumor_coords.float().mean(dim=0).int()
        d1 = max(0, min(center[0] - cut_d // 2, D - cut_d))
        h1 = max(0, min(center[1] - cut_h // 2, H - cut_h))
        w1 = max(0, min(center[2] - cut_w // 2, W - cut_w))

    mixed_data = data.clone()
    mixed_data[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w] = \
        other_data[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w]

    other_target_one_hot = F.one_hot(other_target.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2,
                                                                                                      3).float()
    mixed_target = target_one_hot.clone()
    lam_region = (cut_d * cut_h * cut_w) / (D * H * W)
    # mixed_target[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w] = \
    #     (1 - lam_region) * target_one_hot[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w] + \
    #     lam_region * other_target_one_hot[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w]
    mixed_target[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w] = \
        other_target_one_hot[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w]
    return mixed_data, mixed_target


class LayerDecompositionLoss(torch.nn.Module):
    def __init__(self, lambda_recon_normal=1.0, lambda_recon_tumor=1.0, lambda_seg=2.0,
                 lambda_blend=1.0, use_l3_blend=True, stage_ii_mode=False,
                 confidence_threshold=0.9, ablation_mode=None):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.dice_ce_loss = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0,
                                                    smooth_dr=1e-6)

        self.lambda_recon_normal = lambda_recon_normal
        self.lambda_recon_tumor = lambda_recon_tumor
        self.lambda_seg = lambda_seg
        self.lambda_blend = lambda_blend
        self.use_l3_blend = use_l3_blend
        self.stage_ii_mode = stage_ii_mode
        self.confidence_threshold = confidence_threshold
        self.ablation_mode = ablation_mode

    def forward(self, outputs, image, label, tumor_texture_layer, tumor_mask_layer, alpha=None):
        reconstructed_normal_liver = torch.clamp(outputs[:, 0:1, :, :, :], 0.0, 1.0)
        reconstructed_tumor = torch.clamp(outputs[:, 1:2, :, :, :], 0.0, 1.0)
        predicted_tumor_mask = outputs[:, 2:5, :, :, :]
        target_tumor_normalized = (tumor_texture_layer - (-21)) / (189 - (-21))
        target_tumor_normalized = torch.clamp(target_tumor_normalized, 0.0, 1.0)

        loss_recon_normal = self.l1_loss(reconstructed_normal_liver * (1 - tumor_mask_layer),
                                         image * (1 - tumor_mask_layer))
        loss_recon_tumor = self.l1_loss(reconstructed_tumor, target_tumor_normalized)
        loss_seg = self.dice_ce_loss(predicted_tumor_mask, label)
        loss_blend = torch.tensor(0.0, device=image.device)
        if self.use_l3_blend:
            pred_prob = torch.softmax(predicted_tumor_mask, dim=1)
            pred_tumor_prob = pred_prob[:, 2:3, :, :, :]
            blended_recon = (1 - pred_tumor_prob) * reconstructed_normal_liver + \
                            pred_tumor_prob * reconstructed_tumor
            loss_blend = self.l1_loss(blended_recon, image)

        if self.stage_ii_mode:
            pred_probs = torch.softmax(predicted_tumor_mask, dim=1)
            max_probs, pseudo_labels = torch.max(pred_probs, dim=1)

            high_conf_mask = max_probs > self.confidence_threshold
            if high_conf_mask.sum() > 0:
                loss_seg = 0.5 * loss_seg + 0.5 * self.dice_ce_loss(predicted_tumor_mask, pseudo_labels.unsqueeze(1))

        if self.ablation_mode == 'no_l0':
            loss_recon_normal = loss_recon_normal * 0
        elif self.ablation_mode == 'no_l1':
            loss_recon_tumor = loss_recon_tumor * 0
        elif self.ablation_mode == 'no_l3':
            loss_blend = loss_blend * 0
        elif self.ablation_mode == 'only_l2':
            loss_recon_normal = loss_recon_normal * 0
            loss_recon_tumor = loss_recon_tumor * 0
            loss_blend = loss_blend * 0

        total_loss = (self.lambda_recon_normal * loss_recon_normal +
                      self.lambda_recon_tumor * loss_recon_tumor +
                      self.lambda_seg * loss_seg +
                      self.lambda_blend * loss_blend)

        return total_loss, loss_recon_normal, loss_recon_tumor, loss_seg, loss_blend

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_loss_recon_normal = AverageMeter()
    run_loss_recon_tumor = AverageMeter()
    run_loss_seg = AverageMeter()
    run_loss_blend = AverageMeter()

    for idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda(args.rank)
        target = batch_data['label'].cuda(args.rank)

        tumor_texture_layer = batch_data['tumor_texture_layer'].cuda(args.rank)
        tumor_mask_layer = batch_data['tumor_mask_layer'].cuda(args.rank)
        alpha = batch_data['alpha'].cuda(args.rank)  # Use alpha from data batch

        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            logits = model(data)
            loss, loss_recon_normal, loss_recon_tumor, loss_seg, loss_blend = loss_func(
                logits, data, target, tumor_texture_layer, tumor_mask_layer, alpha
            )

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list, loss_recon_normal_list, loss_recon_tumor_list, loss_seg_list, loss_blend_list = distributed_all_gather(
                [loss, loss_recon_normal, loss_recon_tumor, loss_seg, loss_blend], 
                out_numpy=True, 
                is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(np.mean(np.stack(loss_list, axis=0)), n=args.batch_size * args.world_size)
            run_loss_recon_normal.update(np.mean(np.stack(loss_recon_normal_list, axis=0)), n=args.batch_size * args.world_size)
            run_loss_recon_tumor.update(np.mean(np.stack(loss_recon_tumor_list, axis=0)), n=args.batch_size * args.world_size)
            run_loss_seg.update(np.mean(np.stack(loss_seg_list, axis=0)), n=args.batch_size * args.world_size)
            run_loss_blend.update(np.mean(np.stack(loss_blend_list, axis=0)), n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_loss_recon_normal.update(loss_recon_normal.item(), n=args.batch_size)
            run_loss_recon_tumor.update(loss_recon_tumor.item(), n=args.batch_size)
            run_loss_seg.update(loss_seg.item(), n=args.batch_size)
            run_loss_blend.update(loss_blend.item(), n=args.batch_size)

        if args.rank == 0 and idx % 10 == 0:
            print(
                f"Epoch {epoch}/{args.max_epochs} {idx}/{len(loader)} "
                f"loss: {run_loss.avg:.4f} "
                f"(recon_nl: {run_loss_recon_normal.avg:.4f}, recon_t: {run_loss_recon_tumor.avg:.4f}, seg: {run_loss_seg.avg:.4f}, blend: {run_loss_blend.avg:.4f}) "
                f"time {time.time() - start_time:.2f}s"
            )
        start_time = time.time()

    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def train_epoch_defaut(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    mixup_loader = MixDataLoader(loader)

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]

        data = data.cuda(args.rank, non_blocking=True)
        target = target.cuda(args.rank, non_blocking=True)

        if args.simple_mixup: 
            mixed_input = mixup([data, target])
            data = mixed_input[0]
            target = mixed_input[1]
        elif args.mixup:
            data, target = tumor_weighted_mixup_3d(data, target, mixup_loader, alpha=args.mixup_alpha,
                                                  mixup_prob=args.mixup_prob, num_classes=args.num_classes)
        elif args.cutmix:
            data, target = tumor_aware_cutmix_3d(data, target, mixup_loader, beta=args.cutmix_beta,
                                                 cutmix_prob=args.cutmix_prob, num_classes=args.num_classes)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather(
                [loss],
                out_numpy=True,
                is_valid=idx < loader.sampler.valid_length
            )
            loss_val = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)
            run_loss.update(loss_val, n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        if args.rank == 0 and idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{args.max_epochs} {idx}/{len(loader)} "
                f"loss: {run_loss.avg:.4f} time {elapsed:.2f}s"
            )
        start_time = time.time()

    return run_loss.avg

def train_epoch_validity(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
            valid = torch.ones(data.size(0), dtype=torch.bool)  
        else:
            data, target = batch_data["image"], batch_data["label"]
            valid = batch_data.get("valid", torch.ones(data.size(0), dtype=torch.bool)).bool()

        valid_indices = torch.where(valid)[0]
        if len(valid_indices) == 0:
            continue
        data = data[valid_indices]
        target = target[valid_indices]
        effective_bs = len(valid_indices)

        data = data.cuda(args.rank, non_blocking=True)
        target = target.cuda(args.rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target) if effective_bs > 0 else 0

        if effective_bs > 0:
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        if args.distributed:
            effective_bs_tensor = torch.tensor(effective_bs, dtype=torch.float32, device=data.device)
            loss_sum = loss.detach() * effective_bs
            loss_sum_tensor = torch.tensor(loss_sum.item(), dtype=torch.float32, device=data.device)

            dist.all_reduce(effective_bs_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)

            total_effective = effective_bs_tensor.item()
            total_loss_sum = loss_sum_tensor.item()

            if total_effective > 0:
                avg_loss = total_loss_sum / total_effective
                run_loss.update(avg_loss, n=total_effective)
        else:
            if effective_bs > 0:
                run_loss.update(loss.item(), n=effective_bs)

        if args.rank == 0 and idx % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{args.max_epochs} {idx}/{len(loader)} "
                f"loss: {run_loss.avg:.4f} | eff_bs: {effective_bs} "
                f"time {elapsed:.2f}s"
            )
        start_time = time.time()

    return run_loss.avg


def resample(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def val_epoch(model, loader, val_shape_dict, epoch, loss_func, args, model_inferer=None):
    model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data['image'].cuda(args.rank)
            target = batch_data['label'].cuda(args.rank)
            tumor_texture_layer = batch_data['tumor_texture_layer'].cuda(args.rank)
            tumor_mask_layer = batch_data['tumor_mask_layer'].cuda(args.rank)
            alpha = batch_data['alpha'].cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    torch.cuda.empty_cache()
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            if args.layer_decomposition:
                loss, _, _, _, _ = loss_func(logits, data, target, tumor_texture_layer, tumor_mask_layer, alpha)
                # In layer decomposition mode, segmentation is on channels 2:5
                predicted_tumor_mask = logits[:, 2:5, :, :, :]
            else:
                loss = loss_func(logits, target)
                # In standard mode, segmentation is on all output channels
                predicted_tumor_mask = logits

            # Segmentation evaluation
            predicted_tumor_mask = torch.softmax(predicted_tumor_mask, 1).cpu().numpy()
            predicted_tumor_mask = np.argmax(predicted_tumor_mask, axis=1).astype(np.uint8)
            target = target.cpu().numpy()[:, 0, :, :, :]

            name = batch_data["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
            val_shape = val_shape_dict[name]

            pred = resample(predicted_tumor_mask[0], val_shape)
            y = resample(target[0], val_shape)

            dice_list_sub = []
            for i in range(1, args.num_classes):
                organ_Dice = dice(pred == i, y == i)
                dice_list_sub.append(organ_Dice)

            if args.distributed:
                torch.distributed.barrier()
                gather_list_sub = [[0] * len(dice_list_sub) for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(gather_list_sub, dice_list_sub)

                classes_metriclist = []
                for i in range(args.num_classes - 1):
                    class_metric = [s[i] for s in gather_list_sub]
                    classes_metriclist.append(class_metric)
                avg_classes = np.mean(classes_metriclist, 1)
                loss = loss.cuda(args.rank)
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0)), n=args.batch_size * args.world_size)
                run_acc.update(avg_classes, n=1)
            else:
                avg_classes = np.array(dice_list_sub)
                run_acc.update(avg_classes, n=args.batch_size)
                run_loss.update(loss.item(), n=args.batch_size)

            if args.rank == 0:
                print('Batch mean: Liver: {}, Tumor: {}, all:{}'.format(avg_classes[0], avg_classes[1], np.mean(avg_classes)))
                print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      'loss: {:.4f}'.format(run_loss.avg),
                      'acc', run_acc.avg,
                      'acc_avg: {:.4f}'.format(np.mean(run_acc.avg)),
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()

    return run_loss.avg, run_acc.avg


def save_checkpoint(model, epoch, args, filename='model.pt', best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()

    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 args,
                 val_shape_dict,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 val_channel_names=None
                 ):
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', writer.log_dir)

    scaler = None
    if args.amp:  # new native amp
        scaler = GradScaler()

    val_acc_max = 0.

    for epoch in range(start_epoch, args.max_epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(args.rank, time.ctime(), 'Epoch:', epoch)

        epoch_time = time.time()
        if args.layer_decomposition:
            train_loss = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func,
                                     args=args)
        else:
            # Keep original behavior when layer decomposition is off
            train_loss = train_epoch_defaut(model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func,
                                     args=args)

        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))

        if args.rank == 0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)

        b_new_best = False
        val_acc = 0
        if ((epoch + 1) % args.val_every == 0):

            if args.distributed:
                torch.distributed.barrier()  # sync processes

            epoch_time = time.time()
            # torch.cuda.empty_cache()
            val_loss, val_acc = val_epoch(model, val_loader, val_shape_dict, epoch=epoch, loss_func=loss_func,
                                          model_inferer=model_inferer, args=args)
            if args.rank == 0:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(val_loss),
                      'acc', val_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
                with open("test_online.txt", 'a') as f:
                    print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(val_loss),
                          'acc', val_acc, 'time {:.2f}s'.format(time.time() - epoch_time), file=f)
                if writer is not None:
                    writer.add_scalar('val_loss', val_loss, epoch)
                    writer.add_scalar('val_mean_dice', np.mean(val_acc), epoch)
                    if val_channel_names is not None:
                        for val_channel_ind in range(len(val_channel_names)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(val_channel_names[val_channel_ind], val_acc[val_channel_ind], epoch)

                if np.mean(val_acc) > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, np.mean(val_acc)))
                    val_acc_max = np.mean(val_acc)
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer,
                                        scheduler=scheduler)

        if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            save_checkpoint(model, epoch, args, best_acc=np.mean(val_acc), filename='model_final.pt')
            if b_new_best:
                print('Copying to model.pt new best model!!!!')
                shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        if scheduler is not None:
            scheduler.step()

    print('monai_trainer DONE, best  acc', val_acc_max)

    return val_acc_max