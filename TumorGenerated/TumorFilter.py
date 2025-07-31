import random
from typing import Hashable, Mapping, Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import RandomizableTransform, MapTransform
from scipy.ndimage import label, binary_dilation


class TumorFilter(RandomizableTransform, MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 prob: float = 0.8,
                 rank: int = 0,
                 filter=None,
                 filter_inferer=None,
                 use_inferer=True,
                 threshold=0.5,
                 allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        random.seed(0)
        np.random.seed(0)
        self.rank = rank
        self.filter = filter
        self.filter_inferer = filter_inferer
        self.use_inferer = use_inferer and (filter_inferer is not None)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        image = d['image']
        label = d['label']


        image_tensor = torch.from_numpy(image).unsqueeze(0).cuda(self.rank, non_blocking=True)
        label_tensor = torch.from_numpy(label).unsqueeze(0).cuda(self.rank, non_blocking=True)

        # Process single sample
        filtered_image, filtered_label = filter_synthetic_tumor(
            image_tensor, label_tensor,
            self.filter, self.filter_inferer, self.use_inferer, self.threshold
        )

        if filtered_image is None or filtered_label is None:
            d['valid'] = False
            return d

        filtered_image = filtered_image.cpu().squeeze(0).numpy()
        filtered_label = filtered_label.cpu().squeeze(0).numpy()

        d['image'] = filtered_image
        d['label'] = filtered_label
        return d


def denoise_pred(pred: np.ndarray):
    """
    # 0: background, 1: liver, 2: tumor.
    Input shape: (C, H, W, D)
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

    liver_dilation = binary_dilation(denoise_pred[1, ...], iterations=30).astype(bool)
    denoise_pred[2, ...] = pred[2, ...].astype(bool) * liver_dilation

    denoise_pred[0, ...] = 1 - np.logical_or(denoise_pred[1, ...], denoise_pred[2, ...])
    return denoise_pred


def calculate_quality_proportion(segmentation_output, tumor_mask):
    tumor_mask = (tumor_mask == 2).float()
    tumor_voxels = tumor_mask.sum().item()
    if tumor_voxels == 0:
        return 0
    seg_tumor_prob = segmentation_output[0][2, ...]
    matched_voxels = (seg_tumor_prob * tumor_mask.squeeze(0).squeeze(0)).sum().item()
    return matched_voxels / tumor_voxels


def filter_synthetic_tumor(data, target, model, model_inferer, use_inferer, threshold=0.5):
    # Input data: (1, 1, H, W, D), target: (1, 1, H, W, D)
    with torch.no_grad():
        output = model_inferer(data) if use_inferer else model(data)
        output_np = output.cpu().detach().numpy()

        # Process each sample in the batch (though here batch size should be 1)
        denoised_outputs = []
        for i in range(output_np.shape[0]):
            single_pred = output_np[i]  # (3, H, W, D)
            denoised = denoise_pred(single_pred)
            denoised_outputs.append(denoised)
        denoised_output = torch.tensor(np.stack(denoised_outputs)).cpu()

        quality = calculate_quality_proportion(denoised_output, target.cpu())
        if quality >= threshold:
            return data.cpu(), target.cpu()
        else:
            return None, None