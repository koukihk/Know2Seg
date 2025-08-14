import numpy as np
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform


class AddValidKeyd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        d['valid'] = True
        return d

class AddMissingKeysd(MapTransform):
    """
    Adds specified keys to the data dictionary if they are missing, filling them with zero tensors.
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        # Ensure tumor_texture_layer exists and has the same shape as the image
        if "tumor_texture_layer" not in d:
            image_shape = d["image"].shape
            # Create with same spatial dimensions but single channel
            texture_shape = (1,) + image_shape[1:] if len(image_shape) > 3 else image_shape
            d["tumor_texture_layer"] = np.zeros(texture_shape, dtype=d["image"].dtype)
        
        # Ensure tumor_mask_layer exists and has the same shape as the label
        if "tumor_mask_layer" not in d:
            label_shape = d["label"].shape
            # Create with same spatial dimensions but single channel  
            mask_shape = (1,) + label_shape[1:] if len(label_shape) > 3 else label_shape
            d["tumor_mask_layer"] = np.zeros(mask_shape, dtype=d["label"].dtype)

        # Ensure alpha exists and has the same shape as the label (soft mask)
        if "alpha" not in d:
            label_shape = d["label"].shape
            # Create with same spatial dimensions but single channel
            alpha_shape = (1,) + label_shape[1:] if len(label_shape) > 3 else label_shape
            d["alpha"] = np.zeros(alpha_shape, dtype=d["label"].dtype)

        return d