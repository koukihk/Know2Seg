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
            d["tumor_texture_layer"] = np.zeros_like(d["image"])
        
        # Ensure tumor_mask_layer exists and has the same shape as the label
        if "tumor_mask_layer" not in d:
            d["tumor_mask_layer"] = np.zeros_like(d["label"])

        return d