import logging
import random
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import (SynthesisTumor, get_predefined_texture, get_predefined_texture_b)


class TumorGenerated(RandomizableTransform, MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 prob: float = 0.1,
                 tumor_prob=[0.2, 0.2, 0.2, 0.2, 0.2],
                 ellipsoid_model=None,
                 allow_missing_keys: bool = False,
                 use_enhanced_method: bool = False,
                 hparam_overrides: Optional[Dict[str, Any]] = None,
                 ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.ellipsoid_model = ellipsoid_model
        self.edge_advanced_blur = True
        self.use_enhanced_method = use_enhanced_method
        self.hparams = dict(hparam_overrides or {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self.textures_root = Path(self.hparams.get("texture_log_root", "."))
        self.tumor_types = ['tiny', 'small', 'medium', 'large', 'mix']

        assert len(tumor_prob) == 5
        self.tumor_prob = np.array(tumor_prob)
        # texture shape: 420, 300, 320
        # self.textures = pre_define 10 texture
        self.textures = []
        sigma_as = [3, 6, 9, 12, 15]
        sigma_bs = [4, 7]
        predefined_texture_shape = (420, 300, 320)
        for sigma_a in sigma_as:
            for sigma_b in sigma_bs:
                texture = get_predefined_texture_b(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)
        print("All predefined texture have generated.")

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = random.choice(self.textures)
            image_metadata = d.get("image_meta_dict") or {}
            image_path = Path(str(image_metadata.get("filename_or_obj", "unknown")))
            image_id = f"{image_path.parent.name}/{image_path.name}" if image_path.parent.name else image_path.name
            if not self.logger.handlers:
                logging.basicConfig(level=logging.INFO)
            image, label, tumor_texture_layer, tumor_mask_layer, alpha = SynthesisTumor(
                d['image'][0], d['label'][0], tumor_type, texture,
                self.edge_advanced_blur, self.ellipsoid_model,
                self.use_enhanced_method,
                hyperparams=self.hparams
            )
            d['image'][0] = image
            d['label'][0] = label
            d['tumor_texture_layer'] = tumor_texture_layer[np.newaxis, ...]
            d['tumor_mask_layer'] = tumor_mask_layer[np.newaxis, ...]
            d['alpha'] = alpha[np.newaxis, ...]

        return d
