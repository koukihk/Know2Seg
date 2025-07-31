from monai.transforms import MapTransform


class AddValidKeyd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        d['valid'] = True
        return d