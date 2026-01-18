import os
import nibabel as nib
import numpy as np


class TumorSaver:
    @staticmethod
    def get_datatype(datatype):
        datatype_map = {
            2: 'uint8',
            4: 'int16',
            8: 'int32',
            16: 'float32',
            32: 'complex64',
            64: 'float64'
        }
        if hasattr(datatype, 'item'):
            return datatype_map.get(datatype.item(), 'uint8')
        return datatype_map.get(datatype, 'uint8')

    @staticmethod
    def save_nifti(data, affine_matrix, data_type, output_path, filename):
        os.makedirs(output_path, exist_ok=True)

        if not filename.startswith('synt_'):
            filename = 'synt_' + filename

        full_path = os.path.join(output_path, filename)

        if os.path.exists(full_path):
            counter = 1
            base_name = filename.split('.nii')[0]
            ext = '.nii.gz' if filename.endswith('.gz') else '.nii'

            final_filename = f"{base_name}_{counter}{ext}"
            while os.path.exists(os.path.join(output_path, final_filename)):
                counter += 1
                final_filename = f"{base_name}_{counter}{ext}"
            full_path = os.path.join(output_path, final_filename)

        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()

        nib.save(
            nib.Nifti1Image(data.astype(data_type), affine_matrix),
            full_path
        )
        print(f"Saved {full_path}")

    @staticmethod
    def save_data(d, folder='default'):
        image_meta = d.get('image_meta_dict', None)
        if image_meta is None:
            print("Warning: No image meta dict found, skipping save.")
            return

        image_data_type = TumorSaver.get_datatype(image_meta.get('datatype', 16))
        image_affine_matrix = image_meta.get('original_affine', np.eye(4))

        original_filename = os.path.basename(image_meta.get('filename_or_obj', 'unknown.nii.gz'))

        keys_to_save = {
            'image': ('image', True),
            'label': ('label', False),
            'tumor_texture_layer': ('layer', True),
            'alpha': ('alpha', True),
            'tumor_mask_layer': ('mask', True)
        }

        for key, (sub_folder, use_img_meta) in keys_to_save.items():
            if key in d:
                data = d[key]

                if len(data.shape) == 4:
                    data = data.squeeze(0)

                output_dir = os.path.join('synt', folder, sub_folder)

                if use_img_meta:
                    dtype = image_data_type
                    affine = image_affine_matrix
                else:
                    label_meta = d.get('label_meta_dict', image_meta)
                    dtype = TumorSaver.get_datatype(label_meta.get('datatype', 2))
                    affine = label_meta.get('original_affine', image_affine_matrix)

                TumorSaver.save_nifti(data, affine, dtype, output_dir, original_filename)