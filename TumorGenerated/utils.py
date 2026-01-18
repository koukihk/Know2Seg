### Tumor Generateion
import random

import cv2
import elasticdeform
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, sobel, gaussian_filter1d
from skimage.restoration import denoise_tv_chambolle


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1) 

    return Bj


def get_predefined_texture_b(mask_shape, sigma_a, sigma_b):
    # Step 1: Uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    # a = generate_simplex_noise(mask_shape, 0.5)

    # Step 2: Nonlinear diffusion filtering
    a_denoised = denoise_tv_chambolle(a, weight=0.1, multichannel=False)

    # Step 3: Wavelet transform
    coeffs = pywt.wavedecn(a_denoised, wavelet='db4', level=2)  
    coeffs[1] = {k: 0.3 * v for k, v in coeffs[1].items()}
    for i in range(2, len(coeffs)):
        coeffs[i] = {k: np.zeros_like(v) for k, v in coeffs[i].items()}
    a_wavelet_denoised = pywt.waverecn(coeffs, wavelet='db4')  

    # Normalize to 0-1
    a_wavelet_denoised = (a_wavelet_denoised - np.min(a_wavelet_denoised)) / (
            np.max(a_wavelet_denoised) - np.min(a_wavelet_denoised))

    # Step 4: Gaussian filter
    # a_2 = gaussian_filter(a, sigma=sigma_a)
    a_2 = gaussian_filter(a_wavelet_denoised, sigma=sigma_a)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)  

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5, 5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist()  # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points


def get_absolute_coordinate(relative_coordinate, original_shape, target_volume, start):
    x_ratio = original_shape[0] / target_volume[0]
    y_ratio = original_shape[1] / target_volume[1]
    z_ratio = original_shape[2] / target_volume[2]

    absolute_x = relative_coordinate[0] * x_ratio
    absolute_y = relative_coordinate[1] * y_ratio
    absolute_z = relative_coordinate[2] * z_ratio

    absolute_x += start[0]
    absolute_y += start[1]
    absolute_z += start[2]

    return np.array([absolute_x, absolute_y, absolute_z], dtype=float)


def ellipsoid_select(mask_scan, ellipsoid_model=None, max_attempts=600, edge_op="both", use_optimized=False, margin=1):
    if use_optimized and ellipsoid_model is not None:
        try:
            from ellipsoid_sampler import optimized_ellipsoid_select
            return optimized_ellipsoid_select(mask_scan, ellipsoid_model, max_attempts, edge_op)
        except ImportError:
            print("Warning: optimized_ellipsoid_select not found, falling back to original method")

    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    target_volume = (300, 250, 140)
    start = (x_start, y_start, z_start)

    def has_margin(arr, pt, m):
        # check local cube fully inside organ
        if m <= 0:
            return True
        minb = np.maximum(pt - m, 0)
        maxb = np.minimum(pt + m, np.array(arr.shape) - 1)
        sub = arr[minb[0]:maxb[0] + 1, minb[1]:maxb[1] + 1, minb[2]:maxb[2] + 1]
        return np.all(sub == 1)

    loop_count = 0
    while loop_count < max_attempts:
        potential_point = ellipsoid_model.get_random_point()
        if any(coord < 0 for coord in potential_point):
            loop_count += 1
            continue
        potential_point = get_absolute_coordinate(potential_point, liver_mask.shape, target_volume, start)
        potential_point = np.clip(potential_point, 0, np.array(mask_scan.shape) - 1).astype(int)

        if mask_scan[tuple(potential_point)] == 1:
            # Check margin, not at edge, and within the liver
            if has_margin(mask_scan, potential_point, margin) and (
            not is_edge_point(mask_scan, potential_point, edge_op)):
                return potential_point

        loop_count += 1

    # fallback: pick random interior voxel within shrunken bbox
    interior = np.argwhere(mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] == 1)
    if interior.size > 0:
        ridx = np.random.randint(0, interior.shape[0])
        candidate = interior[ridx] + np.array([x_start, y_start, z_start])
        candidate = np.clip(candidate, 0, np.array(mask_scan.shape) - 1)
        if has_margin(mask_scan, candidate, margin) and (not is_edge_point(mask_scan, candidate, edge_op)):
            return candidate.astype(int)
    # final fallback
    return random_select(mask_scan)

def check_sobel(mask_scan, potential_point, sobel_threshold=405, sobel_neighborhood_size=(7, 7, 7)):
    # Calculate the neighborhood bounds, ensuring it stays within the mask scan limits
    min_bounds = np.maximum(potential_point - np.array(sobel_neighborhood_size) // 2, 0)
    max_bounds = np.minimum(potential_point + np.array(sobel_neighborhood_size) // 2, np.array(mask_scan.shape) - 1)

    # Extract the neighborhood sub-region
    neighborhood_sn = mask_scan[min_bounds[0]:max_bounds[0] + 1,
                                min_bounds[1]:max_bounds[1] + 1,
                                min_bounds[2]:max_bounds[2] + 1]

    # Compute Sobel filters only on the neighborhood region
    sobel_x = sobel(neighborhood_sn, axis=0)
    sobel_y = sobel(neighborhood_sn, axis=1)
    sobel_z = sobel(neighborhood_sn, axis=2)

    # Calculate the gradient magnitude of the Sobel filter
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    # Get the gradient magnitude at the central point of the neighborhood
    central_point = np.array(sobel_neighborhood_size) // 2
    gradient_value = gradient_magnitude[tuple(central_point)]

    # Return True if the gradient magnitude exceeds the threshold
    return gradient_value > sobel_threshold

def is_edge_point(mask_scan, potential_point, edge_op="both", neighborhood_size=(3, 3, 3), volume_threshold=5,
                  sobel_threshold=400, erosion_kernel_size=5):
    def check_volume():
        # Define the boundaries of the neighborhood around the potential point
        min_bounds = np.maximum(potential_point - np.array(neighborhood_size) // 2, 0)
        max_bounds = np.minimum(potential_point + np.array(neighborhood_size) // 2, np.array(mask_scan.shape) - 1)

        # Extract the neighborhood volume from the mask scan
        neighborhood_volume = mask_scan[min_bounds[0]:max_bounds[0] + 1,
                                        min_bounds[1]:max_bounds[1] + 1,
                                        min_bounds[2]:max_bounds[2] + 1]

        # Count the number of liver voxels in the neighborhood
        liver_voxel_count = np.sum(neighborhood_volume == 1)

        # Check if the liver voxel count is below the threshold
        return liver_voxel_count < volume_threshold
        
    def check_erosion():
        z = potential_point[2]
        if z < 0 or z >= mask_scan.shape[2]:
            return True  
            
        mask_slice = mask_scan[..., z].copy()  
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), dtype=np.uint8)
        eroded_mask = cv2.erode(mask_slice.astype(np.uint8), kernel, iterations=1)
        
        x, y = potential_point[0], potential_point[1]
        if x < 0 or x >= mask_scan.shape[0] or y < 0 or y >= mask_scan.shape[1]:
            return True
            
        return mask_slice[x, y] == 1 and eroded_mask[x, y] == 0

    if edge_op == "volume":
        return check_volume()
    elif edge_op == "sobel":
        return check_sobel(mask_scan, potential_point, sobel_threshold)
    elif edge_op == "erosion":
        return check_erosion()
    elif edge_op == "any":
        return check_volume() and check_sobel(mask_scan, potential_point, sobel_threshold)
    elif edge_op == "both":
        return check_volume() or check_sobel(mask_scan, potential_point, sobel_threshold)
    elif edge_op == "volume_erosion":
        return check_volume() or check_erosion()
    elif edge_op == "sobel_erosion":
        return check_sobel(mask_scan, potential_point, sobel_threshold) or check_erosion()
    elif edge_op == "all":
        return check_volume() or check_sobel(mask_scan, potential_point, sobel_threshold) or check_erosion()
    elif edge_op == "none":
        return False
    else:
        raise ValueError("Invalid edge_op option. Choose from 'volume', 'sobel', 'erosion', 'any', 'both', 'volume_erosion', 'sobel_erosion', 'all', or 'none'.")


def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    x = int(round(float(x)))
    y = int(round(float(y)))
    z = int(round(float(z)))
    
    x = max(1, x)
    y = max(1, y)
    z = max(1, z)
    
    sh = (4 * x, 4 * y, 4 * z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2 * x, 2 * y, 2 * z])  # center point

    # calculate the ellipsoid
    bboxl = np.floor(com - radii).clip(0, None).astype(int)
    bboxh = (np.ceil(com + radii) + 1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice, bboxl, bboxh))]
    roiaux = aux[tuple(map(slice, bboxl, bboxh))]
    logrid = *map(np.square, np.ogrid[tuple(
        map(slice, (bboxl - com) / radii, (bboxh - com - 1) / radii, 1j * (bboxh - bboxl)))]),
    dst = (1 - sum(logrid)).clip(0, None)
    mask = dst > roiaux
    roi[mask] = 1
    np.copyto(roiaux, dst, where=mask)

    return out


def get_fixed_geo(mask_scan, tumor_type, ellipsoid_model=None, margin=1):
    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros(
        (mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    # texture_map = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.float16)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            y = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            z = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            sigma = random.uniform(0.5, 1)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            y = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            z = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            sigma = random.randint(1, 2)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            y = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            z = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            sigma = random.randint(3, 6)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'large':
        num_tumor = random.randint(1, 3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            y = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            z = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            sigma = random.randint(5, 10)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == "mix":
        # tiny
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            y = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            z = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            sigma = random.uniform(0.5, 1)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

        # small
        num_tumor = random.randint(5, 10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            y = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            z = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            sigma = random.randint(1, 2)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # medium
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            y = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            z = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            sigma = random.randint(3, 6)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # large
        num_tumor = random.randint(1, 3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            y = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            z = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            point = ellipsoid_select(mask_scan, ellipsoid_model, margin=margin) if ellipsoid_model is not None else random_select(mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    geo_mask = geo_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2]
    # texture_map = texture_map[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    geo_mask = (geo_mask * mask_scan) >= 1

    return geo_mask


def get_tumor(volume_scan, mask_scan, tumor_type, texture, edge_advanced_blur=False, ellipsoid_model=None, margin=1):
    geo_mask = get_fixed_geo(mask_scan, tumor_type, ellipsoid_model, margin=margin)

    sigma = np.random.uniform(1, 2)
    if edge_advanced_blur:
        sigma = np.random.uniform(1.0, 2.2)
    difference = np.random.uniform(65, 145)

    # blur the boundary (alpha)
    geo_blur = gaussian_filter(geo_mask * 1.0, sigma)

    pure_difference = texture * difference * mask_scan
    tumor_diff_with_alpha = pure_difference * geo_blur
    abnormally = (volume_scan - tumor_diff_with_alpha) * mask_scan
    tumor_s_layer = (volume_scan - pure_difference) * mask_scan

    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, abnormally_mask, tumor_s_layer, geo_mask, geo_blur

def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture, edge_advanced_blur, ellipsoid_model=None,
                   margin=1):
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    # input texture shape: 420, 300, 320
    # we need to cut it into the shape of liver_mask
    # for examples, the liver_mask.shape = 286, 173, 46; we should change the texture shape
    x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start

    pad_x = max(0, x_length - texture.shape[0] + 1)
    pad_y = max(0, y_length - texture.shape[1] + 1)
    pad_z = max(0, z_length - texture.shape[2] + 1)

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        texture = np.pad(texture, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='wrap')

    start_x = random.randint(0, texture.shape[
        0] - x_length - 1)  # random select the start point, -1 is to avoid boundary check
    start_y = random.randint(0, texture.shape[1] - y_length - 1)
    start_z = random.randint(0, texture.shape[2] - z_length - 1)
    cut_texture = texture[start_x:start_x + x_length, start_y:start_y + y_length, start_z:start_z + z_length]

    liver_volume, liver_mask, tumor_s_layer, tumor_mask_layer, geo_blur = get_tumor(liver_volume, liver_mask,
                                                                                        tumor_type, cut_texture,
                                                                                        edge_advanced_blur,
                                                                                        ellipsoid_model,
                                                                                        margin=margin)

    # Restore to full size
    full_tumor_texture_layer = np.zeros_like(volume_scan)
    full_tumor_mask_layer = np.zeros_like(mask_scan)
    full_alpha = np.zeros_like(volume_scan)
    full_tumor_texture_layer[x_start:x_end, y_start:y_end, z_start:z_end] = tumor_s_layer
    full_tumor_mask_layer[x_start:x_end, y_start:y_end, z_start:z_end] = tumor_mask_layer
    full_alpha[x_start:x_end, y_start:y_end, z_start:z_end] = geo_blur

    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    return volume_scan, mask_scan, full_tumor_texture_layer, full_tumor_mask_layer, full_alpha