import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.colors as mcolors
from scipy import ndimage
import glob
from skimage import measure
import json
from tqdm import tqdm
import concurrent.futures

def voxel2R(A):
    """Convert voxel volume to sphere radius (unit: mm)"""
    return (np.array(A)/4*3/np.pi)**(1/3)

def pixel2voxel(A, res):
    """Convert pixel count to volume (unit: mm³)"""
    return np.array(A)*(res[0]*res[1]*res[2])

def get_liver_bbox(mask):
    """Get bounding box of liver region"""
    x_start, x_end = np.where(np.any(mask, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask, axis=(0, 1)))[0][[0, -1]]
    
    # Add margin
    margin = 5
    x_start, x_end = max(0, x_start - margin), min(mask.shape[0], x_end + margin)
    y_start, y_end = max(0, y_start - margin), min(mask.shape[1], y_end + margin)
    z_start, z_end = max(0, z_start - margin), min(mask.shape[2], z_end + margin)
    
    return (x_start, x_end), (y_start, y_end), (z_start, z_end)

def find_optimal_slice(tumor_mask, liver_mask):
    """Find the slice containing the largest tumor area"""
    # Ensure input is boolean type
    tumor_mask = tumor_mask.astype(bool)
    liver_mask = liver_mask.astype(bool)
    
    # Find slices containing tumor
    tumor_slices = {}
    for z in range(tumor_mask.shape[2]):
        slice_area = np.sum(tumor_mask[:, :, z])
        if slice_area > 0:
            tumor_slices[z] = slice_area
    
    # If no tumor is found, find the slice with most liver tissue
    if not tumor_slices:
        liver_slices = {}
        for z in range(liver_mask.shape[2]):
            slice_area = np.sum(liver_mask[:, :, z])
            if slice_area > 0:
                liver_slices[z] = slice_area
        if not liver_slices:
            return None
        return max(liver_slices, key=liver_slices.get)
    
    # Return the slice with largest tumor area
    return max(tumor_slices, key=tumor_slices.get)

def analyze_tumors(tumor_mask, spacing_mm):
    """Analyze tumor positions and sizes"""
    tumor_info = []
    if np.sum(tumor_mask) > 0:
        label_cc, label_num = ndimage.label(tumor_mask)
        for i in range(1, label_num + 1):
            tumor_region = (label_cc == i)
            tumor_size = np.sum(tumor_region)
            if tumor_size < 8:  # Ignore too small tumors
                continue
                
            # Calculate tumor radius (mm)
            tumor_volume_mm = pixel2voxel(tumor_size, spacing_mm)
            tumor_radius_mm = voxel2R(tumor_volume_mm)
            
            # Find tumor center
            pos = ndimage.center_of_mass(tumor_region)
            
            # Find 2D slices where this tumor appears
            z_positions = np.unique(np.where(tumor_region)[2])
            
            if len(z_positions) == 0:
                continue
                
            size_category = ""
            if tumor_radius_mm <= 5:
                size_category = "0-5mm"
            elif tumor_radius_mm <= 10:
                size_category = "5-10mm"
            else:
                size_category = ">10mm"
                
            tumor_info.append({
                "center": pos,
                "radius_mm": tumor_radius_mm,
                "volume_mm3": tumor_volume_mm,
                "size_category": size_category,
                "region": tumor_region,
                "z_positions": z_positions
            })
    
    return tumor_info

def find_tumor_contours(tumor_mask, z_slice):
    """Find exact contours of tumor regions in the given slice"""
    # Extract the slice
    slice_tumor = tumor_mask[:, :, z_slice].T
    
    # If no tumor in this slice, return empty list
    if np.sum(slice_tumor) == 0:
        return [], []
    
    # Find connected components
    labeled, num_components = ndimage.label(slice_tumor)
    
    tumors = []
    contours = []
    
    # Process each component
    for idx in range(1, num_components + 1):
        component = (labeled == idx)
        if np.sum(component) < 5:  # Skip very small components
            continue
        
        # Find the contour
        component_contours = measure.find_contours(component.astype(float), 0.5)
        
        if not component_contours:
            continue
            
        # Get the largest contour for this component
        largest_contour = max(component_contours, key=len)
        
        # Calculate area and center
        area = np.sum(component)
        center = ndimage.center_of_mass(component)
        
        # Calculate bounding box
        rows, cols = np.where(component)
        if len(rows) == 0 or len(cols) == 0:
            continue
            
        min_r, max_r = np.min(rows), np.max(rows)
        min_c, max_c = np.min(cols), np.max(cols)
        
        # Calculate tumor size in mm
        area_mm = pixel2voxel(area, (1, 1, 1))  # We'll update this with real spacing later
        radius_mm = voxel2R(area_mm)
        
        tumors.append({
            "area": area,
            "center": center,
            "bbox": (min_c, min_r, max_c, max_r),
            "radius_mm": radius_mm,
            "contour": largest_contour
        })
        
        contours.append(largest_contour)
    
    return tumors, contours

def create_visualization(scan_path, label_path, model_dirs, output_path, case_name=None, verbose=False):
    """Create visualization of tumor segmentation"""
    # Extract case name from file path if not provided
    if case_name is None:
        case_name = os.path.basename(scan_path).split('.')[0]
        
    # Load scan and label
    try:
        scan_nib = nib.load(scan_path)
        label_nib = nib.load(label_path)
    except Exception as e:
        print(f"Error loading {case_name}: {str(e)}")
        return False
    
    scan_data = scan_nib.get_fdata()
    label_data = label_nib.get_fdata()
    
    # Get pixel spacing
    pixdim = label_nib.header['pixdim']
    spacing_mm = tuple(pixdim[1:4])
    
    # Extract liver and tumor masks
    liver_mask = np.zeros_like(label_data)
    liver_mask[label_data == 1] = 1
    liver_mask[label_data == 2] = 1  # Tumor region is also part of the liver
    
    tumor_mask = np.zeros_like(label_data)
    tumor_mask[label_data == 2] = 1
    
    # Check if there are tumors in the label
    if np.sum(tumor_mask) == 0:
        print(f"No tumors found in {case_name}, skipping visualization")
        return False
    
    # Analyze tumor information for entire volume
    tumor_info = analyze_tumors(tumor_mask, spacing_mm)
    
    # Get liver region bounding box
    try:
        x_range, y_range, z_range = get_liver_bbox(liver_mask)
    except IndexError:
        print(f"No liver found in {case_name}, skipping visualization")
        return False
    
    # Find optimal slice
    optimal_slice = find_optimal_slice(tumor_mask, liver_mask)
    if optimal_slice is None:
        print(f"Could not find a suitable slice in {case_name}!")
        return False
    
    # Ensure slice is within liver range
    optimal_slice = max(z_range[0], min(optimal_slice, z_range[1]))
    
    # Find exact tumor contours in the selected slice
    slice_tumors, tumor_contours = find_tumor_contours(tumor_mask, optimal_slice)
    
    # Update tumor radius calculations with proper spacing
    for tumor in slice_tumors:
        area_mm = pixel2voxel(tumor["area"], spacing_mm)
        tumor["radius_mm"] = voxel2R(area_mm)
    
    # Load model predictions
    model_predictions = []
    model_names = []
    
    # Define default model names that will be used if we have exactly 3 models
    default_model_names = ["Model trained on real tumors", "SynTumor", "MedCraft(Ours)"]
    
    if verbose:
        print(f"Attempting to load predictions for {case_name} from {len(model_dirs)} model directories")
    
    for i, model_dir in enumerate(model_dirs):
        if verbose:
            print(f"Processing model directory {i+1}: {model_dir}")
            
        # Find prediction files
        pred_files = glob.glob(os.path.join(model_dir, "*.nii.gz"))
        if not pred_files:
            print(f"No prediction files found in directory {model_dir}")
            continue
            
        # Use prediction file matching scan name
        scan_name = os.path.basename(scan_path).split('.')[0]
        matching_files = [f for f in pred_files if scan_name in os.path.basename(f)]
        
        pred_path = None
        if matching_files:
            pred_path = matching_files[0]
            if verbose:
                print(f"Found matching prediction file: {pred_path}")
        else:
            # If no match found, try alternatives
            print(f"No exact match for {scan_name} in {model_dir}, trying to find a similar file")
            # Try to match the pattern without extension
            for pred_file in pred_files:
                if os.path.basename(pred_file).startswith(scan_name):
                    pred_path = pred_file
                    if verbose:
                        print(f"Found similar prediction file: {pred_path}")
                    break
            
        if pred_path is None:
            print(f"No suitable prediction file found for {scan_name} in {model_dir}")
            continue
            
        try:
            pred_nib = nib.load(pred_path)
            pred_data = pred_nib.get_fdata()
        except Exception as e:
            print(f"Error loading prediction file {pred_path}: {str(e)}")
            continue
        
        # If prediction is class labels, convert to liver and tumor masks
        if len(pred_data.shape) == 3 and pred_data.dtype != bool:
            pred_liver_mask = np.zeros_like(pred_data)
            pred_liver_mask[pred_data == 1] = 1
            pred_liver_mask[pred_data == 2] = 1
            
            pred_tumor_mask = np.zeros_like(pred_data)
            pred_tumor_mask[pred_data == 2] = 1
            
            model_predictions.append((pred_liver_mask, pred_tumor_mask))
            
            # Add model name
            if i < len(default_model_names):
                model_names.append(default_model_names[i])
            else:
                model_names.append(f"Model {i+1}")
                
            if verbose:
                print(f"Successfully loaded model {i+1}: {model_names[-1]}")
        else:
            # If prediction is already in mask form
            model_predictions.append((pred_data, pred_data))
            
            # Add model name
            if i < len(default_model_names):
                model_names.append(default_model_names[i])
            else:
                model_names.append(f"Model {i+1}")
                
            if verbose:
                print(f"Successfully loaded model {i+1}: {model_names[-1]}")
    
    if verbose:
        print(f"Total models loaded: {len(model_predictions)}")
        print(f"Model names: {model_names}")
    
    # Create visualization
    num_models = len(model_predictions)
    
    if num_models == 0:
        print(f"No model predictions could be loaded for {case_name}")
        return False
    
    fig, axes = plt.subplots(1, num_models + 2, figsize=(5 * (num_models + 2), 5))
    
    # Handle the case when only one image is to be generated
    if num_models + 2 == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    
    # 1. Original scan image
    scan_slice = scan_data[:, :, optimal_slice].T
    
    # Normalize scan values to 0-1 range for display
    scan_min, scan_max = -150, 250  # Common CT window settings
    scan_slice_norm = np.clip(scan_slice, scan_min, scan_max)
    scan_slice_norm = (scan_slice_norm - scan_min) / (scan_max - scan_min)
    
    axes[0].imshow(scan_slice_norm, cmap='gray')
    axes[0].set_title("CT Scan", fontsize=25)
    
    # Draw tumor contours and bounding boxes with exact positions
    for tumor, contour in zip(slice_tumors, tumor_contours):
        # Draw contour
        axes[0].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = tumor["bbox"]
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        
        # Add size label at top of bounding box
        radius_mm = tumor["radius_mm"]
        axes[0].text(x_max + 7, y_max + 3, f"{radius_mm:.1f}mm", 
                    color='yellow', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7))
    
    # 2. Ground truth overlaid on scan
    axes[1].imshow(scan_slice_norm, cmap='gray')
    
    # Create liver mask overlay (green semi-transparent)
    liver_slice = liver_mask[:, :, optimal_slice].T
    liver_overlay = np.zeros((*liver_slice.shape, 4))
    liver_overlay[liver_slice == 1] = [0, 1, 0, 0.3]  # Green, 30% transparency
    
    # Create tumor mask overlay (red semi-transparent)
    tumor_slice = tumor_mask[:, :, optimal_slice].T
    tumor_overlay = np.zeros((*tumor_slice.shape, 4))
    tumor_overlay[tumor_slice == 1] = [1, 0, 0, 0.5]  # Red, 50% transparency
    
    axes[1].imshow(liver_overlay)
    axes[1].imshow(tumor_overlay)
    
    # Draw contours on ground truth image as well
    for contour in tumor_contours:
        axes[1].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)
    
    axes[1].set_title("Ground Truth", fontsize=25)
    
    # 3+ Model prediction results
    for i, ((pred_liver, pred_tumor), model_name) in enumerate(zip(model_predictions, model_names)):
        axes[i+2].imshow(scan_slice_norm, cmap='gray')
        
        pred_liver_slice = pred_liver[:, :, optimal_slice].T
        pred_liver_overlay = np.zeros((*pred_liver_slice.shape, 4))
        pred_liver_overlay[pred_liver_slice == 1] = [0, 1, 0, 0.3]  # Green, 30% transparency
        
        pred_tumor_slice = pred_tumor[:, :, optimal_slice].T
        pred_tumor_overlay = np.zeros((*pred_tumor_slice.shape, 4))
        pred_tumor_overlay[pred_tumor_slice == 1] = [1, 0, 0, 0.5]  # Red, 50% transparency
        
        axes[i+2].imshow(pred_liver_overlay)
        axes[i+2].imshow(pred_tumor_overlay)
        
        # Also find and plot contours of predicted tumors
        pred_tumors, pred_contours = find_tumor_contours(pred_tumor, optimal_slice)
        for contour in pred_contours:
            axes[i+2].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5)
        
        axes[i+2].set_title(model_name, fontsize=25)
    
    # Hide axis ticks
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Ensure the file extension is .png
    if not output_path.lower().endswith('.png'):
        output_path = os.path.splitext(output_path)[0] + '.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    return True

def process_batch_from_json(json_file, data_dir, model_dirs, output_dir, limit=None, verbose=False):
    """Process multiple cases from a JSON file"""
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Check if validation data exists
    if 'validation' not in data:
        print(f"No validation data found in {json_file}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get validation cases
    validation_cases = data['validation']
    if limit:
        validation_cases = validation_cases[:limit]
    
    print(f"Processing {len(validation_cases)} cases from {json_file}")
    
    # Verify model directories
    if verbose:
        print(f"Model directories to be used:")
        for i, model_dir in enumerate(model_dirs):
            print(f"  Model {i+1}: {model_dir}")
            if not os.path.exists(model_dir):
                print(f"    Warning: Directory does not exist!")
            else:
                pred_files = glob.glob(os.path.join(model_dir, "*.nii.gz"))
                print(f"    Found {len(pred_files)} prediction files")
    
    # Track successful and failed cases
    successful = []
    failed = []
    
    # Process each case
    for i, case in enumerate(tqdm(validation_cases, desc="Processing cases")):
        # Get image and label paths
        image_path = os.path.join(data_dir, case['image'])
        label_path = os.path.join(data_dir, case['label'])
        
        # Get case name
        case_name = os.path.basename(image_path).split('.')[0]
        
        # Create output path for this case
        case_output_path = os.path.join(output_dir, f"{case_name}.png")
        
        # Process case
        try:
            result = create_visualization(image_path, label_path, model_dirs, case_output_path, case_name, verbose)
            if result:
                successful.append(case_name)
            else:
                failed.append(case_name)
        except Exception as e:
            print(f"Error processing case {case_name}: {str(e)}")
            failed.append(case_name)
    
    # Write summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Total cases: {len(validation_cases)}\n")
        f.write(f"Successfully processed: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        f.write("Successfully processed cases:\n")
        for case in successful:
            f.write(f"- {case}\n")
        
        f.write("\nFailed cases:\n")
        for case in failed:
            f.write(f"- {case}\n")
    
    print(f"Batch processing complete. Successfully processed {len(successful)} out of {len(validation_cases)} cases.")
    print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Tumor Segmentation Visualization Tool')
    parser.add_argument('--scan_path', type=str, help='Path to scan image (for single case)')
    parser.add_argument('--label_path', type=str, help='Path to ground truth label (for single case)')
    parser.add_argument('--model_dirs', type=str, nargs='+', help='Directories of model prediction results')
    parser.add_argument('--output_path', type=str, default='visualization.png', help='Output image path (for single case)')
    
    # Add batch processing arguments
    parser.add_argument('--json_file', type=str, help='JSON file containing validation dataset info')
    parser.add_argument('--data_dir', type=str, help='Base directory for dataset paths in JSON file')
    parser.add_argument('--output_dir', type=str, default='batch_results', help='Output directory for batch results')
    parser.add_argument('--limit', type=int, help='Limit the number of cases to process')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output for debugging')
    
    args = parser.parse_args()
    
    # Check if model directories exist
    if args.model_dirs:
        args.model_dirs = [d for d in args.model_dirs if os.path.exists(d) or print(f"Warning: Directory {d} does not exist!")]
        if len(args.model_dirs) > 3:
            print("Warning: Maximum 3 model directories supported, using only the first 3.")
            args.model_dirs = args.model_dirs[:3]
        
        if args.verbose:
            print(f"Using model directories: {args.model_dirs}")
    else:
        args.model_dirs = []
    
    # Check if we should do batch processing
    if args.json_file and args.data_dir:
        process_batch_from_json(args.json_file, args.data_dir, args.model_dirs, args.output_dir, args.limit, args.verbose)
    
    # Otherwise do single case processing
    elif args.scan_path and args.label_path:
        create_visualization(args.scan_path, args.label_path, args.model_dirs, args.output_path, verbose=args.verbose)
    
    else:
        print("Error: Either provide --scan_path and --label_path for single case, "
              "or --json_file and --data_dir for batch processing")

if __name__ == "__main__":
    main()
