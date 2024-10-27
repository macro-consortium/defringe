from skimage.feature import peak_local_max
from skimage.filters import gaussian
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy.spatial import distance
import click, logging


logger = logging.getLogger(__name__)

def improved_region_selection(image, threshold=3, min_distance=50):
    """
    Dynamically selects regions based on intensity variations, focusing on large-scale fringes.
    """
    # Detect edges and smooth the image
    smooth_image = gaussian(image, sigma=3)

    # Find bright and dark regions in the smoothed image
    bright_coords = peak_local_max(smooth_image, min_distance=min_distance, threshold_abs=threshold)
    dark_coords = peak_local_max(-smooth_image, min_distance=min_distance, threshold_abs=threshold)
    # print (len(bright_coords))
    # print (len(dark_coords))

    return bright_coords, dark_coords


def pair_bright_dark_regions(bright_coords, dark_coords, max_distance=100):
    """
    Pairs each bright region with its closest dark region, ensuring each region is only used once.
    Rejects pairs where the distance exceeds max_distance.
    """
    paired_bright = []
    paired_dark = []
    
    # Create a list to track which dark regions have been used
    used_dark_indices = set()

    # For each bright region, find the nearest dark region that hasn't been used yet
    for bright in bright_coords:
        # Calculate the distance from this bright region to all dark regions
        distances = distance.cdist([bright], dark_coords, metric='euclidean')[0]
        
        # Filter out distances to already used dark regions
        for idx in used_dark_indices:
            distances[idx] = np.inf  # Set distance to infinity for used dark regions
        
        # Check if there are valid distances left to pair
        if np.all(np.isinf(distances)):  # If all distances are inf, continue to next bright region
            continue
        
        # Find the index of the closest unused dark region
        closest_idx = np.argmin(distances)
        
        # Check if the closest dark region is within the max distance
        if distances[closest_idx] != np.inf and distances[closest_idx] <= max_distance:
            paired_bright.append(bright)
            paired_dark.append(dark_coords[closest_idx])
            used_dark_indices.add(closest_idx)

    return np.array(paired_bright), np.array(paired_dark)


def extract_region(data, center, half_size=10):
    """
    Extract a square region from the image data around the specified center coordinate.
    Handles cases where the region extends beyond the image boundaries.
    
    half_size: Half the size of the square region (i.e., the distance from the center to any edge of the square).
    """
    x_center, y_center = int(center[0]), int(center[1])
    
    # Calculate the bounds of the square region
    x_min = max(0, x_center - half_size)
    x_max = min(data.shape[0], x_center + half_size)
    y_min = max(0, y_center - half_size)
    y_max = min(data.shape[1], y_center + half_size)
    
    #    Return the square region from the image
    return data[x_min:x_max, y_min:y_max]

@click.command()
@click.argument('image_file', type=click.Path(exists=True))
@click.argument('fringe_file', type=click.Path(exists=True))
@click.argument('region_file', type=click.Path(exists=True))

def remove_fringe_cli(image_file, fringe_file, max_distance=100):
    """
    Compare regions and perform fringe subtraction using the same regions for both the fringe image
    and the science image.
    """
    # Load the fringe data
    fringe = fits.getdata(fringe_file)

    # Select bright and dark regions from the fringe image only
    bright_coords, dark_coords = improved_region_selection(fringe)

    # Pair the regions (ensuring same bright/dark pairs across both images)
    paired_bright, paired_dark = pair_bright_dark_regions(bright_coords, dark_coords, max_distance)

    # Load the image to subtract fringes from (science image)
    fits_data = fits.getdata(image_file)
    
    # Compute differences between paired regions in both the image and fringe data
    diffs = []
    fringe_diffs = []
    region_half_size = 16  # Define the half-size of the square regions (for a 32x32 region)
    
    for bright, dark in zip(paired_bright, paired_dark):
        # Extract regions from the fringe image
        region_bright_fringe = extract_region(fringe, bright, half_size=region_half_size)
        region_dark_fringe = extract_region(fringe, dark, half_size=region_half_size)
        
        # Extract the exact same regions from the science image
        region_bright_image = extract_region(fits_data, bright, half_size=region_half_size)
        region_dark_image = extract_region(fits_data, dark, half_size=region_half_size)
        
        # Check if the regions have valid data before computing the median
        if region_bright_image.size == 0 or region_dark_image.size == 0:
            print(f"Skipping region at {bright} or {dark} due to invalid region size.")
            continue

        # Compute the median difference in the image
        median_bright_image = np.median(region_bright_image)
        median_dark_image = np.median(region_dark_image)
        diffs.append(median_bright_image - median_dark_image)
        
        # Compute the median difference in the fringe
        median_bright_fringe = np.median(region_bright_fringe)
        median_dark_fringe = np.median(region_dark_fringe)
        fringe_diffs.append(median_bright_fringe - median_dark_fringe)
    
    clipped_diffs = sigma_clip(diffs, sigma=2, maxiters=5).compressed()
    clipped_fringe_diffs = sigma_clip(fringe_diffs, sigma=2, maxiters=5).compressed()

    # Compute the scale factor for fringe subtraction using both sets of differences
    if len(clipped_diffs) > 0 and len(clipped_fringe_diffs) > 0:
        scale_factor = np.median(clipped_diffs) / np.median(clipped_fringe_diffs)
        print (scale_factor)
    else:
        print("Invalid differences found. Setting scale factor to 1.0.")
        scale_factor = 1.0

    # Subtract the scaled fringe pattern
    data_fringe_sub = fits_data - fringe * scale_factor

    # Save the fringe-subtracted data
    im_name = image_file.split('/')[-1]
    im_name = im_name.split('.')[0]
    fringe_sub_file = f"{im_name}_sub_fringe.fits"

    image_header = fits.getheader(image_file)
    image_header['DEFRINGE'] = 'True'
    fits.writeto(fringe_sub_file, data_fringe_sub, header=image_header, overwrite=True) 
    print(f"Fringe-subtracted data saved to: {fringe_sub_file}")

    return data_fringe_sub

remove_fringe = remove_fringe_cli.callback


if __name__ == '__main__':
    remove_fringe_cli()
