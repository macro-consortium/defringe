import numpy as np
import click
import logging

from regions import Regions
from astropy.io import fits
from pathlib import Path

logger = logging.getLogger(__name__)

def average_pixel_value(fringe_data, region_data):
    """
    Calculate the average pixel value of the regions in the fringe frame.
    """
    pixel_averages = []
    
    for region in range(1, len(region_data), 2):
        region_mask = region_data[region].to_mask()
        average_pixel = np.average(region_mask.cutout(fringe_data), weights=region_mask)
        pixel_averages.append(average_pixel)

    average_of_averages = np.average(pixel_averages)
    return average_of_averages

def region_differences(image_data, region_data):
    """
    Calculate the median pixel difference between the bright and dark regions.
    """
    pixel_differences = []

    for region in range(0, len(region_data)-1, 2):
        
        bright_mask = region_data[region].to_mask()
        dark_mask = region_data[region+1].to_mask()
        bright_median = np.median(bright_mask.cutout(image_data))
        dark_median = np.median(dark_mask.cutout(image_data))
        pixel_differences.append(bright_median - dark_median)

    difference_median = np.median(pixel_differences)
    return difference_median

@click.command()

# @click.option(
#     "-f",
#     "--fringe-frame",
# )
@click.argument('image_file', type=click.Path(exists=True))
@click.argument('fringe_file', type=click.Path(exists=True))
@click.argument('region_file', type=click.Path(exists=True))

def remove_fringe_cli(image_file, fringe_file, region_file):
    """
    Remove the fringe pattern from an image using a fringe frame and regions.
    """
    fringe_data = fits.getdata(fringe_file)*65535
    region_data = Regions.read(region_file, format='ds9')

    average_fringe = average_pixel_value(fringe_data, region_data)
    fringe_data -= average_fringe
    fringe_differences = region_differences(fringe_data, region_data)
    image_data = fits.getdata(image_file)
    im_differences = region_differences(image_data, region_data)
    fringe_ratio = im_differences / fringe_differences
    fringe_data *= fringe_ratio
    corrected_image = image_data - fringe_data
    
    header = fits.getheader(image_file)
    header['FRNGCORR'] = 'T'
    im_path = Path(image_file)
    out_file = im_path.with_name(im_path.stem + '_defrng.fts')
    fits.writeto(out_file, corrected_image, header, overwrite=True)
    logger.info(f'Image defringed and saved to {out_file}')

    return corrected_image


remove_fringe = remove_fringe_cli.callback


if __name__ == '__main__':
    remove_fringe_cli()