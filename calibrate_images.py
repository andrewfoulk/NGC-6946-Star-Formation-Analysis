import os
import numpy as np
from astropy.io import fits

# Base path to the directory where your images are stored
base_path = '/Users/andrewfoulk/astr310/images' 

# Directory paths within each observing date folder
science_path = 'science'
bias_path = 'bias'
dark_path = 'darks'
flat_path = 'flats'

# Function to calibrate images for a specific observing date
def calibrate_images_for_date(observation_date):
    """
    Calibrates images for a specific observing date by applying bias, dark, and flat-field corrections.
    
    Function is split into two parts:
        1. Loads all the images.
        2. Calibrates the science images.
        
    """
    # Initialize lists to hold different types of images
    h_alpha_sci, sloan_r_sci, h_alpha_flats, sloan_r_flats, bias, darks = [], [], [], [], [], []

    # Directory path for the observing date
    dated_path = os.path.join(base_path, observation_date)

    ##### Load Images #####

    # Load science images
    files = [f for f in os.listdir(os.path.join(dated_path, science_path)) if f.endswith(('.fit', '.fits'))]
    for file in files:
        file_path = os.path.join(dated_path, science_path, file)
        data = fits.getdata(file_path, ignore_missing_simple=True)
        if 'ha' in file.lower():
            h_alpha_sci.append(data)
        elif 'r' in file.lower():
            sloan_r_sci.append(data)

    # Load flat images
    files = [f for f in os.listdir(os.path.join(dated_path, flat_path)) if f.endswith(('.fit', '.fits'))]
    for file in files:
        file_path = os.path.join(dated_path, flat_path, file)
        data = fits.getdata(file_path, ignore_missing_simple=True)
        if 'ha' in file.lower():
            h_alpha_flats.append(data)
        elif 'r' in file.lower():
            sloan_r_flats.append(data)

    # Load bias images
    files = [f for f in os.listdir(os.path.join(dated_path, bias_path)) if f.endswith(('.fit', '.fits'))]
    for file in files:
        file_path = os.path.join(dated_path, bias_path, file)
        data = fits.getdata(file_path, ignore_missing_simple=True)
        bias.append(data)

    # Load dark images
    files = [f for f in os.listdir(os.path.join(dated_path, dark_path)) if f.endswith(('.fit', '.fits'))]
    for file in files:
        file_path = os.path.join(dated_path, dark_path, file)
        data = fits.getdata(file_path, ignore_missing_simple=True)
        darks.append(data)

    ##### Calibrate Science Images #####

    # Create master bias by stacking and taking the median across the third dimension
    master_bias = np.median(np.dstack(bias), axis=2)

    # Subtract the master bias from all images (Bias is subtracted from the flats)
    h_alpha_sci = [image - master_bias for image in h_alpha_sci]
    sloan_r_sci = [image - master_bias for image in sloan_r_sci]
    darks = [image - master_bias for image in darks]
    h_alpha_flats = [image - master_bias for image in h_alpha_flats]
    sloan_r_flats = [image - master_bias for image in sloan_r_flats]

    # Create master dark by stacking and taking the median across the third dimension
    master_dark = np.median(np.dstack(darks), axis=2)

    # Subtract the master dark from the science images
    h_alpha_sci = [image - master_dark for image in h_alpha_sci]
    sloan_r_sci = [image - master_dark for image in sloan_r_sci]

    # Subtract the master dark from the flats
    h_alpha_flats = [flat - master_dark for flat in h_alpha_flats]
    sloan_r_flats = [flat - master_dark for flat in sloan_r_flats]

    # Create master flats by stacking and taking the median across the third dimension
    master_flat_h_alpha = np.median(np.dstack(h_alpha_flats), axis=2)
    master_flat_sloan_r = np.median(np.dstack(sloan_r_flats), axis=2)

    # Normalize the master flats
    master_flat_h_alpha /= np.median(master_flat_h_alpha)
    master_flat_sloan_r /= np.median(master_flat_sloan_r)

    # Divide each science image by the master flat for flat-field correction
    h_alpha_calibrated = [image / master_flat_h_alpha for image in h_alpha_sci]
    sloan_r_calibrated = [image / master_flat_sloan_r for image in sloan_r_sci]

    return h_alpha_calibrated, sloan_r_calibrated

# Function to save calibrated images
def save_calibrated_images(images, filter_type, observation_date):
    """
    Saves calibrated images in a specific directory based on date and filter.
    """
    # Define directory path based on filter type and observing date
    dir_path = os.path.join(base_path, f"{filter_type}_calibrated", observation_date)
    os.makedirs(dir_path, exist_ok=True)
    for i, image in enumerate(images):
        file_path = os.path.join(dir_path, f"{filter_type}_calibrated_{i+1}.fit")
        hdu = fits.PrimaryHDU(image)
        hdu.writeto(file_path, overwrite=True)
        print(f"{filter_type} image {i+1} saved to {file_path}")

# Observing sessions with flat exposure times for each filter
observing_sessions = [
    ('10-03-24'),  
    ('10-09-24'),
    ('10-18-24')
]

# Calibrate and save images for each observing session
for observation_date in observing_sessions:
    print(f"Processing date: {observation_date}")
    h_alpha, sloan_r = calibrate_images_for_date(observation_date)
    save_calibrated_images(h_alpha, 'h_alpha', observation_date)
    save_calibrated_images(sloan_r, 'sloan_r', observation_date)
