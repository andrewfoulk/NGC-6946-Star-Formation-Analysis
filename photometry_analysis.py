import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings

# Suppress warnings for clarity in debugging
warnings.filterwarnings("ignore")

# Define constants
egain = 1.2999999523162842  # Electron/ADU for the CCD gain calibration
Kccd = 1 / egain            # CCD gain in ADU/electron

# File paths to the cropped Hα and Sloan-R images of NGC 6946
ha_image_path = "/Users/andrewfoulk/astr310/output/final_coadds/cropped/h_alpha_final_coadded_cropped.fit"
sloan_r_image_path = "/Users/andrewfoulk/astr310/output/final_coadds/cropped/sloan_r_final_coadded_cropped.fit"

# Load the images
ha_image = fits.getdata(ha_image_path)
sloan_r_image = fits.getdata(sloan_r_image_path)

# Generate aperture sizes
aperture_sizes = np.arange(290, 300 + 2 * 30, 2)  # Starting at 345, step of 2, 30 points

# Updated function with negative pixel handling and debugging statements
def aperE(im, col, row, rad1, rad2, ir1, ir2, or1, or2, Kccd, saturation=np.inf):
    """
    Performs aperture photometry on an image, replacing negative pixel values
    in both the sky annulus and source aperture with the median of surrounding pixels.
    """
    # Copy the image to avoid modifying the original data
    im = im.copy()
    a, b = im.shape
    xx, yy = np.meshgrid(np.arange(b), np.arange(a))

    # Define source aperture and sky annulus masks
    ixsrc = ((xx - col) / rad1) ** 2 + ((yy - row) / rad2) ** 2 <= 1
    ixsky = np.logical_and(
        (((xx - col) / or1) ** 2 + ((yy - row) / or2) ** 2) <= 1,
        (((xx - col) / ir1) ** 2 + ((yy - row) / ir2) ** 2) >= 1
    )

    # Replace negative values in the sky annulus
    negative_sky_mask = ixsky & (im < 0)
    negative_y_indices_sky, negative_x_indices_sky = np.where(negative_sky_mask)
    num_negative_sky = len(negative_y_indices_sky)
    print(f"Number of negative values in sky annulus: {num_negative_sky}")

    for y, x in zip(negative_y_indices_sky, negative_x_indices_sky):
        y_min = max(y - 1, 0)
        y_max = min(y + 1, a - 1)
        x_min = max(x - 1, 0)
        x_max = min(x + 1, b - 1)
        neighborhood = im[y_min:y_max+1, x_min:x_max+1].flatten()
        center_index = (y - y_min) * (x_max - x_min + 1) + (x - x_min)
        neighborhood = np.delete(neighborhood, center_index)
        im[y, x] = np.median(neighborhood)

    # Replace negative values in the source aperture
    imixsrc = im[ixsrc]
    src_y_indices, src_x_indices = np.where(ixsrc)
    negative_src_mask = imixsrc < 0
    num_negative_src = np.sum(negative_src_mask)
    print(f"Number of negative values in imixsrc: {num_negative_src}")

    negative_indices_src = np.where(negative_src_mask)[0]

    for idx in negative_indices_src:
        y = src_y_indices[idx]
        x = src_x_indices[idx]
        y_min = max(y - 1, 0)
        y_max = min(y + 1, a - 1)
        x_min = max(x - 1, 0)
        x_max = min(x + 1, b - 1)
        neighborhood = im[y_min:y_max+1, x_min:x_max+1].flatten()
        center_index = (y - y_min) * (x_max - x_min + 1) + (x - x_min)
        neighborhood = np.delete(neighborhood, center_index)
        im[y, x] = np.median(neighborhood)

    # Update imixsrc after replacing negative values
    imixsrc = im[ixsrc]

    # Proceed with calculations
    length = max(ixsky.shape)
    sky = np.median(im[ixsky], axis=0)
    pix = imixsrc - sky

    # Ensure imixsrc is non-negative for sqrt
    imixsrc_positive = np.where(imixsrc < 0, 0, imixsrc)
    sig = np.sqrt(imixsrc_positive / Kccd)
    ssig = np.std(im[ixsky]) / np.sqrt(length) / Kccd
    flx = np.sum(pix) / Kccd
    err = np.sqrt(np.sum(sig**2) + ssig**2)

    # Debugging statements for variables
    print(f"Flux (flx): {flx}")
    print(f"Error (err): {err}")

    return flx, err

# Initialize lists to store fluxes and errors
ha_fluxes = []
ha_errors = []
sloan_fluxes = []
sloan_errors = []

# Specify the centers of the images
row_center_ha, col_center_ha = 588.3, 992.8
row_center_sloan, col_center_sloan = 609.3, 1000.6

# Perform aperture photometry for H-alpha image
for r in aperture_sizes:
    rad1 = rad2 = r
    ir1 = ir2 = r + 75
    or1 = or2 = r + 150

    print(f"\nProcessing H-alpha image with aperture radius {rad1}:")
    flx_ha, err_ha = aperE(
        ha_image.astype(np.float64),
        col_center_ha,
        row_center_ha,
        rad1,
        rad2,
        ir1,
        ir2,
        or1,
        or2,
        Kccd
    )
    ha_fluxes.append(flx_ha)
    ha_errors.append(err_ha)
    print(f"H-alpha flux at aperture radius {rad1}: {flx_ha} ± {err_ha}")

# Perform aperture photometry for Sloan-R image
for r in aperture_sizes:
    rad1 = rad2 = r
    ir1 = ir2 = r + 75
    or1 = or2 = r + 150

    print(f"\nProcessing Sloan-R image with aperture radius {rad1}:")
    flx_sloan, err_sloan = aperE(
        sloan_r_image.astype(np.float64),
        col_center_sloan,
        row_center_sloan,
        rad1,
        rad2,
        ir1,
        ir2,
        or1,
        or2,
        Kccd
    )
    sloan_fluxes.append(flx_sloan)
    sloan_errors.append(err_sloan)
    print(f"Sloan-R flux at aperture radius {rad1}: {flx_sloan} ± {err_sloan}")

# Plot flux vs. aperture size with error bars for each filter
plt.figure(figsize=(12, 8))
plt.errorbar(
    aperture_sizes,
    ha_fluxes,
    yerr=ha_errors,
    fmt='o',
    color='blue',
    label='H-alpha',
    markersize=4  # Reduced marker size for H-alpha
)
plt.errorbar(
    aperture_sizes,
    sloan_fluxes,
    yerr=sloan_errors,
    fmt='s',
    color='red',
    label='Sloan-R',
    markersize=4  # Reduced marker size for Sloan-R
)
plt.xlabel('Aperture Radius (pixels)')
plt.ylabel('Flux (ADU)')
plt.title("Flux vs. Aperture Size for Hα and Sloan r' Filters")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()