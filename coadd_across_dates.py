import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Base path to the directory where the co-added images are stored
output_dir_base = "/Users/andrewfoulk/astr310/output"

# Function to shift an image by a specified number of rows and columns
def imshift(im, nr, nc):
    """Shifts an image by nr rows and nc columns (positive or negative)."""
    a, b = im.shape
    imr = np.zeros(im.shape)
    ir1 = max(0, -nr)
    ir2 = min(a, a - nr)
    it1 = max(0, -nc)
    it2 = min(b, b - nc)
    r1 = max(0, nr)
    r2 = min(a, nr + a)
    c1 = max(0, nc)
    c2 = min(b, nc + b)
    imr[r1:r2, c1:c2] = im[ir1:ir2, it1:it2]
    return imr

# Function to display an image with dynamic color scaling
def show_image_with_dynamic_colorscale(image_data, title="", save_path=None):
    """
    Displays an image with a dynamic color scale based on percentiles.
    """
    plt.figure(figsize=(8, 8))
    vmin = np.percentile(image_data, 5)
    vmax = np.percentile(image_data, 95)
    plt.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    if save_path:
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")
    plt.show()

# List of observing sessions
observing_sessions = ['10-03-24', '10-18-24'] # 10-09-24 has bad images

# List of filters
filters = ['h_alpha', 'sloan_r']

# Shifts to apply between dates for each filter
# Format: date_shifts[filter_type][observation_date] = (shift_rows, shift_cols)
date_shifts = {
    'h_alpha': {
        '10-03-24': (0, 0),    # Reference date for h_alpha
        '10-09-24': (-9, 26),   # Example shifts; adjust as needed
        '10-18-24': (-97, -21),
    },
    'sloan_r': {
        '10-03-24': (0, 0),    # Reference date for sloan_r
        '10-09-24': (0, 0),   # Example shifts; adjust as needed
        '10-18-24': (-61, -12),
    }
}

# Output directory for final co-added images
final_output_dir = "/Users/andrewfoulk/astr310/output/final_coadds"
os.makedirs(final_output_dir, exist_ok=True)

# Co-add images across dates for each filter
for filter_type in filters:
    print(f"\nProcessing filter: {filter_type}")
    cumulative_image = None

    for observation_date in observing_sessions:
        # Path to the co-added image for this date and filter
        coadded_image_path = os.path.join(output_dir_base, observation_date, f"{filter_type}_coadded_shifted.fit")

        if not os.path.exists(coadded_image_path):
            print(f"Co-added image {coadded_image_path} does not exist. Skipping.")
            continue

        image_data = fits.getdata(coadded_image_path).astype(np.float64)

        # Apply the shift for this date
        shift_rows, shift_cols = date_shifts[filter_type].get(observation_date, (0, 0))
        print(f"Shifting date {observation_date} by ({shift_rows}, {shift_cols}) for filter {filter_type}")
        shifted_image = imshift(image_data, int(shift_rows), int(shift_cols))

        # Initialize or accumulate the images
        if cumulative_image is None:
            cumulative_image = shifted_image
        else:
            cumulative_image += shifted_image

        # Show the cumulative image after each addition
        title = f"Cumulative Image After Adding {observation_date}, {filter_type}"
        show_image_with_dynamic_colorscale(cumulative_image, title=title)

    # Save the final co-added image for this filter
    if cumulative_image is not None:
        output_path = os.path.join(final_output_dir, f"{filter_type}_final_coadded.fit")
        fits.PrimaryHDU(cumulative_image).writeto(output_path, overwrite=True)
        print(f"Final co-added image for filter {filter_type} saved to {output_path}")
    else:
        print(f"No images found for filter {filter_type}.")
