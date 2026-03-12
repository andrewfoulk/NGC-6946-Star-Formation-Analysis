import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Base path to the directory where your images are stored
base_path = '/Users/andrewfoulk/astr310/images'
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

# Function to plot an image with a dynamic color scale
def plot_with_dynamic_colorscale(image, title):
    """
    Plots the image with a dynamic color scale based on ADU values.
    """
    vmin = np.percentile(image, 20)  # Lower bound: 20th percentile
    vmax = np.percentile(image, 95)  # Upper bound: 95th percentile

    plt.figure(figsize=(8, 8))
    fig = plt.imshow(image, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(fig, fraction=0.036, pad=0.04, label='Pixel Value (ADU)')
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()

# Dictionary to store shifts for each image
shifts = {
    '10-03-24': {
        'h_alpha': {
            'h_alpha_calibrated_1.fit': (0, 0),
            'h_alpha_calibrated_2.fit': (21, 21),
            'h_alpha_calibrated_3.fit': (12, 13),
            'h_alpha_calibrated_4.fit': (-10, -8),
            'h_alpha_calibrated_5.fit': (4, 8),
            'h_alpha_calibrated_6.fit': (20, 21),
            'h_alpha_calibrated_7.fit': (15, 20),
            'h_alpha_calibrated_8.fit': (-6, -7),
            'h_alpha_calibrated_9.fit': (-10, -6),
        },
        'sloan_r': {
            'sloan_r_calibrated_1.fit': (0, 0),
            'sloan_r_calibrated_2.fit': (6, 5),
            # Has a satellite
            # Has a satellite
            'sloan_r_calibrated_5.fit': (3, 1),
            'sloan_r_calibrated_6.fit': (30, 19),
            'sloan_r_calibrated_7.fit': (13, 11),
            'sloan_r_calibrated_8.fit': (35, 27),
            'sloan_r_calibrated_9.fit': (30, 22),
        }
    },
    '10-09-24': {
        'h_alpha': {
            'h_alpha_calibrated_1.fit': (0, 0),  # Reference image
            'h_alpha_calibrated_2.fit': (-18, -16),
                                                    # This image had a streak in the lower left
                                                    # This image had a streak in the lower left
            'h_alpha_calibrated_5.fit': (-35, -33),
        },
        'sloan_r': {
                                                    # Reference image and has streaks
            'sloan_r_calibrated_2.fit': (-17, -29),
            'sloan_r_calibrated_3.fit': (0, -13),
            'sloan_r_calibrated_4.fit': (17, 4),
        }
    },
    '10-18-24': {
        'h_alpha': {
            'h_alpha_calibrated_1.fit': (0, 0),
            'h_alpha_calibrated_10.fit': (7, 7),
            'h_alpha_calibrated_11.fit': (4, 2),
            'h_alpha_calibrated_12.fit': (2, 1),
            'h_alpha_calibrated_13.fit': (-19, 111),
            # Low image quality
            'h_alpha_calibrated_15.fit': (2, 3),
            'h_alpha_calibrated_2.fit': (8, 3),
            'h_alpha_calibrated_3.fit': (10, 4),
            # Low image quality
            'h_alpha_calibrated_5.fit': (2, 1),
            # Low image quality
            'h_alpha_calibrated_7.fit': (4, 3),
            'h_alpha_calibrated_8.fit': (-1, -1),
            'h_alpha_calibrated_9.fit': (7, 4),
        },
        'sloan_r': {
            'sloan_r_calibrated_1.fit': (0, 0),
            'sloan_r_calibrated_10.fit': (5, 9),
            'sloan_r_calibrated_11.fit': (116, 1),
            'sloan_r_calibrated_12.fit': (118, 0),
            'sloan_r_calibrated_13.fit': (5, 7),
            'sloan_r_calibrated_14.fit': (4, 6),
            # Low image quality
            'sloan_r_calibrated_2.fit': (3, 0),
            'sloan_r_calibrated_3.fit': (3, 7),
            'sloan_r_calibrated_4.fit': (2, 3),
            'sloan_r_calibrated_5.fit': (1, -1),
            'sloan_r_calibrated_6.fit': (6, 2),
            'sloan_r_calibrated_7.fit': (7, 1),
            'sloan_r_calibrated_8.fit': (5, 1),
            'sloan_r_calibrated_9.fit': (4, 8),
        }
    }
}

# Function to apply shifts, co-add images, and display cumulative results
def apply_shifts_and_coadd(observing_sessions, filter_type):
    for observation_date in observing_sessions:
        print(f"\nProcessing date: {observation_date}")
        date_shifts = shifts[observation_date][filter_type]

        image_dir = os.path.join(base_path, f"{filter_type}_calibrated", observation_date)
        cumulative_coadded_image = None

        for filename, (shift_rows, shift_cols) in date_shifts.items():
            file_path = os.path.join(image_dir, filename)
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist. Skipping.")
                continue

            image_data = fits.getdata(file_path).astype(np.float64)

            print(f"Shifting {filename} by ({shift_rows}, {shift_cols})")
            shifted_image = imshift(image_data, int(shift_rows), int(shift_cols))

            if cumulative_coadded_image is None:
                cumulative_coadded_image = shifted_image
            else:
                cumulative_coadded_image += shifted_image

            title = f"Cumulative Co-added Image After Adding {filename}"
            plot_with_dynamic_colorscale(cumulative_coadded_image, title)

        if cumulative_coadded_image is not None:
            output_path = os.path.join(output_dir_base, observation_date, f"{filter_type}_coadded_shifted.fit")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fits.PrimaryHDU(cumulative_coadded_image).writeto(output_path, overwrite=True)
            print(f"Final co-added image for date {observation_date} saved to {output_path}")

# Observing sessions
observing_sessions = ['10-03-24', '10-09-24', '10-18-24']

# Apply shifts and co-add for H-alpha
apply_shifts_and_coadd(observing_sessions, 'h_alpha')

# Apply shifts and co-add for Sloan-r
apply_shifts_and_coadd(observing_sessions, 'sloan_r')
