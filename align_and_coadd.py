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

# Function to fine-tune the shift by checking small shifts around the initial shift
def fine_tune_shift(ref_image, shifted_image, max_shift=3):
    """Fine-tunes the alignment by checking small shifts around the initial shift."""
    best_shift = (0, 0)
    best_score = -np.inf

    shift_range = range(-max_shift, max_shift + 1)
    crop_margin = max_shift
    ref_crop = ref_image[crop_margin:-crop_margin, crop_margin:-crop_margin]

    for dy in shift_range:
        for dx in shift_range:
            fine_shifted_image = imshift(shifted_image, dy, dx)
            shifted_crop = fine_shifted_image[crop_margin:-crop_margin, crop_margin:-crop_margin]
            score = np.sum(ref_crop * shifted_crop)
            if score > best_score:
                best_score = score
                best_shift = (dy, dx)
    print(f"Fine-tuned shift: {best_shift}")
    return imshift(shifted_image, best_shift[0], best_shift[1])

# Function to display an image with dynamic color scaling
def show_image_with_dynamic_colorscale(image_data, title="", save_path=None):
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

# Function to align and co-add images within a single observing date
def align_and_coadd_images_click_center(image_dir, output_path, filter_type, crop_margin=50, zoom_margin=150):
    images = []
    shifted_images = []
    initial_center = None
    filenames = []

    def onclick(event):
        nonlocal ref_center
        x, y = int(event.xdata), int(event.ydata)
        print(f"Selected center: x={x}, y={y}")
        ref_center = (y, x)
        plt.close()

    max_shift = 5 if filter_type == 'h_alpha' else 15

    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".fit"):
            file_path = os.path.join(image_dir, filename)
            image_data = fits.getdata(file_path).astype(np.float64)
            images.append(image_data)
            filenames.append(filename)

            center_y, center_x = image_data.shape[0] // 2, image_data.shape[1] // 2
            y_min, y_max = max(0, center_y - zoom_margin), min(image_data.shape[0], center_y + zoom_margin)
            x_min, x_max = max(0, center_x - zoom_margin), min(image_data.shape[1], center_x + zoom_margin)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(image_data[y_min:y_max, x_min:x_max], cmap='gray', origin='lower',
                      vmin=np.percentile(image_data, 5), vmax=np.percentile(image_data, 95))
            plt.colorbar(ax.images[0], ax=ax)
            ax.set_title(f"Click to select center for: {filename} (Zoomed In)")
            ax.set_xlabel("X Pixel")
            ax.set_ylabel("Y Pixel")

            ref_center = None
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            if ref_center is None:
                print("Error: No center selected. Please click on the image.")
                return
            x_center, y_center = ref_center
            x_center += x_min
            y_center += y_min

            if initial_center is None:
                initial_center = (y_center, x_center)
                ref_image = image_data
                shifted_images.append(image_data)
                print(f"Image {filename} used as reference.")
                continue

            shift_rows = initial_center[0] - y_center
            shift_cols = initial_center[1] - x_center
            shifted_image = imshift(image_data, int(shift_rows), int(shift_cols))
            print(f"Image {filename} initial shift by ({shift_rows}, {shift_cols})")

            fine_shifted_image = fine_tune_shift(ref_image, shifted_image, max_shift=max_shift)
            shifted_images.append(fine_shifted_image)

    if not shifted_images:
        print("No images to co-add.")
        return

    final_image = np.sum(shifted_images, axis=0)
    final_image_cropped = final_image[crop_margin:-crop_margin, crop_margin:-crop_margin]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fits.PrimaryHDU(final_image_cropped).writeto(output_path, overwrite=True)
    print(f"Co-added image saved to {output_path}")

    title = f"Co-added Image for {os.path.basename(image_dir)}"
    show_image_with_dynamic_colorscale(final_image_cropped, title=title)
    return final_image_cropped

# Observing sessions and paths to the directories with images for each date
observing_sessions = ['10-03-24', '10-09-24', '10-18-24']

# Step 1: Align and co-add images for each observing date for both H-alpha and Sloan-r
for observation_date in observing_sessions:
    print(f"Processing date: {observation_date}")

    h_alpha_dir = os.path.join(base_path, 'h_alpha_calibrated', observation_date)
    sloan_r_dir = os.path.join(base_path, 'sloan_r_calibrated', observation_date)
    h_alpha_output_path = os.path.join(output_dir_base, observation_date, "h_alpha_coadded.fit")
    sloan_r_output_path = os.path.join(output_dir_base, observation_date, "sloan_r_coadded.fit")

    os.makedirs(os.path.dirname(h_alpha_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(sloan_r_output_path), exist_ok=True)

    align_and_coadd_images_click_center(h_alpha_dir, h_alpha_output_path, filter_type='h_alpha')
    align_and_coadd_images_click_center(sloan_r_dir, sloan_r_output_path, filter_type='sloan_r')
