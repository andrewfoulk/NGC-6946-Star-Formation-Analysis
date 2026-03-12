import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Paths to the final co-added images
final_output_dir = "/Users/andrewfoulk/astr310/output/final_coadds"
ha_image_path = os.path.join(final_output_dir, "h_alpha_final_coadded.fit")
sloan_r_image_path = os.path.join(final_output_dir, "sloan_r_final_coadded.fit")

# Load the images
ha_image = fits.getdata(ha_image_path)
sloan_r_image = fits.getdata(sloan_r_image_path)

# Number of pixels to crop from each side
crop_top = 120    # Adjust as needed
crop_bottom = 120  # Adjust as needed
crop_left = 100   # Adjust as needed
crop_right = 100  # Adjust as needed

# Ensure that cropping does not exceed image dimensions
image_height, image_width = ha_image.shape
assert crop_top + crop_bottom < image_height, "Total vertical crop exceeds image height."
assert crop_left + crop_right < image_width, "Total horizontal crop exceeds image width."

# Crop the images
ha_image_cropped = ha_image[crop_top:image_height - crop_bottom, crop_left:image_width - crop_right]
sloan_r_image_cropped = sloan_r_image[crop_top:image_height - crop_bottom, crop_left:image_width - crop_right]

# Save the cropped images
cropped_output_dir = os.path.join(final_output_dir, "cropped")
os.makedirs(cropped_output_dir, exist_ok=True)

ha_cropped_path = os.path.join(cropped_output_dir, "h_alpha_final_coadded_cropped.fit")
sloan_r_cropped_path = os.path.join(cropped_output_dir, "sloan_r_final_coadded_cropped.fit")

fits.writeto(ha_cropped_path, ha_image_cropped, overwrite=True)
fits.writeto(sloan_r_cropped_path, sloan_r_image_cropped, overwrite=True)

print(f"Cropped Hα image saved to {ha_cropped_path}")
print(f"Cropped Sloan r' image saved to {sloan_r_cropped_path}")

# Define the new plotting function
def plot_with_dynamic_colorscale(image, title):
    """
    Plots the image with a dynamic color scale based on ADU values.
    """
    # Calculate dynamic color scale bounds
    vmin = np.percentile(image, 20)  # Lower bound: 20th percentile
    vmax = np.percentile(image, 95)  # Upper bound: 95th percentile

    # Create the plot
    plt.figure(figsize=(8, 8))
    fig = plt.imshow(image, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(fig, fraction=0.036, pad=0.04, label='Pixel Value (ADU)')

    # Add annotations and formatting
    plt.title(f"{title}")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()

# Plot the cropped images using the new plotting function
plot_with_dynamic_colorscale(ha_image_cropped, "Hα Image")
plot_with_dynamic_colorscale(sloan_r_image_cropped, "Sloan r' Image")
