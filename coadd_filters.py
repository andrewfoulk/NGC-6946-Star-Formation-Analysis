import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage.morphology import disk, closing

def imshift(im, nr, nc):
    """Shifts an image by nr rows and nc columns (positive or negative)."""
    return shift(im, shift=(nr, nc), mode='constant', cval=0)

def colorize_with_radius_limited_gamma(sloan_r_image, h_alpha_image, output_path_png, output_path_fits, 
                                       star_threshold=0.12, gas_threshold=0.4345, galaxy_threshold=0.12, 
                                       galaxy_radius=350, star_gamma=0.95, gamma_radius=200, 
                                       blue_intensity_factor=1, blue_red_mix=0.4, blue_green_mix=0.4,
                                       red_intensity_factor=1, red_green_mix=0, red_blue_mix=0.2,
                                       black_threshold=0.1):
    
    ref_center = [None, None]  # To store centers of each image

    # Callback for clicking on Sloan-r image
    def onclick_sloan(event):
        ref_center[0] = (int(event.ydata), int(event.xdata))
        print(f"Sloan-r center selected at: {ref_center[0]}")
        plt.close()  # Close Sloan-r image after the click

    # Callback for clicking on H-alpha image
    def onclick_halpha(event):
        ref_center[1] = (int(event.ydata), int(event.xdata))
        print(f"H-alpha center selected at: {ref_center[1]}")
        plt.close()  # Close H-alpha image after the click

    # Display the Sloan-r image for center selection
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sloan_r_image, cmap='gray', origin='lower',
              vmin=np.percentile(sloan_r_image, 5), vmax=np.percentile(sloan_r_image, 95))
    ax.set_title("Click on the center of the Sloan-r image")
    fig.canvas.mpl_connect('button_press_event', onclick_sloan)
    plt.show()

    # Ensure the Sloan-r center is selected
    if ref_center[0] is None:
        print("Error: Sloan-r center not selected. Please try again.")
        return

    # Display the H-alpha image for center selection
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(h_alpha_image, cmap='gray', origin='lower',
              vmin=np.percentile(h_alpha_image, 5), vmax=np.percentile(h_alpha_image, 95))
    ax.set_title("Click on the center of the H-alpha image")
    fig.canvas.mpl_connect('button_press_event', onclick_halpha)
    plt.show()

    # Ensure the H-alpha center is selected
    if ref_center[1] is None:
        print("Error: H-alpha center not selected. Please try again.")
        return

    # Calculate the shift based on clicked centers
    dy, dx = ref_center[0][0] - ref_center[1][0], ref_center[0][1] - ref_center[1][1]
    print(f"Calculated shift: (dy={dy}, dx={dx})")

    # Shift the H-alpha image to align with the Sloan-r image
    h_alpha_shifted = imshift(h_alpha_image, dy, dx)

    # Normalize images
    sloan_r_norm = (sloan_r_image - np.min(sloan_r_image)) / (np.max(sloan_r_image) - np.min(sloan_r_image))
    h_alpha_norm = (h_alpha_shifted - np.min(h_alpha_shifted)) / (np.max(h_alpha_shifted) - np.min(h_alpha_shifted))

    # Create the galaxy mask before gamma correction
    galaxy_mask = sloan_r_norm > galaxy_threshold  # Basic intensity threshold

    # Restrict galaxy mask to a circular region centered on the galaxy
    h, w = sloan_r_image.shape
    center = (h // 2, w // 2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    central_mask = dist_from_center <= galaxy_radius
    galaxy_mask &= central_mask  # Combine with intensity mask

    # Apply morphological closing to clean up the galaxy mask
    galaxy_mask = closing(galaxy_mask, disk(5))

    # Display the galaxy mask
    plt.figure(figsize=(8, 8))
    plt.imshow(galaxy_mask, cmap='gray', origin='lower')
    plt.title("Galaxy Mask")
    plt.show()

    # Create a radius-limited mask for applying gamma correction to stars
    gamma_mask = dist_from_center <= gamma_radius

    # Apply gamma correction within the radius-limited mask
    sloan_r_norm_gamma = np.where(gamma_mask, sloan_r_norm ** star_gamma, sloan_r_norm)

    # Create masks for high-intensity regions within the galaxy region
    sloan_r_mask = (sloan_r_norm_gamma > star_threshold) & galaxy_mask  # Mask for stars in galaxy region
    h_alpha_mask = (h_alpha_norm > gas_threshold) & galaxy_mask         # Mask for gas in galaxy region

    # Initialize RGB arrays for colorized images
    sloan_r_colored = np.zeros((sloan_r_image.shape[0], sloan_r_image.shape[1], 3))
    h_alpha_colored = np.zeros((h_alpha_image.shape[0], h_alpha_image.shape[1], 3))

    # Apply custom blue and red shades
    sloan_r_colored[..., 2] = blue_intensity_factor * sloan_r_norm_gamma * sloan_r_mask  # Blue channel
    sloan_r_colored[..., 0] = blue_red_mix * sloan_r_norm_gamma * sloan_r_mask           # Red tint for custom blue
    sloan_r_colored[..., 1] = blue_green_mix * sloan_r_norm_gamma * sloan_r_mask         # Green tint for custom blue

    h_alpha_colored[..., 0] = red_intensity_factor * h_alpha_norm * h_alpha_mask         # Red channel
    h_alpha_colored[..., 1] = red_green_mix * h_alpha_norm * h_alpha_mask                # Green tint for custom red
    h_alpha_colored[..., 2] = red_blue_mix * h_alpha_norm * h_alpha_mask                 # Blue tint for custom red

    # Display the individual colorized images with the galaxy mask applied
    plt.figure(figsize=(10, 10))
    plt.imshow(sloan_r_colored, origin='lower')
    plt.title("Radius-Limited Gradient Colorized Sloan-r Stars in Galaxy (Blue)")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(h_alpha_colored, origin='lower')
    plt.title("Gradient Colorized H-alpha Gas in Galaxy (Red)")
    plt.axis('off')
    plt.show()

    # Combine grayscale background with colorized stars and gas within the galaxy region
    composite_rgb = np.stack([sloan_r_norm, sloan_r_norm, sloan_r_norm], axis=-1)  # Grayscale background
    composite_rgb += sloan_r_colored + h_alpha_colored  # Add gradient colorized stars and gas in galaxy

    # Clip the composite to avoid any intensity overflow
    composite_rgb = np.clip(composite_rgb, 0, 1)

    # Apply the black background threshold
    composite_rgb[composite_rgb < black_threshold] = 0

    # Display and save the final co-added image
    plt.figure(figsize=(10, 10))
    plt.imshow(composite_rgb, origin='lower')
    plt.title("Co-added Image with Black Background")
    plt.axis('off')
    plt.savefig(output_path_png)
    plt.show()

    # Save the composite as a FITS file
    hdu = fits.PrimaryHDU(composite_rgb)
    hdu.writeto(output_path_fits, overwrite=True)

    print(f"Final colorized image saved as PNG to {output_path_png}")
    print(f"Final colorized image saved as FITS to {output_path_fits}")

# Paths to the final co-added H-alpha and Sloan-r images
sloan_r_path = '/Users/andrewfoulk/astr310/output/final_combined_sloan_r.fit'
h_alpha_path = '/Users/andrewfoulk/astr310/output/final_combined_h_alpha.fit'
output_path_png = '/Users/andrewfoulk/astr310/output/ngc6946_radius_limited_gradient_colorized_stars_gas_in_galaxy.png'
output_path_fits = '/Users/andrewfoulk/astr310/output/ngc6946_radius_limited_gradient_colorized_stars_gas_in_galaxy.fits'

# Load the images
sloan_r_image = fits.getdata(sloan_r_path, ignore_missing_simple=True)
h_alpha_image = fits.getdata(h_alpha_path, ignore_missing_simple=True)

# Colorize and co-add the images with a radius-limited gradient for stars within the galaxy region
colorize_with_radius_limited_gamma(sloan_r_image, h_alpha_image, output_path_png, output_path_fits)
