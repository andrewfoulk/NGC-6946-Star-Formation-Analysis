import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
import warnings

# File paths to the cropped Hα and Sloan-R images of NGC 6946
ha_image_path = "/Users/andrewfoulk/astr310/output/final_coadds/cropped/h_alpha_final_coadded_cropped.fit"
sloan_r_image_path = "/Users/andrewfoulk/astr310/output/final_coadds/cropped/sloan_r_final_coadded_cropped.fit"

# Load Hα and Sloan-R images
ha_image = fits.getdata(ha_image_path)
sloan_r_image = fits.getdata(sloan_r_image_path)

# Define the galaxy center (using DS9 or another tool)
# Coordinates are in (row, column) format
galaxy_center_h_alpha = (588.3, 992.8)
galaxy_center_sloan_r = (609.3, 1000.6)

def plot_with_dynamic_colorscale(image, annulus_pixels_plot, galaxy_radius, title):
    """
    Plots the image with a dynamic color scale based on ADU values.
    """
    # Calculate dynamic color scale bounds
    finite_values = image[np.isfinite(image)]
    if len(finite_values) > 0:
        vmin = np.percentile(finite_values, 20)
        vmax = np.percentile(finite_values, 95)
    else:
        vmin, vmax = 0, 1  # Default values if no finite values are present

    # Create the plot
    plt.figure(figsize=(8, 8))
    fig = plt.imshow(image, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(fig, fraction=0.036, pad=0.04, label='Pixel Value (ADU)')

    # Overlay the annulus contour
    if annulus_pixels_plot is not None:
        plt.contour(annulus_pixels_plot, colors='cyan', linewidths=1)

    # Add annotations and formatting
    plt.title(f"{title}: Radius = {galaxy_radius:.2f} pixels")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()

def mask_stars(image, center, estimated_galaxy_radius=325, star_threshold_factor=1.25):
    """
    Masks stars in the image by identifying pixels brighter than a threshold outside the galaxy.
    """
    y_center, x_center = center

    # Calculate the radial distance from the galaxy center for each pixel
    y_indices, x_indices = np.indices(image.shape)
    r = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)

    # Create a mask for pixels inside the estimated galaxy radius
    galaxy_mask = r <= estimated_galaxy_radius

    # Create a mask to isolate pixels within a radius of 50 from the galaxy center
    galaxy_center_emission = r <= 25

    # Get the maximum pixel value within the galaxy
    median_galaxy_value = np.median(image[galaxy_center_emission])

    # Define the threshold for star detection
    star_threshold = median_galaxy_value * star_threshold_factor

    # Create a mask for pixels outside the galaxy that are brighter than the threshold
    star_mask = (image > star_threshold) & (~galaxy_mask)

    # Apply the star mask to the image
    image_masked = np.copy(image)
    image_masked[star_mask] = 0  # Replace star pixels with NaN

    return image_masked, star_mask

def radial_brightness_profile(image, r, max_radius):
    """
    Computes the radial brightness profile in a vectorized manner.
    """
    # Flatten the arrays
    r = r.flatten()
    image = image.flatten()

    # Mask invalid (NaN) pixels
    valid = np.isfinite(image)
    r_valid = r[valid]
    image_valid = image[valid]

    # Only consider pixels within max_radius
    within_radius = r_valid < max_radius
    r_valid = r_valid[within_radius]
    image_valid = image_valid[within_radius]

    # Convert radial distances to integer bin indices
    r_int = r_valid.astype(int)

    # Use np.bincount to sum the pixel values and counts per radius
    radial_sum = np.bincount(r_int, weights=image_valid, minlength=max_radius)
    radial_count = np.bincount(r_int, minlength=max_radius)

    # Avoid division by zero
    radial_mean = np.zeros(max_radius)
    nonzero = radial_count > 0
    radial_mean[nonzero] = radial_sum[nonzero] / radial_count[nonzero]
    radial_mean[~nonzero] = np.nan

    # The radial bins are integers from 0 to max_radius - 1
    return radial_mean

def find_radius_from_profile(radial_mean, threshold, radial_bins_centers, consecutive_required=60, r=None, max_radius=500):
    """
    Finds the galaxy radius based on the radial brightness profile and a threshold.
    """
    galaxy_radius = None
    annulus_pixels_plot = None
    consecutive_below_threshold = 0
    first_below_index = None

    for i, brightness in enumerate(radial_mean):
        if np.isnan(brightness):
            consecutive_below_threshold = 0
            first_below_index = None
            continue
        if brightness < threshold:
            if consecutive_below_threshold == 0:
                first_below_index = i
            consecutive_below_threshold += 1
            if consecutive_below_threshold == consecutive_required:
                galaxy_radius = radial_bins_centers[first_below_index]
                if r is not None:
                    annulus_mask = (r >= galaxy_radius) & (r < galaxy_radius + 3)
                    annulus_pixels_plot = annulus_mask
                break
        else:
            consecutive_below_threshold = 0
            first_below_index = None

    if galaxy_radius is None:
        galaxy_radius = max_radius

    return galaxy_radius, annulus_pixels_plot

def find_galaxy_radius_and_uncertainty(image, image_name, center, perturbation_x_range, perturbation_y_range, max_radius=450, num_trials=50, plot_annulus=True):
    """
    Determines the galaxy size and its uncertainty by moving from the center outwards.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    y_center, x_center = center  # (row, col)

    # Mask stars outside the galaxy
    image_masked, star_mask = mask_stars(image, center)

    # Plot the masked stars in black and white
    plt.figure(figsize=(8, 8))
    plt.imshow(star_mask, cmap='gray', origin='lower', interpolation='nearest', vmin=0, vmax=1)
    plt.title(f"Masked Stars for {image_name}")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.show()

    # Calculate the radial distance from the galaxy center for each pixel
    y_indices, x_indices = np.indices(image.shape)
    r = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)

    # Background estimation using pixels beyond max_radius and not masked
    background_mask = (r > max_radius) & (~np.isnan(image_masked))
    background_pixels = image_masked[background_mask]
    clipped_background = sigma_clip(background_pixels, sigma=3, maxiters=5)
    good_background_mask = ~clipped_background.mask

    background_mean = np.nanmean(clipped_background.data[good_background_mask])
    background_std = np.nanstd(clipped_background.data[good_background_mask])
    threshold = background_mean + background_std

    # Compute radial brightness profile
    radial_mean = radial_brightness_profile(image_masked, r, max_radius)
    radial_bins_centers = np.arange(max_radius)

    # Plot the radial brightness profile
    plt.figure(figsize=(10, 6))
    plt.plot(radial_bins_centers, radial_mean, label='Radial Brightness')
    plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Mean Brightness (ADU)')
    plt.title(f'Radial Brightness Profile for {image_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Find galaxy radius with original threshold
    galaxy_radius, annulus_pixels_plot = find_radius_from_profile(
        radial_mean, threshold, radial_bins_centers, r=r, max_radius=max_radius
    )

    # Plot the annulus if required
    if plot_annulus and annulus_pixels_plot is not None:
        plot_with_dynamic_colorscale(
            image_masked,
            annulus_pixels_plot,
            galaxy_radius,
            title=f"Galaxy Radius Determination for {image_name}"
        )

    # Calculate uncertainties due to threshold variation
    threshold_plus = threshold + 0.1 * background_std
    threshold_minus = threshold - 0.1 * background_std

    galaxy_radius_plus, _ = find_radius_from_profile(
        radial_mean, threshold_plus, radial_bins_centers, max_radius=max_radius
    )
    galaxy_radius_minus, _ = find_radius_from_profile(
        radial_mean, threshold_minus, radial_bins_centers, max_radius=max_radius
    )

    radius_uncertainty_threshold = (abs(galaxy_radius_plus - galaxy_radius) + abs(galaxy_radius - galaxy_radius_minus)) / 2

    # Monte Carlo simulation for uncertainty estimation due to center perturbation
    radii_samples = []

    for _ in range(num_trials):
        perturbed_center_y = y_center + np.random.uniform(0, perturbation_y_range)  # Perturb center y-coordinate
        perturbed_center_x = x_center + np.random.uniform(0, perturbation_x_range)  # Perturb center x-coordinate
        r_perturbed = np.sqrt((x_indices - perturbed_center_x)**2 + (y_indices - perturbed_center_y)**2)
        radial_mean_perturbed = radial_brightness_profile(image_masked, r_perturbed, max_radius)
        radius_perturbed, _ = find_radius_from_profile(
            radial_mean_perturbed, threshold, radial_bins_centers, max_radius=max_radius
        )
        radii_samples.append(radius_perturbed)

    if radii_samples:
        radius_uncertainty_center = np.std(radii_samples)
    else:
        radius_uncertainty_center = 0

    # Combine uncertainties using quadrature addition
    total_uncertainty = np.sqrt(radius_uncertainty_center**2 + radius_uncertainty_threshold**2)

    print(f"Uncertainty from threshold: {radius_uncertainty_threshold:.2f}")
    print(f"Uncertainty from center selection: {radius_uncertainty_center}")
    print(f"Galaxy radius: {galaxy_radius:.2f} ± {total_uncertainty:.2f} pixels")

    return galaxy_radius, total_uncertainty

# Analysis for Hα image with specific perturbations
ha_radius, ha_total_uncertainty = find_galaxy_radius_and_uncertainty(
    ha_image, "Hα Image", galaxy_center_h_alpha,
    perturbation_x_range=2.074, perturbation_y_range=1.851
)

# Analysis for Sloan-R image with specific perturbations
sloan_r_radius, sloan_r_total_uncertainty = find_galaxy_radius_and_uncertainty(
    sloan_r_image, "Sloan r' Image", galaxy_center_sloan_r,
    perturbation_x_range=3.574, perturbation_y_range=2.834
)
