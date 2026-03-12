# NGC 6946 Star Formation Analysis
### Multi-Filter CCD Photometry with Python

**Technologies:** Python, Astropy, NumPy, Matplotlib, SciPy

This project investigates the **star formation activity and structural properties of NGC 6946** (the Fireworks Galaxy) using photometric observations in **Hα and Sloan r′ filters**. Data were collected over three nights in October 2024 using a 7-inch Astro-Physics refractor telescope with an SBIG ST-10XME CCD camera.

The Hα filter isolates emission from active star-forming regions, while the Sloan r′ filter captures light from older stellar populations and interstellar matter — enabling a direct comparison of recent star formation against the galaxy's broader structure.

---

# Data Processing Pipeline

## 1. Image Calibration
Raw CCD frames are calibrated using the standard reduction equation:

```
S_calibrated[x,y] = (S_raw[x,y] - (t_s / t_d) * D[x,y] - B[x,y]) / F[x,y]
```

where:
- `B[x,y]` — master bias frame (removes fixed-pattern readout noise)
- `D[x,y]` — master dark frame (removes thermal noise)
- `F[x,y]` — normalized master flat frame (corrects pixel sensitivity variations and dust)
- `t_s / t_d` — ratio of science to dark exposure times

Each master frame is produced by median-combining all frames of that type to suppress random noise.

---

## 2. Image Alignment and Co-addition
A two-stage alignment process handles both intra-night and inter-night positional shifts:

1. **Within each night:** Images are aligned to a reference frame by manually identifying the galaxy core and iteratively refining pixel shifts until stars are pinpoint and structural features are sharp.
2. **Across nights:** Night-level co-adds are aligned to a common reference and combined into a single high-SNR composite per filter.

Several images were excluded due to telescope pier collisions, satellite streaks, and a rotation artifact caused by the alignment algorithm converging on a phantom center. The final dataset used 15 Hα frames (Oct 3 + Oct 18) and 15 Sloan r′ frames (Oct 18 only).

---

## 3. Star Masking
Foreground stars outside the galaxy introduce brightness peaks that skew radial profiles. Stars are masked by:

1. Estimating the galaxy radius as 325 px from center
2. Measuring the median intensity of the inner 25 px of the galaxy core
3. Flagging any pixel beyond 325 px that exceeds 125% of that median as a star and setting it to zero

This approach references the galaxy's own brightness distribution, ensuring no diffuse galaxy emission is removed.

---

## 4. Galaxy Size Determination
The apparent size of NGC 6946 is determined from radial brightness profiles:

1. The galaxy center is identified by manually selecting coordinates 20 times in DS9 per filter and taking the mean
2. A sigma-clipping procedure (3σ, 5 iterations) is applied to pixels beyond 450 px to derive a clean background mean and standard deviation
3. The galaxy radius is defined as the first point where mean annular brightness falls below `background_mean + 1σ` and **remains below that threshold for 60 consecutive pixels** — preventing noise spikes or spiral arm gaps from triggering a premature edge detection

---

## 5. Uncertainty Analysis — Galaxy Size
Two independent sources of uncertainty are estimated and combined in quadrature:

```
σ_total = sqrt(σ_center² + σ_threshold²)
```

- **σ_center:** Monte Carlo simulation (50 trials) perturbing the galaxy center uniformly within the standard deviation of the 20 DS9 selections
- **σ_threshold:** Galaxy radius recalculated at ±10% of σ_background; uncertainty taken as the average deviation from the nominal radius

---

## 6. Aperture Photometry
Total flux is measured by summing background-subtracted pixel intensities within a circular aperture:

```
F = sum(I_aperture - I_background) / K_ccd
```

where:
- `I_aperture` — total ADU within the aperture
- `I_background` — median ADU of an annular background region (inner radius = r + 75 px, outer = r + 150 px)
- `K_ccd` — CCD gain factor converting ADU to electrons

Negative pixel values (from calibration residuals or cosmic rays) are replaced by the median of their 8 neighboring pixels before flux summation. Aperture sizes from 290 to 350 px (2 px steps, 30 trials) are tested to confirm flux stability.

---

## 7. Uncertainty Analysis — Flux
Total flux uncertainty combines photon noise and background noise in quadrature:

```
σ_total = sqrt(σ_photon² + σ_background²)
```

- **σ_photon:** Poisson noise from detected electrons within the aperture
- **σ_background:** Standard deviation of pixel intensities in the background annulus, scaled by aperture area

---

# Results

## Galaxy Size

| Filter | Galaxy Radius (pixels) |
|--------|----------------------|
| Hα | 261.00 ± 0.66 |
| Sloan r′ | 298.00 ± 2.35 |

NGC 6946 appears larger in Sloan r′ due to its sensitivity to older stellar populations and diffuse gas extending beyond the compact star-forming regions traced by Hα.

## Total Flux (aperture radius = 310 px)

| Filter | Total Flux (electrons) | Uncertainty (electrons) |
|--------|----------------------|------------------------|
| Hα | 1.875 × 10⁷ | 2.394 × 10⁴ |
| Sloan r′ | 2.289 × 10⁸ | 1.167 × 10⁵ |

The Sloan r′ flux is roughly 12× greater than Hα, reflecting the broader wavelength range captured and the larger contribution of older stellar populations to the galaxy's total emission.

---

# Key Findings
- NGC 6946 is measurably larger and brighter in Sloan r′ than in Hα, confirming that older stellar populations extend well beyond the active star-forming regions
- Hα emission is concentrated in the spiral arms and core, tracing ionized hydrogen from recently formed massive stars
- Flux growth curves plateau near the galaxy's outer edge in both filters, validating the chosen aperture size
