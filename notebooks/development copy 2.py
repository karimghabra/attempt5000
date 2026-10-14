# %% [markdown]
# # Burst pipeline: hold the vessels still, then keep only what changes
#
# A burst is a few seconds of camera frames showing blood vessels in the eye, dark on a bright background. The eye never holds still: it jitters by a few pixels from one frame to the next, and now and then it jumps by a hundred pixels or more (a saccade).
#
# This notebook takes one burst and
#
# 1. removes the eye's motion, so that a pixel shows the same spot of tissue in every frame, and
# 2. removes the static picture, so that what is left is what changes over time.
#
# | Step | What happens to the data | Result |
# |---|---|---|
# | 1. Load | The TIFF frames are read into one stack. | `raw_burst`, `fps` |
# | 2. Preview | A helper that plays any stack as a GIF. | `create_animation` |
# | 3. Gate | Blurred frames are dropped. | `gated_frames`, `sharpness`, `vesselness`, `gated_indices`, `frame_times` |
# | 4. Measure motion | The shift between consecutive frames is measured by phase correlation. | `step_shifts`, `frame_positions` |
# | 5. Split into fixations | The frame sequence is cut wherever the eye jumped. | `fixation_members` |
# | 6. Align each fixation | Every frame's shift from its fixation's reference frame is measured, and the aligned frames are averaged. | `fixations` |
# | 7. Align the fixations | Each fixation's average is mapped onto the largest fixation, the anchor. | `aligned_fixation_averages` |
# | 8. Remove dust and glare | Dust on the sensor is divided out of every frame, and glare is marked as no data. | `sensor_map`, `glare_pixels`, `clean_frame` |
# | 9. Stabilize | Every cleaned frame is resampled once, so that the vessels stay in place. | `stabilized_frames`, `stabilized_fixations` |
# | 10. Subtract the temporal mean | In each fixation, every pixel's average over time is removed. | `fixation_means`, `mean_subtracted_fixations` |
# | 11. Remove flicker | Each frame's overall brightness is evened out. | `brightness_factors`, `flicker_corrected_fixations` |
# | 12. Remove the background | The pixels that do not change over time are found. In each fixation a smooth background is fitted through them, and every frame is measured against it. | `quiet_pixels`, `background_fits`, `density_fixations` |
# | 13. Compare | The temporal mean subtraction is played with and without the background removal, side by side. | `density_means` |
# | 14. Follow the shading | The background, fitted without temporal data, is refitted to every frame. This removes the shadows that drift from frame to frame. | `frame_shadings`, `frame_centres` |
# | 15. Average and smear | For each fixation: what the temporal burst averages to over time (0), and how much it fluctuates at each pixel. | `burst_smears` |
# | 16. Clean the smear | The smear's own background is fitted and removed, and the result is compared with the background-cleaned average frame. | `smear_fits`, `cleaned_smears` |
# | 17. Sum and average | The background-cleaned average frame and the cleaned smear are added, each in multiples of its own background noise. | `combined_sums` |
# | 18. Vessel graph | A binary vessel mask is made from each of several pictures by one rule, its centerlines are traced, and the graphs are compared. | `traced`, `graph_table` |
# | 19. Contrast layer | The mean frame's density is put in multiples of the local noise. | `contrast_maps`, `valid_maps`, `quiet_maps` |
# | 20. Hypercolumns | 16 orientation neurons at 8 widths at every pixel answer to lines, not to edges, blobs or line ends. | `orientation_scores`, `radius_maps` |
# | 21. Interactions | The neurons sharpen each other's tuning, inhibit each other across a vessel and support each other along it; crests are read out. | `static_fields`, `static_crests` |
# | 22. Chains | Crest neurons are strung into chains without a mask or a skeleton, gaps are bridged, and the pruning is taken from a plateau. | `static_chains`, `pruning_table` |
# | 23. Is the motion visible? | The line mass along a chain in one frame is correlated with the next frame, shifted along the chain. | `flow_visible` |
# | 24. Flow field | The hypercolumns are given the change between frames; where things keep changing is traced like the static field. | `flow_energy`, `flow_chains` |
# | 25. Signals per chain | Kymograph, velocity, pulse, blur and lucky frames of every chain. | `segment_signals` |
# | 26. One field, one graph | Both fields are compared and combined; the chains become a graph whose junction arms are read on a ring outside the core. | `vessel_chains`, `vessel_graphs` |
# | 27. Junctions | Every way of dividing a junction's arms into vessels is scored; the most specific one that fits is chosen. | `junctions` |
# | 28. Synthetic bursts | The same code reads bursts made with everything known, and is compared with the truth. | printed numbers |
# | 29. Vessels from the static score | Centerlines are read straight from the ridges of section 20's score, fitted with splines, and given their flow: flow data, velocity and direction. Junctions are read with flow as evidence. | `flow_masks`, `vessel_lines`, `spline_vessels`, `ridge_junctions` |
# | 30. Harness | Every way of reading a burst is scored on synthetic bursts with the truth known, with 95% intervals, on bursts for developing and on bursts held out. | `harness`, `ablation`, `harness_junctions`, `harness_arms` |
# | 31. Tube rule | A rule that drops only pieces of centerline with a free end, by their length outside their parent's tube. Tried, not adopted. | `tube_sweep` |
# | 32. Whole vessels | Pieces of one vessel are joined, and one speed is read per whole vessel, with an interval and a flag above the trackable limit. | `whole_graphs`, `whole_vessels`, `whole_readings` |
# | 33. Flow evidence | How reliable a measured direction is, without truth; flow as a likelihood ratio between two readings of a junction; the trunk of a bifurcation by direction. | `direction_reliability`, `evidence_junctions` |
# | 34. Ablation table | Every row of the harness, and what sections 32 and 33 have to show where the truth is known. | `harness`, `ablation` |
#
# **Terms used below**
#
# - **Fixation**: a run of frames between two saccades, in which the eye only jitters.
# - **Reference frame**: the one frame of a fixation that its other frames are aligned to.
# - **Anchor**: the fixation with the most frames. Its reference frame is the common frame for the whole burst.
# - **Shift**: `(dy, dx)` in pixels, rows first.
# - **DN**: digital number, the camera's raw intensity unit.
# - **Optical density**: `B - ln(I)`, how far a pixel's log intensity lies below the fitted background `B`. About 0 on background, positive in vessels.
# - **Quiet pixel**: a pixel whose brightness changes over time by no more than the camera noise does. Quiet pixels are background.
#
# Run the cells from top to bottom. Previews are saved next to the notebook under a name that includes a hash of their content, so a later run finds them instead of rendering them again.

# %% [markdown]
# ## Setup

# %%
import csv
import functools
import hashlib
import itertools
import json
import os
import pickle
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.fft
import scipy.signal
import scipy.sparse.csgraph
import scipy.spatial
import scipy.special
import scipy.stats
import scipy.interpolate
from IPython.display import Image as IPImage, Video, display
from PIL import Image
from scipy.ndimage import distance_transform_edt, percentile_filter
from skimage.filters import apply_hysteresis_threshold, frangi, laplace
from skimage.morphology import remove_small_objects, skeletonize
from skimage.transform import downscale_local_mean

# %% [markdown]
# ## 1. Load the burst
#
# The burst is the first `burst_*` folder found next to the notebook or in one of its parent folders. Its TIFF files are read in name order into one array. Nothing is changed here: the values are the camera's own.
#
# **Result:** `raw_burst`, shape `(frames, rows, columns)`, `uint16`.

# %%
project_dir = Path.cwd()
search_roots = (project_dir, *project_dir.parents)
burst_folders = []
for search_root in search_roots:
    burst_folders = sorted(path for path in search_root.glob("burst_*") if path.is_dir())
    if burst_folders:
        project_dir = search_root
        break
if not burst_folders:
    raise FileNotFoundError(
        f"No burst_* directory found in {Path.cwd()} or its parent directories. "
        "Use a local Python kernel with the burst folder in the workspace."
    )

target_folder = burst_folders[0]
tiff_files = sorted(target_folder.glob("*.tif*"))
if not tiff_files:
    raise FileNotFoundError(f"No TIFF files found in {target_folder}")

print(f"Loading {len(tiff_files)} TIFF frames from {target_folder}...")
with Image.open(tiff_files[0]) as image:
    first_frame = np.asarray(image)
    raw_burst = np.empty((len(tiff_files), *first_frame.shape), dtype=first_frame.dtype)
    raw_burst[0] = first_frame

for index, file_path in enumerate(tiff_files[1:], start=1):
    with Image.open(file_path) as image:
        raw_burst[index] = np.asarray(image)

print(f"Loaded burst with shape {raw_burst.shape} and dtype {raw_burst.dtype}.")

# %% [markdown]
# The frame rate comes from the recorder's `manifest.json`. It is only used to play the previews at the speed of the recording.
#
# **Result:** `fps`.

# %%
manifest_path = target_folder / "manifest.json"
if manifest_path.exists():
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest_data = json.load(manifest_file)
    capture_data = manifest_data.get("capture", {})
    requested_data = manifest_data.get("acquisition_requested", {})
    fps = capture_data.get("effective_fps", requested_data.get("requested_fps", 15.0))
    print(f"Using {fps:.3f} FPS from the capture manifest.")
else:
    fps = 15.0
    print(f"manifest.json not found in {target_folder}; using {fps:.1f} FPS.")

# %% [markdown]
# ## 2. A way to look at a stack
#
# `create_animation` plays any `(frames, rows, columns)` stack as a GIF or MP4. It changes what you see, never the data:
#
# - the stack's full range, from its smallest to its largest value, is stretched to 8 bits;
# - frames are reduced to `width` pixels across;
# - the file name carries a hash of the frames, so each distinct stack is rendered once and found again on later runs.

# %%
def create_animation(frames, output_filename="burst_animation.gif", fps=30.0, embed_html5=True, width=800):
    """Save a grayscale frame stack as a cached GIF or MP4 and display it."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError("frames must be a non-empty NumPy array with shape (N, H, W)")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    base, ext = os.path.splitext(output_filename)
    ext = ext.lower()
    if ext not in {".gif", ".mp4"}:
        raise ValueError("Output filename must end with either '.gif' or '.mp4'")

    # Output is at most `width` pixels wide, with even dimensions as H.264 requires.
    width = max(2, 2 * round(min(width, frames.shape[2]) / 2))
    height = max(2, 2 * round(width * frames.shape[1] / frames.shape[2] / 2))

    # SHA-1 is only a cache key here; it hashes the stack several times faster than SHA-256.
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(memoryview(frames))
    digest.update(f"{frames.shape}:{frames.dtype}:{fps}:{ext}:{width}x{height}".encode("utf-8"))
    cached_filename = f"{base}_{digest.hexdigest()[:16]}{ext}"

    if os.path.exists(cached_filename):
        print(f"Found cached animation: {cached_filename}")
    else:
        print(f"Rendering {len(frames)} frames at {fps:.2f} FPS...")
        # Frames are scaled to 8-bit over the stack's full range and handed straight to the
        # encoder; drawing each one through matplotlib took minutes on this machine.
        low, high = float(frames.min()), float(frames.max())
        scale = 255.0 / max(high - low, 1e-8)
        preview = np.empty((len(frames), height, width), dtype=np.uint8)
        for index, frame in enumerate(frames):
            small = cv2.resize(frame.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
            preview[index] = np.clip((small - low) * scale + 0.5, 0, 255)

        if ext == ".gif":
            images = [Image.fromarray(frame) for frame in preview]
            images[0].save(
                cached_filename, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0
            )
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                 "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                 "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", cached_filename],
                input=preview.tobytes(), check=True,
            )
        print(f"Saved animation to {cached_filename}")

    if ext == ".mp4":
        display(Video(cached_filename, embed=embed_html5, width=640, height=400))
    else:
        display(IPImage(filename=cached_filename))

# %% [markdown]
# The raw burst. Watch for the jitter, and for any sudden jump (a saccade).

# %%
create_animation(raw_burst, output_filename="burst_preview.gif", fps=fps)

# %% [markdown]
# ## 3. Gate: drop the blurred frames
#
# Some frames are blurred, for example when the eye jumps during the exposure. They are removed before anything is aligned or averaged.
#
# Each frame gets two scores:
#
# - **Sharpness**: the variance of the frame's Laplacian. It is high when edges are crisp.
# - **Vesselness**: the mean Frangi response to dark ridges. It is high when vessels stand out.
#
# Both are computed on a reduced copy of the frame (the mean of every 4 × 4 block, `downsample`), divided by its mean brightness so that exposure changes do not move the scores. The reduction matters: at full resolution both scores measure sensor noise and do not respond to blur.
#
# A frame is kept when both scores reach 80% of the burst's median (`sharpness_cutoff`, `vesselness_cutoff`).

# %%
def score_frame(frame, downsample=4, sigmas=(1, 2, 3)):
    """Return the Laplacian-variance sharpness and mean Frangi vesselness of one frame."""
    # Scored as in ingest_stabilize.gate_frames, on downsample x downsample block means: at full resolution
    # both scores measure sensor noise and do not respond to blur. sigmas are in downsampled pixels.
    small = downscale_local_mean(frame, (downsample, downsample))
    small = small / small.mean()  # divide out brightness so exposure changes don't move the scores
    return laplace(small).var(), frangi(small, sigmas=sigmas, black_ridges=True).mean()


def gate_burst_frames(burst_array, downsample=4, sigmas=(1, 2, 3), sharpness_cutoff=0.8, vesselness_cutoff=0.8):
    """Keep the frames whose sharpness and vesselness both reach the given fraction of the burst median.

    Returns the kept frames, their two scores, the two thresholds, and each kept frame's number in the burst."""
    num_frames = len(burst_array)
    workers = max(1, (os.cpu_count() or 2) // 2)
    print(f"Scoring {num_frames} frames for sharpness and vesselness on {workers} threads...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(lambda frame: score_frame(frame, downsample, sigmas), burst_array))
    all_sharpness, all_vesselness = np.array(scores).T

    sharpness_threshold = sharpness_cutoff * np.median(all_sharpness)
    vesselness_threshold = vesselness_cutoff * np.median(all_vesselness)
    keep = (all_sharpness >= sharpness_threshold) & (all_vesselness >= vesselness_threshold)
    keep_indices = np.where(keep)[0]

    gated_frames = burst_array[keep_indices]
    sharpness_scores = all_sharpness[keep_indices]
    vesselness_scores = all_vesselness[keep_indices]

    print(f"Original frames: {num_frames} | Gated frames kept: {len(gated_frames)} | "
          f"dropped: {np.where(~keep)[0].tolist()}")
    print(f"Sharpness threshold: {sharpness_threshold:.3e} | Vesselness threshold: {vesselness_threshold:.3e}")

    return gated_frames, sharpness_scores, vesselness_scores, (sharpness_threshold, vesselness_threshold), keep_indices

# %% [markdown]
# **Result:** `gated_frames`, the kept frames in their original order, with their scores in `sharpness` and `vesselness`. `gated_indices` holds each kept frame's number in the recorded burst, and `frame_times` the time it was recorded at (seconds from the first frame). Where a frame was dropped, the time between two kept frames is longer.
#
# From here on, a frame number is a position in `gated_frames`, not in the original burst.

# %%
gated_frames, sharpness, vesselness, thresholds, gated_indices = gate_burst_frames(raw_burst)
frame_times = gated_indices / fps  # seconds from the first frame of the burst; dropped frames leave gaps

# render the gated frames as a GIF to visualize the effect of the gating
create_animation(gated_frames, output_filename="gated_burst_preview.gif", fps=fps)

# %% [markdown]
# ## 4. Measure the motion between frames
#
# Motion is measured by phase correlation. Shifting an image leaves the magnitude of its Fourier transform alone and adds a phase that grows linearly with frequency (the Fourier shift theorem). The shift between two frames can therefore be read from the phase difference of their spectra:
#
# 1. `frame_spectrum` removes the frame's mean, tapers its edges with a Hann window and takes the Fourier transform.
# 2. `phase_shift` multiplies one spectrum by the conjugate of the other and scales every frequency to unit magnitude. Only the phase ramp of the shift is left.
# 3. The ramp is low-passed and transformed back. The result is one Gaussian-shaped peak whose position is the shift. A three-point fit along each axis places it to a fraction of a pixel.
#
# `phase_shift` returns the shift as `(dy, dx)`, together with a coherence between 0 and 1: the height of the peak relative to that of a perfect match.

# %%
PEAK_SIGMA = 1.5  # pixels; width given to the correlation peak by low-passing the phase ramp

num_frames, frame_height, frame_width = gated_frames.shape
frame_shape = (frame_height, frame_width)
workers = max(1, (os.cpu_count() or 2) // 2)  # threads for the per-frame work


@functools.lru_cache(maxsize=None)
def correlation_filters(shape):
    """Return the Hann window, the low-pass for the phase ramp and its peak height for images of one shape."""
    window = np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)
    frequency_y = scipy.fft.fftfreq(shape[0])[:, None]
    frequency_x = scipy.fft.rfftfreq(shape[1])[None, :]
    lowpass = np.exp(-2 * np.pi**2 * PEAK_SIGMA**2 * (frequency_y**2 + frequency_x**2)).astype(np.float32)
    return window, lowpass, scipy.fft.irfft2(lowpass, s=shape)[0, 0]


# compute the fast fourier transform of all frames
# (a spectrum takes ~0.05 s, so each is recomputed where it is used instead of holding 1.2 GB of them)
def frame_spectrum(frame):
    """Return the half-plane FFT of a mean-subtracted, Hann-windowed frame."""
    frame = frame.astype(np.float32)
    return scipy.fft.rfft2((frame - frame.mean()) * correlation_filters(frame.shape)[0])


# use fourier shift theorem and normalized cross power spectrum to
# compute the phase ramp, and use the inverse fourier transform to compute the subpixel shift between frames
def phase_shift(reference_spectrum, moving_spectrum, shape=None):
    """Return the (dy, dx) shift of the moving image from the reference, and the peak coherence (0 to 1)."""
    shape = shape or frame_shape  # the images are full frames unless a tile shape is given
    _, lowpass, peak_scale = correlation_filters(shape)
    cross_power = moving_spectrum * np.conj(reference_spectrum)
    cross_power /= np.abs(cross_power) + 1e-12  # unit magnitude leaves only the phase ramp of the shift
    # Low-passing the ramp drops the noise-dominated frequencies and turns the correlation
    # peak into a Gaussian, which a 3-point fit along each axis then locates below a pixel.
    correlation = scipy.fft.irfft2(cross_power * lowpass, s=shape)
    peak = np.unravel_index(np.argmax(correlation), shape)

    shift = np.zeros(2)
    for axis, size in enumerate(shape):
        line = correlation[:, peak[1]] if axis == 0 else correlation[peak[0], :]
        samples = line.take([peak[axis] - 1, peak[axis], peak[axis] + 1], mode="wrap")
        below, centre, above = np.log(np.maximum(samples, 1e-12))
        curvature = below - 2 * centre + above
        offset = 0.5 * (below - above) / curvature if curvature < 0 else 0.0
        position = peak[axis] + float(np.clip(offset, -0.5, 0.5))
        shift[axis] = position - size if position > size / 2 else position
    return shift, float(correlation[peak] / peak_scale)

# %% [markdown]
# The shift is measured between every frame and the one before it. Adding the steps up gives the path of the eye over the burst.
#
# **Result:** `step_shifts` (one `(dy, dx)` per frame), `step_sizes` (their lengths in pixels) and `frame_positions` (the running sum of the steps).

# %%
def step_shift(index):
    return phase_shift(frame_spectrum(gated_frames[index - 1]), frame_spectrum(gated_frames[index]))


with ThreadPoolExecutor(max_workers=workers) as pool:
    steps = list(pool.map(step_shift, range(1, num_frames)))
step_shifts = np.array([(0.0, 0.0), *(shift for shift, _ in steps)])  # row i: shift of frame i from frame i - 1
step_sizes = np.hypot(step_shifts[:, 0], step_shifts[:, 1])
frame_positions = np.cumsum(step_shifts, axis=0)

# %% [markdown]
# ## 5. Split the burst into fixations
#
# Ordinary steps are a few pixels long. A step that is more than `GREAT_SHIFT_FACTOR` (10) times the median step, and more than `GREAT_SHIFT_FLOOR` (20) pixels, is treated as a saccade. The frame sequence is cut at every saccade, and each piece is one fixation.
#
# **Result:** `fixation_members`, a list holding the frame numbers of each fixation.

# %%
GREAT_SHIFT_FACTOR = 10.0  # a frame-to-frame shift this many times the burst's median shift is "great"...
GREAT_SHIFT_FLOOR = 20.0   # ...as long as it is also at least this many pixels

# when subpixel shifts are great, break the frame sequence into multiple fixations
great_shift = max(GREAT_SHIFT_FACTOR * np.median(step_sizes[1:]), GREAT_SHIFT_FLOOR)
fixation_members = np.split(np.arange(num_frames), np.flatnonzero(step_sizes > great_shift))
print(f"Frame-to-frame shift: median {np.median(step_sizes[1:]):.2f} px, largest {step_sizes.max():.2f} px | "
      f"great-shift limit {great_shift:.1f} px -> {len(fixation_members)} fixation(s)")

# %% [markdown]
# ## 6. Align the frames of each fixation and average them
#
# Within a fixation the eye only jitters, so shifting the frames is enough to line them up. For each fixation:
#
# 1. **Choose the reference frame.** It is the frame with the best quality, where quality is sharpness plus vesselness, each standardized over the gated burst. If no frame of the fixation is better than the burst's average, the frame closest to the fixation's median position is used instead.
# 2. **Measure each frame's shift from the reference.** This is a phase correlation directly against the reference frame, not a sum of the steps from section 4.
# 3. **Average the aligned frames.** Each frame is shifted back onto the reference with cubic interpolation. A frame only counts where it really has data, so the border of the average is made from fewer frames.
#
# **Result:** `fixations`, one dictionary per fixation:
#
# | Key | Content |
# |---|---|
# | `frames` | its frame numbers (positions in `gated_frames`) |
# | `times` | when each of its frames was recorded, in seconds |
# | `reference` | the frame number of its reference frame |
# | `shifts` | `(dy, dx)` of each frame from the reference |
# | `coherence` | how well each frame matched the reference, from 0 to 1 |
# | `average` | the mean of the aligned frames |
# | `coverage` | the number of frames averaged at each pixel |

# %%
# choose the reference frame according to the highest sharpness and vesselness score, or the median frame if all scores are low
# (both scores are standardized over the gated burst and summed; "low" means no frame of the fixation beats
# the burst average, and the median frame is then the one closest to the fixation's median position)
def standardize(scores):
    return (scores - scores.mean()) / (scores.std() + 1e-12)


def covered_area(dy, dx, margin=0):
    """Return the (rows, columns) of the reference frame that a frame shifted by (dy, dx) really covers."""
    rows = slice(max(0, int(np.ceil(margin - dy))),
                 min(frame_height, int(np.floor(frame_height - 1 - margin - dy)) + 1))
    columns = slice(max(0, int(np.ceil(margin - dx))),
                    min(frame_width, int(np.floor(frame_width - 1 - margin - dx)) + 1))
    return rows, columns


frame_quality = standardize(sharpness) + standardize(vesselness)

fixations = []
for members in fixation_members:
    reference = members[np.argmax(frame_quality[members])]
    if frame_quality[reference] <= 0:
        distances = np.linalg.norm(frame_positions[members] - np.median(frame_positions[members], axis=0), axis=1)
        reference = members[np.argmin(distances)]

    # each fixation is a set of frames that are aligned to a single reference frame, and the fixations are then aligned to each other
    reference_spectrum = frame_spectrum(gated_frames[reference])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        matches = list(pool.map(
            lambda index: phase_shift(reference_spectrum, frame_spectrum(gated_frames[index])), members
        ))
    shifts = np.array([shift for shift, _ in matches])

    # compute the average of each fixation, once fixation frames are aligned to their reference
    total = np.zeros(frame_shape, dtype=np.float32)
    coverage = np.zeros(frame_shape, dtype=np.uint16)
    for index, (dy, dx) in zip(members, shifts):
        aligned = cv2.warpAffine(
            gated_frames[index].astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]),
            (frame_width, frame_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        # Only the pixels that the shifted frame really covers count towards the average.
        area = covered_area(dy, dx)
        total[area] += aligned[area]
        coverage[area] += 1

    fixations.append({
        "frames": members,  # indices into gated_frames
        "times": frame_times[members],  # seconds; when each frame was recorded (gaps where frames were dropped)
        "reference": reference,
        "shifts": shifts,  # (dy, dx) of each frame from the reference
        "coherence": np.array([coherence for _, coherence in matches]),
        "average": total / np.maximum(coverage, 1),
        "coverage": coverage,  # number of frames averaged at each pixel
    })
    print(f"Fixation {len(fixations) - 1}: frames {members[0]}-{members[-1]} ({len(members)} frames), "
          f"reference {reference}, largest shift from reference {np.linalg.norm(shifts, axis=1).max():.1f} px")

# %% [markdown]
# ## 7. Align the fixations to each other
#
# After a saccade the view differs by more than a shift, so a translation alone does not line two fixations up. The fixation with the most frames is the anchor, and the average of every other fixation is mapped onto the anchor's average:
#
# 1. Phase correlation gives a starting translation.
# 2. ECC (enhanced correlation coefficient, `cv2.findTransformECC`) refines it into an affine map. ECC is given only the fine detail of the averages: the structure smaller than about `DETAIL_SIGMA` pixels. Between fixations the thick out-of-focus vessels shift against the sharp ones, and they would otherwise pull the fit away from the sharp ones.
# 3. The fit is judged by its local misfit: the two averages are compared tile by tile, and the misfit is the median shift still left between them. If the affine map leaves more than `MISFIT_TOLERANCE` pixels, a homography (8 parameters) is tried as well. A fit is kept only if it lowers the misfit.
#
# **Result:** `aligned_fixation_averages`, the averages resampled into the anchor's frame, and three more entries in each fixation's dictionary:
#
# | Key | Content |
# |---|---|
# | `to_anchor` | the map from a point `(x, y)` in the anchor's frame to the same point in this fixation's reference frame (2 × 3 for a translation or an affine map, 3 × 3 for a homography) |
# | `model` | `"translation"`, `"affine"` or `"homography"`, whichever was kept |
# | `misfit` | its median local misfit in pixels |

# %%
DETAIL_SIGMA = 2.0      # pixels; ECC fits the structure finer than this, i.e. the in-focus vessels
MISFIT_TILE = 256       # pixels; side of the tiles on which the leftover local shift is measured
MISFIT_TOLERANCE = 0.5  # pixels; a median leftover shift above this makes the affine insufficient

# then try to identify an affine transformation that aligns average fixation frames to each other
# (every average is mapped onto the fixation with the most frames: phase correlation gives the starting
# translation and ECC refines it. ECC is given only the fine detail, because between fixations the thick
# out-of-focus vessels shift against the sharp ones and would otherwise pull the fit away from them.)
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
anchor = int(np.argmax([len(fixation["frames"]) for fixation in fixations]))
anchor_average = fixations[anchor]["average"]
anchor_spectrum = frame_spectrum(anchor_average)


def fine_detail(image):
    """Keep the structure finer than DETAIL_SIGMA, which is what the frame alignment locks onto."""
    return image - cv2.GaussianBlur(image, (0, 0), DETAIL_SIGMA)


def warp_to_anchor(average, transform):
    """Resample an average into the anchor's frame; return it with its median local misfit in pixels."""
    warp = cv2.warpAffine if len(transform) == 2 else cv2.warpPerspective
    size = (frame_width, frame_height)
    warped = warp(average, transform, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    inside = warp(
        np.ones(frame_shape, dtype=np.uint8), transform, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    ) > 0
    # The misfit is the shift still left between the two images, measured on half-overlapping tiles.
    residuals = []
    for top in range(0, frame_height - MISFIT_TILE + 1, MISFIT_TILE // 2):
        for left in range(0, frame_width - MISFIT_TILE + 1, MISFIT_TILE // 2):
            tile = (slice(top, top + MISFIT_TILE), slice(left, left + MISFIT_TILE))
            if inside[tile].all():
                shift, _ = phase_shift(
                    frame_spectrum(anchor_average[tile]), frame_spectrum(warped[tile]), (MISFIT_TILE, MISFIT_TILE)
                )
                residuals.append(np.hypot(shift[0], shift[1]))
    return warped, float(np.median(residuals)) if residuals else np.inf


anchor_detail = fine_detail(anchor_average)
aligned_fixation_averages = np.empty((len(fixations), *frame_shape), dtype=np.float32)
for number, fixation in enumerate(fixations):
    average = fixation["average"]
    (dy, dx), _ = phase_shift(anchor_spectrum, frame_spectrum(average))
    transform, model = np.float32([[1, 0, dx], [0, 1, dy]]), "translation"  # maps anchor (x, y) to this average's (x, y)
    aligned, misfit = warp_to_anchor(average, transform)
    if number != anchor:
        average_detail = fine_detail(average)
        tried = [f"translation {misfit:.2f} px"]
        # if the affine map is insufficient, we can use an enhanced cross correlation to find a warping transform
        # (ECC is already what fits the affine, so the richer warp it is asked for is its 8-parameter homography;
        # each fit is kept only if it lowers the misfit)
        for candidate_model, motion in (("affine", cv2.MOTION_AFFINE), ("homography", cv2.MOTION_HOMOGRAPHY)):
            if candidate_model == "homography" and misfit <= MISFIT_TOLERANCE:
                break
            start = transform if motion == cv2.MOTION_AFFINE else np.vstack([transform, [0, 0, 1]]).astype(np.float32)
            try:
                _, candidate = cv2.findTransformECC(
                    anchor_detail, average_detail, start.copy(), motion, ECC_CRITERIA, None, 5
                )
            except cv2.error:
                tried.append(f"{candidate_model} did not converge")
                continue
            candidate_aligned, candidate_misfit = warp_to_anchor(average, candidate)
            tried.append(f"{candidate_model} {candidate_misfit:.2f} px")
            if candidate_misfit < misfit:
                transform, model, aligned, misfit = candidate, candidate_model, candidate_aligned, candidate_misfit
        verdict = "within" if misfit <= MISFIT_TOLERANCE else "still above"
        print(f"Fixation {number} -> fixation {anchor}: local misfit {' -> '.join(tried)}; "
              f"kept the {model}, {verdict} the {MISFIT_TOLERANCE} px tolerance")
    aligned_fixation_averages[number] = aligned
    fixation.update({"to_anchor": transform, "model": model, "misfit": misfit})

# %% [markdown]
# Blinking between the aligned averages checks the alignment by eye: whatever jumps between the images is not aligned.

# %%
# blink between the aligned fixation averages to check the alignment by eye
create_animation(aligned_fixation_averages, output_filename="fixation_averages.gif", fps=2)

# %% [markdown]
# ## 8. Remove dust and glare
#
# Two things in the frames are not tissue. Both are dealt with here, before the frames are stabilized, so that every later step works on clean frames.
#
# - **Dust on the sensor** sits on the same sensor pixels in every frame, while the tissue moves under it. It dims the light by a fixed fraction, so it can be divided out.
# - **Glare** is light reflected off the wet surface. Where it clips, the pixel holds no tissue signal at all, so glare cannot be corrected. It is left out instead: those pixels count as "no data" in that frame.
#
# **Glare first.** In each frame a pixel is glare when
#
# - it is clipped: at least `SATURATION_LEVEL` of the burst's largest value, or
# - it is more than `GLARE_EXCESS` brighter than the bright tissue around it. That bright level (`bright_level`) is the 95th percentile of the brightness in a window around the pixel. Vessels are darker than it and open tissue just reaches it, so only glare stands clear of it.
#
# Glare is then grown by `GLARE_GROW` pixels to take in the halo around it. A pixel that is too bright in nearly every frame is not glare but a stuck sensor pixel, and is kept apart.
#
# **Result:** `glare_pixels`, for each frame the pixels left out as glare (as flat indices into the frame), and `stuck_pixels`, an image that is `True` at the stuck pixels.

# %%
SATURATION_LEVEL = 0.95  # fraction of the burst's largest value; from here up a pixel counts as clipped
GLARE_EXCESS = 0.08      # a pixel this much brighter than the bright tissue around it is glare as well
GLARE_GROW = 12          # pixels by which glare is grown, to take in its halo
STUCK_FRACTION = 0.9     # a pixel that is too bright in this fraction of the frames is stuck, not glare
BRIGHT_SCALE = 8         # the bright level of the tissue is read from frames reduced by this factor...
BRIGHT_WINDOW = 11       # ...over a window of this many reduced pixels each way

clipped_level = SATURATION_LEVEL * gated_frames.max()


def disc_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def bright_level(frame):
    """Return a smooth picture of how bright the tissue between the vessels is around each pixel."""
    reduced = cv2.resize(
        frame, (frame_width // BRIGHT_SCALE, frame_height // BRIGHT_SCALE), interpolation=cv2.INTER_AREA
    )
    # the 95th percentile sits on the open tissue: vessels are darker, and glare fills too little of the window
    level = cv2.GaussianBlur(percentile_filter(reduced, 95, size=BRIGHT_WINDOW, mode="nearest"), (0, 0), 1.5)
    return cv2.resize(level, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)


def too_bright(index):
    """Return the pixels of a frame that are clipped or stand out above the tissue, as flat indices."""
    frame = gated_frames[index].astype(np.float32)
    stands_out = cv2.GaussianBlur(frame, (0, 0), 1) > (1 + GLARE_EXCESS) * bright_level(frame)
    return np.flatnonzero((frame >= clipped_level) | stands_out)


with ThreadPoolExecutor(max_workers=workers) as pool:
    bright_pixels = list(pool.map(too_bright, range(num_frames)))

# a pixel that is too bright in nearly every frame is the sensor's doing (a hot pixel), not glare
times_bright = np.bincount(np.concatenate(bright_pixels), minlength=frame_height * frame_width)
stuck_pixels = (times_bright >= STUCK_FRACTION * num_frames).reshape(frame_shape)

glare_pixels = []  # per frame: the pixels left out as glare, as flat indices into the frame
for bright in bright_pixels:
    glare = np.zeros(frame_shape, dtype=np.uint8)
    glare.flat[bright] = 1
    glare[stuck_pixels] = 0
    glare_pixels.append(np.flatnonzero(cv2.dilate(glare, disc_kernel(GLARE_GROW))))

bright_counts = [len(bright) - stuck_pixels.sum() for bright in bright_pixels]
glare_counts = [len(glare) for glare in glare_pixels]
print(f"Glare: {np.median(bright_counts):.0f} pixels per frame are clipped or stand out above the tissue "
      f"({min(bright_counts)} to {max(bright_counts)}); grown by {GLARE_GROW} px, "
      f"{np.median(glare_counts):.0f} pixels per frame are left out "
      f"({np.median(glare_counts) / (frame_height * frame_width) * 100:.2f}% of a frame)")
print(f"Stuck pixels (too bright in at least {STUCK_FRACTION:.0%} of the frames): {stuck_pixels.sum()}"
      + "".join(f" at (x {x}, y {y})" for y, x in np.argwhere(stuck_pixels)[:5]))

# %% [markdown]
# **Then the sensor.** If a frame showed nothing but tissue, it would equal the tissue picture, moved to where that frame saw it and scaled by the frame's brightness. So each frame is divided by exactly that, and the ratios are averaged on the sensor, pixel by pixel. The tissue cancels, because it lies somewhere else on the sensor in every frame. What is left is each sensor pixel's own response: 1 for a sound pixel, less under dust.
#
# The tissue picture is the average of a fixation's frames, each shifted onto the reference frame as in section 6, with the glare left out. Dust leaves a dark trail on that picture, so the calculation runs twice: the second round also leaves out the sensor pixels that the first round found flawed.
#
# - Where the response is off by more than `DUST_DEPTH`, the spot is grown by `DUST_GROW` pixels and the frames are divided by the response there. Everywhere else the frames stay exactly as recorded.
# - A pixel that passes less than `DARKEST_GAIN` of the light cannot be restored. It is left out in every frame, like the stuck pixels.
#
# **Result:**
#
# - `sensor_map`: the response of every sensor pixel, exactly 1 where the sensor is sound.
# - `unusable_pixels`: `True` at the sensor pixels that hold no data in any frame.
# - `clean_frame(index)`: gated frame `index` with the sensor map divided out, and 0 where it holds no data (glare and unusable pixels). The cleaned frames are computed when they are asked for, instead of being stored as one more stack.
# - `tissue` and `frame_factors`: each fixation's tissue picture (`NaN` where no frame saw the tissue clear) and each frame's brightness factor.
# - `viewing_tissue`: the tissue pictures with those `NaN` spots smoothed over from their surroundings. The previews use it to show something in the pixels that were left out.
#
# From here on, 0 in a frame means "no data".

# %%
DUST_DEPTH = 0.03    # a sensor pixel whose response is off by more than this is flawed
DUST_GROW = 4        # pixels by which the flawed spots are grown, to take in their soft edges
DARKEST_GAIN = 0.25  # a pixel that passes less light than this cannot be restored and is left out instead
MIN_VIEWS = 0.25     # a pixel's response is trusted when this fraction of the frames could measure it


def glare_mask(index):
    """Return where a frame is left out as glare, as a boolean image."""
    mask = np.zeros(frame_shape, dtype=bool)
    mask.flat[glare_pixels[index]] = True
    return mask


def moved_to_reference(image, dy, dx):
    """Move a frame's image to where it lies in its fixation's reference frame (0 where the frame has nothing)."""
    return cv2.warpAffine(image, np.float32([[1, 0, -dx], [0, 1, -dy]]), (frame_width, frame_height))


def seen_from_frame(picture, dy, dx):
    """Move a picture in a fixation's reference frame to where a frame shifted by (dy, dx) saw it (NaN beyond it)."""
    return cv2.warpAffine(
        picture, np.float32([[1, 0, dx], [0, 1, dy]]), (frame_width, frame_height), borderValue=np.nan
    )


def tissue_pictures(left_out):
    """Average each fixation's frames on the tissue, without glare and without the sensor pixels in `left_out`."""
    pictures = []
    for fixation in fixations:
        total = np.zeros(frame_shape, dtype=np.float32)
        weight = np.zeros(frame_shape, dtype=np.float32)
        for index, (dy, dx) in zip(fixation["frames"], fixation["shifts"]):
            good = ~(left_out | glare_mask(index))
            total += moved_to_reference(np.where(good, gated_frames[index], 0).astype(np.float32), dy, dx)
            weight += moved_to_reference(good.astype(np.float32), dy, dx)
        pictures.append(np.where(weight >= 1, total / np.maximum(weight, 1), np.nan).astype(np.float32))
    return pictures


def sensor_response(pictures):
    """Return each sensor pixel's mean of frame / expected tissue, and every frame's brightness factor."""
    total = np.zeros(frame_shape, dtype=np.float32)
    views = np.zeros(frame_shape, dtype=np.float32)
    factors = np.ones(num_frames)
    sample = (slice(None, None, 4), slice(None, None, 4))  # every 4th pixel is plenty to fit one number per frame
    for fixation, picture in zip(fixations, pictures):
        for index, (dy, dx) in zip(fixation["frames"], fixation["shifts"]):
            expected = seen_from_frame(picture, dy, dx)  # the tissue as this frame saw it; NaN where unknown
            frame = gated_frames[index].astype(np.float32)
            usable = np.isfinite(expected) & ~glare_mask(index)
            expected[~usable] = 1
            fit = usable[sample]
            factors[index] = (np.dot(frame[sample][fit], expected[sample][fit])
                              / np.dot(expected[sample][fit], expected[sample][fit]))
            total += np.where(usable, frame / (factors[index] * expected), 0)
            views += usable
    response = np.where(views >= MIN_VIEWS * num_frames, total / np.maximum(views, 1), 1)
    return response.astype(np.float32), factors


# The tissue picture is needed to find the flawed sensor pixels, and the flawed pixels should stay out of the
# tissue picture, because dust leaves a dark trail on it. Two rounds settle this: the second leaves out what
# the first found.
flawed = stuck_pixels.copy()
for _ in range(2):
    tissue = tissue_pictures(left_out=flawed)
    response, frame_factors = sensor_response(tissue)
    flawed = cv2.dilate((np.abs(response - 1) > DUST_DEPTH).astype(np.uint8), disc_kernel(DUST_GROW)) > 0
    flawed |= stuck_pixels

sensor_map = np.where(flawed, response, 1).astype(np.float32)  # 1 wherever the sensor is fine
unusable_pixels = stuck_pixels | (sensor_map < DARKEST_GAIN)   # sensor pixels that hold no data in any frame
# for the previews: the tissue pictures, with the few spots that no frame saw clear smoothed over from around them
viewing_tissue = [
    cv2.inpaint(np.nan_to_num(picture), np.isnan(picture).astype(np.uint8), 5, cv2.INPAINT_TELEA) for picture in tissue
]


def clean_frame(index):
    """Return a gated frame with the sensor's flaws divided out, and 0 where it holds no data."""
    frame = np.rint(gated_frames[index] / sensor_map).astype(gated_frames.dtype)
    frame[unusable_pixels] = 0
    frame.flat[glare_pixels[index]] = 0
    return frame


off = (np.abs(sensor_map - 1) > DUST_DEPTH).astype(np.uint8)
spots, _, spot_stats, spot_centres = cv2.connectedComponentsWithStats(off)
print(f"Sensor map: {spots - 1} spots where the response is off by more than {DUST_DEPTH:.0%}; "
      f"{(sensor_map != 1).sum()} pixels are corrected and {unusable_pixels.sum()} left out as unusable")
for spot in np.argsort(-spot_stats[1:, cv2.CC_STAT_AREA])[:8] + 1:
    left, top, width, height, area = spot_stats[spot]
    inside = sensor_map[top:top + height, left:left + width]
    extreme = inside.flat[np.argmax(np.abs(inside - 1))]
    print(f"  (x {spot_centres[spot][0]:.0f}, y {spot_centres[spot][1]:.0f}): {area} px, "
          f"response {extreme:.2f} at its {'darkest' if extreme < 1 else 'brightest'}")

# %% [markdown]
# The burst before and after. The frames are shown as recorded, not stabilized, so the dust stays in place while the tissue moves.
#
# - **Top:** the whole frame, as recorded (left) and cleaned (right).
# - **Bottom left pair:** a close-up of the darkest dust spot at full resolution, as recorded and cleaned.
# - **Bottom right pair:** a close-up of the place most often under glare, as recorded and cleaned.
#
# For viewing only, a pixel that was left out shows what the other frames saw at that spot of tissue (`viewing_tissue`). In the data it stays 0.

# %%
PANEL_WIDTH = 480           # pixels; width of each whole-frame panel
CLOSEUP = PANEL_WIDTH // 2  # pixels; side of each close-up, shown at full resolution


def frame_for_viewing(index, number, dy, dx):
    """Return a cleaned frame of fixation `number`, its left-out pixels showing what the other frames saw there."""
    frame = clean_frame(index).astype(np.float32)
    expected = frame_factors[index] * seen_from_frame(viewing_tissue[number], dy, dx)
    expected = np.where(np.isfinite(expected), expected, gated_frames[index])  # beyond the picture: as recorded
    return np.where(frame > 0, frame, expected)


def closeup(row, column):
    """Return the window of CLOSEUP x CLOSEUP pixels around a point, kept inside the frame."""
    top = int(np.clip(row - CLOSEUP // 2, 0, frame_height - CLOSEUP))
    left = int(np.clip(column - CLOSEUP // 2, 0, frame_width - CLOSEUP))
    return slice(top, top + CLOSEUP), slice(left, left + CLOSEUP)


# the close-ups look at the darkest spot of the sensor map and at the place most often under glare
# (away from the frame's edge, so that the close-up is centred on it)
times_glare = np.bincount(np.concatenate(glare_pixels), minlength=frame_height * frame_width).reshape(frame_shape)
busiest = np.zeros(frame_shape, dtype=np.float32)
inner = (slice(CLOSEUP // 2, -CLOSEUP // 2), slice(CLOSEUP // 2, -CLOSEUP // 2))
busiest[inner] = cv2.blur(times_glare.astype(np.float32), (CLOSEUP // 4, CLOSEUP // 4))[inner]
windows = [closeup(*np.unravel_index(np.argmin(sensor_map), frame_shape)),
           closeup(*np.unravel_index(np.argmax(busiest), frame_shape))]

# render the burst as recorded next to the cleaned burst: the whole frame on top, the close-ups below
panel_height = round(PANEL_WIDTH * frame_height / frame_width)
preview_frames = np.empty((num_frames, panel_height + CLOSEUP, 2 * PANEL_WIDTH), dtype=np.uint16)
for number, fixation in enumerate(fixations):
    for index, (dy, dx) in zip(fixation["frames"], fixation["shifts"]):
        pair = (gated_frames[index], frame_for_viewing(index, number, dy, dx))  # as recorded, cleaned
        whole = [cv2.resize(image.astype(np.float32), (PANEL_WIDTH, panel_height), interpolation=cv2.INTER_AREA)
                 for image in pair]
        close = [image[window] for window in windows for image in pair]
        preview_frames[index] = np.vstack([np.hstack(whole), np.hstack(close)])
darkest = preview_frames.min()  # thin lines between the panels, at a value that leaves the gray scale alone
preview_frames[:, panel_height] = darkest
preview_frames[:, :panel_height, PANEL_WIDTH] = darkest
preview_frames[:, panel_height:, CLOSEUP::CLOSEUP] = darkest
create_animation(preview_frames, output_filename="dust_and_glare_removed_preview.gif", fps=fps, width=2 * PANEL_WIDTH)
del preview_frames
# %% [markdown]
# ## 9. Stabilize every frame
#
# Each frame now has two pieces of motion: its own shift within its fixation (section 6) and its fixation's map onto the anchor (section 7). They are combined into one transform, so every frame is resampled only once, with cubic interpolation.
#
# **Result:** the stabilized frames, in two layouts. Where a frame has no data, the value is 0: outside the frame, and where section 8 left pixels out. The cubic resampling blends a pixel without data into the 2 pixels around it, so those are left out as well.
#
# - `stabilized_frames` holds every cleaned frame (section 8) in the anchor's frame. It has the shape and type of `gated_frames`. Use it to look at the burst as a whole.
# - `stabilized_fixations` is a list with one stack per fixation. Each stack is aligned to its own reference frame only (section 6, without the map of section 7), so a fixation can be worked on without relying on the alignment between fixations.
#
# `stabilized_left_out` lists, for each frame of `stabilized_frames`, the pixels inside the frame that hold no data (as flat indices). The preview uses it.
#
# The anchor's entry in `stabilized_fixations` is a view of `stabilized_frames`, not a second copy. Copy it before changing it in place.

# %%
def anchor_to_reference(fixation):
    """Return the 3x3 map from anchor (x, y) to the same point's (x, y) in this fixation's reference frame."""
    to_reference = np.eye(3)
    to_reference[:len(fixation["to_anchor"])] = fixation["to_anchor"]
    return to_reference


def warp_frame(index, to_frame):
    """Resample a cleaned frame through a 3x3 map from output (x, y) to that frame's (x, y).

    Returns the resampled frame, and where inside that frame there is no data (the frame is 0 there)."""
    frame = clean_frame(index)
    size = (frame_width, frame_height)
    warped = cv2.warpPerspective(frame, to_frame, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    # the cubic resampling blends a pixel without data (0) into the 2 pixels around it, so those are left out too
    empty = cv2.warpPerspective(
        (frame == 0).astype(np.uint8), to_frame, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    left_out = cv2.dilate(empty, np.ones((5, 5), np.uint8)) > 0
    warped[left_out] = 0
    return warped, left_out


# a frame's own shift and its fixation's map onto the anchor are combined into one transform, so each frame
# is resampled only once. Pixels a frame does not cover stay 0.
stabilized_frames = np.zeros_like(gated_frames)
stabilized_fixations = []
stabilized_left_out = [None] * num_frames  # per frame of stabilized_frames: the pixels inside the frame without data
for number, fixation in enumerate(fixations):
    members = fixation["frames"]
    to_reference = anchor_to_reference(fixation)
    if number == anchor:
        # The anchor's own reference is the common frame, so its entry is a view of stabilized_frames
        # rather than a second copy; copy it before changing it in place.
        own_frames = stabilized_frames[members[0]:members[-1] + 1]
    else:
        own_frames = np.zeros((len(members), *frame_shape), dtype=gated_frames.dtype)
    for position, (index, (dy, dx)) in enumerate(zip(members, fixation["shifts"])):
        to_frame = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])  # this fixation's reference (x, y) -> this frame
        stabilized_frames[index], left_out = warp_frame(index, to_frame @ to_reference)
        stabilized_left_out[index] = np.flatnonzero(left_out)
        if number != anchor:
            own_frames[position], _ = warp_frame(index, to_frame)
    stabilized_fixations.append(own_frames)

print(f"stabilized_frames: {stabilized_frames.shape} | "
      f"stabilized_fixations: {[frames.shape for frames in stabilized_fixations]}")

# %% [markdown]
# The stabilized burst. The vessels should now stay in place while the edge of each frame moves around them. For viewing only, a pixel that section 8 left out shows what the other frames saw at that spot of tissue, as in that section's preview. In `stabilized_frames` it is 0.

# %%
VIEW_WIDTH = 800  # pixels; the preview is put together at the width it is shown at

# render the stabilized frames as a GIF to visualize the effect of the registration. For viewing only, a pixel
# that was left out shows what the other frames saw at that spot of tissue.
view_height = round(VIEW_WIDTH * frame_height / frame_width)
preview_frames = np.empty((num_frames, view_height, VIEW_WIDTH), dtype=np.uint16)
for number, fixation in enumerate(fixations):
    picture = viewing_tissue[number]
    if number != anchor:  # this fixation's tissue picture, mapped into the anchor's frame
        picture = cv2.warpPerspective(
            picture, anchor_to_reference(fixation), (frame_width, frame_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
    for index in fixation["frames"]:
        shown = stabilized_frames[index].astype(np.float32)
        left_out = stabilized_left_out[index]
        shown.flat[left_out] = frame_factors[index] * picture.flat[left_out]
        preview_frames[index] = cv2.resize(shown, (VIEW_WIDTH, view_height), interpolation=cv2.INTER_AREA)
create_animation(preview_frames, output_filename="stabilized_burst_preview.gif", fps=fps, width=VIEW_WIDTH)
del preview_frames

# %% [markdown]
# ## 10. Subtract the temporal mean
#
# After stabilization, a pixel shows the same spot of tissue in every frame of a fixation. Its mean over time is the static picture: the anatomy. Subtracting that mean leaves what changes at each spot: noise, blood flow and whatever motion is left.
#
# This is done within each fixation, on `stabilized_fixations`. A frame only counts where it has data, which leaves out a margin of `EDGE_MARGIN` pixels inside its border, where the resampling mixed in the empty outside. It also leaves out the pixels that section 8 marked as no data (glare and unusable sensor pixels).
#
# **Result:**
#
# - `fixation_means`: per fixation, a `(rows, columns)` image of the mean, `NaN` where no frame has data.
# - `mean_subtracted_fixations`: per fixation, a `(frames, rows, columns)` stack of frame minus mean in DN, `NaN` where that frame has no data.

# %%
# after stabilization, within each fixation, subtract the per-pixel temporal mean. The mean holds the static
# anatomy; what is left is whatever changes over time at each tissue location (noise, flow, leftover motion).
EDGE_MARGIN = 2  # pixels dropped inside each frame's border, where the cubic resampling mixed in the empty outside

fixation_means = []             # per fixation: (rows, cols) mean over the frames that cover each pixel
mean_subtracted_fixations = []  # per fixation: (frames, rows, cols) float32, NaN where a frame has no data
for number, (fixation, frames) in enumerate(zip(fixations, stabilized_fixations)):
    # the part of the fixation's reference frame that each shifted frame really covers
    areas = [covered_area(dy, dx, EDGE_MARGIN) for dy, dx in fixation["shifts"]]

    total = np.zeros(frame_shape, dtype=np.float32)
    count = np.zeros(frame_shape, dtype=np.uint16)
    for frame, area in zip(frames, areas):
        total[area] += frame[area]
        count[area] += frame[area] > 0  # 0 marks the pixels left out as glare or as unusable
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)

    residual = np.full(frames.shape, np.nan, dtype=np.float32)
    energy = 0.0
    for position, (frame, area) in enumerate(zip(frames, areas)):
        residual[position][area] = np.where(frame[area] > 0, frame[area] - mean[area], np.nan)
        energy += float(np.nansum(np.square(residual[position][area], dtype=np.float64)))

    fixation_means.append(mean)
    mean_subtracted_fixations.append(residual)
    print(f"Fixation {number}: mean subtracted from {len(frames)} frames, "
          f"{count.sum() / count.size / len(frames) * 100:.1f}% of pixels have data, "
          f"what is left has an RMS of {np.sqrt(energy / count.sum()):.1f} DN")

# %% [markdown]
# To look at the result, all frames are mapped into the anchor's frame, like the stabilized preview. The display range is limited to ± `DISPLAY_SIGMAS` times the noise level. Without that limit the few most extreme pixels would set the contrast, and everything else would come out flat gray. Pixels without data are shown as 0, which is mid-gray.
#
# `render_in_anchor_frame` does this for any list of per-fixation stacks, and is used again in section 11.

# %%
# render the mean-subtracted frames as a GIF to visualize what changes over time
DISPLAY_SIGMAS = 4.0  # the previews show this many noise standard deviations either side of 0
noise_sigma = 1.4826 * np.nanmedian(np.abs(mean_subtracted_fixations[anchor][::8, ::8, ::8]))
display_limit = DISPLAY_SIGMAS * noise_sigma


def render_in_anchor_frame(fixation_stacks, output_filename, process=None, width=800):
    """Render per-fixation frames as one video in the anchor's frame, clipped to +-display_limit."""
    preview_frames = np.zeros(stabilized_frames.shape, dtype=np.int16)
    for number, (fixation, stack) in enumerate(zip(fixations, fixation_stacks)):
        to_reference = anchor_to_reference(fixation)
        for index, frame in zip(fixation["frames"], stack):
            if process is not None:
                frame = process(frame)
            if number != anchor:
                frame = cv2.warpPerspective(
                    frame, to_reference, (frame_width, frame_height),
                    flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=np.nan,
                )
            preview_frames[index] = np.clip(np.nan_to_num(frame), -display_limit, display_limit)
    create_animation(preview_frames, output_filename=output_filename, fps=fps, width=width)


print(f"Showing -{display_limit:.0f} to +{display_limit:.0f} DN ({DISPLAY_SIGMAS:g} x the noise of {noise_sigma:.1f} DN)")
render_in_anchor_frame(mean_subtracted_fixations, "mean_subtracted_burst_preview.gif")

# %% [markdown]
# ## 11. Remove the flicker
#
# Whole frames differ in brightness by about 1%, which shows as flicker in the video above. Each frame gets one brightness factor: the scale that best fits the stabilized frame to its fixation's mean (a least-squares fit on every 4th pixel along each axis). The factors of a fixation are scaled to average 1. Each stabilized frame is divided by its factor, and the temporal mean is then subtracted again.
#
# **Result:**
#
# - `brightness_factors`: per fixation, one factor per frame.
# - `flicker_corrected_fixations`: per fixation, a `(frames, rows, columns)` stack in DN, laid out like `mean_subtracted_fixations`.

# %%
# reduce the flicker: whole frames differ in brightness by about 1%, so each stabilized frame is divided by one
# brightness factor, fitted against its fixation's mean, before the temporal mean is subtracted again.
brightness_factors = []           # per fixation: one factor per frame, averaging 1
flicker_corrected_fixations = []  # per fixation: (frames, rows, cols) float32, NaN where a frame has no data
sample = (slice(None, None, 4), slice(None, None, 4))  # every 4th pixel is plenty to fit one number per frame
for number, (mean, residual) in enumerate(zip(fixation_means, mean_subtracted_fixations)):
    factors = np.empty(len(residual))
    for position, frame in enumerate(residual):
        has_data = np.isfinite(frame[sample])
        reference = mean[sample][has_data].astype(np.float64)
        values = frame[sample][has_data] + reference  # the stabilized frame itself
        factors[position] = np.dot(values, reference) / np.dot(reference, reference)
    factors /= factors.mean()

    corrected = np.empty_like(residual)
    total = np.zeros(frame_shape, dtype=np.float32)
    count = np.zeros(frame_shape, dtype=np.uint16)
    for position, frame in enumerate(residual):
        corrected[position] = (frame + mean) / factors[position]
        total += np.nan_to_num(corrected[position])
        count += np.isfinite(corrected[position])
    corrected -= np.where(count > 0, total / np.maximum(count, 1), np.nan)

    level_before = np.std([np.nanmean(frame[sample]) for frame in residual])
    level_after = np.std([np.nanmean(frame[sample]) for frame in corrected])
    brightness_factors.append(factors)
    flicker_corrected_fixations.append(corrected)
    print(f"Fixation {number}: brightness factors {factors.min():.4f} to {factors.max():.4f}, "
          f"whole-frame level {level_before:.1f} DN -> {level_after:.1f} DN")

# %% [markdown]
# Two previews of the corrected frames:
#
# 1. as they are;
# 2. with each frame's own smooth part removed as well. The smooth part is the structure larger than about `SMOOTH_SIGMA` pixels: slow shading that drifts as the eye moves.
#
# The second step is for looking only. `flicker_corrected_fixations` keeps the smooth part.

# %%
SMOOTH_SIGMA = 32    # pixels; structure larger than about this counts as a frame's smooth part
SMOOTH_SCALE = 8     # the smooth part is computed on frames reduced by this factor
PREVIEW_WIDTH = 480  # pixels; these previews are mostly noise and compress badly, so they are kept small


def remove_smooth_part(frame):
    """Subtract a frame's own large-scale part, ignoring the pixels without data."""
    reduced = (frame_width // SMOOTH_SCALE, frame_height // SMOOTH_SCALE)
    has_data = np.isfinite(frame)
    values = cv2.resize(np.where(has_data, frame, 0), reduced, interpolation=cv2.INTER_AREA)
    weight = cv2.resize(has_data.astype(np.float32), reduced, interpolation=cv2.INTER_AREA)
    sigma = SMOOTH_SIGMA / SMOOTH_SCALE
    smooth = cv2.GaussianBlur(values, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-3)
    return frame - cv2.resize(smooth, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)


# render the flicker-corrected frames as GIFs, without and with the smooth-part step
# (the smooth-part step is for looking only; flicker_corrected_fixations keeps the smooth part)
render_in_anchor_frame(flicker_corrected_fixations, "flicker_corrected_burst_preview.gif", width=PREVIEW_WIDTH)
render_in_anchor_frame(
    flicker_corrected_fixations, "flicker_corrected_smooth_removed_burst_preview.gif",
    process=remove_smooth_part, width=PREVIEW_WIDTH,
)

# %% [markdown]
# ## 12. Remove the background
#
# The illumination is uneven: on this burst the background is about twice as bright in some places as in others, and the vessels sit on top of it. This step fits the background and measures every pixel against it.
#
# **Background is what does not change over time.** Blood moving through a vessel makes a pixel's brightness fluctuate from frame to frame by more than the camera noise does. Tissue without flow shows only the noise. Pixels that show only the noise are called *quiet*, and the background is fitted through them.
#
# The step works on the logarithm of the image. There, a vessel that absorbs a fixed fraction of the light lowers the value by a fixed amount, whatever the brightness behind it.
#
# - `L = ln(I)` is the log image.
# - `B` is the fitted background in the same units: what `L` would be without the vessels.
# - `B - L` is the **optical density**: about 0 on background and positive in vessels. Where vessels overlap, their densities add.
#
# It has three parts:
#
# 1. **Find the quiet pixels**, from how much each pixel fluctuates over time.
# 2. **Fit the background of each fixation's mean frame.** `fit_background` makes a first fit from the mean frame alone, judging a pixel by how much it stands out from its surroundings. One more refit then takes the quiet pixels as the background.
# 3. **Measure every frame** against its fixation's background.
#
# `fit_background` works in these stages:
#
# 1. **First guess.** An upper envelope of `L`: a grey closing fills in every dark structure that a disc of `radius` pixels cannot fit inside, so the vessels disappear. A pixel counts as background when its density against this envelope is below the median plus `cutoff` robust standard deviations.
# 2. **Refit**, done `iterations` times. A mask of everything that stands out against the local noise is built from the current density, and widened by `dilation` pixels. `B` is then refitted as a local mean of `L` over the pixels outside the mask, with wider windows bridging the wide vessels.
# 3. **Density refit.** Every pixel whose density is above `od_threshold` times the noise joins the mask, and `B` is refitted.
# 4. **Glare refit.** Pixels far brighter than the background (density below minus `glare_threshold` times the noise) are dropped, and `B` is refitted.
#
# The functions in the next cell are those of `background.py`, unchanged; their docstrings give the details. They name functions of that pipeline (`register_burst`, `stabilize_burst`, `vessels.vessel_seeds`) that this notebook does not have. `fixation_backgrounds` and `fixation_density` are left out, because they need one common reference frame and `ingest_stabilize.mean_frame`; the fit cell below does their job with this notebook's fixations.

# %%
# background estimation in log space, following step 1 of the pipeline report
# (pipeline_report.html): a first guess of the background from an upper
# envelope, and the first guess of which pixels are background
#
#   L   = ln I, the log image
#   B0  = upper envelope of L: L with every dark line narrower than the
#         closing disc filled in
#   OD0 = B0 - L, the first optical density: ~0 on background, positive in
#         vessels (overlapping vessels add)
#
# then the refit (fit_background): a vessel mask from the contrast-to-noise
# map (the report's step 2) takes the vessels out, and B is fitted on the
# background pixels alone. frame_density applies a fitted B to single frames.


def log_image(image, valid=None, saturation=4095):
    '''
    natural log of the image. empty (<= 0), saturated (>= saturation) and
    invalid pixels are not data; they take the log of the nearest good pixel

    input: 2d image, optional mask of valid pixels, saturation level (4095
    for 12-bit camera data)

    output: L = ln I as float32, boolean mask of the good pixels
    '''
    image = np.asarray(image, dtype=np.float64)
    ok = np.isfinite(image) & (image > 0) & (image < saturation)
    if valid is not None:
        ok &= valid
    if not ok.any():
        raise ValueError('no usable pixels')

    L = np.log(np.where(ok, image, 1))
    if not ok.all():
        # indices of the nearest good pixel, for every pixel
        nearest = distance_transform_edt(~ok, return_distances=False, return_indices=True)
        L = L[tuple(nearest)]

    return L.astype(np.float32), ok


def upper_envelope(L, radius=60, block=4, smooth=None):
    '''
    first guess of the background: grey closing (max filter, then min filter
    over a disc) fills every dark structure the disc can't fit inside, so
    vessels disappear and the background around them remains. the result sits
    slightly above the true background (the max filter latches onto noise
    peaks), and below L only on bright details smaller than the block grid
    and smoothing

    for speed the closing runs on a grid `block` times smaller, then is
    smoothed and resized back

    input: log image, disc radius in full-size pixels (by the usual definition
    of closing, dark structures up to about 2 x radius wide get filled), block
    size, smoothing sigma in grid pixels (defaults to radius / block / 2, as
    in the report)

    output: B0, same shape as L
    '''
    rows, cols = L.shape

    # pad to a multiple of the block by reflection so block averages stay honest
    pad_rows, pad_cols = (-rows) % block, (-cols) % block
    padded = np.pad(L, ((0, pad_rows), (0, pad_cols)), mode='reflect')
    small = downscale_local_mean(padded, (block, block)).astype(np.float32)

    grid_radius = max(1, round(radius / block))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grid_radius + 1,) * 2)
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, disc, borderType=cv2.BORDER_REFLECT)

    if smooth is None:
        smooth = grid_radius / 2
    closed = cv2.GaussianBlur(closed, (0, 0), smooth, borderType=cv2.BORDER_REFLECT)

    full = cv2.resize(closed, (padded.shape[1], padded.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return full[:rows, :cols]


def robust_sd(x):
    '''
    standard deviation estimated from the median absolute deviation, so a
    minority of extreme values (vessels) barely affect it
    '''
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def first_background(image, valid=None, radius=60, block=4, cutoff=2, saturation=4095):
    '''
    the report's first guess of the background and of which pixels are
    background

    a pixel counts as background when its (lightly blurred) optical density is
    below median + cutoff x robust sd: most pixels are background, so the
    median is typical background and vessels are the outliers above it

    input: 2d image, optional valid mask, envelope settings (see
    upper_envelope), cutoff in robust standard deviations, saturation level

    output: dict with L (log image), ok (usable pixels), B0 (first background,
    log units), OD0 (B0 - L) and bg (first background pixels)
    '''
    L, ok = log_image(image, valid, saturation)
    B0 = upper_envelope(L, radius, block)
    OD0 = B0 - L

    S = cv2.GaussianBlur(OD0, (0, 0), 1)
    values = S[ok]
    bg = ok & (S < np.median(values) + cutoff * robust_sd(values))

    return {'L': L, 'ok': ok, 'B0': B0, 'OD0': OD0, 'bg': bg}


# background fitting, the rest of the report's step 1. the background is
# refitted on background pixels only, after a vessel mask built from the
# contrast-to-noise map (the report's step 2) has removed the vessels


def blur(x, sigma):
    '''
    gaussian blur with reflected borders. wide blurs are smooth, so above
    8 px they run on a smaller grid and are resized back, which is much faster
    and visually identical
    '''
    x = np.asarray(x, dtype=np.float32)
    if sigma <= 8:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    factor = int(sigma // 4)
    rows, cols = x.shape
    small = cv2.resize(x, (max(1, cols // factor), max(1, rows // factor)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (cols, rows), interpolation=cv2.INTER_LINEAR)


def masked_mean(x, weight, sigma, octaves=4, falloff=0.03):
    '''
    local mean of x over the pixels where weight is 1 ("normalized
    convolution"). wider windows, each counting falloff times less than the
    one before, fill holes the narrow one can't reach, so wide vessels are
    bridged from the background around them

    input: 2d image, boolean / 0-1 weight, base sigma in pixels

    output: local mean, same shape
    '''
    weight = np.asarray(weight, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    for k in range(octaves):
        numerator += falloff ** k * blur(x * weight, sigma * 2 ** k)
        denominator += falloff ** k * blur(weight, sigma * 2 ** k)

    # with no background anywhere near, fall back to the global background mean
    overall = x[weight > 0].mean() if (weight > 0).any() else x.mean()
    eps = falloff ** octaves
    return (numerator + eps * overall) / (denominator + eps)


def local_rms(d, bg, window):
    '''
    typical size of d near each pixel, measured on background pixels only.
    values beyond 3 robust standard deviations are capped, and the result is
    floored at 5 % of the global level

    input: 2d signal, boolean background mask, window sigma in pixels

    output: local rms, same shape
    '''
    level = 1.4826 * np.median(np.abs(d[bg]))
    capped = np.minimum(d ** 2, (3 * level) ** 2)
    return np.sqrt(np.maximum(masked_mean(capped, bg, window), (0.05 * level) ** 2))


def band_cnr(od, bg, bands=(1, 2, 4, 8, 16, 32)):
    '''
    contrast against the local noise (the report's step 2): at each size s,
    centre minus a surround twice as wide, divided by its local noise on
    background pixels. a thin vessel stands out at a small size, a wide one at
    a large size

    input: optical density, boolean background mask, band sizes in pixels

    output: Z, the contrast-to-noise ratio of the best band
    '''
    z = np.full(od.shape, -np.inf, dtype=np.float32)
    for s in bands:
        d = blur(od, s) - blur(od, 2 * s)
        z = np.maximum(z, d / local_rms(d, bg, max(24, 4 * s)))
    return z


def fit_background(image, valid=None, radius=60, block=4, cutoff=2, iterations=2,
                   z_threshold=2, z_low=1, noise_threshold=3, dilation=5, close_radius=0,
                   od_threshold=2, glare_threshold=5, sigma=8, noise_window=48,
                   bands=(1, 2, 4, 8, 16, 32), saturation=4095):
    """
    the report's full step 1, with a wider vessel mask: start from the upper
    envelope (first_background), then repeat `iterations` times:
      1. Z = contrast-to-noise of the current OD (band_cnr)
      2. vessel mask M = Z above z_threshold, plus weaker Z above z_low that
         connects to it (hysteresis: faint stretches of a visible vessel count,
         isolated texture doesn't), plus lightly blurred OD above
         noise_threshold x the local noise. optionally closed with a disc of
         close_radius (fills a gap along an artery's central light reflex),
         then dilated by `dilation` px so the vessels' soft edges don't leak
         into the background
      3. refit the background B on the pixels outside M (masked_mean, sigma)
      4. OD = B - L
    then two optional extra refits:
      - od refit: add every pixel whose OD is above od_threshold x noise to
        the mask, refit (the report's "change 1": the density itself finds
        the faint vessel stretches the contrast map missed)
      - glare: pixels far brighter than the background (OD below
        -glare_threshold x noise) are not background either; drop them and
        refit

    on the test burst, the wider mask (hysteresis, 5 px dilation, od refit)
    keeps ~40 % more density on faint vessels and removes the dip of B under
    wide vessels, while the background residual gets flatter, not worse

    input: 2d image, optional valid mask (pass ~saturated, from the frames:
    a pixel saturated in some frames averages below the saturation level),
    first-guess settings (see first_background), number of refits, mask
    thresholds (z_low=None for a plain threshold, od_threshold=None or
    glare_threshold=None to skip those refits), mask closing and dilation,
    background sigma, noise window, contrast bands, saturation level (all
    sizes in pixels)

    output: dict with L, ok, B0, OD0, bg0 (the first guess and its background
    pixels), B (fitted background, log units), OD (B - L, the target),
    M (pixels kept out of the fit: deliberately generous, not a vessel
    segmentation; see vessels.vessel_seeds for that), bg (background
    pixels), Z (contrast-to-noise), noise (local noise of OD), glare (pixels
    dropped as glare), and history: one dict per refit with its name and that
    refit's M, bg, B and OD
    """
    first = first_background(image, valid, radius, block, cutoff, saturation)
    L, ok, bg = first['L'], first['ok'], first['bg']
    B, od = first['B0'], first['OD0']
    vessels, z = ~bg, None
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
    history = []

    def refit(name):
        nonlocal bg, B, od
        bg = ok & ~vessels
        B = masked_mean(L, bg, sigma)
        od = B - L
        history.append({'name': name, 'M': vessels, 'bg': bg, 'B': B, 'OD': od})

    def noise_of(od, bg):
        smooth = blur(od, 1)
        return smooth, local_rms(smooth - np.median(smooth[bg]), bg, noise_window)

    smooth, noise = noise_of(od, bg)
    for i in range(iterations):
        z = band_cnr(od, bg, bands)
        smooth, noise = noise_of(od, bg)
        if z_low is None:
            vessels = z > z_threshold
        else:
            vessels = apply_hysteresis_threshold(z, z_low, z_threshold)
        vessels |= smooth > noise_threshold * noise
        if close_radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
            vessels = cv2.morphologyEx(vessels.astype(np.uint8), cv2.MORPH_CLOSE,
                                       kernel).astype(bool)
        vessels = cv2.dilate(vessels.astype(np.uint8), disc).astype(bool)
        refit(f'refit {i + 1}')

    if od_threshold is not None:
        vessels = vessels | (blur(od, 1) > od_threshold * noise)
        refit('od refit')

    glare = np.zeros(od.shape, bool)
    if glare_threshold is not None:
        glare = ok & (od < -glare_threshold * noise)
        if glare.any():
            ok = ok & ~glare
            refit('glare refit')

    if not ok.all():
        nearest = distance_transform_edt(~ok, return_distances=False, return_indices=True)
        od = od[tuple(nearest)]
        if history:
            history[-1]['OD'] = od

    return {**first, 'bg0': first['bg'], 'ok': ok, 'B': B, 'OD': od, 'M': vessels,
            'bg': bg, 'Z': z, 'noise': noise, 'glare': glare, 'history': history}


def frame_density(frame, background, transform=None, background_pixels=None,
                  saturation=4095):
    """
    optical density of a single frame against a background fitted on the
    stabilized average. the background is warped into the frame with the
    frame's own registration transform (the inverse of what stabilized it),
    so every frame gets exactly the same subtraction. if background_pixels
    is given, the frame's median density on those pixels is subtracted, so a
    frame exposed a little differently from the average still centres on zero

    input: 2d frame (raw, in its own coordinates), B from fit_background
    (reference coordinates), the frame's (2, 3) transform from register_burst
    (None if the frame is already in reference coordinates), optional
    background-pixel mask in reference coordinates, saturation level

    output: OD of the frame in its own coordinates (nan where the background
    doesn't cover or the frame isn't data), the offset that was subtracted
    """
    frame = np.asarray(frame, dtype=np.float32)
    rows, cols = frame.shape
    ok = np.isfinite(frame) & (frame > 0) & (frame < saturation)
    L = np.log(np.where(ok, frame, 1))

    if transform is None:
        B, pixels = background, background_pixels
    else:
        # transforms map reference -> frame, which is the forward map here
        B = cv2.warpAffine(background.astype(np.float32), transform, (cols, rows),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=np.nan)
        pixels = None
        if background_pixels is not None:
            pixels = cv2.warpAffine(background_pixels.astype(np.uint8), transform,
                                    (cols, rows), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0).astype(bool)

    od = np.where(ok, B - L, np.nan).astype(np.float32)
    offset = 0.0
    if pixels is not None:
        values = od[pixels & np.isfinite(od)]
        if values.size:
            offset = float(np.median(values))
            od -= offset
    return od, offset


def burst_density(frames, background, transforms=None, background_pixels=None,
                  saturation=4095, workers=None):
    """
    optical density of every frame of a burst against one fitted background:
    frame_density applied to the whole burst, so the vessels of every frame
    are measured against the same background

    pass the stabilized frames (reference coordinates, from stabilize_burst)
    with transforms=None to get a density burst in reference coordinates,
    where the vessels sit still and the background is flat: the burst to
    watch, average, or follow over time. pass the raw frames with their
    transforms to get each frame's density in its own coordinates instead

    input: burst of shape (frames, rows, cols), B from fit_background, the
    (frames, 2, 3) transforms from register_burst or None, optional
    background-pixel mask (each frame is centred on zero over it), saturation
    level

    output: density burst as float32 (nan where a frame has no data), and the
    (frames,) offsets subtracted
    """
    density = np.empty(frames.shape, dtype=np.float32)
    offsets = np.zeros(len(frames))

    def one(i):
        transform = None if transforms is None else transforms[i]
        density[i], offsets[i] = frame_density(frames[i], background, transform,
                                               background_pixels, saturation)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, range(len(frames))))

    return density, offsets


def saturated_pixels(frames, transforms, saturation=4095):
    """
    reference-coordinate mask of the pixels saturated in any frame of a
    burst, for fit_background's `valid`. tested on the raw frames: the
    stabilized frames are interpolated, which overshoots near bright spots
    and would flag pixels that never saturated, and the average can't tell,
    since a pixel saturated in some frames averages below the saturation level

    input: raw burst of shape (frames, rows, cols), its (frames, 2, 3)
    transforms from register_burst, saturation level

    output: boolean mask, same shape as one frame
    """
    rows, cols = frames.shape[1:]
    saturated = np.zeros((rows, cols), bool)
    for frame, transform in zip(frames, transforms):
        hit = frame >= saturation
        if hit.any():
            warped = cv2.warpAffine(hit.astype(np.uint8), transform, (cols, rows),
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            saturated |= warped.astype(bool)
    return saturated

# %% [markdown]
# **Which pixels are quiet.** The fluctuation is measured at each spot of tissue, over the frames of all fixations:
#
# 1. Each frame's relative change from its fixation's mean is taken: `ln(frame / mean)`.
# 2. The frame's own smooth part is removed (`remove_smooth_part`, section 11). This takes out the flicker and the slowly drifting shading.
# 3. The rest is blurred by `FLUCTUATION_BLUR` pixels to beat the pixel noise, squared, multiplied by the brightness and added up in the anchor's frame.
#
# Shot noise makes a relative change of about `1 / sqrt(brightness)`, so multiplying by the brightness makes the result level wherever only noise acts. That level is the **noise floor**, read off as the peak of the histogram. `fluctuation_ratio` is the fluctuation divided by the noise floor: about 1 on quiet tissue, and 2 to 4 in vessels on this burst.
#
# A pixel **changes** when its ratio is above `ACTIVE_LEVELS[1]`, or above `ACTIVE_LEVELS[0]` and connected to a pixel that is. This is the rule `fit_background` uses for its own mask: faint stretches of something that changes count, isolated specks do not. A pixel is **quiet** when it is more than `ACTIVE_GROW` pixels away from every pixel that changes, so that the soft edges of vessels are not taken for background. Where fewer than `MIN_FRAMES` frames have data, a pixel is not judged: it is neither.
#
# On this burst, leftover sub-pixel motion explains only 3–10% of the fluctuation above the noise, so the map is not an alignment artifact. It has not been separated into flow, pulsation and changes of focus.
#
# **Result:**
#
# - `fluctuation_ratio`: the fluctuation over the noise floor in the anchor's frame, `NaN` where not judged.
# - `changing_pixels` and `quiet_pixels`: per fixation, boolean images in the fixation's own reference frame.

# %%
FLUCTUATION_BLUR = 2.0        # pixels; each frame's change is blurred by this to beat the pixel noise
ACTIVE_LEVELS = (1.25, 1.5)   # x the noise floor; a pixel changes above the second, or above the first next to one that does
ACTIVE_GROW = 5               # pixels; a quiet pixel is farther than this from every pixel that changes
MIN_FRAMES = 40               # a pixel is judged only where at least this many frames have data


def blur_valid(frame, sigma):
    """Gaussian blur that ignores the pixels without data, which stay NaN."""
    has_data = np.isfinite(frame)
    values = cv2.GaussianBlur(np.where(has_data, frame, 0), (0, 0), sigma)
    weight = cv2.GaussianBlur(has_data.astype(np.float32), (0, 0), sigma)
    return np.where(has_data, values / np.maximum(weight, 1e-3), np.nan)


# the fluctuation at each spot of tissue, added up over the frames of all fixations in the anchor's frame
squares = np.zeros(frame_shape, dtype=np.float32)
frames_counted = np.zeros(frame_shape, dtype=np.float32)
for number, (fixation, mean, residual) in enumerate(zip(fixations, fixation_means, mean_subtracted_fixations)):
    to_reference = anchor_to_reference(fixation)
    for frame in residual:
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.log1p(frame / mean)  # ln(frame / mean): the frame's relative change from the mean
        change = blur_valid(remove_smooth_part(change), FLUCTUATION_BLUR)
        has_data = np.isfinite(change)
        energy = np.where(has_data, change**2 * mean, 0)  # times the brightness, which levels the shot noise
        has_data = has_data.astype(np.float32)
        if number != anchor:
            energy, has_data = (
                cv2.warpPerspective(image, to_reference, (frame_width, frame_height),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                for image in (energy, has_data)
            )
        squares += energy
        frames_counted += has_data

judged = frames_counted >= MIN_FRAMES
activity = np.sqrt(squares / np.maximum(frames_counted, 1))
# the noise floor is where the quiet pixels sit: the peak of the histogram
histogram, edges = np.histogram(activity[judged], bins=400, range=(0, np.percentile(activity[judged], 99)))
noise_floor = edges[np.argmax(histogram):][:2].mean()
fluctuation_ratio = np.where(judged, activity / noise_floor, np.nan).astype(np.float32)

changing = apply_hysteresis_threshold(np.nan_to_num(fluctuation_ratio), *ACTIVE_LEVELS)
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ACTIVE_GROW + 1,) * 2)
quiet = judged & ~cv2.dilate(changing.astype(np.uint8), disc).astype(bool)

changing_pixels = []  # per fixation: True where the tissue changes over time, in the fixation's own frame
quiet_pixels = []     # per fixation: True where it is quiet, clear of everything that changes
for number, fixation in enumerate(fixations):
    masks = [changing, quiet]
    if number != anchor:
        masks = [
            cv2.warpPerspective(mask.astype(np.uint8), anchor_to_reference(fixation), (frame_width, frame_height),
                                flags=cv2.INTER_NEAREST).astype(bool)
            for mask in masks
        ]
    changing_pixels.append(masks[0])
    quiet_pixels.append(masks[1])

print(f"Fluctuation measured on up to {frames_counted.max():.0f} frames per pixel, {judged.mean() * 100:.1f}% of the "
      f"pixels judged | of those, {changing[judged].mean() * 100:.1f}% change and {quiet[judged].mean() * 100:.1f}% are quiet")

# show the fluctuation and the quiet pixels, in the anchor's frame
figure, axes = plt.subplots(1, 2, figsize=(16, 5.3))
axes[0].imshow(np.nan_to_num(fluctuation_ratio), cmap="gray", vmin=0.8, vmax=2.5)
axes[0].set_title("fluctuation over time / noise floor (black: 0.8 or less, white: 2.5 or more)")
axes[1].imshow(quiet, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("quiet pixels (white)")
for axis in axes:
    axis.axis("off")
figure.tight_layout()
plt.show()

# %% [markdown]
# Each fixation gets its own background, fitted on its mean frame (`fixation_means`). The illumination is fixed to the camera and stabilization fixes the tissue, so after a saccade the lighting pattern lies somewhere else on the tissue.
#
# The fit is made in two steps:
#
# 1. `fit_background` fits the mean frame on its own. Pixels that saturate in any frame are kept out; they are looked up in `gated_frames`, before any resampling.
# 2. **Quiet refit.** The background is fitted once more in the same way, as a local mean of `L` over the background pixels, `BACKGROUND_SIGMA` pixels wide. This time the temporal data decide what is background: a pixel is background when it is quiet. So that there is background close to the vessels as well, a pixel that the first step took as background stays background unless it changes. Where a pixel was not judged, the first step's choice stays.
#
# Every stabilized frame of the fixation is then measured against that `B`. The frame's median density on the background pixels, its offset, is subtracted. A frame that is a little brighter or darker than the mean therefore still centres on 0, which also takes out the whole-frame flicker of section 11.
#
# **Result:**
#
# - `background_fits`: per fixation, the dictionary from `fit_background`, in the fixation's own reference frame. `B` is the fitted background, `OD` the density of the mean frame, `bg` the background pixels and `M` the pixels kept out of the fit, all after the quiet refit. `history` lists every refit; its entry before the last holds the fit made without the temporal data.
# - `density_fixations`: per fixation, a `(frames, rows, columns)` stack of optical density, laid out like `mean_subtracted_fixations` (`NaN` where a frame has no data).
# - `density_offsets`: per fixation, the offset subtracted from each frame.

# %%
BACKGROUND_SIGMA = 8  # pixels; the fitted background is a local mean of this width over the background pixels

background_fits = []    # per fixation: the dict from fit_background (B, OD, bg, M, ...), in the fixation's own frame
density_fixations = []  # per fixation: (frames, rows, cols) float32 optical density, NaN where a frame has no data
density_offsets = []    # per fixation: what was subtracted from each frame to centre its background on 0
for number, (fixation, frames) in enumerate(zip(fixations, stabilized_fixations)):
    members = fixation["frames"]
    # a pixel that saturates in any frame is not background data; these are looked up in the frames as recorded
    to_frames = np.float64([[[1, 0, dx], [0, 1, dy]] for dy, dx in fixation["shifts"]])  # reference (x, y) -> frame
    saturated = saturated_pixels(gated_frames[members[0]:members[-1] + 1], to_frames)
    fit = fit_background(fixation_means[number], valid=~saturated, sigma=BACKGROUND_SIGMA)

    # quiet refit: one more refit like those inside fit_background, with the temporal data deciding what is
    # background. The quiet pixels are background, and so are the pixels fit_background chose unless they change
    # (which is also what stays where a pixel was not judged).
    usable = fit["ok"]
    background = usable & (quiet_pixels[number] | (fit["bg"] & ~changing_pixels[number]))
    log_background = masked_mean(fit["L"], background, BACKGROUND_SIGMA)
    mean_density = log_background - fit["L"]
    if not usable.all():
        # as in fit_background: a pixel that is not data takes the density of the nearest one that is
        nearest = distance_transform_edt(~usable, return_distances=False, return_indices=True)
        mean_density = mean_density[tuple(nearest)]
    from_quiet = (background & quiet_pixels[number]).sum() / background.sum()
    fit.update({"B": log_background, "OD": mean_density, "bg": background, "M": ~background})
    fit["history"].append({"name": "quiet refit", "M": fit["M"], "bg": background, "B": log_background,
                           "OD": mean_density})

    density, offsets = burst_density(frames, fit["B"], background_pixels=fit["bg"], workers=workers)
    # same data area as the mean-subtracted frames, which leaves out the rim that resampling blended with the outside
    density[np.isnan(mean_subtracted_fixations[number])] = np.nan

    background_fits.append(fit)
    density_fixations.append(density)
    density_offsets.append(offsets)
    print(f"Fixation {number}: background fitted on {background.mean() * 100:.0f}% of the pixels, "
          f"{from_quiet * 100:.0f}% of them quiet pixels "
          f"({saturated.sum()} saturated and {fit['glare'].sum()} glare pixels left out), "
          f"density of the mean on them {robust_sd(fit['OD'][fit['bg']]):.4f} (robust sd) | "
          f"frame offsets {offsets.min():+.4f} to {offsets.max():+.4f}")

# %% [markdown]
# A check of each fit. From left to right: the fixation's mean frame, the fitted background on the same gray scale, the density of the mean frame, and the pixels kept out of the fit (white).
#
# The background should be smooth and free of vessels, and the density should be flat between the vessels. The pixels kept out are everything that changes, with a margin; this is not a vessel segmentation.

# %%
DENSITY_RANGE = (-0.1, 0.5)  # optical density shown from black to white, here and in the preview below

# show each fit: the fixation's mean, the fitted background, the density of the mean and the pixels kept out of the fit
figure, axes = plt.subplots(len(fixations), 4, figsize=(16, 2.9 * len(fixations)), squeeze=False)
for number, (fit, mean, row) in enumerate(zip(background_fits, fixation_means, axes)):
    brightness = np.nanpercentile(mean[::4, ::4], [0.5, 99.5])
    panels = (
        (f"fixation {number}: mean frame", mean, brightness),
        ("fitted background, exp(B)", np.exp(fit["B"]), brightness),
        ("density of the mean, B - ln(mean)", fit["OD"], DENSITY_RANGE),
        ("kept out of the fit", ~fit["bg"], (0, 1)),
    )
    for axis, (title, image, (low, high)) in zip(row, panels):
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(title)
        axis.axis("off")
figure.tight_layout()
plt.show()

# %% [markdown]
# The density burst, in the anchor's frame like the previews above. The vessels are now bright on a flat dark background. Black to white is `DENSITY_RANGE`; pixels without data are shown as 0, the background level.
#
# `preview_tile` prepares one frame at preview size, and is used again in section 13.

# %%
preview_height = round(PREVIEW_WIDTH * frame_height / frame_width)


def preview_tile(frame, number, low, high):
    """Return a frame of fixation `number` in the anchor's frame at preview size, 8-bit with low..high as black..white."""
    if number != anchor:
        frame = cv2.warpPerspective(
            frame, anchor_to_reference(fixations[number]), (frame_width, frame_height),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=np.nan,
        )
    frame = np.clip(np.nan_to_num(frame), low, high)  # pixels without data show as 0
    small = cv2.resize(frame, (PREVIEW_WIDTH, preview_height), interpolation=cv2.INTER_AREA)
    return np.uint8((small - low) * (255 / (high - low)) + 0.5)


# render the density frames as a GIF to visualize the effect of the background removal
preview_frames = np.empty((num_frames, preview_height, PREVIEW_WIDTH), dtype=np.uint8)
for number, (fixation, density) in enumerate(zip(fixations, density_fixations)):
    for index, frame in zip(fixation["frames"], density):
        preview_frames[index] = preview_tile(frame, number, *DENSITY_RANGE)
create_animation(preview_frames, output_filename="density_burst_preview.gif", fps=fps, width=PREVIEW_WIDTH)
del preview_frames

# %% [markdown]
# The background-cleaned burst on its own, with the quiet pixels removed: every stabilized frame divided by its fixation's fitted background, in the anchor's frame. This is `exp(-density)`, the fraction of the background's brightness that reaches each pixel. The quiet pixels are background, so they are set to the background level, 1. What is left is everything that changes over time, with a margin of `ACTIVE_GROW` pixels around it, on a flat background.
#
# Black to white is `CLEANED_RANGE`. Pixels without data are also shown at the background level. To see every pixel instead, delete the line marked `remove the quiet pixels`.

# %%
CLEANED_RANGE = (0.4, 1.1)  # fraction of the background's brightness shown from black to white

# render the background-cleaned stabilized frames as a GIF: the stabilized burst without its uneven lighting
low, high = CLEANED_RANGE
cleaned_frames = np.empty(stabilized_frames.shape, dtype=np.uint8)
for number, (fixation, density) in enumerate(zip(fixations, density_fixations)):
    to_reference = anchor_to_reference(fixation)
    for index, frame in zip(fixation["frames"], density):
        frame = np.where(quiet_pixels[number], 0, frame)  # remove the quiet pixels: they are background, density 0
        if number != anchor:
            frame = cv2.warpPerspective(
                frame, to_reference, (frame_width, frame_height),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=np.nan,
            )
        cleaned = np.exp(-np.nan_to_num(frame))  # a pixel without data gets density 0, the background level
        cleaned_frames[index] = np.clip((cleaned - low) * (255 / (high - low)) + 0.5, 0, 255)
create_animation(cleaned_frames, output_filename="background_cleaned_burst_preview.gif", fps=fps)
del cleaned_frames

# %% [markdown]
# ## 13. Compare the temporal mean subtraction with and without the background removal
#
# Both versions of the burst go through the step of section 10: within each fixation, every pixel's mean over time is subtracted. They play side by side, in the anchor's frame.
#
# - **Left:** the stabilized frames minus their fixation's mean. This is `mean_subtracted_fixations`, as in the preview of section 10.
# - **Right:** the density frames minus their fixation's mean density. It is shown negated, because density rises where a frame gets darker. This way dark means less light on both sides.
#
# Each side spans ± `DISPLAY_SIGMAS` times its own noise level, since the units differ (DN on the left, optical density on the right).
#
# What to expect: a fixation's background `B` is the same in every frame, so it drops out when the mean over time is subtracted. The right side is therefore the left side divided by the local brightness, without the whole-frame flicker, which the offsets of section 12 took out. Dark vessels look noisier on the right, because the same noise in DN is a larger fraction of less light.
#
# **Result:** `density_means`, per fixation a `(rows, columns)` image of the mean density over the frames that have data at each pixel. The mean-subtracted density is not stored; it is `density_fixations[n] - density_means[n]`.

# %%
density_means = []  # per fixation: (rows, cols) mean density over the frames that cover each pixel
for density in density_fixations:
    total = np.zeros(frame_shape, dtype=np.float32)
    count = np.zeros(frame_shape, dtype=np.uint16)
    for frame in density:
        total += np.nan_to_num(frame)
        count += np.isfinite(frame)
    density_means.append(np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32))

density_noise = 1.4826 * np.nanmedian(
    np.abs(density_fixations[anchor][::8, ::8, ::8] - density_means[anchor][::8, ::8])
)
density_limit = DISPLAY_SIGMAS * density_noise
print(f"Left, without background removal: -{display_limit:.0f} to +{display_limit:.0f} DN | "
      f"right, with background removal: -{density_limit:.3f} to +{density_limit:.3f} optical density "
      f"({DISPLAY_SIGMAS:g} x the noise of {density_noise:.4f})")

# render both mean-subtracted versions side by side as one GIF: left without, right with the background removal
comparison_frames = np.empty((num_frames, preview_height, 2 * PREVIEW_WIDTH), dtype=np.uint8)
for number, fixation in enumerate(fixations):
    for position, index in enumerate(fixation["frames"]):
        plain = mean_subtracted_fixations[number][position]
        cleaned = density_means[number] - density_fixations[number][position]  # negated, so dark means less light
        comparison_frames[index, :, :PREVIEW_WIDTH] = preview_tile(plain, number, -display_limit, display_limit)
        comparison_frames[index, :, PREVIEW_WIDTH:] = preview_tile(cleaned, number, -density_limit, density_limit)
create_animation(
    comparison_frames, output_filename="mean_subtracted_with_and_without_background.gif", fps=fps,
    width=2 * PREVIEW_WIDTH,
)
del comparison_frames

# %% [markdown]
# ## 14. Follow the shading from frame to frame
#
# In the right-hand video of section 13 the flicker is gone, but large soft shadows still drift across the picture. They are not vessels. The illumination is fixed to the camera, and within a fixation the eye still moves by tens of pixels, so the lighting pattern slides over the tissue from frame to frame. Section 12 fits one background per fixation, and one picture cannot follow something that moves.
#
# **Which pixels the background was fitted on makes no difference here.** A background that is the same in every frame drops out when the mean over time is subtracted, whether it was fitted on the quiet pixels or without any temporal data. The two differ by one constant per frame; the cell below prints how large it is.
#
# **What removes the shadows is a background for every frame.** Each frame's density is brought to 0 on the background pixels around each pixel, not only over the frame as a whole: the local mean of the frame's density over the background pixels, `SHADING_SIGMA` pixels wide, is subtracted. This local mean is the frame's *shading*. Under a vessel it is bridged from the background on either side, so what happens inside the vessel is left alone.
#
# No temporal data is used for the background in this section. The background and its pixels are those `fit_background` found from the mean frame alone (the fit before the quiet refit), and each frame's shading is measured in that frame by itself.
#
# The video shows the temporal mean subtraction of section 13 for both, on the gray scale of section 13's right-hand side:
#
# - **Left:** one background per fixation.
# - **Right:** the same background, refitted to every frame.
#
# **Result:** `frame_shadings`, per fixation a `(frames, rows / SHADING_SCALE, columns / SHADING_SCALE)` stack: what each frame's shading adds to the fixation's average shading, on a coarse grid. `enlarge` brings one of them to full size; subtracting that from a frame of `density_fixations` takes the moving shading out of it. `density_fixations` itself is unchanged. `frame_centres` holds, per fixation, the one constant that goes with each frame's shading.

# %%
SHADING_SIGMA = 32  # pixels; shading wider than about this is followed from frame to frame
SHADING_SCALE = 8   # the shading is smooth, so it is worked out and kept on a grid this many times coarser


def frame_shading(density, background):
    """Return one density frame's local mean over its background pixels, on the coarse grid: that frame's shading."""
    weight = (background & np.isfinite(density)).astype(np.float32)
    size = (frame_width // SHADING_SCALE, frame_height // SHADING_SCALE)
    values = cv2.resize(np.nan_to_num(density) * weight, size, interpolation=cv2.INTER_AREA)
    weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)  # share of background pixels in each block
    return masked_mean(values / np.maximum(weight, 1e-6), weight, SHADING_SIGMA / SHADING_SCALE)


def enlarge(shading):
    """Bring a shading from the coarse grid to full size."""
    return cv2.resize(shading, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)


def spatial_frame(number, position):
    """Return what centres one frame on the background fitted without temporal data, and the frame's shading."""
    fit = background_fits[number]
    spatial = fit["history"][-2]  # the last fit made from the mean frame alone, before the quiet refit
    density = density_fixations[number][position] + (spatial["B"] - fit["B"])
    centre = np.median(density[spatial["bg"] & np.isfinite(density)])  # centred on 0 over its background, as in section 12
    return centre, frame_shading(density - centre, spatial["bg"])


frame_shadings = []  # per fixation: (frames, rows / 8, cols / 8), each frame's shading minus the fixation's average
frame_centres = []   # per fixation: (frames,), the constant that goes with each frame's shading
comparison_frames = np.empty((num_frames, preview_height, 2 * PREVIEW_WIDTH), dtype=np.uint8)
largest_step = 0.0
for number, fixation in enumerate(fixations):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda position: spatial_frame(number, position), range(len(fixation["frames"]))))
    centres = np.array([centre for centre, _ in results])
    shadings = np.array([shading for _, shading in results])
    shadings -= shadings.mean(axis=0)
    frame_shadings.append(shadings)
    centres -= centres.mean()  # over all frames; at the rim, where some frames have no data, off by under a gray level
    frame_centres.append(centres)
    largest_step = max(largest_step, np.abs(centres).max())

    for position, index in enumerate(fixation["frames"]):
        # the frame minus the mean over time, negated so dark means less light. The difference between the two
        # backgrounds is the same in every frame and has dropped out; only the frame's constant is left of it.
        fixed = density_means[number] - density_fixations[number][position] + centres[position]
        followed = fixed + enlarge(shadings[position])
        comparison_frames[index, :, :PREVIEW_WIDTH] = preview_tile(fixed, number, -density_limit, density_limit)
        comparison_frames[index, :, PREVIEW_WIDTH:] = preview_tile(followed, number, -density_limit, density_limit)

moving = np.sqrt(np.mean(np.concatenate([shadings.ravel() for shadings in frame_shadings]) ** 2))
print(f"Without the temporal data, a frame differs from section 13's by one constant: at most {largest_step:.4f} "
      f"of density, against a gray scale of -{density_limit:.3f} to +{density_limit:.3f}")
print(f"Shading that moves from frame to frame: {moving:.4f} of density (rms), about {moving * 100:.1f}% of the "
      f"brightness. It is on the left and removed on the right.")
create_animation(
    comparison_frames, output_filename="mean_subtracted_fixed_and_per_frame_background.gif", fps=fps,
    width=2 * PREVIEW_WIDTH,
)
del comparison_frames
# %% [markdown]
# ## 15. The average and the smear of the temporal burst, per fixation
#
# The temporal burst is what section 14 shows on its right-hand side: each frame's density minus its fixation's mean density, with the frame's own shading taken out. Three pictures per fixation, in the fixation's own reference frame:
#
# - **Left: the mean density.** This is what was subtracted: the static picture, in which the vessels are bright.
# - **Middle: the average of the temporal burst over time.** It is 0, because each pixel's own mean over time is exactly what was subtracted. The picture is flat, and the cell prints how far from 0 it gets (the shading leaves a trace at the rim, where some frames have no data).
# - **Right: the smear.** How far the temporal burst swings around 0 at each pixel: its standard deviation over time. Each frame is first blurred by `SMEAR_BLUR` pixels to beat the pixel noise, and the result is multiplied by the square root of the brightness, which evens out the shot noise. Vessels with flow stand out; tissue without flow stays at the noise level.
#
# **Result:** `burst_smears`, per fixation a `(rows, columns)` image of the smear, `NaN` where fewer than `SMEAR_MIN_FRAMES` of the fixation's frames have data.

# %%
SMEAR_BLUR = 2.0         # pixels; each frame is blurred by this before its fluctuation is added up
SMEAR_MIN_FRAMES = 0.25  # a pixel needs data in this fraction of its fixation's frames to appear in the smear

gray = plt.get_cmap("gray").copy()
gray.set_bad("black")  # pixels without a value are drawn black

burst_smears = []  # per fixation: the temporal burst's standard deviation over time, shot noise evened out
burst_averages = []
for number, (density, mean_density, changes) in enumerate(zip(density_fixations, density_means, frame_shadings)):
    summed, soft_sum, soft_squares = (np.zeros(frame_shape, dtype=np.float64) for _ in range(3))
    seen = np.zeros(frame_shape, dtype=np.uint16)
    for frame, change in zip(density, changes):
        temporal = mean_density - frame + enlarge(change)  # negated as in the videos: dark means less light
        soft = np.nan_to_num(blur_valid(temporal, SMEAR_BLUR))
        summed += np.nan_to_num(temporal)
        soft_sum += soft
        soft_squares += soft**2
        seen += np.isfinite(temporal)
    frames_seen = np.maximum(seen, 1)
    burst_averages.append(np.where(seen > 0, summed / frames_seen, np.nan))
    # the spread around each pixel's own mean, times the square root of the brightness to even out the shot noise
    spread = np.sqrt(np.maximum(soft_squares / frames_seen - (soft_sum / frames_seen) ** 2, 0))
    enough = seen >= SMEAR_MIN_FRAMES * len(density)
    burst_smears.append(
        np.where(enough, spread * np.sqrt(np.nan_to_num(fixation_means[number])), np.nan).astype(np.float32)
    )
    print(f"Fixation {number}: {len(density)} frames | the temporal burst averages to 0 within "
          f"{np.nanmax(np.abs(burst_averages[number])):.0e} of density")

smear_range = np.nanpercentile(burst_smears[anchor], [1, 99.7])  # one gray scale for the smears of all fixations
figure, axes = plt.subplots(len(fixations), 3, figsize=(16, 3.8 * len(fixations)), squeeze=False)
for number, row in enumerate(axes):
    shown = (
        (f"fixation {number}: mean density, which was subtracted", density_means[number], DENSITY_RANGE),
        ("average of the temporal burst over time: 0", burst_averages[number], (-density_limit, density_limit)),
        ("smear of the temporal burst: its fluctuation over time", burst_smears[number], smear_range),
    )
    for axis, (title, image, (low, high)) in zip(row, shown):
        axis.imshow(image, cmap=gray, vmin=low, vmax=high)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
figure.tight_layout()
plt.show()
del burst_averages

# %% [markdown]
# ## 16. Clean up the smear's background and compare it with the background-cleaned average frame
#
# The smear of section 15 has a background of its own. On tissue without flow it sits at the noise floor, not at 0, and that level is not perfectly even. Here it is taken out with the same fit that cleans the average frame, `fit_background`.
#
# That fit expects dark vessels on a bright background, in log space. So it is given `exp(-smear / noise floor)`: the log of that is `-smear` in multiples of the noise floor, in which the vessels are dark. The density the fit returns is then
#
# `(smear - fitted background of the smear) / noise floor`.
#
# That is the **cleaned smear**: 0 where nothing flows, and on a vessel the fluctuation above the background, in multiples of the noise floor. The smear's background is fitted as a much wider local mean than the average frame's (`SMEAR_SIGMA` pixels against 8). The noise floor is nearly flat, and a narrow fit creeps up the widest vessels and hollows them out.
#
# It is compared with the **background-cleaned average frame**, the density of each fixation's mean frame (`background_fits[...]["OD"]`). The average frame shows how much light a place takes away on average; the smear shows how much that changes from frame to frame. A pixel *stands out* in either picture when it is more than `STANDS_OUT` times that picture's background noise above 0.
#
# Two figures:
#
# 1. **The cleaning**, per fixation: the smear, its fitted background, what is left, and the pixels kept out of the fit (white).
# 2. **The comparison**, per fixation: the background-cleaned average frame, the cleaned smear, and both in one picture. There the average frame is magenta and the smear green: a place that both mark is white, one that takes light away without changing is magenta, one that only changes is green.
#
# **Result:**
#
# - `smear_fits`: per fixation, the dictionary from `fit_background`, fitted on the smear.
# - `cleaned_smears`: per fixation, the smear with its background removed, in multiples of its noise floor (`NaN` where the smear has no value).

# %%
SMEAR_SIGMA = 64  # pixels; width of the local mean that is the smear's background. The noise floor is nearly flat,
                  # and a narrower one (8, as for the average frame) creeps up the widest vessels and hollows them out
STANDS_OUT = 5    # a pixel stands out when it is this many times its picture's background noise above 0

smear_fits = []      # per fixation: the dict from fit_background, fitted on the smear
cleaned_smears = []  # per fixation: the smear minus its fitted background, in multiples of the smear's noise floor
compared = []        # per fixation: what the figures below show
for number, smear in enumerate(burst_smears):
    # the noise floor is where the pixels without flow sit: the peak of the smear's histogram
    values = smear[np.isfinite(smear)]
    histogram, edges = np.histogram(values, bins=400, range=(0, np.percentile(values, 99)))
    floor = edges[np.argmax(histogram):][:2].mean()
    # exp(-smear / floor) is bright where the smear is low and dark on the vessels; its log is -smear / floor
    fit = fit_background(np.exp(-smear / floor), sigma=SMEAR_SIGMA)
    seen = np.isfinite(smear)
    cleaned = np.where(seen, fit["OD"], np.nan)
    smear_fits.append(fit)
    cleaned_smears.append(cleaned)

    average_fit = background_fits[number]
    average = np.where(seen, average_fit["OD"], np.nan)  # the background-cleaned average frame, where the smear has data
    average_noise = robust_sd(average[average_fit["bg"] & seen])
    smear_noise = robust_sd(cleaned[fit["bg"] & seen])
    dark = seen & (average > STANDS_OUT * average_noise)   # takes light away on average
    moving = seen & (cleaned > STANDS_OUT * smear_noise)   # changes from frame to frame
    compared.append((smear / floor, -fit["B"], cleaned, ~fit["bg"], average))
    print(f"Fixation {number}: background of the smear fitted on {fit['bg'].mean() * 100:.0f}% of the pixels; the cleaned "
          f"smear has a background noise of {smear_noise:.2f} noise floors | correlation with the background-cleaned "
          f"average frame {np.corrcoef(average[seen], cleaned[seen])[0, 1]:.2f}")
    print(f"   stands out ({STANDS_OUT} x its own background noise): in the average frame on {dark.sum() / seen.sum() * 100:.0f}% "
          f"of the field, in the smear on {moving.sum() / seen.sum() * 100:.0f}%, in both on "
          f"{(dark & moving).sum() / seen.sum() * 100:.0f}% | where both do, the average frame is "
          f"{np.median(average[dark & moving]) / average_noise:.0f} x and the smear "
          f"{np.median(cleaned[dark & moving]) / smear_noise:.0f} x its noise (median)")

# show the cleaning of each smear: the smear, its fitted background, what is left, and the pixels kept out of the fit
smear_top = np.nanpercentile(compared[anchor][0], 99.7)
figure, axes = plt.subplots(len(fixations), 4, figsize=(16, 2.9 * len(fixations)), squeeze=False)
for number, (row, (smear, background, cleaned, kept_out, average)) in enumerate(zip(axes, compared)):
    shown = (
        (f"fixation {number}: smear, in noise floors", smear, (0, smear_top)),
        ("fitted background of the smear", background, (0, smear_top)),
        ("smear with its background removed", cleaned, (-1, smear_top - 1)),
        ("kept out of the fit", kept_out, (0, 1)),
    )
    for axis, (title, image, (low, high)) in zip(row, shown):
        axis.imshow(image, cmap=gray, vmin=low, vmax=high)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
figure.tight_layout()
plt.show()

# show the comparison: the background-cleaned average frame, the cleaned smear, and both in one picture.
# In the last one the average frame is magenta and the smear green, each up to its 99.5th percentile, so a place
# that both mark comes out white, one that takes light away without changing magenta, one that only changes green.
figure, axes = plt.subplots(len(fixations), 3, figsize=(16, 3.8 * len(fixations)), squeeze=False)
for number, (row, (smear, background, cleaned, kept_out, average)) in enumerate(zip(axes, compared)):
    average_share = np.clip(np.nan_to_num(average) / np.nanpercentile(average, 99.5), 0, 1)
    smear_share = np.clip(np.nan_to_num(cleaned) / np.nanpercentile(cleaned, 99.5), 0, 1)
    shown = (
        (f"fixation {number}: background-cleaned average frame", average, DENSITY_RANGE),
        ("smear with its background removed", cleaned, (-1, smear_top - 1)),
        ("magenta: average frame, green: smear, white: both", np.dstack([average_share, smear_share, average_share]), (0, 1)),
    )
    for axis, (title, image, (low, high)) in zip(row, shown):
        axis.imshow(image, cmap=gray, vmin=low, vmax=high)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
figure.tight_layout()
plt.show()
del compared

# %% [markdown]
# ## 17. The sum and the average of the two vessel pictures
#
# The two pictures of section 16 put together, for each fixation: the background-cleaned average frame and the smear with its background removed, their sum and their average.
#
# The two are in different units (optical density, noise floors). So each is first divided by its own background noise: a value then says how many times the noise a pixel stands above 0.
#
# - **The sum** adds the two. A vessel that either picture shows is in it, and one that both show is twice as strong.
# - **The average** is half the sum, so it is the same picture on half the scale. The cell after this section shows it inverted for one fixation, with dark vessels on a light background.
#
# All four panels share one gray scale, that of the two pictures on their own. The average can therefore be read against either of them directly, and the sum shows how much the two add up to (it goes white where the scale ends).
#
# **Result:** `combined_sums`, per fixation the sum, in multiples of the background noise of one picture (`NaN` where the smear has no value). The average is `combined_sums[n] / 2`.

# %%
combined_sums = []  # per fixation: average frame / its background noise + cleaned smear / its background noise
shown_together = []
for number, cleaned in enumerate(cleaned_smears):
    seen = np.isfinite(cleaned)
    average_fit, smear_fit = background_fits[number], smear_fits[number]
    in_average = np.where(seen, average_fit["OD"], np.nan) / robust_sd(average_fit["OD"][average_fit["bg"] & seen])
    in_smear = cleaned / robust_sd(cleaned[smear_fit["bg"] & seen])
    combined_sums.append(in_average + in_smear)
    shown_together.append((in_average, in_smear, combined_sums[number], combined_sums[number] / 2))

    # where both pictures see background the sum is noise alone; that is the level the sum is judged against
    sum_noise = robust_sd(combined_sums[number][average_fit["bg"] & smear_fit["bg"] & seen])
    share = lambda marked: (marked & seen).sum() / seen.sum() * 100
    print(f"Fixation {number}: stands out ({STANDS_OUT} x the background noise) in the sum on "
          f"{share(combined_sums[number] > STANDS_OUT * sum_noise):.0f}% of the field | in the average frame alone on "
          f"{share(in_average > STANDS_OUT):.0f}%, in the smear alone on {share(in_smear > STANDS_OUT):.0f}%, "
          f"in either on {share((in_average > STANDS_OUT) | (in_smear > STANDS_OUT)):.0f}% | "
          f"background noise of the sum {sum_noise:.2f} (two independent pictures would give 1.41)")

# one gray scale for every panel: up to the 99.5th percentile of the fainter of the two pictures in the anchor,
# so that the fine vessels show; the widest vessels of the other picture then go white
top = min(np.nanpercentile(picture, 99.5) for picture in shown_together[anchor][:2])
figure, axes = plt.subplots(len(fixations), 4, figsize=(16, 2.9 * len(fixations)), squeeze=False)
for number, (row, pictures) in enumerate(zip(axes, shown_together)):
    titles = (f"fixation {number}: background-cleaned average frame", "smear with its background removed",
              "sum of the two", "average of the two")
    for axis, title, image in zip(row, titles, pictures):
        axis.imshow(image, cmap=gray, vmin=0, vmax=top)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
figure.suptitle(f"each picture in multiples of its own background noise; black to white is 0 to {top:.0f}", fontsize=10)
figure.tight_layout()
plt.show()

# the figure shows the panels much reduced; save each one at full resolution too, on the same gray scale,
# and the average also inverted (dark vessels on a light background)
os.makedirs("vessel_pictures", exist_ok=True)
for number, pictures in enumerate(shown_together):
    for name, picture in zip(("average_frame", "smear", "sum", "average"), pictures):
        shade = (np.clip(np.nan_to_num(picture) / top, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(shade).save(f"vessel_pictures/fixation_{number}_{name}.png")
        if name == "average":
            Image.fromarray(255 - shade).save(f"vessel_pictures/fixation_{number}_{name}_inverted.png")
print(f"Saved {5 * len(shown_together)} pictures at full resolution in the folder vessel_pictures")
del shown_together

# %%
# invert the density of the average of the two for fixation 1
# (the average of the two is half of combined_sums; inverted on the gray scale of the figure above, the vessels
# come out dark on a light background, the way the frames themselves look)
inverted_average = 1 - np.clip(np.nan_to_num(combined_sums[1] / 2) / top, 0, 1)  # 1 is background, 0 the top of the scale

figure, axis = plt.subplots(figsize=(16, 10))
axis.imshow(inverted_average, cmap="gray", vmin=0, vmax=1)
axis.set_title("fixation 1: average of the two, inverted")
axis.axis("off")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## 18. A binary vessel mask and the vessel graph, tried on several pictures
#
# Everything from here on is for the model of the flow. Its first need is the structure: where the vessels run, where they meet, how wide they are. The structure is taken from a **binary mask**, not from gray values. A mask says only "vessel" or "not vessel", so its geometry is not tangled up with how dark, how blurred or how noisy a vessel is. That is what makes a centerline and a width well defined.
#
# Which picture the mask should come from is not obvious, so the same recipe is run on all of them and the results are compared.
#
# **From a picture to a mask** (`vessel_mask`), the same rule for every picture:
#
# 1. *Ridge strength.* A vessel is a bright ridge. At each of `RIDGE_WIDTHS` the picture is smoothed and the curvature across a ridge is measured; the best width at a pixel is taken as the vessel's half-width there.
# 2. *Lines.* A pixel is on a vessel's line when the ridge strength is at its largest there, going across the ridge. A line counts when it is stronger than 99% of what the background produces (`RIDGE_LEVELS[0]` times that), grown from places at `RIDGE_LEVELS[1]` times that.
# 3. *Paint.* Each line pixel is painted as a disc of `PAINT_WIDTH` times its half-width. The discs together are the mask.
#
# **From a mask to a graph** (`centerlines`, `graph_numbers`): the mask is thinned to lines one pixel wide, side branches shorter than `SPUR_LENGTH` are pruned, and the lines are cut at their forks into segments.
#
# **The pictures tried**, all with bright vessels:
#
# | Picture | What it is |
# |---|---|
# | mean frame, minus its log | the fixation's mean frame as it is, no background removed |
# | density, first guess of background | density against the upper envelope (`OD0`) |
# | density, background cleaned | the background-cleaned average frame (`OD`) |
# | mean of the frames' densities | `density_means` |
# | smear | the temporal burst's fluctuation (`burst_smears`) |
# | smear, background removed | `cleaned_smears` |
# | average of density and smear | section 17's average |
# | larger of density and smear | whichever of the two stands out more at a pixel |
# | both: root of density x smear | large only where both stand out |
# | flow mask | the vessel mask that `development.ipynb` saves as `flow_vessel_mask.png`, if the file is there |
#
# **How they are compared**, for the anchor's fixation:
#
# - *mask*: the share of the field the mask covers.
# - *length, segments, forks*: the size of the graph.
# - *loose ends*: ends of a centerline inside the field. A real vessel does not just stop, so these are breaks. *Per 1000 px* of centerline is the number to compare.
# - *networks, largest*: how many unconnected pieces the graph falls into, and the share of the length in the largest one.
# - *found again, and back*: the share of the anchor's centerlines that the other fixation's centerlines pass within `REPEAT_WITHIN` pixels, and the other way round. The two fixations are separate sets of frames, so this is how repeatable the tracing is. The other fixation has far fewer frames here, which counts against the pictures that need many frames (the smear).
#
# **Result:**
#
# - `traced`: for every `(picture name, fixation)`, its binary `mask` and its `centerlines` (both boolean images in the fixation's own reference frame).
# - `graph_table`: the numbers of every picture's graph in the anchor's fixation.
# - Three pictures of every candidate are saved at full size in `vessel_pictures/graphs`: the picture alone (`..._picture.png`), the picture with its centerlines (`..._centerlines.png`) and its binary mask with the centerlines (`..._mask.png`). `all_pictures_with_centerlines.png` is the figure below.

# %%
RIDGE_WIDTHS = (1.5, 2.1, 3, 4.2, 6, 8.5, 12, 17, 24)  # pixels; half-widths at which a vessel is looked for
RIDGE_LEVELS = (1.0, 3.0)  # x the ridge strength 99% of the background stays under: a line above the first, grown from the second
PAINT_WIDTH = 1.7          # the mask reaches this many ridge widths from a line: where a vessel with a flat profile ends
EDGE_MARGIN_LINES = 6      # pixels; no line is started this close to where the data ends
MIN_LINE = 15              # pixels; shorter pieces of line are dropped
SPUR_LENGTH = 12           # pixels; a side branch that ends within this length is pruned from the centerlines
REPEAT_WITHIN = 3          # pixels; a centerline counts as found again when the other one passes this close


def ridge_strength(picture, has_data):
    """Return how strongly each pixel lies on a bright ridge at the best of RIDGE_WIDTHS, that width, and the
    direction across the ridge (radians)."""
    rows, columns = picture.shape
    # where there is no data the picture is continued smoothly from around it, so that the edge of the data is no ridge
    known = cv2.GaussianBlur(has_data.astype(np.float32), (0, 0), 16)
    around = cv2.GaussianBlur(np.where(has_data, picture, 0).astype(np.float32), (0, 0), 16) / np.maximum(known, 1e-3)
    around[known < 1e-3] = np.median(picture[has_data])
    filled = np.where(has_data, picture, around).astype(np.float32)
    strength = np.zeros((rows, columns), dtype=np.float32)
    width = np.zeros((rows, columns), dtype=np.float32)
    across = np.zeros((rows, columns), dtype=np.float32)
    for sigma in RIDGE_WIDTHS:
        factor = max(1, int(sigma // 3))  # a wide ridge is smooth, so it is looked for on a coarser grid
        small = filled if factor == 1 else cv2.resize(
            filled, (columns // factor, rows // factor), interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(small, (0, 0), sigma / factor)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3) / 4
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3) / 4
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3) / 4
        # the curvature across a bright ridge is the more negative eigenvalue of the second derivatives. Times
        # width^1.5 it is largest at the width that equals the ridge's own (Lindeberg's ridge normalization)
        bend = 0.5 * (dxx + dyy - np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2))
        ridge = np.maximum(-bend, 0) * (sigma / factor) ** 1.5 * factor ** -0.5
        angle = 0.5 * np.arctan2(2 * dxy, dxx - dyy) + np.pi / 2  # direction of that eigenvalue: across the ridge
        if factor > 1:
            ridge = cv2.resize(ridge, (columns, rows), interpolation=cv2.INTER_LINEAR)
            angle = cv2.resize(angle, (columns, rows), interpolation=cv2.INTER_NEAREST)
        better = ridge > strength
        width[better], across[better] = sigma, angle[better]
        strength = np.maximum(strength, ridge)
    return strength, width, across


def vessel_mask(picture, has_data, background):
    """Return the binary vessel mask of a picture in which vessels are bright, and the lines it was painted from.

    A pixel is on a vessel's line when the ridge strength is at its largest there across the ridge, and clear of
    what the background produces. Each line pixel is then painted as a disc of the half-width found there.
    """
    rows, columns = picture.shape
    strength, width, across = ridge_strength(picture, has_data)
    x, y = np.meshgrid(np.arange(columns, dtype=np.float32), np.arange(rows, dtype=np.float32))
    step_x, step_y = np.cos(across), np.sin(across)
    one_side = cv2.remap(strength, x + step_x, y + step_y, cv2.INTER_LINEAR)
    other_side = cv2.remap(strength, x - step_x, y - step_y, cv2.INTER_LINEAR)
    inside = cv2.erode(has_data.astype(np.uint8), np.ones((2 * EDGE_MARGIN_LINES + 1,) * 2, np.uint8),
                       borderValue=0) > 0
    crest = (strength >= one_side) & (strength >= other_side) & inside
    level = np.percentile(strength[background & has_data], 99)
    lines = apply_hysteresis_threshold(np.where(crest, strength, 0), RIDGE_LEVELS[0] * level, RIDGE_LEVELS[1] * level)
    lines = remove_small_objects(lines, MIN_LINE, connectivity=2)
    # paint: a pixel is vessel when it lies within the half-width of its nearest line pixel
    distance, nearest = cv2.distanceTransformWithLabels((~lines).astype(np.uint8), cv2.DIST_L2, 5,
                                                        labelType=cv2.DIST_LABEL_PIXEL)
    width_of = np.zeros(nearest.max() + 1, dtype=np.float32)
    width_of[nearest[lines]] = width[lines]
    return (distance <= PAINT_WIDTH * width_of[nearest]) & has_data, lines


def skeleton_parts(skeleton):
    """Return the end pixels and fork pixels of a one-pixel-wide skeleton, and its pieces between forks."""
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.float32),
                              borderType=cv2.BORDER_CONSTANT) - 1
    ends = skeleton & (neighbours == 1)
    forks = skeleton & (neighbours >= 3)
    count, pieces = cv2.connectedComponents((skeleton & ~forks).astype(np.uint8), connectivity=8)
    return ends, forks, pieces, count


def centerlines(mask):
    """Return the centerlines of a binary mask: its skeleton, with the short side branches pruned."""
    skeleton = skeletonize(mask)
    for _ in range(2):  # pruning a spur can leave a shorter one behind it
        ends, forks, pieces, count = skeleton_parts(skeleton)
        length = np.bincount(pieces.ravel(), minlength=count)
        has_end = np.bincount(pieces[ends], minlength=count) > 0
        near_fork = cv2.dilate(forks.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        at_fork = np.bincount(pieces[near_fork & (pieces > 0)], minlength=count) > 0
        drop = has_end & (length < SPUR_LENGTH)  # a short side branch, or a short piece on its own
        drop[0] = False
        skeleton = skeletonize(skeleton & ~drop[pieces])
    return skeleton


def graph_numbers(skeleton, has_data):
    """Return what describes the graph of a set of centerlines: its length, segments, forks, loose ends, pieces."""
    ends, forks, pieces, count = skeleton_parts(skeleton)
    inside = cv2.distanceTransform(has_data.astype(np.uint8), cv2.DIST_L2, 5) > 20  # away from the edge of the data
    networks, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel(), minlength=networks)[1:]
    return {
        "length": int(skeleton.sum()),
        "segments": count - 1,
        "forks": cv2.connectedComponents(forks.astype(np.uint8), connectivity=8)[0] - 1,
        "loose ends": int((ends & inside).sum()),  # ends inside the field: a real vessel does not just stop
        "networks": networks - 1,
        "largest network": float(sizes.max() / max(sizes.sum(), 1)) if len(sizes) else 0.0,
    }


def distance_to(skeleton):
    """Return every pixel's distance to the nearest centerline pixel."""
    return cv2.distanceTransform((~skeleton).astype(np.uint8), cv2.DIST_L2, 5)


def found_again(skeleton, has_data, other_skeleton, other_has_data, to_other):
    """Return the share of one fixation's centerlines that the other fixation's pass within REPEAT_WITHIN pixels.

    to_other is the 3x3 map from this fixation's (x, y) to the other's; None when they share a frame.
    """
    size = skeleton.shape[::-1]
    distance, seen = distance_to(other_skeleton), other_has_data.astype(np.float32)
    if to_other is not None:  # bring the other fixation's distances and data area into this fixation's frame
        distance = cv2.warpPerspective(distance, to_other, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                       borderValue=1e3)
        seen = cv2.warpPerspective(seen, to_other, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    both = has_data & (cv2.erode(np.uint8(seen > 0.99), np.ones((9, 9), np.uint8)) > 0)
    here = skeleton & both
    return float((distance[here] <= REPEAT_WITHIN).mean()) if here.any() else np.nan

# %%
FLOW_MASK_FILE = "flow_vessel_mask.png"  # the vessel mask from the flow, saved by development.ipynb
FLOW_MASK_NAME = "flow mask"

def candidate_pictures(number):
    """Return the pictures of one fixation that a vessel graph could be traced from (vessels bright in all of them),
    where they have data, and the pixels that count as background."""
    fit, smear_fit, cleaned = background_fits[number], smear_fits[number], cleaned_smears[number]
    has_data = np.isfinite(cleaned) & fit["ok"]
    in_density = fit["OD"] / robust_sd(fit["OD"][fit["bg"] & has_data])        # in multiples of its background noise
    in_smear = cleaned / robust_sd(cleaned[smear_fit["bg"] & has_data])        # likewise
    pictures = {
        "mean frame, minus its log": -fit["L"],
        "density, first guess of background": fit["OD0"],
        "density, background cleaned": fit["OD"],
        "mean of the frames' densities": density_means[number],
        "smear": burst_smears[number],
        "smear, background removed": cleaned,
        "average of density and smear": (in_density + in_smear) / 2,
        "larger of density and smear": np.maximum(in_density, in_smear),
        "both: root of density x smear": np.sqrt(np.clip(in_density, 0, None) * np.clip(in_smear, 0, None)),
    }
    return pictures, has_data, quiet_pixels[number] & fit["bg"] & has_data


def trace(job):
    """Make one picture's binary mask and trace its centerlines."""
    name, number, picture, has_data, background = job
    mask, _ = vessel_mask(np.nan_to_num(picture), has_data & np.isfinite(picture), background)
    return (name, number), {"mask": mask, "centerlines": centerlines(mask)}


jobs, data_area = [], []
for number in range(len(fixations)):
    pictures, has_data, background = candidate_pictures(number)
    data_area.append(has_data)
    jobs += [(name, number, picture, has_data, background) for name, picture in pictures.items()]
with ThreadPoolExecutor(max_workers=workers) as pool:
    traced = dict(pool.map(trace, jobs))  # (picture name, fixation) -> its binary mask and its centerlines
del jobs, pictures

# one more candidate, if it is there: the vessel mask from the flow that development.ipynb saves. It is a binary
# mask already, for the whole burst in the anchor's frame, so it is only traced.
if os.path.exists(FLOW_MASK_FILE):
    flow_mask = np.asarray(Image.open(FLOW_MASK_FILE).convert("L")) > 0
    if flow_mask.shape == frame_shape:
        traced[FLOW_MASK_NAME, anchor] = {"mask": flow_mask, "centerlines": centerlines(flow_mask)}

# the numbers of every graph in the anchor's fixation, and how much of it the other fixation traces again
names = [name for name, number in traced if number == anchor]
others = [number for number in range(len(fixations)) if number != anchor]
graph_table = {}
print(f"{'picture':36s} {'mask':>5s} {'length':>7s} {'segments':>8s} {'forks':>6s} {'loose ends':>10s} {'per 1000 px':>11s} "
      f"{'networks':>8s} {'largest':>8s} {'found again':>11s} {'and back':>8s}")
for name in names:
    here = traced[name, anchor]
    numbers = graph_numbers(here["centerlines"], data_area[anchor])
    numbers["mask"] = here["mask"][data_area[anchor]].mean()
    again, back = [], []
    for number in others:
        if (name, number) in traced:
            to_other = anchor_to_reference(fixations[number])  # anchor (x, y) -> that fixation's reference frame
            there = traced[name, number]["centerlines"]
            again.append(found_again(here["centerlines"], data_area[anchor], there, data_area[number], to_other))
            back.append(found_again(there, data_area[number], here["centerlines"], data_area[anchor],
                                    np.linalg.inv(to_other)))
    numbers["found again"] = float(np.mean(again)) if again else np.nan
    numbers["and back"] = float(np.mean(back)) if back else np.nan
    graph_table[name] = numbers
    print(f"{name:36s} {numbers['mask'] * 100:4.0f}% {numbers['length']:7d} {numbers['segments']:8d} {numbers['forks']:6d} "
          f"{numbers['loose ends']:10d} {numbers['loose ends'] / max(numbers['length'], 1) * 1000:11.1f} "
          f"{numbers['networks']:8d} {numbers['largest network'] * 100:7.0f}% {numbers['found again'] * 100:10.0f}% "
          f"{numbers['and back'] * 100:7.0f}%")

# how much of each picture's centerlines the other pictures trace as well
print(f"\nShare of the row's centerlines that the column's pass within {REPEAT_WITHIN} px (anchor fixation):")
print(" " * 40 + " ".join(f"{column:3d}" for column in range(len(names))))
for row, name in enumerate(names):
    shares = [found_again(traced[name, anchor]["centerlines"], data_area[anchor],
                          traced[other, anchor]["centerlines"], data_area[anchor], None) for other in names]
    print(f"{row:3d} {name:36s} " + " ".join(f"{share * 100:3.0f}" for share in shares))

# show every picture with its centerlines drawn on it (yellow), and save three pictures of each at full size:
# the picture alone, the picture with its centerlines, and its binary mask with the centerlines
os.makedirs("vessel_pictures/graphs", exist_ok=True)
shown, has_data, _ = candidate_pictures(anchor)
if (FLOW_MASK_NAME, anchor) in traced:
    shown[FLOW_MASK_NAME] = traced[FLOW_MASK_NAME, anchor]["mask"].astype(np.float32)
columns = 2
figure, axes = plt.subplots(-(-len(names) // columns), columns, figsize=(16, 5.2 * -(-len(names) // columns)),
                            squeeze=False)
for axis in axes.ravel():
    axis.axis("off")
for axis, name in zip(axes.ravel(), names):
    here, picture = traced[name, anchor], shown[name]
    seen = np.isfinite(picture) if name == FLOW_MASK_NAME else has_data & np.isfinite(picture)
    low, high = (0, 1) if name == FLOW_MASK_NAME else np.percentile(picture[seen], [1, 99.5])  # black to white
    gray = np.uint8(np.clip((np.nan_to_num(picture) - low) / (high - low), 0, 1) * 255 * seen)
    drawn = cv2.dilate(here["centerlines"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    with_lines = np.dstack([gray, gray, gray])
    with_lines[drawn] = (255, 220, 0)
    on_mask = np.zeros_like(with_lines)
    on_mask[here["mask"]] = 110
    on_mask[drawn] = (255, 220, 0)
    file_name = "_".join("".join(letter if letter.isalnum() else " " for letter in name.lower()).split())
    for kind, image in (("picture", gray), ("centerlines", with_lines), ("mask", on_mask)):
        Image.fromarray(image).save(f"vessel_pictures/graphs/{file_name}_{kind}.png")
    axis.imshow(with_lines, interpolation="antialiased")
    axis.set_title(f"{name}: {graph_table[name]['length'] / 1000:.0f} thousand px of centerline", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/graphs/all_pictures_with_centerlines.png", dpi=120)
plt.show()
del shown
print(f"Saved the {len(names)} pictures, alone, with their centerlines and as masks, at full resolution in the "
      f"folder vessel_pictures/graphs")

# %%
#  we shouldnt be using binary masks to form our centerlines...
#
#  define a function that computes the convolution of orientation sensing neurons.
#
# it is important that we figure out how to perform lateral inhibition, as this is what will let us compute centerlines.
# they need to be local contrast-sensitive, so we may need a layer of contrast sensing neurons first.
#
# the functions below are the layers a picture passes through, in this order. No mask is made anywhere.
#   contrast_neurons     centre minus surround: how far a pixel stands out from the tissue around it
#   orientation_neurons  the convolution: 16 neurons per pixel (a hypercolumn), each tuned to one orientation of a line
#   firing_rates         responses become rates between 0 and 1, measured against what the background produces
#   lateral_inhibition   a neuron is silenced by a better placed or better oriented neighbour; a line's crest remains
#   string_neurons       neighbouring active neurons of nearly the same orientation are strung into centerlines,
#                        and the strings are pruned by three numbers
ORIENTATIONS = 16                         # neurons in every hypercolumn, one per orientation of a line
ORIENTATION_SIGMA = np.pi / ORIENTATIONS  # radians; width of a neuron's Gaussian tuning over orientation
NEURON_WIDTHS = (1.5, 3, 6, 12, 24)       # pixels; half-widths of the lines the neurons look for
CONTRAST_SURROUND = 64                    # pixels; width of the surround of a contrast neuron
FLANK_REACH = 2.0                         # a line neuron checks its two flanks this many line widths to either side
HALF_RATE = 2.0                           # a response this many times the noise level makes a neuron fire at half its maximum
EDGE_SILENT = 12                          # pixels; neurons this close to where the data ends stay silent
FIELD_PAD = 72                            # pixels of mirrored picture added around the field, so its edges do not meet


def contrast_neurons(picture, has_data, surround=CONTRAST_SURROUND):
    """Centre-surround neurons: how far each pixel stands out from the tissue around it.

    The pictures given to it are densities (logs), so centre minus surround is a contrast: it does not depend on
    how bright the place is. Where there is no data the output is continued smoothly from the pixels around, so
    that the edge of the data does not look like a line to the next layer.
    """
    weight = has_data.astype(np.float32)
    known = np.where(has_data, picture, 0)
    contrast = known - blur(known, surround) / np.maximum(blur(weight, surround), 1e-3)
    nearby = blur(np.where(has_data, contrast, 0), 16) / np.maximum(blur(weight, 16), 1e-3)
    return np.where(has_data, contrast, nearby).astype(np.float32)


def line_frequencies(shape, width):
    """Return, for every frequency of a field of the given shape, the across-the-line factor of a receptive
    field for lines of one half-width, and the direction of the frequency."""
    rows, columns = shape
    fy = 2 * np.pi * scipy.fft.fftfreq(rows)[:, None].astype(np.float32)
    fx = 2 * np.pi * scipy.fft.rfftfreq(columns)[None, :].astype(np.float32)
    radius = np.hypot(fx, fy)
    return (radius * width) ** 2 * np.exp(-0.5 * (radius * width) ** 2) * width ** -0.5, np.arctan2(fy, fx)


def orientation_tuning(across, direction, k):
    """Return the receptive field of orientation neuron k, as a transfer function.

    A neuron's receptive field is built in the frequency domain as a product of two smooth factors:
      across the line   a centre with an inhibiting flank on either side (second derivative of a Gaussian of the
                        line's width), which is what makes it answer to a line and not to a slope
      over orientation  a Gaussian of ORIENTATION_SIGMA around the neuron's own orientation
    Both factors are smooth, so the receptive field has no side lobes: one centre, two flanks, nothing beyond.
    The 16 neurons of a hypercolumn add up to the across factor alone: the unoriented centre-surround response.
    """
    # a line running along k * pi / 16 has its frequencies at right angles to it
    off = np.angle(np.exp(2j * (direction - k * np.pi / ORIENTATIONS - np.pi / 2))) / 2  # folded to +-90 degrees
    return across * np.exp(-0.5 * (off / ORIENTATION_SIGMA) ** 2) / np.sqrt(2 * np.pi)


def orientation_neurons(picture, widths=NEURON_WIDTHS, both_signs=False, threads=None):
    """Convolve a picture with the orientation neurons: returns, for each of the 16 orientations and each pixel,
    the largest response over the line widths, and which of the widths gave it (its place in `widths`).

    Two things keep a neuron honest, at every width:
      its hypercolumn   it cannot answer more than its whole hypercolumn does. The 16 add up to the unoriented
                        response, which is large only where the pixel itself stands out. A long receptive field
                        that merely points at a bright spot from a distance is silenced by this.
      its flanks        the smoothed picture has to fall off on both sides of it, FLANK_REACH widths away across
                        the line. On the shoulder of a wide vessel only one side falls off, so no line is seen there.
    With both_signs the size of the response counts, whichever its sign (for pictures of change, where a line
    can be brighter or darker than what is around it); otherwise only bright lines are answered.
    """
    rows, columns = picture.shape
    padded = cv2.copyMakeBorder(picture.astype(np.float32), FIELD_PAD, FIELD_PAD, FIELD_PAD, FIELD_PAD, cv2.BORDER_REFLECT)
    full = padded.shape
    best = np.zeros((ORIENTATIONS, rows, columns), dtype=np.float32)
    best_width = np.zeros((ORIENTATIONS, rows, columns), dtype=np.uint8)
    inner = (slice(FIELD_PAD, FIELD_PAD + rows), slice(FIELD_PAD, FIELD_PAD + columns))
    for place, width in enumerate(widths):
        factor = max(1, int(width // 3))  # a wide line is smooth, so it is looked for on a coarser grid
        small = padded if factor == 1 else cv2.resize(
            padded, (full[1] // factor, full[0] // factor), interpolation=cv2.INTER_AREA)
        size = small.shape[::-1]
        spectrum = scipy.fft.rfft2(small, workers=threads)
        across, direction = line_frequencies(small.shape, width / factor)
        column = scipy.fft.irfft2(spectrum * across, s=small.shape, workers=threads)  # the hypercolumn as a whole
        smooth = cv2.GaussianBlur(small, (0, 0), width / factor)
        reach = FLANK_REACH * width / factor
        for k in range(ORIENTATIONS):
            response = scipy.fft.irfft2(spectrum * orientation_tuning(across, direction, k), s=small.shape, workers=threads)
            theta = k * np.pi / ORIENTATIONS
            step = (-np.sin(theta) * reach, np.cos(theta) * reach)  # across a line of this orientation, as (x, y)
            sides = [cv2.warpAffine(smooth, np.float32([[1, 0, sign * step[0]], [0, 1, sign * step[1]]]), size,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for sign in (1, -1)]
            if both_signs:
                polarity = np.sign(column)
                layer = np.where(np.sign(response) == polarity, np.minimum(np.abs(response), np.abs(column)), 0)
                flanked = (polarity * (smooth - sides[0]) > 0) & (polarity * (smooth - sides[1]) > 0)
            else:
                layer = np.minimum(response, column)
                flanked = (smooth > sides[0]) & (smooth > sides[1])
            layer = np.where(flanked, layer, 0).astype(np.float32) * factor ** -0.5
            if factor > 1:
                layer = cv2.resize(layer, (full[1], full[0]), interpolation=cv2.INTER_LINEAR)
            layer = layer[inner]
            best_width[k][layer > best[k]] = place
            np.maximum(best[k], layer, out=best[k])
    return best, best_width


def firing_rates(responses, noise, has_data):
    """Turn responses into firing rates between 0 and 1: silent at 0, half the maximum at HALF_RATE x the noise
    level. Neurons within EDGE_SILENT pixels of where the data ends stay silent."""
    drive = np.maximum(responses / noise, 0) ** 2
    inside = cv2.erode(has_data.astype(np.uint8), np.ones((2 * EDGE_SILENT + 1,) * 2, np.uint8), borderValue=0) > 0
    return drive / (drive + HALF_RATE ** 2) * inside


def lateral_inhibition(rates):
    """Return which neurons stay active after lateral inhibition.

    Every neuron is inhibited by the neurons that see the same line less well placed or less well oriented: the
    two neurons of its own orientation one pixel to either side across the line, and its two neighbours in
    orientation in its own hypercolumn. Inhibition is winner-take-all: a neuron stays only if none of those four
    fires more. What stays is one neuron per place across a line: its crest.
    """
    rows, columns = rates.shape[1:]
    stays = np.zeros(rates.shape, dtype=bool)
    for k in range(ORIENTATIONS):
        theta = k * np.pi / ORIENTATIONS
        across = (-np.sin(theta), np.cos(theta))  # one pixel across a line of this orientation, as (x, y)
        sides = [cv2.warpAffine(rates[k], np.float32([[1, 0, sign * across[0]], [0, 1, sign * across[1]]]),
                                (columns, rows), flags=cv2.INTER_LINEAR) for sign in (1, -1)]
        stays[k] = ((rates[k] > 0) & (rates[k] >= sides[0]) & (rates[k] >= sides[1])
                    & (rates[k] >= rates[k - 1]) & (rates[k] >= rates[(k + 1) % ORIENTATIONS]))
    return stays


def string_neurons(rates, stays, low, high, min_length):
    """String the active neurons into centerlines.

    Two active neurons belong to one string when they are neighbours in the picture and at most one orientation
    step apart, so two vessels that cross stay two strings. Pruning has three numbers: neurons firing below `low`
    are not strung; a string is kept only if one of its neurons fires at `high` or more; and only if it has at
    least `min_length` neurons.

    Returns the kept neurons (orientation x rows x columns, boolean) and the number of strings.
    """
    active = stays & (rates >= low)
    wrapped = np.concatenate([active, active[:1]])  # orientation is a ring: the last one neighbours the first
    labels, count = scipy.ndimage.label(wrapped, structure=np.ones((3, 3, 3), dtype=bool))
    # the copy of the first orientation at the end joins the strings that run through the wrap
    parent = np.arange(count + 1)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    joined = wrapped[0]
    for a, b in np.unique(np.stack([labels[0][joined], labels[-1][joined]], axis=1), axis=0):
        parent[root(a)] = root(b)
    roots = np.array([root(a) for a in range(count + 1)])
    labels = roots[labels[:-1]]
    length = np.bincount(labels.ravel(), minlength=count + 1)
    strong = np.zeros(count + 1, dtype=bool)
    strong[np.unique(labels[active & (rates >= high)])] = True
    keep = strong & (length >= min_length)
    keep[0] = False
    return keep[labels], int(keep.sum())


def centerline_picture(kept):
    """Return the centerlines as an image (True on a line) and the orientation index of the line at each pixel."""
    return kept.any(axis=0), np.argmax(kept, axis=0)


def orientation_colors(lines, orientation):
    """Paint centerlines in the color of their orientation (a full turn of hue for half a turn of a line)."""
    hsv = np.zeros((*lines.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (orientation.astype(np.int32) * 180 // ORIENTATIONS).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(lines) * 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# the receptive fields of the 16 neurons of one hypercolumn, for lines 3 pixels wide: white excites, black inhibits
across, direction = line_frequencies((96, 96), 3.0)
figure, axes = plt.subplots(2, ORIENTATIONS // 2, figsize=(16, 4.3))
for k, axis in enumerate(axes.ravel()):
    field = np.fft.fftshift(scipy.fft.irfft2(orientation_tuning(across, direction, k), s=(96, 96)))[24:72, 24:72]
    axis.imshow(field, cmap="gray", vmin=-np.abs(field).max(), vmax=np.abs(field).max())
    axis.set_title(f"{k * 180 / ORIENTATIONS:.0f} degrees", fontsize=9)
    axis.axis("off")
figure.tight_layout()
plt.show()

# %%
#  deploy a neural field whose pixels have 16 orientations (neurons), and whose convolution kernel is a Gaussian in the orientation domain, with side lobes suppressed
#  once we compute the orientation score in each hypercolumn and each pixel, we can begin stringing adjacent, strong-scoring orientation pixels together to form centerlines,
#  since pruning is parametric, try different pruning thresholds and see how the centerlines change.
#
# this field looks at the still: the background-cleaned density of the anchor's fixation.
# a pruning is three numbers: the firing rate below which a neuron is not strung, the rate one neuron of a string
# has to reach, and the least number of neurons in a string
PRUNINGS = ((0.2, 0.5, 25), (0.35, 0.65, 25), (0.5, 0.8, 25), (0.65, 0.9, 25),  # the firing rates varied
            (0.35, 0.65, 10), (0.35, 0.65, 60), (0.35, 0.65, 150))              # the length varied
CHOSEN_PRUNING = (0.35, 0.65, 25)  # the pruning the comparisons further down use


def drawn(lines, orientation, under, low, high):
    """Draw centerlines in the colors of their orientation over a dimmed gray picture."""
    gray = np.uint8(np.clip((np.nan_to_num(under) - low) / (high - low), 0, 1) * 110)
    picture = np.dstack([gray, gray, gray])
    thick = cv2.dilate(lines.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    colors = orientation_colors(thick, cv2.dilate(orientation.astype(np.uint8), np.ones((3, 3), np.uint8)))
    picture[thick] = colors[thick]
    return picture


still_fit = background_fits[anchor]
still_data = np.isfinite(cleaned_smears[anchor]) & still_fit["ok"]   # the data area the pictures of the sections above share
still_background = quiet_pixels[anchor] & still_fit["bg"] & still_data
responses, still_widths = orientation_neurons(contrast_neurons(still_fit["OD"], still_data), threads=workers)
still_noise = np.percentile(responses[:, still_background][:, ::3], 99)  # the level 99% of the background neurons stay under
still_rates = firing_rates(responses, still_noise, still_data)       # (orientation, rows, columns): the orientation score
still_stays = lateral_inhibition(still_rates)
del responses
print(f"Orientation-sensitive field: {ORIENTATIONS} neurons at each of {still_data.sum()} pixels; "
      f"{still_stays.mean() * 100:.1f}% of them stay active after lateral inhibition")

# string and prune, with every pruning in turn. For reference: how the lines sit on the centerlines that the
# vessel-graph section traced from the same picture by way of a binary mask
reference = traced["density, background cleaned", anchor]["centerlines"]
still_prunings = {}
print(f"{'not strung below':>16s} {'a string reaches':>16s} {'least length':>12s} {'strings':>8s} {'length (px)':>11s} "
      f"{'on the mask lines':>17s} {'mask lines found':>16s}")
for pruning in PRUNINGS:
    kept, strings = string_neurons(still_rates, still_stays, *pruning)
    lines, orientation = centerline_picture(kept)
    still_prunings[pruning] = (lines, orientation)
    if pruning == CHOSEN_PRUNING:
        still_kept = kept  # (orientation, rows, columns): the neurons on the centerlines
    print(f"{pruning[0]:16.2f} {pruning[1]:16.2f} {pruning[2]:12d} {strings:8d} {int(lines.sum()):11d} "
          f"{found_again(lines, still_data, reference, still_data, None) * 100:16.0f}% "
          f"{found_again(reference, still_data, lines, still_data, None) * 100:15.0f}%")
del kept

# show how the centerlines change with the pruning: the firing rates varied on top, the length varied below.
# The color of a line is its orientation.
os.makedirs("vessel_pictures/neural", exist_ok=True)
shown = (PRUNINGS[0], PRUNINGS[2], PRUNINGS[3], PRUNINGS[4], PRUNINGS[5], PRUNINGS[6])
figure, axes = plt.subplots(2, 3, figsize=(16, 7.2))
for axis, pruning in zip(axes.ravel(), shown):
    picture = drawn(*still_prunings[pruning], still_fit["OD"], *DENSITY_RANGE)
    Image.fromarray(picture).save("vessel_pictures/neural/still_pruning_{}_{}_{}.png".format(*pruning))
    axis.imshow(picture, interpolation="antialiased")
    axis.set_title("not strung below {}, a string reaches {}, at least {} long".format(*pruning), fontsize=9)
    axis.axis("off")
figure.tight_layout()
plt.show()
still_lines, still_orientation = still_prunings[CHOSEN_PRUNING]
Image.fromarray(drawn(still_lines, still_orientation, still_fit["OD"], *DENSITY_RANGE)).save(
    "vessel_pictures/neural/still_centerlines.png")
del still_prunings

# %%
#  deploy another neural field whose pixels have 16 orientations (neurons), and whose convolution kernel is a Gaussian in the orientation domain, with side lobes suppressed.
#  this layer, however, is passed the difference between two frames as input, divided by the timestep between frames, so that it is sensitive to the flow of the vessels and gating of frames
#
# the second layer outputs the firing rates of each hypercolumn for the entire frame for every frame, and we can then compute the centerlines of the flow-sensitive layer, and compare them to the centerlines of the orientation-sensitive layer.
#  we can prune these too. prune them parametrically
#
# this field looks at change: for every two frames of the anchor's fixation that follow each other, the density of
# the later one minus that of the earlier one, divided by the time between them (longer where frames were gated out
# in between). A line of change can be lighter or darker, so the size of a response counts, whatever its sign.
# The field's answers to one pair of frames are 16 pictures; for the whole burst that is far too much to keep
# (19 GB at full size), so they are added up as they come.
# Inside a vessel that flows, the change between two frames is a texture of cells arriving and leaving. It has no
# direction of its own, so the 16 neurons of a hypercolumn answer it about equally, and a centerline cannot be read
# off them directly. What does have a line is where the hypercolumns keep answering. So the field works in two stages:
#   1. for every pair of frames, each hypercolumn's answer: its 16 neurons added up. These are added over the pairs.
#   2. the mean of that over the burst is a picture with a bright ridge along every vessel that flows. It goes
#      through the same layers as the still: contrast neurons, orientation neurons, firing rates, lateral inhibition.
FLOW_SCALE = 2          # the flow-sensitive field works on frames reduced by this factor
FLOW_WIDTHS = (1.5, 3)  # reduced pixels; half-widths of the lines of change stage 1 looks for (3 and 6 pixels of a frame)
FLOW_LINE_WIDTHS = tuple(width / FLOW_SCALE for width in NEURON_WIDTHS[1:])  # reduced pixels; the lines stage 2 looks for

# the original number of every gated frame, so that the time between two frames is known
original_number, position = [], 0
for index, frame in enumerate(raw_burst):
    if position < num_frames and np.array_equal(frame[::16, ::16], gated_frames[position][::16, ::16]):
        original_number.append(index)
        position += 1

flow_members = fixations[anchor]["frames"]
flow_density = density_fixations[anchor]
flow_size = (frame_width // FLOW_SCALE, frame_height // FLOW_SCALE)
flow_total = np.zeros(flow_size[::-1], dtype=np.float32)  # the hypercolumns' answers, added up over the pairs
flow_seen = np.zeros(flow_size[::-1], dtype=np.float32)
flow_activity = np.zeros(len(flow_members) - 1)  # how strongly the whole field answered each pair of frames
flow_timesteps = np.zeros(len(flow_members) - 1)
adding = threading.Lock()


def answer_pair(position):
    """Pass one pair of frames through the flow-sensitive field and add its answers to the totals."""
    timestep = (original_number[flow_members[position + 1]] - original_number[flow_members[position]]) / fps
    change = (flow_density[position + 1] - flow_density[position]) / timestep  # density per second
    has_data = np.isfinite(change)
    seen = cv2.resize(has_data.astype(np.float32), flow_size, interpolation=cv2.INTER_AREA)
    change = cv2.resize(np.where(has_data, change, 0), flow_size, interpolation=cv2.INTER_AREA) / np.maximum(seen, 1e-3)
    seen = seen > 0.99
    answers, _ = orientation_neurons(contrast_neurons(change, seen, CONTRAST_SURROUND / FLOW_SCALE), FLOW_WIDTHS,
                                     both_signs=True)
    # a hypercolumn close to where this pair's data ends answers to that edge, so it is not counted for this pair
    margin = 2 * (EDGE_SILENT // FLOW_SCALE) + 1
    seen = cv2.erode(seen.astype(np.uint8), np.ones((margin, margin), np.uint8), borderValue=0) > 0
    column = answers.sum(axis=0) * seen  # how strongly each hypercolumn answers this pair of frames
    with adding:
        np.add(flow_total, column, out=flow_total)
        np.add(flow_seen, seen, out=flow_seen)
    flow_activity[position], flow_timesteps[position] = column.mean(), timestep


with ThreadPoolExecutor(max_workers=workers) as pool:
    list(pool.map(answer_pair, range(len(flow_members) - 1)))

# stage 2. Each hypercolumn's mean answer over the pairs that had data is the picture of where things keep changing.
# Noise alone makes every hypercolumn answer a little, the same amount everywhere; the median over the background
# is that amount, and it is taken off.
flow_data = flow_seen >= 0.5 * (len(flow_members) - 1)
flow_background = cv2.resize(still_background.astype(np.uint8), flow_size, interpolation=cv2.INTER_NEAREST) > 0
flow_background &= flow_data
flow_picture = np.where(flow_data, flow_total / np.maximum(flow_seen, 1), 0)
flow_picture = np.where(flow_data, flow_picture - np.median(flow_picture[flow_background]), 0).astype(np.float32)
responses, flow_widths = orientation_neurons(
    contrast_neurons(flow_picture, flow_data, CONTRAST_SURROUND / FLOW_SCALE), FLOW_LINE_WIDTHS, threads=workers)
flow_noise = np.percentile(responses[:, flow_background][:, ::3], 99)
flow_rates = firing_rates(responses, flow_noise, flow_data)  # (orientation, rows / 2, columns / 2)
flow_stays = lateral_inhibition(flow_rates)
del flow_total, responses
longer = flow_timesteps > 1.5 / fps
print(f"Flow-sensitive field: {len(flow_activity)} pairs of frames of fixation {anchor}, {longer.sum()} of them with "
      f"a dropped frame in between | the field's answer per pair varies by "
      f"{flow_activity.std() / flow_activity.mean() * 100:.0f}% around its mean, and is "
      f"{flow_activity[longer].mean() / flow_activity[~longer].mean() if longer.any() else np.nan:.2f} x as large across "
      f"a dropped frame as otherwise")


def full_size(lines, orientation):
    """Bring centerlines of the flow-sensitive field to the size of a frame."""
    size = (frame_width, frame_height)
    return (cv2.resize(lines.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0,
            cv2.resize(orientation.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST))


# string and prune as above (lengths count reduced pixels here, so the least length is divided by FLOW_SCALE),
# and compare with the centerlines of the orientation-sensitive field at its chosen pruning
flow_prunings = {}
print(f"{'not strung below':>16s} {'a string reaches':>16s} {'least length':>12s} {'strings':>8s} {'length (px)':>11s} "
      f"{'on the still lines':>18s} {'still lines found':>17s}")
for pruning in PRUNINGS:
    kept, strings = string_neurons(flow_rates, flow_stays, pruning[0], pruning[1], pruning[2] // FLOW_SCALE)
    lines, orientation = full_size(*centerline_picture(kept))
    flow_prunings[pruning] = (lines, orientation)
    if pruning == CHOSEN_PRUNING:
        flow_kept = kept  # (orientation, rows / 2, columns / 2): the neurons on the centerlines
    print(f"{pruning[0]:16.2f} {pruning[1]:16.2f} {pruning[2]:12d} {strings:8d} "
          f"{int(centerline_picture(kept)[0].sum()) * FLOW_SCALE:11d} "
          f"{found_again(lines, still_data, still_lines, still_data, None) * 100:17.0f}% "
          f"{found_again(still_lines, still_data, lines, still_data, None) * 100:16.0f}%")
del kept

# show how the flow-sensitive centerlines change with the pruning, over the picture of where things keep changing
flow_under = cv2.resize(flow_picture, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
flow_top = np.percentile(flow_picture[flow_data], 99.5)
figure, axes = plt.subplots(2, 3, figsize=(16, 7.2))
for axis, pruning in zip(axes.ravel(), shown):
    picture = drawn(*flow_prunings[pruning], flow_under, 0, flow_top)
    Image.fromarray(picture).save("vessel_pictures/neural/flow_pruning_{}_{}_{}.png".format(*pruning))
    axis.imshow(picture, interpolation="antialiased")
    axis.set_title("not strung below {}, a string reaches {}, at least {} long".format(*pruning), fontsize=9)
    axis.axis("off")
figure.tight_layout()
plt.show()
flow_lines, flow_orientation = flow_prunings[CHOSEN_PRUNING]
del flow_prunings

# %%
# show combinations, oriented flow-sensitive centerlines, oriented orientation-sensitive centerlines, and the intersection of the two.
#
# the intersection is taken neuron by neuron: a neuron on a centerline of the still is in it when the flow-sensitive
# field has a centerline neuron within REPEAT_WITHIN pixels of it whose orientation is the same or one step away
flow_kept_full = np.zeros(still_kept.shape, dtype=bool)
reach = np.ones((2 * REPEAT_WITHIN + 1,) * 2, np.uint8)
for k in range(ORIENTATIONS):
    layer = cv2.resize(flow_kept[k].astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    flow_kept_full[k] = cv2.dilate(layer, reach) > 0
both_kept = np.zeros(still_kept.shape, dtype=bool)
for k in range(ORIENTATIONS):
    both_kept[k] = still_kept[k] & (flow_kept_full[k - 1] | flow_kept_full[k] | flow_kept_full[(k + 1) % ORIENTATIONS])
both_lines, both_orientation = centerline_picture(both_kept)
del flow_kept_full

# the combination: every centerline, colored by which field found it
thick = np.ones((3, 3), np.uint8)
combination = np.zeros((*frame_shape, 3), dtype=np.uint8)
combination[cv2.dilate(flow_lines.astype(np.uint8), thick) > 0] = (255, 60, 220)    # the flow-sensitive field only: magenta
combination[cv2.dilate(still_lines.astype(np.uint8), thick) > 0] = (60, 220, 255)   # the orientation-sensitive field only: cyan
combination[cv2.dilate(both_lines.astype(np.uint8), thick) > 0] = (255, 255, 255)   # both: white

pictures = (
    ("orientation-sensitive field (the still): centerlines in the color of their orientation",
     drawn(still_lines, still_orientation, still_fit["OD"], *DENSITY_RANGE), "still_centerlines"),
    ("flow-sensitive field (change between frames): centerlines in the color of their orientation",
     drawn(flow_lines, flow_orientation, flow_under, 0, flow_top), "flow_centerlines"),
    ("intersection: on a centerline of both fields, with the same orientation",
     drawn(both_lines, both_orientation, still_fit["OD"], *DENSITY_RANGE), "both_centerlines"),
    ("combination: cyan the still only, magenta the flow only, white both", combination, "combination"),
)
figure, axes = plt.subplots(2, 2, figsize=(16, 10.4))
for axis, (title, picture, file_name) in zip(axes.ravel(), pictures):
    Image.fromarray(picture).save(f"vessel_pictures/neural/{file_name}.png")
    axis.imshow(picture, interpolation="antialiased")
    axis.set_title(title, fontsize=10)
    axis.axis("off")
figure.tight_layout()
plt.show()
print(f"Centerlines at the pruning {CHOSEN_PRUNING}: {int(still_lines.sum())} px from the orientation-sensitive field, "
      f"{int(flow_lines.sum()) // FLOW_SCALE} px from the flow-sensitive field, {int(both_lines.sum())} px in both "
      f"({both_lines.sum() / max(still_lines.sum(), 1) * 100:.0f}% of the still's)")
print("Saved the four pictures and the prunings at full resolution in the folder vessel_pictures/neural")

# %%
#  the unpruned still centerlines look best, but many vessels are heavily fractured and have continuations nearby.
#  consolidate them: a horizontally recurrent layer on the orientation-sensitive field, to connect the pieces of a
#  vessel that are separated by a gap, a junction or a crossover.
#
# the neurons on the centerlines talk to each other within their own layer, along the lines they lie on:
#   horizontal_support     what a neuron hears, ahead and behind, from the neurons that lie on one smooth arc with it;
#                          its support is what its two sides agree on
#   horizontal_recurrence  silent neurons with enough support start to fire, and speak in the next round: bridges
#   thin_lines             the neurons' lines as lines one pixel wide, pieces a pixel or two apart joined
#   run_into_lines         what the layer does not do: an end that stops just short of the side of another line
#                          is extended to it (a branch meeting the vessel it comes from)
# three things turned out to be needed, each found by its failure:
#   only kept strings speak. When every active neuron spoke, the noise fragments supported each other and wove a
#   web over the plain tissue.
#   two neurons support each other only along a smooth arc. With a plain fan around the axis, two lines running
#   side by side filled the space between them with a honeycomb.
#   a bridge has to lie on something. Bridges over tissue where the picture shows nothing lay on fluctuating
#   tissue no more often than any pixel does; the others as often as the centerlines themselves.
CONSOLIDATED_PRUNING = PRUNINGS[0]  # the loosest pruning: these are the centerlines that are consolidated
HORIZONTAL_REACH = 16.0        # pixels; how far along a line a neuron listens to either side
HORIZONTAL_WIDTH = 0.8         # pixels; how far off the line it listens, next to itself
HORIZONTAL_TURNS = (-1, 0, 1)  # orientation steps; it listens to neurons turned this far from itself
HORIZONTAL_ROUNDS = 3          # times the field is passed through its horizontal connections
ALONGSIDE = 3                  # pixels; a line silences bridging neurons of its own orientation this far to either side
BRIDGE_EVIDENCE = 0.1          # firing rate; the hypercolumns under a bridge fire at least this much on the picture (median)
JOIN_WITHIN = 2                # pixels; pieces of line this close to each other are drawn as one
SPUR = 6                       # pixels; side branches shorter than this are pruned from the thin lines
JUNCTION_REACH = 16            # pixels; a free end is extended straight ahead to a line it meets within this distance


def line_patch(angle, radius):
    """Return, for a square patch around a neuron, how far every place lies along and across a line through it
    at the given angle, and that line drawn through the middle of the patch."""
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    line = np.zeros(along.shape, dtype=np.uint8)
    tip = (int(round(radius * np.cos(angle))), int(round(radius * np.sin(angle))))
    cv2.line(line, (radius - tip[0], radius - tip[1]), (radius + tip[0], radius + tip[1]), 1)
    return along, across, line


def horizontal_lobes(k, turn, reach=HORIZONTAL_REACH):
    """Return the two lobes through which orientation neuron k listens to the neurons turned `turn` orientation
    steps from it: ahead and behind.

    Two neurons support each other when one smooth arc passes through both. A neuron turned by some angle lies
    on such an arc where it is seen at half that angle off the listener's axis, so the lobe for neurons turned by
    `turn` steps is a narrow fan around the ray half as many steps off the axis. It fades with distance (a
    Gaussian of `reach`) and widens with it by as much as half an orientation step allows. A line running
    alongside, parallel and a few pixels off, lies on no such arc and is not heard. The weights are scaled so
    that a straight line along the ray, firing at 1 all the way, is heard as 1.
    """
    step = np.pi / ORIENTATIONS
    along, across, line = line_patch(k * step + turn * step / 2, int(3 * reach))
    width = HORIZONTAL_WIDTH + np.tan(step / 2) * np.abs(along)
    weight = np.exp(-0.5 * (along / reach) ** 2 - 0.5 * (across / width) ** 2)
    lobes = []
    for side in (1, -1):
        lobe = np.where(side * along > 0.5, weight, 0)
        lobes.append((lobe / (lobe * line).sum()).astype(np.float32))
    return lobes


def horizontal_support(level, reach=HORIZONTAL_REACH, turns=HORIZONTAL_TURNS):
    """Return what every neuron hears through its horizontal connections from the neurons that speak (`level`
    is their firing; silent neurons are 0).

    What a neuron hears on each side is a weighted sum of the firing of the neurons that lie on a smooth arc
    with it. Its support is what its two sides agree on, the smaller of the two: a gap between two pieces of one
    line is supported from both ends, and so is a line that disappears under another one and comes out on the
    far side. The stretch beyond a free end is heard from one side only: no support.
    """
    support = np.empty(level.shape, dtype=np.float32)

    def hear(k):
        ahead, behind = 0, 0
        for turn in turns:
            speakers = level[(k + turn) % ORIENTATIONS]
            to_ahead, to_behind = horizontal_lobes(k, turn, reach)
            ahead = ahead + cv2.filter2D(speakers, -1, to_ahead, borderType=cv2.BORDER_CONSTANT)
            behind = behind + cv2.filter2D(speakers, -1, to_behind, borderType=cv2.BORDER_CONSTANT)
        support[k] = np.clip(np.minimum(ahead, behind), 0, 1)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(hear, range(ORIENTATIONS)))
    return support


def alongside(kept, reach=ALONGSIDE):
    """Return, for every neuron, whether a kept neuron of its own orientation (or one step from it) lies within
    `reach` pixels of it across its axis: the place is taken."""
    taken = np.zeros(kept.shape, dtype=bool)
    for k in range(ORIENTATIONS):
        near = (kept[k - 1] | kept[k] | kept[(k + 1) % ORIENTATIONS]).astype(np.uint8)
        _, _, line = line_patch((k + ORIENTATIONS // 2) * np.pi / ORIENTATIONS, reach)  # a short line across the axis
        taken[k] = cv2.dilate(near, line) > 0
    return taken


def horizontal_recurrence(rates, kept, low, rounds=HORIZONTAL_ROUNDS, reach=HORIZONTAL_REACH,
                          turns=HORIZONTAL_TURNS, evidence=BRIDGE_EVIDENCE):
    """Pass the field through its horizontal connections `rounds` times. Returns the kept neurons with the
    bridging neurons added, and the level at which every one of them fires.

    Only kept neurons speak. In every round a silent neuron starts to fire, at the level of its support, when
      its support is `low` or more,
      it is the crest of that support (lateral inhibition, as in the feedforward field),
      no kept line of its own orientation runs alongside it (a longer-range lateral inhibition), and
      the bridge it is part of lies on something: the hypercolumns under the bridge fire, on the picture itself,
      at `evidence` or more (their median). A bridge over plain tissue is dropped as a whole.
    A neuron that was kept is never lost, and a bridge built in one round speaks in the next.
    """
    seen = rates.max(axis=0)
    kept = kept.copy()
    level = np.where(kept, rates, 0).astype(np.float32)
    for round_number in range(rounds):
        support = horizontal_support(level, reach, turns)
        bridge = lateral_inhibition(support) & (support >= low) & ~alongside(kept)
        pieces, count = scipy.ndimage.label(bridge.any(axis=0), structure=np.ones((3, 3), dtype=bool))
        accepted = np.ones(count + 1, dtype=bool)
        if evidence and count:
            accepted[1:] = scipy.ndimage.median(seen, pieces, index=np.arange(1, count + 1)) >= evidence
            bridge &= accepted[pieces]
        level[bridge] = support[bridge]
        kept |= bridge
        print(f"   round {round_number + 1}: {int(accepted[1:].sum())} bridges, {int(bridge.sum())} neurons start to fire; "
              f"{count - int(accepted[1:].sum())} more bridges dropped, over tissue where the picture shows nothing")
        del support, bridge
    return kept, level


def thin_lines(lines, join=JOIN_WITHIN, spur=SPUR):
    """Return a picture of lines as lines one pixel wide. Pieces within `join` pixels of each other are joined
    first, and side branches shorter than `spur` pixels (free at one end, at a fork at the other) are pruned."""
    if join:
        lines = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join + 1,) * 2)) > 0
    skeleton = skeletonize(lines)
    for _ in range(2):  # pruning a spur can leave a shorter one behind it
        ends, forks, pieces, count = skeleton_parts(skeleton)
        length = np.bincount(pieces.ravel(), minlength=count)
        has_end = np.bincount(pieces[ends], minlength=count) > 0
        near_fork = cv2.dilate(forks.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        at_fork = np.bincount(pieces[near_fork & (pieces > 0)], minlength=count) > 0
        drop = has_end & at_fork & (length < spur)
        drop[0] = False
        skeleton = skeletonize(skeleton & ~drop[pieces])
    return skeleton


def run_into_lines(lines, seen, evidence=BRIDGE_EVIDENCE, reach=JUNCTION_REACH):
    """Extend every free end of a picture of thin lines straight ahead to the first other line it meets within
    `reach` pixels, if the hypercolumns under the extension fire at `evidence` or more (`seen`, their median).
    Returns the lines with the extensions drawn in, and the extensions alone."""
    ends, forks, pieces, count = skeleton_parts(lines)
    rows, columns = lines.shape
    added = np.zeros(lines.shape, dtype=np.uint8)
    for y, x in zip(*np.nonzero(ends)):
        top, left = max(y - 10, 0), max(x - 10, 0)
        own_y, own_x = np.nonzero(pieces[top:y + 11, left:x + 11] == pieces[y, x])  # the last stretch of its line
        outward = np.array([x - (own_x.mean() + left), y - (own_y.mean() + top)])   # the end points away from it
        if len(own_y) < 4 or np.hypot(*outward) < 1:
            continue  # a speck: no direction
        outward /= np.hypot(*outward)
        for step in np.arange(2, reach + 0.5, 0.5):
            px, py = int(round(x + step * outward[0])), int(round(y + step * outward[1]))
            if not (0 <= px < columns and 0 <= py < rows):
                break
            if lines[py, px] and pieces[py, px] != pieces[y, x]:
                extension = np.zeros(lines.shape, dtype=np.uint8)
                cv2.line(extension, (int(x), int(y)), (px, py), 1)
                if np.median(seen[extension > 0]) >= evidence:
                    added |= extension
                break
    return lines | (added > 0), (added > 0) & ~lines


def over_gray(under, low, high, layers):
    """Draw sets of lines, each in its own color, over a dimmed gray picture."""
    gray = np.uint8(np.clip((np.nan_to_num(under) - low) / (high - low), 0, 1) * 110)
    picture = np.dstack([gray, gray, gray])
    for lines, color in layers:
        picture[cv2.dilate(lines.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = color
    return picture


# the centerlines as they are, the same with pieces a pixel or two apart joined, after the horizontal layer, and
# with the ends extended that run into a line
kept, _ = string_neurons(still_rates, still_stays, *CONSOLIDATED_PRUNING)
as_they_are = thin_lines(kept.any(axis=0), join=0)
joined = thin_lines(kept.any(axis=0))
print(f"Horizontal layer on the centerlines at the pruning {CONSOLIDATED_PRUNING}: {int(kept.sum())} neurons speak")
consolidated_kept, consolidated_level = horizontal_recurrence(still_rates, kept, CONSOLIDATED_PRUNING[0])
bridged = thin_lines(consolidated_kept.any(axis=0))
consolidated_lines, extensions = run_into_lines(bridged, still_rates.max(axis=0))
del kept

# two checks that do not use the still: does what was added lie where the tissue fluctuates from frame to frame
# (the cleaned smear stands out), and does a centerline of the flow-sensitive field pass within REPEAT_WITHIN px?
smear = cleaned_smears[anchor]
moving = np.nan_to_num(smear) > STANDS_OUT * robust_sd(smear[smear_fits[anchor]["bg"] & np.isfinite(smear)])
near_flow = distance_to(cv2.resize(flow_lines.astype(np.uint8), (frame_width, frame_height),
                                   interpolation=cv2.INTER_NEAREST) > 0) <= REPEAT_WITHIN
not_new = cv2.dilate(joined.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # within 2 px of a line that was there
steps = (("as they are", as_they_are), (f"pieces within {JOIN_WITHIN} px joined", joined),
         ("after the horizontal layer", bridged), ("ends extended into lines", consolidated_lines))
consolidation_table = {}
print(f"{'':28s} {'length (px)':>11s} {'forks':>6s} {'loose ends':>10s} {'per 1000 px':>11s} {'networks':>8s} "
      f"{'largest':>7s} {'added (px)':>10s} {'fluctuates':>10s} {'flow line':>9s}")
for place, (name, lines) in enumerate(steps):
    numbers = graph_numbers(lines, still_data)
    added = lines & ~not_new & (place >= 2)  # what the two consolidating steps added to the joined centerlines
    numbers["added"] = int(added.sum())
    numbers["added fluctuates"] = float(moving[added].mean()) if added.any() else np.nan
    numbers["added near a flow line"] = float(near_flow[added].mean()) if added.any() else np.nan
    consolidation_table[name] = numbers
    print(f"{name:28s} {numbers['length']:11d} {numbers['forks']:6d} {numbers['loose ends']:10d} "
          f"{numbers['loose ends'] / max(numbers['length'], 1) * 1000:11.1f} {numbers['networks']:8d} "
          f"{numbers['largest network'] * 100:6.0f}% {numbers['added']:10d} "
          + (f"{numbers['added fluctuates'] * 100:9.0f}% {numbers['added near a flow line'] * 100:8.0f}%" if added.any() else ""))
print(f"For comparison: of the joined centerlines {moving[joined].mean() * 100:.0f}% lie where the tissue fluctuates and "
      f"{near_flow[joined].mean() * 100:.0f}% near a flow line; of the field as a whole {moving[still_data].mean() * 100:.0f}% "
      f"and {near_flow[still_data].mean() * 100:.0f}%")

# show before and after: the whole field on top, the middle of it enlarged below. White: the centerlines as they
# were (joined); magenta: added by the horizontal layer; cyan: ends extended into a line
WHITE, MAGENTA, CYAN = (255, 255, 255), (255, 60, 220), (60, 220, 255)
before = over_gray(still_fit["OD"], *DENSITY_RANGE, [(joined, WHITE)])
after = over_gray(still_fit["OD"], *DENSITY_RANGE, [(consolidated_lines, MAGENTA), (consolidated_lines & not_new, WHITE),
                                                    (extensions, CYAN)])
Image.fromarray(before).save("vessel_pictures/neural/consolidated_before.png")
Image.fromarray(after).save("vessel_pictures/neural/consolidated_after.png")
middle = (slice(frame_height // 2 - 150, frame_height // 2 + 150), slice(frame_width // 2 - 240, frame_width // 2 + 240))
figure, axes = plt.subplots(2, 2, figsize=(16, 10.4))
for axis, (title, picture) in zip(axes.ravel(), (
        (f"centerlines at the pruning {CONSOLIDATED_PRUNING}, pieces within {JOIN_WITHIN} px joined", before),
        ("consolidated: magenta added by the horizontal layer, cyan ends extended into a line", after),
        ("the middle of the field, before", before[middle]), ("the middle of the field, after", after[middle]))):
    axis.imshow(picture, interpolation="antialiased" if picture.shape[0] > 400 else "nearest")
    axis.set_title(title, fontsize=10)
    axis.axis("off")
figure.tight_layout()
plt.show()
print("Saved both pictures at full resolution in the folder vessel_pictures/neural (consolidated_before, consolidated_after)")

# %% [markdown]
# # From the density burst to typed vessel junctions
#
# Sections 19 to 28 carry the burst from its density frames to a graph of the vessels whose junctions are read: which arms belong to one vessel, where a vessel splits, where two vessels merely cross. They follow the notes at the end of section 18 and replace the five cells above this one (the first neuron fields and their consolidation), which nothing below needs.
#
# **Principles**
#
# - **No binary masks for structure.** Centerlines are crests of an orientation field. They are thinned by lateral inhibition and strung together in position and orientation. Masks are only drawn for display.
# - **Nothing is trained.** Every receptive field is fixed. Thresholds are multiples of the noise measured on this burst's quiet pixels. Sizes are in pixels.
# - **Two fields, one mechanism.** The static field sees density, the flow field sees how density changes. Both go through the same hypercolumns, the same interactions and the same stringing.
# - **Every step ends with a check**: a picture and a number.
#
# | Section | What happens | Result |
# |---|---|---|
# | 19. Contrast layer | The mean frame's density is put in multiples of the local noise. | `contrast_maps`, `valid_maps`, `quiet_maps` |
# | 20. Hypercolumns | 16 orientation neurons at 8 widths at every pixel answer to lines, not to edges, blobs or line ends. | `orientation_scores`, `radius_maps` |
# | 21. Interactions | The neurons sharpen each other's tuning, inhibit each other across a vessel and support each other along it. Crests are read out. | `static_fields`, `static_crests` |
# | 22. Chains | Crest neurons are strung into chains, gaps are bridged, and the chains are pruned at a setting on a plateau. | `static_chains`, `pruning_table` |
# | 23. Is the motion visible? | The line mass along a chain in one frame is correlated with the next frame, shifted along the chain. | `flow_visible` |
# | 24. Flow field | The hypercolumns are given the change between frames; where things keep changing is traced like the static field. | `flow_energy`, `flow_chains` |
# | 25. Signals per chain | Kymograph, velocity, pulse, blur and lucky frames of every chain. | `segment_signals` |
# | 26. One field, one graph | Both fields are compared and combined; the chains become a graph, and every junction's arms are read on a ring outside its core. | `vessel_chains`, `vessel_graphs` |
# | 27. Junctions | Every way of dividing a junction's arms into vessels is scored; the most specific one that fits is chosen. | `junctions` |
# | 28. Synthetic bursts | The same code reads bursts made with everything known. | the numbers it prints |
# | 29. Vessels from the static score | Centerlines read straight from section 20's score, as splines, each with its flow; junctions read with flow as evidence. | `spline_vessels`, `ridge_junctions` |
#
# **A note on `vascnet`.** The plan these sections follow reuses a package called `vascnet` for its receptive fields, association field, thinning, junction rules, synthetic data and metrics. That package is not on this machine, so everything is written out here, from the plan's description of it. Where a detail was not given, the cell that makes the choice says so.
#
# **Terms used below**
#
# - **Hypercolumn**: the neurons at one pixel, one for each of 16 orientations of a line.
# - **Noise unit**: the spread of what a neuron answers on the quiet pixels. A neuron that answers 4.5 stands 4.5 of those above them.
# - **Half-width**: the Gaussian width (sigma) of the line a neuron sees, as in section 18. A round vessel of radius R shows as a line of half-width 0.836 R.
# - **Chain**: a centerline as a list of points, each with a place, an orientation, a half-width and a strength.
# - **Line mass**: a frame's density added up across a vessel. It hardly changes with defocus and is what is conserved at a junction.
# - **Arm**: the part of a vessel that leaves a junction, read on a ring outside the junction's core.

# %% [markdown]
# ### What the sections below share
#
# Three things are set up once. The first cell below defines, the second runs.
#
# - **Frame times.** The gate of section 3 now returns which recorded frames it kept (`gated_indices`), so every frame has a time (`frame_times`, and `times` in every fixation's dictionary). Where a frame was dropped the next one comes later. Anything that compares a frame with the next uses only the pairs that are `MAX_GAP_FRAMES` recorded frames apart: `frame_pairs`.
# - **The data area of a fixation**, `fixation_has_data`: where the mean density and the background fit both have data, less the rim of `EDGE_MARGIN` pixels.
# - **One temporal frame.** `temporal_frame(number, position)` is a frame's density with the shading that moves from frame to frame taken out (section 14): positive in vessels, about 0 on background. All work on flow starts from it. The cell checks that the mean density minus a temporal frame is the right-hand picture of section 14's video.
# - **Memory.** `memory_note(step)` prints what the kernel holds after a step and the most it has held.

# %%
MAX_GAP_FRAMES = 1  # recorded frames; two kept frames are compared as a pair only when they are this far apart, at most


def temporal_frame(number, position):
    """Return one frame of a fixation as the flow sections use it: its density with the shading that moves from
    frame to frame taken out. Positive in vessels, about 0 on background, NaN where the frame has no data."""
    shading = enlarge(frame_shadings[number][position]) + np.float32(frame_centres[number][position])
    return density_fixations[number][position] - shading


def memory_note(step):
    """Print how much memory this kernel holds after a step, and the most it has held so far."""
    held = psutil.Process().memory_info()
    print(f"[memory] {step}: {held.private / 2**30:.2f} GB now, {held.peak_pagefile / 2**30:.2f} GB at the most so far")

# %%
fixation_has_data = []  # per fixation: True where the mean density and the background fit both have data, off the rim
frame_pairs = []        # per fixation: (earlier position, later position, seconds between them) of the pairs compared
rim = np.ones((2 * EDGE_MARGIN + 1,) * 2, np.uint8)
for number, (fixation, fit) in enumerate(zip(fixations, background_fits)):
    inside = np.isfinite(density_means[number]) & fit["ok"]
    fixation_has_data.append(cv2.erode(inside.astype(np.uint8), rim, borderValue=0) > 0)
    steps = np.diff(fixation["times"])
    frame_pairs.append([(position, position + 1, float(step)) for position, step in enumerate(steps)
                        if step <= (MAX_GAP_FRAMES + 0.5) / fps])
    print(f"Fixation {number}: {len(fixation['frames'])} frames from {fixation['times'][0]:.3f} s to "
          f"{fixation['times'][-1]:.3f} s | {len(frame_pairs[number])} of {len(steps)} pairs of neighbouring frames are "
          f"{MAX_GAP_FRAMES} frame apart, the others have dropped frames between them | data on "
          f"{fixation_has_data[number].mean() * 100:.1f}% of the field")

# check: the mean density minus a temporal frame is the right-hand picture of section 14's video
largest = 0.0
for number, fixation in enumerate(fixations):
    for position in range(0, len(fixation["frames"]), max(1, len(fixation["frames"]) // 6)):
        fixed = density_means[number] - density_fixations[number][position] + frame_centres[number][position]
        followed = fixed + enlarge(frame_shadings[number][position])
        largest = max(largest, float(np.nanmax(np.abs(followed - (density_means[number] - temporal_frame(number, position))))))
print(f"temporal_frame against section 14's right-hand video, on a sample of frames: largest difference {largest:.1e} of density")

# %% [markdown]
# ## 19. Contrast layer
#
# The first layer of neurons says how many times the local noise a pixel stands above the background.
#
# - **Input:** the density of each fixation's mean frame, `background_fits[n]["OD"]`.
# - **Where there is no data** the picture is continued smoothly from around the hole and falls to 0, the density of background, far from any data. The edge of the data is then no line to the next layer.
# - **Normalizing:** the density is divided by the local noise, `background_fits[n]["noise"]`. Density has already divided the illumination out; this divides the noise out. A faint vessel in a dim corner and a dark vessel in the centre are then judged on one scale.
#
# **Check:** on the quiet pixels the contrast should have a median of 0 and a robust spread of 1. The fit's noise map was measured on the density after a 1 pixel blur, so it is off by a constant factor that differs between fixations; the cell prints it and divides it out, one number per fixation (`contrast_gains`).
#
# **Result:** `contrast_maps`, `valid_maps` (where there is data) and `quiet_maps` (the quiet background pixels with data, on which every noise level below is measured).

# %%
CONTRAST_FILL = 16  # pixels; how far the picture is continued smoothly into the places without data


def contrast_layer(density, noise, has_data):
    """Contrast-sensing neurons: how many times the local noise each pixel stands above the background.

    Density has already divided the illumination out of the picture; dividing by the local noise puts a faint
    vessel in a dim corner and a dark vessel in the centre on one scale. Where there is no data the picture is
    continued smoothly from the pixels around the hole and falls to 0, the background's density, far from any
    data, so that the edge of the data is no line to the next layer.
    """
    known = blur(has_data.astype(np.float32), CONTRAST_FILL)
    around = blur(np.where(has_data, density, 0), CONTRAST_FILL) / np.maximum(known, 0.05)
    return (np.where(has_data, density, around) / noise).astype(np.float32)

# %%
contrast_maps = []   # per fixation: the mean frame's density in multiples of the local noise
valid_maps = []      # per fixation: True where that is data
quiet_maps = []      # per fixation: the quiet background pixels with data, on which every noise level below is measured
contrast_gains = []  # per fixation: what the fit's noise map was multiplied by to make the quiet pixels' spread 1
for number, fit in enumerate(background_fits):
    valid = fixation_has_data[number]
    quiet = quiet_pixels[number] & fit["bg"] & valid
    contrast = contrast_layer(fit["OD"], fit["noise"], valid)
    centre, spread = np.median(contrast[quiet]), robust_sd(contrast[quiet])
    # the fit's noise map is the spread of the density after a 1 pixel blur; the layer works on the density as it is,
    # so the map is brought to this picture's own level by one number per fixation
    contrast /= spread
    contrast_maps.append(contrast)
    valid_maps.append(valid)
    quiet_maps.append(quiet)
    contrast_gains.append(float(spread))
    print(f"Fixation {number}: on the {quiet.sum()} quiet pixels the contrast has median {centre:+.2f} and robust sd "
          f"{spread:.2f} with the fit's noise map as it is ({'within' if abs(spread - 1) <= 0.1 else 'not within'} 10% "
          f"of 1); after dividing by {spread:.2f}: median {np.median(contrast[quiet]):+.2f}, robust sd "
          f"{robust_sd(contrast[quiet]):.2f}, mean {contrast[quiet].mean():+.2f}")

# show the layer for the anchor's fixation: what goes in, the noise it is divided by, and what comes out
fit = background_fits[anchor]
figure, axes = plt.subplots(1, 3, figsize=(16, 3.6))
shown = (
    ("density of the mean frame", fit["OD"], DENSITY_RANGE),
    ("local noise of the density", fit["noise"], (0, np.percentile(fit["noise"], 99))),
    ("contrast: density / local noise (black -5, white +25)", contrast_maps[anchor], (-5, 25)),
)
for axis, (title, image, (low, high)) in zip(axes, shown):
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## 20. Orientation hypercolumns
#
# At every pixel sit `N_ORIENT` = 16 neurons, one per orientation, at each of 8 half-widths (`HYPER_WIDTHS`, 1.5 to 17 pixels). Widths of 6 pixels and more are worked out on reduced grids, as in section 18.
#
# **Three receptive fields per neuron** (`line_pair`), each `ELONGATION` = 2.5 times as long as the line is wide. They are built in the frequency domain, so applying one is a product and an inverse transform:
#
# - **even**: a centre with an inhibiting flank on either side. It answers to a line.
# - **odd**: one side excites, the other inhibits. It answers to an edge.
# - **halves**: the front half of the even field minus its back half. A line that runs on gives 0; a blob, a line end or a junction does not.
#
# **What a neuron passes on** (`complex_unit`): `[even - alpha |odd| - beta |halves|]+` with alpha = beta = 1. The odd term keeps the walls of a wide vessel from being read as lines. The halves term is end-stopping: it keeps spurs from growing out of every junction.
#
# **Competition between widths.** At every pixel and orientation the strongest width wins. A parabola through its answer and its two neighbours' gives the half-width between the steps. The winner's answer is divided by its channel's noise (the spread of the even field over the quiet pixels) and has its channel's floor taken off.
#
# **Result:**
#
# - `orientation_scores[n]`: `(16, rows, columns)`, in noise units.
# - `radius_maps[n]`: the half-width of the line each neuron sees, in pixels.
#
# **Checks**, printed for the anchor's fixation:
#
# - On the large vessels the winning orientation should be the structure tensor's.
# - The half-widths should agree with those section 18 found, on section 18's centerlines.
# - Section 18's centerlines should lie where the hypercolumns answer.

# %%
N_ORIENT = 16                    # neurons per hypercolumn; neuron k looks for a line running along k x 180 / 16 degrees
ELONGATION = 2.5                 # a receptive field is this many times as long as the line it looks for is wide
EDGE_SUPPRESSION = 1.0           # alpha: how strongly the odd (edge) field silences a neuron
END_STOPPING = 1.0               # beta: how strongly a difference between the two halves of its field silences it
HYPER_WIDTHS = RIDGE_WIDTHS[:8]  # pixels; the half-widths (Gaussian sigma) of the lines looked for, as in section 18
CYLINDER = 0.836                 # a round vessel of radius R shows as a line of half-width 0.836 R
BANK_PAD = 32                    # pixels of mirrored picture put around a field, so that its edges do not meet
VALID_SHARE = 0.9                # a neuron answers only where this share of its receptive field has data
NOISE_SAMPLE = 200_000           # quiet pixels on which a channel's noise level is measured

DAWSON_AT = np.linspace(0, 40, 8001)
DAWSON = scipy.special.dawsn(DAWSON_AT)  # the transform of half a Gaussian needs Dawson's function


def reduction(width):
    """A wide line is smooth, so it is looked for on a grid reduced by this factor (as section 18 does)."""
    return max(1, int(width // 3))


@functools.lru_cache(maxsize=16)
def bank_frequencies(shape):
    """Return the frequencies (radians per pixel) of the half-plane FFT of a field of the given shape."""
    return ((2 * np.pi * scipy.fft.fftfreq(shape[0])).astype(np.float32)[:, None],
            (2 * np.pi * scipy.fft.rfftfreq(shape[1])).astype(np.float32)[None, :])


def line_pair(shape, width, k):
    """Return the three receptive fields of the neuron for orientation k and half-width `width`, as transfer
    functions (real arrays; the second and third are to be multiplied by i).

      even   a line detector: a centre with an inhibiting flank on either side across the line (second derivative
             of a Gaussian of the line's width), a Gaussian ELONGATION times longer along it
      odd    an edge detector of the same size: first derivative across, the same Gaussian along. It is scaled to
             answer to noise as strongly as the even field does
      halves the front half of the even field minus its back half. A line that runs on gives 0; a blob, a line end
             or a junction does not
    All three are in the units of section 18's ridge strength: curvature across the line times width^1.5, which is
    largest at the width that equals the line's own.
    """
    fy, fx = bank_frequencies(shape)
    theta = k * np.pi / N_ORIENT
    along = (np.cos(theta) * fx + np.sin(theta) * fy) * np.float32(ELONGATION * width)
    across = (-np.sin(theta) * fx + np.cos(theta) * fy) * np.float32(width)
    profile = np.exp(-0.5 * across**2) * np.float32(width**-0.5)
    even = across**2 * profile
    length = np.exp(-0.5 * along**2)
    half = np.interp(np.abs(along) / np.sqrt(2), DAWSON_AT, DAWSON).astype(np.float32) * np.sign(along)
    return even * length, np.float32(np.sqrt(1.5)) * across * profile * length, np.float32(2 / np.sqrt(np.pi)) * even * half


def padded_spectrum(picture, factor):
    """Reduce a picture by `factor`, put a mirrored border around it and return its half-plane FFT, the shape that
    was transformed and where the picture sits in it."""
    rows, columns = picture.shape
    small = picture if factor == 1 else cv2.resize(picture, (columns // factor, rows // factor),
                                                   interpolation=cv2.INTER_AREA)
    rows, columns = small.shape
    shape = tuple(scipy.fft.next_fast_len(size + 2 * BANK_PAD, real=True) for size in (rows, columns))
    padded = cv2.copyMakeBorder(small.astype(np.float32), BANK_PAD, shape[0] - rows - BANK_PAD,
                                BANK_PAD, shape[1] - columns - BANK_PAD, cv2.BORDER_REFLECT)
    return scipy.fft.rfft2(padded), shape, (slice(BANK_PAD, BANK_PAD + rows), slice(BANK_PAD, BANK_PAD + columns))


def field_answers(spectrum, shape, inner, width, k):
    """Return what the even field, the odd field and the difference of the two halves answer, for one neuron."""
    even, odd, halves = line_pair(shape, width, k)
    return (scipy.fft.irfft2(spectrum * even, s=shape)[inner],
            scipy.fft.irfft2(spectrum * (1j * odd), s=shape)[inner],
            scipy.fft.irfft2(spectrum * (1j * halves), s=shape)[inner])


def complex_unit(even, odd, halves):
    """What a neuron passes on: its even field's answer, less what the edge field and the end-stopping take away."""
    return np.maximum(even - EDGE_SUPPRESSION * np.abs(odd) - END_STOPPING * np.abs(halves), 0)


def orientation_hypercolumns(contrast, valid, quiet, widths=HYPER_WIDTHS, threads=None):
    """Pass a contrast picture through the hypercolumns: N_ORIENT neurons at every pixel and every width.

    At every pixel and orientation the widths compete and the strongest wins; its width, refined by a parabola
    through its two neighbours in log width, is the line's half-width there. The winner's answer is then divided by
    the noise of its channel (the spread of its even field over the quiet pixels) and has the floor of its channel
    taken off (what the quiet pixels' winners of that width answer, in the median).

    Returns the orientation score (N_ORIENT, rows, columns) in noise units, the half-width (same shape, pixels),
    and the noise level and floor of every width.
    """
    rows, columns = contrast.shape
    size = (columns, rows)
    grids, seen, sample = {}, {}, {}
    for width in widths:
        factor = reduction(width)
        if factor not in grids:
            grids[factor] = padded_spectrum(contrast, factor)
        small = (columns // factor, rows // factor)
        # the share of a receptive field that has data; the field's long axis sets how far a hole is felt
        share = blur(valid.astype(np.float32), ELONGATION * width)
        seen[width] = cv2.resize(share, small, interpolation=cv2.INTER_AREA) >= VALID_SHARE
        places = np.flatnonzero(cv2.resize((quiet & (share >= VALID_SHARE)).astype(np.uint8), small,
                                           interpolation=cv2.INTER_NEAREST))
        sample[width] = places[::max(1, len(places) // NOISE_SAMPLE)]
    scores = np.zeros((N_ORIENT, rows, columns), dtype=np.float32)
    half_width = np.zeros((N_ORIENT, rows, columns), dtype=np.float16)
    winner = np.zeros((N_ORIENT, rows, columns), dtype=np.uint8)
    noise = np.zeros((N_ORIENT, len(widths)))
    step = np.log(widths[-1] / widths[0]) / (len(widths) - 1)  # the widths are evenly spaced in log width

    def one_orientation(k):
        best, below, above = (np.zeros((rows, columns), dtype=np.float32) for _ in range(3))
        previous = best
        for place, width in enumerate(widths):
            factor = reduction(width)
            even, odd, halves = field_answers(*grids[factor], width / factor, k)
            noise[k, place] = robust_sd(even.ravel()[sample[width]]) * factor**-0.5
            answer = np.where(seen[width], complex_unit(even, odd, halves), 0).astype(np.float32) * np.float32(factor**-0.5)
            if factor > 1:
                answer = cv2.resize(answer, size, interpolation=cv2.INTER_LINEAR)
            if place:
                behind = winner[k] == place - 1
                above[behind] = answer[behind]  # the answer one width up from the best so far
            better = answer > best
            below[better], above[better] = previous[better], 0
            best[better], winner[k][better] = answer[better], place
            previous = answer
        # the parabola through the best width's answer and its two neighbours' gives the width between the steps
        curve = below - 2 * best + above
        inside = (winner[k] > 0) & (winner[k] < len(widths) - 1) & (curve < 0)
        offset = np.where(inside, np.clip(0.5 * (below - above) / np.where(inside, curve, 1), -0.5, 0.5), 0)
        half_width[k] = np.asarray(widths, dtype=np.float32)[winner[k]] * np.exp(step * offset)
        scores[k] = best

    with ThreadPoolExecutor(max_workers=threads or 1) as pool:
        list(pool.map(one_orientation, range(N_ORIENT)))

    noise = np.median(noise, axis=0)
    for k in range(N_ORIENT):
        scores[k] /= noise.astype(np.float32)[winner[k]]
    floor = np.zeros(len(widths))
    where_quiet = np.flatnonzero(quiet)[::max(1, int(quiet.sum()) // NOISE_SAMPLE)]
    quiet_scores, quiet_winner = scores.reshape(N_ORIENT, -1)[:, where_quiet], winner.reshape(N_ORIENT, -1)[:, where_quiet]
    for place in range(len(widths)):
        values = quiet_scores[quiet_winner == place]
        floor[place] = np.median(values) if len(values) else 0.0
    for k in range(N_ORIENT):
        scores[k] = np.maximum(scores[k] - floor.astype(np.float32)[winner[k]], 0)
    return scores, half_width, noise, floor


def orientation_hues(orientation, strength):
    """Paint orientations as hues (a full turn of hue for half a turn of a line), as bright as `strength` (0 to 1)."""
    hsv = np.zeros((*orientation.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (orientation.astype(np.float32) * (180 / N_ORIENT)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.clip(strength, 0, 1) * 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def tensor_orientation(picture, inner, outer):
    """Return the orientation of the lines in a picture by its structure tensor (radians, 0 to pi): the gradient
    at the scale `inner`, its products averaged over `outer` pixels. Lines run at right angles to the gradient."""
    smooth = cv2.GaussianBlur(picture.astype(np.float32), (0, 0), inner)
    gx, gy = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(product, (0, 0), outer) for product in (gx * gx, gy * gy, gx * gy))
    return (0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2) % np.pi


def winning_neurons(scores, half_widths):
    """Return, per pixel, the strongest neuron's answer, its orientation number and its half-width."""
    orientation = np.argmax(scores, axis=0)
    return (np.take_along_axis(scores, orientation[None], axis=0)[0], orientation,
            np.take_along_axis(half_widths, orientation[None], axis=0)[0].astype(np.float32))

# %% [markdown]
# The cell below passes every fixation's contrast through the hypercolumns and makes the three checks. In the picture, the hue is the winning orientation and the brightness its answer.

# %%
# the three receptive fields of one neuron (orientation 2, lines 3 pixels wide): white excites, black inhibits
even, odd, halves = line_pair((96, 96), 3.0, 2)
figure, axes = plt.subplots(1, 3, figsize=(9, 3.2))
shown = (("even: a line", even), ("odd: an edge", 1j * odd), ("front half minus back half", 1j * halves))
for axis, (title, transfer) in zip(axes, shown):
    field = np.fft.fftshift(scipy.fft.irfft2(transfer, s=(96, 96)))[24:72, 24:72]
    axis.imshow(field, cmap="gray", vmin=-np.abs(field).max(), vmax=np.abs(field).max())
    axis.set_title(title, fontsize=10)
    axis.axis("off")
figure.tight_layout()
plt.show()

LARGE_VESSEL = 6.0  # pixels; from this half-width up a vessel counts as large in the check of the orientations

orientation_scores = []  # per fixation: (N_ORIENT, rows, columns) float16, the hypercolumns' answers in noise units
radius_maps = []         # per fixation: (N_ORIENT, rows, columns) float16, the half-width of the line each neuron sees
channel_noise = []       # per fixation: the noise level and the floor of each width
for number in range(len(fixations)):
    scores, half_width, noise, floor = orientation_hypercolumns(
        contrast_maps[number], valid_maps[number], quiet_maps[number], threads=max(1, workers // 2))
    strongest = scores.max(axis=0)
    orientation_scores.append(scores.astype(np.float16))
    radius_maps.append(half_width)
    channel_noise.append((noise, floor))
    quiet, valid = quiet_maps[number], valid_maps[number]
    print(f"Fixation {number}: the strongest neuron of a hypercolumn answers above 2 on "
          f"{(strongest[quiet] > 2).mean() * 100:.1f}% of the quiet pixels and above 4.5 on "
          f"{(strongest[quiet] > 4.5).mean() * 100:.2f}%; over the whole field on {(strongest[valid] > 2).mean() * 100:.0f}% "
          f"and {(strongest[valid] > 4.5).mean() * 100:.0f}%")
    print("   noise level per width: " + " ".join(f"{value:.2f}" for value in noise)
          + " | floor per width: " + " ".join(f"{value:.2f}" for value in floor))
    del scores


# checks, on the anchor's fixation
fit, valid = background_fits[anchor], valid_maps[anchor]
strongest, orientation, half_width = winning_neurons(orientation_scores[anchor], radius_maps[anchor])
strongest = strongest.astype(np.float32)

# 1. on the large vessels the winning orientation agrees with the structure tensor's
large = valid & (strongest > 4.5) & (half_width >= LARGE_VESSEL)
off = np.abs(orientation[large] * np.pi / N_ORIENT - tensor_orientation(fit["OD"], 3, 12)[large])
off = np.degrees(np.minimum(off, np.pi - off))
print(f"Orientation, on the {large.sum()} pixels of large vessels (half-width {LARGE_VESSEL:g} px or more): the winning "
      f"neuron is {np.median(off):.1f} degrees from the structure tensor's orientation (median), within one step of "
      f"{180 / N_ORIENT:.2f} degrees on {(off <= 180 / N_ORIENT).mean() * 100:.0f}%")

# 2. the half-widths agree with those section 18 found, on section 18's centerlines of the same picture
_, width_before, _ = ridge_strength(fit["OD"], valid)
lines = traced["density, background cleaned", anchor]["centerlines"] & valid & (width_before > 0)
ratio = np.log2(half_width[lines] / width_before[lines])
print(f"Half-width, on the {lines.sum()} px of section 18's centerlines: {2 ** np.median(ratio):.2f} x section 18's "
      f"(median), within one step of the widths (x 1.41) on {(np.abs(ratio) <= 0.5).mean() * 100:.0f}%, correlation of "
      f"the logs {np.corrcoef(np.log(half_width[lines]), np.log(width_before[lines]))[0, 1]:.2f}")

# 3. section 18's centerlines lie where the hypercolumns answer
near = cv2.dilate(strongest, np.ones((2 * REPEAT_WITHIN + 1,) * 2, np.uint8))
print(f"Of section 18's centerlines, {(near[lines] > 2).mean() * 100:.0f}% have a neuron answering above 2 within "
      f"{REPEAT_WITHIN} px and {(near[lines] > 4.5).mean() * 100:.0f}% one above 4.5")

# show the orientation score (hue: the winning orientation, brightness: its answer) and the half-widths
os.makedirs("vessel_pictures/hypercolumns", exist_ok=True)
score_picture = orientation_hues(orientation, strongest / 20)
width_picture = np.where(strongest > 2, half_width, np.nan)
Image.fromarray(score_picture).save("vessel_pictures/hypercolumns/orientation_score.png")
figure, axes = plt.subplots(1, 2, figsize=(16, 5.3))
axes[0].imshow(score_picture, interpolation="antialiased")
axes[0].set_title("orientation score: hue is the winning orientation, full brightness at 20 x the noise", fontsize=10)
shown = axes[1].imshow(width_picture, cmap="turbo", vmin=HYPER_WIDTHS[0], vmax=HYPER_WIDTHS[-1])
axes[1].set_title("half-width of the winning neuron's line, where it answers above 2 (pixels)", fontsize=10)
figure.colorbar(shown, ax=axes[1], fraction=0.03)
for axis in axes:
    axis.axis("off")
figure.tight_layout()
plt.show()
del strongest, orientation, half_width, near, score_picture, width_picture

# %% [markdown]
# ## 21. Interactions between the neurons, and the crests
#
# This is the step that makes centerlines. Three fixed interactions are applied `LATERAL_STEPS` = 4 times, each as a relaxed update `U <- (1 - eta) U + eta f(U)` with `eta` = 0.7.
#
# 1. **Tuning over orientation** (`tune_orientation`). Each hypercolumn's 16 answers are convolved over orientation with a centre (1 step wide) minus a surround (3 steps wide). This sharpens the tuning and silences blobs, which drive every orientation alike. A gain, measured once on a plain line, gives the best neuron of a line its answer back.
# 2. **Inhibition across the vessel** (`flank_activity`). A neuron is divided by the activity on its two flanks: `U <- U / (1 + flanks / FLANK_HALF)`. The flanks are two lobes `FLANK_OFFSET` = 1.5 half-widths to either side, longer along the line than across. A neuron speaks to the flanks of its own line's width, so a wide vessel's crest reaches the side ridges along its walls, and a thin one only its own shoulders. Repeating this pulls a ridge in towards its crest.
# 3. **Support along the vessel** (`association_support`). A neuron is raised by neurons that lie along its line and continue it on a smooth arc: `U <- U (1 + a F / (h + F))`. It multiplies, so it strengthens what is there and draws nothing where there was no answer.
#
# **Crossings are protected.** Where two orientations far apart are active in one hypercolumn, the part of the surround that comes from the other orientation is held back. Otherwise a thick vessel would erase a thin one inside the overlap, and a crossing would become two T's.
#
# **Reading the crests** (`crest_neurons`). A crest neuron answers above `T_LOW`, more than its two neighbours in orientation, and more than the neurons one pixel to either side across the line, of its own orientation and the two next to it. Each crest neuron carries its place (to a fraction of a pixel), orientation, strength and half-width. Its strength is its answer before the interactions. The support along a line raises a chance row of noise as well, so only that answer is still in noise units, and it is what the pruning of section 22 goes by.
#
# **Not as in the plan:** the plan asks for crests that are connected within one orientation to a neuron above `T_HIGH`. A line that turns changes orientation along its length, so here that test is made on whole chains in section 22: a chain needs one neuron above `T_HIGH`.
#
# **Checks:** the width at half maximum of the answer across five vessels, before and after; and the share of crest neurons with a parallel duplicate alongside, before and after.
#
# **Result:** `static_fields[n]` (the field after the interactions) and `static_crests[n]` (its crest neurons).

# %%
THETA_CENTRE = 1.0       # orientation steps; width of the centre of the tuning over orientation
THETA_SURROUND = 3.0     # orientation steps; width of the surround that is taken off it
CROSSING_APART = 3       # orientation steps; two lines at least this far apart in orientation are a crossing
CROSSING_HALF = 2.0      # noise units; a crossing this strong halves the inhibition between its two orientations
FLANK_OFFSET = 1.5       # half-widths; where across a line the inhibiting flanks of a neuron sit
FLANK_SPREAD = 1 / 3     # of that offset; how wide each flank is across the line (it is ELONGATION times longer along)
FLANK_HALF = 4.0         # noise units; flank activity at which a neuron's answer is halved (1 / k)
ASSOCIATION_REACH = 10   # pixels; how far along its line a neuron hears its neighbours
ASSOCIATION_WIDTH = 1.5  # pixels; how far to the side of its line it still hears them
ASSOCIATION_TURNS = (-1, 0, 1)  # orientation steps; a neighbour may be turned by this much (on one arc with the neuron)
MODULATION = 0.5         # a: the neighbours along the line can raise a neuron's answer by this share, at most
HALF_SAT = 3.0           # h, noise units; support along the line at which half of that is reached
LATERAL_STEPS = 4        # times the three interactions are applied
LATERAL_ETA = 0.7        # eta: the share of each update that is taken
LATERAL_SLAB = 200       # rows worked on at a time by the tuning, to keep the temporary arrays small
SILENT = 0.02            # noise units; a hypercolumn whose neurons all answer less than this is left alone
T_LOW = 2.0              # noise units; a crest neuron below this is not read out
T_HIGH = 4.5             # noise units; a centerline needs one neuron above this


def orientation_weights(sigma):
    """A Gaussian over the ring of orientations, as a matrix: row k holds neuron k's weights for all N_ORIENT."""
    apart = np.minimum(np.arange(N_ORIENT), N_ORIENT - np.arange(N_ORIENT))
    row = np.exp(-0.5 * (apart / sigma) ** 2)
    return np.stack([np.roll(row / row.sum(), k) for k in range(N_ORIENT)]).astype(np.float32)


def ideal_tuning(width=3.0, size=128, between=4):
    """What the neurons of one hypercolumn answer on a straight line of their own width: one row of N_ORIENT
    answers for each of `between` line directions from one neuron's orientation to the next."""
    y, x = np.mgrid[:size, :size].astype(np.float32) - size / 2
    whole = (slice(None), slice(None))
    curves = []
    for part in range(between):
        angle = part / between * np.pi / N_ORIENT
        line = np.exp(-0.5 * ((-np.sin(angle) * x + np.cos(angle) * y) / width) ** 2)
        spectrum = scipy.fft.rfft2(line)
        curves.append([complex_unit(*field_answers(spectrum, line.shape, whole, width, k))[size // 2, size // 2]
                       for k in range(N_ORIENT)])
    return np.array(curves)


ring = np.minimum(np.arange(N_ORIENT), N_ORIENT - np.arange(N_ORIENT))
FAR_APART = np.stack([np.roll(ring >= CROSSING_APART, k) for k in range(N_ORIENT)]).astype(np.float32)
TUNING_CENTRE, TUNING_SURROUND = orientation_weights(THETA_CENTRE), orientation_weights(THETA_SURROUND)
# three sums over a hypercolumn in one product: its centre, and the surround's parts from near and from far orientations
TUNING_PARTS = np.concatenate([TUNING_CENTRE, TUNING_SURROUND * (1 - FAR_APART), TUNING_SURROUND * FAR_APART])
tuning_curves = ideal_tuning()
# the tuning takes a surround off every neuron; this gain gives the best neuron of a plain line its answer back
TUNING_GAIN = float(np.mean(tuning_curves.max(axis=1)
                            / ((TUNING_CENTRE - TUNING_SURROUND) @ tuning_curves.T).max(axis=0)))


def tune_orientation(columns, protect_crossings=True):
    """Interaction 1: sharpen every hypercolumn's tuning over orientation, and silence what drives all its neurons.

    Each hypercolumn's N_ORIENT answers are convolved over orientation with a centre minus a surround. A blob, which
    drives every orientation alike, is left with nothing. Where two lines of clearly different orientation lie in
    one hypercolumn (a crossing), the surround's part that comes from orientations far apart is held back, or the
    stronger line would erase the weaker one inside the overlap.

    `columns` is (N_ORIENT, hypercolumns); returns what they would answer, same shape.
    """
    centre, near, far = (TUNING_PARTS @ columns).reshape(3, N_ORIENT, -1)
    centre -= near
    if protect_crossings:
        # the co-occurrence of two orientations far apart, after each has been sharpened among its neighbours
        roots = np.sqrt(np.maximum(centre, 0))
        crossing = 0.5 * (roots * (FAR_APART @ roots)).sum(axis=0)
        far *= CROSSING_HALF / (crossing + CROSSING_HALF)
    centre -= far
    return np.maximum(TUNING_GAIN * centre, 0)


def lobe_pair(shape, k, turn, offset_along, offset_across, sigma_along, sigma_across):
    """Return, as a transfer function, a receptive field of two Gaussian lobes that sit opposite each other, either
    along a line of orientation k + turn / 2 (offset_along) or across it (offset_across). A thin line of strength 1
    that runs through the lobes' middles (through both for lobes along, through one for lobes across) is heard as 1.
    """
    fy, fx = bank_frequencies(shape)
    theta = (k + turn / 2) * np.pi / N_ORIENT
    along = np.cos(theta) * fx + np.sin(theta) * fy
    across = -np.sin(theta) * fx + np.cos(theta) * fy
    lobes = np.exp(-0.5 * ((sigma_along * along) ** 2 + (sigma_across * across) ** 2))
    if offset_along:
        return lobes * np.cos(np.float32(offset_along) * along) * np.float32(np.sqrt(2 * np.pi) * sigma_across)
    return lobes * np.cos(np.float32(offset_across) * across) * np.float32(2 * np.sqrt(2 * np.pi) * sigma_across)


def padded_shape(rows, columns, reach):
    """The shape a field is transformed at: with room for a receptive field's reach, and fast to transform."""
    return tuple(scipy.fft.next_fast_len(int(size + reach), real=True) for size in (rows, columns))


def flank_activity(field, classes, k):
    """Interaction 2, the inhibiting input of the neurons of orientation k: how active the neurons of their own
    orientation are on their two flanks. A neuron speaks to the flanks FLANK_OFFSET of its own line's half-widths
    away, so a wide vessel's crest reaches the side ridges along its walls and a thin one only its own shoulders."""
    rows, columns = field.shape[1:]
    flanks = np.zeros((rows, columns), dtype=np.float32)
    heard = {}  # per reduction factor: what the neurons of its widths say, added up as spectra
    for place, width in enumerate(HYPER_WIDTHS):
        speaking = np.where(classes[k] == place, field[k], 0)
        if not speaking.any():
            continue
        factor = reduction(width)
        if factor > 1:
            speaking = cv2.resize(speaking, (columns // factor, rows // factor), interpolation=cv2.INTER_AREA)
        offset = FLANK_OFFSET * width / factor
        spread = max(FLANK_SPREAD * offset, 0.6)
        shape = padded_shape(*speaking.shape, BANK_PAD)
        said = scipy.fft.rfft2(speaking, s=shape) * lobe_pair(shape, k, 0, 0, offset, ELONGATION * spread, spread)
        heard[factor] = (heard[factor][0] + said if factor in heard else said, shape, speaking.shape)
    for factor, (said, shape, small) in heard.items():
        answer = scipy.fft.irfft2(said, s=shape)[:small[0], :small[1]]
        # on a reduced grid a pixel stands for `factor` pixels across the line
        flanks += answer if factor == 1 else cv2.resize(answer * factor, (columns, rows), interpolation=cv2.INTER_LINEAR)
    return flanks


def association_support(spectra, shape, k):
    """Interaction 3, the facilitating input of the neurons of orientation k: how active the neurons are that lie
    along their line, ahead and behind, with orientations that continue it smoothly (co-circular: a neighbour
    turned by some steps is heard along the ray half that turn off the neuron's own orientation).

    `spectra` are the transforms of the field's N_ORIENT layers at the padded `shape`."""
    said = sum(spectra[(k + turn) % N_ORIENT]
               * lobe_pair(shape, k, turn, 0.6 * ASSOCIATION_REACH, 0, 0.25 * ASSOCIATION_REACH, ASSOCIATION_WIDTH)
               for turn in ASSOCIATION_TURNS)
    return scipy.fft.irfft2(said, s=shape)


def lateral_interaction(scores, half_widths, steps=LATERAL_STEPS, tuning=True, inhibition=True, facilitation=True,
                        protect_crossings=True, threads=None):
    """Section 21: let the neurons of an orientation score act on each other, `steps` times.

    Every interaction is a relaxed update U <- (1 - eta) U + eta f(U):
      1. tuning over orientation   f(U) = [centre - surround over orientation]+, see tune_orientation
      2. inhibition across a line  f(U) = U / (1 + flank activity / FLANK_HALF), see flank_activity
      3. facilitation along a line f(U) = U (1 + MODULATION F / (HALF_SAT + F)), see association_support. It
         multiplies, so it raises what is there and draws nothing where there was no answer.
    The switches leave single interactions out, to see what each one does.

    Returns the field after the interactions (noise units, like the score that went in).
    """
    field = scores.astype(np.float32)
    rows, columns = field.shape[1:]
    step = np.log(HYPER_WIDTHS[-1] / HYPER_WIDTHS[0]) / (len(HYPER_WIDTHS) - 1)
    classes = np.clip(np.rint(np.log(half_widths.astype(np.float32) / HYPER_WIDTHS[0]) / step), 0,
                      len(HYPER_WIDTHS) - 1).astype(np.uint8)
    shape = padded_shape(rows, columns, 2 * ASSOCIATION_REACH)

    def over_orientations(update):
        with ThreadPoolExecutor(max_workers=threads or 1) as pool:
            changed = list(pool.map(update, range(N_ORIENT)))  # all are worked out from the field as it stands...
        for k, layer in enumerate(changed):
            field[k] = layer                                   # ...and only then put in its place

    for _ in range(steps):
        if tuning:
            for top in range(0, rows, LATERAL_SLAB):
                slab = field[:, top:top + LATERAL_SLAB]
                hypercolumns = slab.reshape(N_ORIENT, -1)  # a copy, put back below
                active = np.flatnonzero(hypercolumns.max(axis=0) > SILENT)
                answers = hypercolumns[:, active]
                answers += LATERAL_ETA * (tune_orientation(answers, protect_crossings) - answers)
                hypercolumns[:, active] = answers
                slab[...] = hypercolumns.reshape(slab.shape)
        if inhibition:
            over_orientations(lambda k: field[k] * (1 - LATERAL_ETA + LATERAL_ETA
                                                    / (1 + np.maximum(flank_activity(field, classes, k), 0) / FLANK_HALF)))
        if facilitation:
            with ThreadPoolExecutor(max_workers=threads or 1) as pool:
                spectra = list(pool.map(lambda k: scipy.fft.rfft2(field[k], s=shape), range(N_ORIENT)))

            def raised(k):
                support = np.maximum(association_support(spectra, shape, k)[:rows, :columns], 0)
                return field[k] * (1 + LATERAL_ETA * MODULATION * support / (HALF_SAT + support))
            over_orientations(raised)
            del spectra
    return field


def crest_neurons(field, half_widths, valid, low=T_LOW, scores=None):
    """Read the crests out of a field (thinning in position and orientation): the neurons that answer above `low`,
    more than their two neighbours in orientation, and more than the neurons one pixel to either side across the
    line, of their own orientation and of the two next to it. So where a line's crest shifts from one orientation
    to the next as the line turns, one neuron is left across the line, not one per orientation.

    Returns one record per crest neuron, as a dict of arrays: its place to a fraction of a pixel (x, y), its pixel
    (column, row), its orientation number k, its strength and its line's half-width (radius). With `scores`, the
    orientation score the field was made from, the strength is the neuron's answer before the interactions: the
    support along a line raises the field of a chance row of noise as well, so only that answer is still in noise
    units, and it is what the pruning of section 22 goes by.
    """
    rows, columns = field.shape[1:]
    parts = []
    for k in range(N_ORIENT):
        theta = k * np.pi / N_ORIENT
        normal = (-np.sin(theta), np.cos(theta))  # one pixel across the line, as (x, y)
        here = field[k]
        beside = [[cv2.warpAffine(field[(k + turn) % N_ORIENT],
                                  np.float32([[1, 0, -side * normal[0]], [0, 1, -side * normal[1]]]),
                                  (columns, rows), flags=cv2.INTER_LINEAR) for turn in (0, -1, 1)] for side in (1, -1)]
        plus, minus = beside[0][0], beside[1][0]
        crest = ((here > low) & (here >= np.maximum.reduce(beside[1])) & (here > np.maximum.reduce(beside[0]))
                 & (here >= field[k - 1]) & (here > field[(k + 1) % N_ORIENT]) & valid)
        row, column = np.nonzero(crest)
        below, top, above = minus[row, column], here[row, column], plus[row, column]
        # the top of the parabola through the three answers across the line places the crest within the pixel
        shift = np.clip(0.5 * (below - above) / np.minimum(below - 2 * top + above, -1e-6), -0.5, 0.5)
        parts.append((column + shift * normal[0], row + shift * normal[1], column, row, np.full(len(row), k),
                      top if scores is None else scores[k][row, column], half_widths[k][row, column]))
    names = ("x", "y", "column", "row", "k", "strength", "radius")
    kinds = (np.float32, np.float32, np.int32, np.int32, np.int16, np.float32, np.float32)
    return {name: np.concatenate([part[place] for part in parts]).astype(kind)
            for place, (name, kind) in enumerate(zip(names, kinds))}


print(f"Tuning over orientation: on a plain line the best neuron of a hypercolumn keeps {1 / TUNING_GAIN * 100:.0f}% of "
      f"its answer when the surround is taken off; a gain of {TUNING_GAIN:.2f} gives it back")

# %%
PROFILE_WIDTHS = (2, 3, 4.5, 7, 11)  # pixels; half-widths of the five vessels whose profiles are shown


def parallel_duplicates(crests, shape):
    """Return, for every crest neuron, whether another crest neuron of nearly its orientation runs alongside it:
    within one orientation step, straight across its line, between 2 pixels and 2 of its half-widths away."""
    rows, columns = shape
    taken = np.zeros((N_ORIENT, rows, columns), dtype=bool)
    taken[crests["k"], crests["row"], crests["column"]] = True
    theta = crests["k"] * np.pi / N_ORIENT
    has = np.zeros(len(theta), dtype=bool)
    for distance in range(2, int(2 * HYPER_WIDTHS[-1]) + 1):
        for side in (1, -1):
            column = np.rint(crests["x"] - side * distance * np.sin(theta)).astype(int)
            row = np.rint(crests["y"] + side * distance * np.cos(theta)).astype(int)
            inside = np.flatnonzero((2 * crests["radius"] >= distance) & (row >= 0) & (row < rows)
                                    & (column >= 0) & (column < columns))
            for turn in (-1, 0, 1):
                has[inside] |= taken[(crests["k"][inside] + turn) % N_ORIENT, row[inside], column[inside]]
    return has


def profile_across(layer, x, y, k, reach):
    """Return a layer's values along the line through (x, y) at right angles to orientation k, at -reach..reach."""
    theta = k * np.pi / N_ORIENT
    steps = np.arange(-reach, reach + 1, dtype=np.float32)
    return cv2.remap(layer, np.float32(x - np.sin(theta) * steps)[None], np.float32(y + np.cos(theta) * steps)[None],
                     cv2.INTER_LINEAR)[0]


def width_at_half(profile):
    """Return the width of a profile's middle peak at half its height, in samples (to a fraction of one)."""
    middle = len(profile) // 2
    half = profile[middle] / 2
    sides = []
    for direction in (1, -1):
        place = middle
        while 0 < place < len(profile) - 1 and profile[place + direction] > half:
            place += direction
        inner, outer = profile[place], profile[min(max(place + direction, 0), len(profile) - 1)]
        sides.append(abs(place - middle) + ((inner - half) / (inner - outer) if inner > outer else 0))
    return sum(sides)


static_fields = []  # per fixation: (N_ORIENT, rows, columns) float16, the orientation score after the interactions
static_crests = []  # per fixation: the crest neurons of that field (dict of arrays, see crest_neurons)
for number in range(len(fixations)):
    scores = orientation_scores[number].astype(np.float32)
    before = crest_neurons(scores, radius_maps[number], valid_maps[number])
    field = lateral_interaction(scores, radius_maps[number], threads=workers)
    crests = crest_neurons(field, radius_maps[number], valid_maps[number], scores=scores)
    static_fields.append(field.astype(np.float16))
    static_crests.append(crests)
    doubled_before, doubled = parallel_duplicates(before, frame_shape), parallel_duplicates(crests, frame_shape)
    print(f"Fixation {number}: {len(before['x'])} crest neurons above {T_LOW:g} before the interactions, "
          f"{doubled_before.mean() * 100:.1f}% of them with a parallel duplicate alongside | after: {len(crests['x'])}, "
          f"{doubled.mean() * 100:.1f}% | {(crests['strength'] >= T_HIGH).mean() * 100:.0f}% of them answer above {T_HIGH:g}")
    if number == anchor:
        # five vessels of different widths: the strongest crest neuron of each width that has no duplicate alongside
        examples = []
        for width in PROFILE_WIDTHS:
            fits = (np.abs(np.log(crests["radius"] / width)) < 0.15) & ~doubled
            if fits.any():
                best = np.flatnonzero(fits)[np.argmax(crests["strength"][fits])]
                x, y, k = float(crests["x"][best]), float(crests["y"][best]), int(crests["k"][best])
                reach = int(3 * width + 4)
                examples.append((width, x, y, profile_across(scores[k], x, y, k, reach), profile_across(field[k], x, y, k, reach)))
    del scores, field, before

# check: profiles across five vessels, before and after the interactions
figure, axes = plt.subplots(1, len(examples), figsize=(16, 3), squeeze=False)
print("Width at half maximum of the answer across a vessel, before -> after the interactions:")
for axis, (width, x, y, profile_before, profile_after) in zip(axes[0], examples):
    steps = np.arange(len(profile_before)) - len(profile_before) // 2
    axis.plot(steps, profile_before / profile_before.max(), color="gray", label="before")
    axis.plot(steps, profile_after / profile_after.max(), color="black", label="after")
    axis.set_title(f"half-width {width:g} px, at (x {x:.0f}, y {y:.0f})", fontsize=10)
    axis.set_xlabel("pixels across the vessel")
    print(f"   half-width {width:4g} px at (x {x:4.0f}, y {y:4.0f}): {width_at_half(profile_before):5.1f} px -> "
          f"{width_at_half(profile_after):5.1f} px")
axes[0][0].set_ylabel("answer / its largest")
axes[0][0].legend(frameon=False)
figure.tight_layout()
plt.show()

# %% [markdown]
# ## 22. String the crests into chains, without a skeleton
#
# **Stringing** (`string_crests`). From every crest neuron one pixel is stepped along its line, both ways. A crest neuron within one pixel of where the step lands, with an orientation within one step, is a candidate. Two neurons are linked when each is the other's best candidate. A run of links is a chain: a list of points that carry orientation, half-width and strength.
#
# **Crossings.** Two lines that cross have different orientations where they cross, so they stay two chains. A skeleton cannot do this.
#
# **Bridging gaps** (`find_bridges`). Two chain ends are joined when all of these hold:
#
# - they are no more than `GAP_MAX` vessel radii apart;
# - a smooth curve that bends one way only and turns by no more than 90 degrees joins them (Kellman and Shipley's relatable edges);
# - the neurons of the bridge's own orientation answer above `T_LOW / 2` along it;
# - a bridge across a clearly wider vessel has both its ends flush against that vessel's wall.
#
# `GAP_MAX` is 6 here, not the plan's starting value of 3. At full size 3 leaves 11.7 loose ends per 1000 px and 6 leaves 8.0, while the chains are found again in the other fixation just as well.
#
# **Junction candidates** (`junction_candidates`): a chain end that comes within `JUNCTION_MARGIN` pixels of another vessel's wall, and a place where two chains of different orientation pass within `MEET_WITHIN` pixels of each other.
#
# **Pruning is parametric.** A chain is dropped when it is shorter than `L_MIN`, answers less than `Z_MIN` on average, or has no neuron above `T_HIGH`. All 48 combinations are tried, and for each the cell records the length, the chains, the junction candidates, the loose ends per 1000 px, the share of the length in the largest network, and how much of the chains the other fixation's chains trace again.
#
# **Found again, with chance taken off.** A dense set of lines passes within 3 pixels of another dense set often by chance alone. So the comparison is repeated with the other fixation's lines moved aside by `CHANCE_SHIFT` pixels; `above chance` is how far the real share lies above that one, from 0 (chance) to 1 (everything found).
#
# **The setting is taken from a plateau** (`plateau`), not by its looks: where the numbers move least when the setting is changed by one step. Among the settings about as flat as the flattest, the one that prunes most is taken.
#
# **Result:** `strung_static[n]` (all chains with their junction candidates), `static_chains[n]` (the chains at the chosen pruning), `pruning_table`, `CHAIN_PRUNING`, and a new row in section 18's `graph_table`.

# %%
STRING_ROUNDS = 3       # times the neurons that are still free on one side look for a partner
STRING_TURNS = (0, -1, 1, -2, 2)  # orientation steps a neighbour may be turned by...
STRING_TURN_COST = (0, 0, 0, 1, 1)  # ...and what a turn adds to the squared distance (pixels) that ranks candidates
END_TANGENT = 6         # points at a chain's end from which the direction it leaves in is taken
GAP_MAX = 6.0           # vessel radii; the longest bridge between two chain ends
BRIDGE_BEND = 45.0      # degrees; the most either end may bend away from the straight line between the two
BRIDGE_TURN = 90.0      # degrees; the most a bridge may turn in all (Kellman and Shipley's relatable edges)
BRIDGE_STRAIGHT = 15.0  # degrees; two ends that both bend less than this may bend either way
BRIDGE_WIDER = 1.4      # a vessel a bridge crosses counts as clearly wider from this many times the ends' half-width
BRIDGE_FLUSH = 1.5      # of that vessel's wall distance; both ends of a bridge across it have to be this close to it
JUNCTION_MARGIN = 2.0   # pixels; a chain end this close to another vessel's wall is a junction candidate
MEET_WITHIN = 1.5       # pixels; two chains of different orientation that pass this close meet
FIELD_EDGE = 20         # pixels; a chain end this close to where the data ends is not counted as a loose end
CHANCE_SHIFT = 25              # pixels; the other fixation's lines are moved aside by this to see what chance finds again


def string_crests(crests, shape):
    """String crest neurons into chains, in position and orientation, without a skeleton.

    From every neuron, one pixel is stepped along its line, both ways. A neuron found within one pixel of where
    the step lands, with an orientation within one step of this one's, is a candidate, so a line may turn by one
    orientation step per pixel; a neuron two steps off is taken only when nothing nearer in orientation is as near
    in place (STRING_TURNS, STRING_TURN_COST). Two neurons are linked when each is the other's best candidate (the
    nearest to where its step lands). Neurons left free on a side try again among the others still free,
    STRING_ROUNDS times. Two lines that cross have different orientations at the crossing, so they stay two chains.

    Returns the chains as lists of neuron numbers in order along the chain, and which of them close on themselves.
    """
    rows, columns = shape
    count = len(crests["x"])
    x, y, k = crests["x"], crests["y"], crests["k"].astype(np.int64)
    key = (k * rows + crests["row"]) * columns + crests["column"]
    order = np.argsort(key)
    sorted_keys = key[order]
    heading = np.stack([np.cos(k * np.pi / N_ORIENT), np.sin(k * np.pi / N_ORIENT)]).astype(np.float32)
    linked = np.full((2, count), -1, dtype=np.int64)  # per neuron: its neighbour ahead and its neighbour behind

    def side_facing(here, there):
        """Which side of neuron `there` (0 ahead, 1 behind) looks towards neuron `here`."""
        return np.where((x[here] - x[there]) * heading[0][there] + (y[here] - y[there]) * heading[1][there] > 0, 0, 1)

    for _ in range(STRING_ROUNDS):
        best = np.full((2, count), -1, dtype=np.int64)
        for side, sign in enumerate((1, -1)):
            free = np.flatnonzero(linked[side] < 0)
            lands_x, lands_y = x[free] + sign * heading[0][free], y[free] + sign * heading[1][free]
            column, row = np.rint(lands_x).astype(np.int64), np.rint(lands_y).astype(np.int64)
            cost = np.full(len(free), np.inf, dtype=np.float32)
            for dy, dx, turn in np.ndindex(3, 3, len(STRING_TURNS)):
                there_row, there_column = row + dy - 1, column + dx - 1
                wanted = (((k[free] + STRING_TURNS[turn]) % N_ORIENT) * rows + there_row) * columns + there_column
                at = np.minimum(np.searchsorted(sorted_keys, wanted), count - 1)
                other = order[at]
                found = ((sorted_keys[at] == wanted) & (there_row >= 0) & (there_row < rows)
                         & (there_column >= 0) & (there_column < columns))
                ahead = sign * ((x[other] - x[free]) * heading[0][free] + (y[other] - y[free]) * heading[1][free]) > 0
                still_free = linked[side_facing(free, other), other] < 0
                apart = ((x[other] - lands_x) ** 2 + (y[other] - lands_y) ** 2 + STRING_TURN_COST[turn]
                         - 1e-3 * crests["strength"][other])
                better = found & ahead & still_free & (other != free) & (apart < cost)
                cost[better] = apart[better]
                best[side][free[better]] = other[better]
        for side in (0, 1):
            here = np.flatnonzero(best[side] >= 0)
            there = best[side][here]
            facing = side_facing(here, there)
            mutual = best[facing, there] == here
            linked[side, here[mutual]] = there[mutual]

    # walk the links: first from the neurons with a free side (chain ends), then around what is left (rings)
    neighbours = linked.T.tolist()
    degree = (linked >= 0).sum(axis=0)
    seen = [False] * count
    chains, closed = [], []
    for start in np.flatnonzero(degree < 2).tolist() + np.flatnonzero(degree == 2).tolist():
        if seen[start]:
            continue
        chain, previous, current = [start], -1, start
        seen[start] = True
        while True:
            following = next((n for n in neighbours[current] if n >= 0 and n != previous and not seen[n]), -1)
            if following < 0:
                break
            chain.append(following)
            seen[following] = True
            previous, current = current, following
        chains.append(chain)
        closed.append(len(chain) > 2 and start in neighbours[current])
    return chains, closed


def chains_from(crests, strings, closed):
    """Turn strings of neuron numbers into chains: polylines that carry, at every point, the place (x, y), the
    orientation theta (radians), the line's half-width (radius), the neuron's answer (strength) and whether the
    point was put in by a bridge."""
    return [{"x": crests["x"][string], "y": crests["y"][string],
             "theta": (crests["k"][string] * (np.pi / N_ORIENT)).astype(np.float32),
             "radius": crests["radius"][string], "strength": crests["strength"][string],
             "bridge": np.zeros(len(string), dtype=bool), "closed": ring}
            for string, ring in zip(strings, closed)]


def chain_length(chain):
    """The length of a chain's polyline in pixels (a single point counts 1)."""
    return float(np.hypot(np.diff(chain["x"]), np.diff(chain["y"])).sum()) + 1.0


def chain_ends(chains):
    """Return the ends of the open chains, as arrays: chain number, which end (0 first, 1 last), place, the unit
    direction the chain leaves in, and the half-width and answer there (medians over END_TANGENT points)."""
    flat = flat_points(chains)
    sizes = np.array([len(chain["x"]) for chain in chains], dtype=int)
    starts = np.cumsum(sizes) - sizes
    is_open = np.flatnonzero(~np.array([chain["closed"] for chain in chains], dtype=bool))
    chain, end = np.repeat(is_open, 2), np.tile([0, 1], len(is_open))
    tip = starts[chain] + end * (sizes[chain] - 1)
    inward = 1 - 2 * end  # along the chain's points, away from the tip
    back = np.minimum(END_TANGENT, sizes[chain] - 1)
    inner = tip + inward * back
    leaving = np.stack([flat["x"][tip] - flat["x"][inner], flat["y"][tip] - flat["y"][inner]]).astype(np.float64)
    # a single neuron leaves along its own orientation, one way for each "end"
    alone = back == 0
    leaving[:, alone] = (2 * end[alone] - 1) * np.stack([np.cos(flat["theta"][tip[alone]]), np.sin(flat["theta"][tip[alone]])])
    leaving /= np.maximum(np.hypot(*leaving), 1e-6)
    near = tip[:, None] + inward[:, None] * np.minimum(np.arange(END_TANGENT + 1)[None], back[:, None])
    return {"chain": chain, "end": end, "x": flat["x"][tip].astype(np.float64), "y": flat["y"][tip].astype(np.float64),
            "dx": leaving[0], "dy": leaving[1], "radius": np.median(flat["radius"][near], axis=1),
            "strength": np.median(flat["strength"][near], axis=1)}


def flat_points(chains):
    """All points of all chains in one set of arrays, with the chain each belongs to and its place along it."""
    sizes = [len(chain["x"]) for chain in chains]
    flat = {name: np.concatenate([chain[name] for chain in chains]) if chains else np.zeros(0)
            for name in ("x", "y", "theta", "radius", "strength")}
    flat["chain"] = np.repeat(np.arange(len(chains)), sizes)
    flat["place"] = np.concatenate([np.arange(size) for size in sizes]) if chains else np.zeros(0, dtype=int)
    return flat


def points_along(x0, y0, x1, y1):
    """Return points a pixel apart along straight lines, between their two ends: the line each point belongs to,
    its x and y, and the number of points of every line (the points of a line follow each other)."""
    length = np.hypot(x1 - x0, y1 - y0)
    samples = np.maximum(np.rint(length).astype(int) - 1, 1)
    line = np.repeat(np.arange(len(length)), samples)
    part = (np.arange(len(line)) - np.repeat(np.cumsum(samples) - samples, samples) + 1) / (samples[line] + 1)
    return line, x0[line] + part * (x1 - x0)[line], y0[line] + part * (y1 - y0)[line], samples


def evidence_along(scores, x0, y0, x1, y1):
    """Return the oriented evidence along straight lines from (x0, y0) to (x1, y1): the median, over a point per
    pixel of each line, of what the neurons of the line's own orientation answer there (the largest within one
    pixel and one orientation step)."""
    rows, columns = scores.shape[1:]
    line, x, y, samples = points_along(x0, y0, x1, y1)
    k = np.rint(np.arctan2(y1 - y0, x1 - x0)[line] / (np.pi / N_ORIENT)).astype(int)
    best = np.zeros(len(line), dtype=np.float32)
    for dy, dx, turn in np.ndindex(3, 3, 3):
        row = np.clip(np.rint(y).astype(int) + dy - 1, 0, rows - 1)
        column = np.clip(np.rint(x).astype(int) + dx - 1, 0, columns - 1)
        best = np.maximum(best, scores[(k + turn - 1) % N_ORIENT, row, column].astype(np.float32))
    # the median of each line's points: the middle one after sorting within the line
    return best[np.lexsort((best, line))][np.cumsum(samples) - samples + samples // 2]


def find_bridges(chains, scores, low=T_LOW):
    """Find the gaps between chain ends that a bridge may close.

    Two ends are joined when
      - they are no farther apart than GAP_MAX radii of the wider of the two vessels;
      - a smooth curve joins them that bends one way only and turns by no more than BRIDGE_TURN (relatability):
        each end leaves within BRIDGE_BEND of the straight line to the other, and the two bend to the same side
        unless both are nearly straight;
      - the neurons of the bridge's orientation answer above low / 2 along it (median);
      - where the bridge would cross a clearly wider vessel, both ends sit flush against that vessel's wall. An end
        that stops short of a wider vessel is not carried across it.
    Every end takes part in one bridge at most; shorter and straighter bridges go first.

    Returns the bridges as rows (end number, end number), the evidence along each, the ends themselves, and how
    many pairs of ends a smooth curve could join.
    """
    ends = chain_ends(chains)
    places = np.column_stack([ends["x"], ends["y"]])
    lengths = np.array([chain_length(chain) for chain in chains])
    if len(places) < 2:
        return np.zeros((0, 2), dtype=int), np.zeros(0), ends, 0
    # every end looks as far as GAP_MAX of its own radii; a pair counts when either end reaches the other
    tree = scipy.spatial.cKDTree(places)
    reached = tree.query_ball_point(places, GAP_MAX * ends["radius"] / CYLINDER)
    first = np.repeat(np.arange(len(places)), [len(found) for found in reached])
    second = np.concatenate(reached).astype(int)
    first, second = np.minimum(first, second), np.maximum(first, second)
    first, second = np.unique(np.stack([first, second])[:, first != second], axis=1)
    gap = places[second] - places[first]
    length = np.hypot(gap[:, 0], gap[:, 1])
    unit = gap / np.maximum(length, 1e-6)[:, None]
    widest = np.maximum(ends["radius"][first], ends["radius"][second])
    # how far each end bends off the straight line between them, signed (left positive)
    bend_first = np.arctan2(ends["dx"][first] * unit[:, 1] - ends["dy"][first] * unit[:, 0],
                            ends["dx"][first] * unit[:, 0] + ends["dy"][first] * unit[:, 1])
    bend_second = np.arctan2(ends["dx"][second] * -unit[:, 1] - ends["dy"][second] * -unit[:, 0],
                             ends["dx"][second] * -unit[:, 0] + ends["dy"][second] * -unit[:, 1])
    straight = np.radians(BRIDGE_STRAIGHT)
    one_way = (bend_first * bend_second >= 0) | ((np.abs(bend_first) < straight) & (np.abs(bend_second) < straight))
    relatable = ((np.abs(bend_first) <= np.radians(BRIDGE_BEND)) & (np.abs(bend_second) <= np.radians(BRIDGE_BEND))
                 & (np.abs(bend_first) + np.abs(bend_second) <= np.radians(BRIDGE_TURN)) & one_way)
    same_chain = ends["chain"][first] == ends["chain"][second]
    long_enough = lengths[ends["chain"][first]] > 4 * length
    keep = np.flatnonzero(relatable & (length > 0) & (~same_chain | long_enough))
    tried = len(keep)
    seen = evidence_along(scores, *places[first[keep]].T, *places[second[keep]].T) if tried else np.zeros(0)
    keep, seen = keep[seen >= low / 2], seen[seen >= low / 2]
    if len(keep):
        # the widest clearly wider vessel each bridge would cross, and where it crosses it
        labels = draw_chains(chains, scores.shape[1:])
        radius_of = np.array([np.median(chain["radius"]) for chain in chains])
        line, x, y, samples = points_along(*places[first[keep]].T, *places[second[keep]].T)
        crossed = labels[np.rint(y).astype(int), np.rint(x).astype(int)]
        wider = np.where((crossed >= 0) & (crossed != ends["chain"][first[keep]][line])
                         & (crossed != ends["chain"][second[keep]][line]), radius_of[np.maximum(crossed, 0)], 0)
        wider[wider < BRIDGE_WIDER * widest[keep][line]] = 0
        at = np.lexsort((wider, line))[np.cumsum(samples) - 1]  # per bridge: its point on the widest such vessel
        wall = wider[at] / CYLINDER + JUNCTION_MARGIN
        flush = ((np.hypot(x[at] - places[first[keep], 0], y[at] - places[first[keep], 1]) <= BRIDGE_FLUSH * wall)
                 & (np.hypot(x[at] - places[second[keep], 0], y[at] - places[second[keep], 1]) <= BRIDGE_FLUSH * wall))
        keep, seen = keep[(wider[at] == 0) | flush], seen[(wider[at] == 0) | flush]
    # shorter and straighter first; an end is used once
    cost = length[keep] * (1 + np.abs(bend_first[keep]) + np.abs(bend_second[keep]))
    used = np.zeros(len(places), dtype=bool)
    bridges, evidence = [], []
    for at in np.argsort(cost):
        pair = keep[at]
        if not used[first[pair]] and not used[second[pair]]:
            used[first[pair]] = used[second[pair]] = True
            bridges.append((first[pair], second[pair]))
            evidence.append(seen[at])
    return np.array(bridges, dtype=int).reshape(-1, 2), np.array(evidence), ends, tried


def join_chains(chains, bridges, evidence, ends):
    """Join chains end to end along the bridges. The bridge's own points are put in a pixel apart, with the
    orientation of the bridge, half-widths that run from one end's to the other's, the evidence along the bridge as
    their strength, and `bridge` set."""
    partner = {}
    for (a, b), seen in zip(bridges, evidence):
        partner[int(ends["chain"][a]), int(ends["end"][a])] = (int(ends["chain"][b]), int(ends["end"][b]), seen)
        partner[int(ends["chain"][b]), int(ends["end"][b])] = (int(ends["chain"][a]), int(ends["end"][a]), seen)
    names = ("x", "y", "theta", "radius", "strength", "bridge")

    def filling(before, after, strength):
        """The points of a bridge between the last point of `before` and the first point of `after`."""
        x0, y0, x1, y1 = before["x"][-1], before["y"][-1], after["x"][0], after["y"][0]
        steps = max(int(np.rint(np.hypot(x1 - x0, y1 - y0))) - 1, 0)
        part = (np.arange(steps) + 1) / (steps + 1)
        return {"x": x0 + part * (x1 - x0), "y": y0 + part * (y1 - y0),
                "theta": np.full(steps, np.arctan2(y1 - y0, x1 - x0) % np.pi),
                "radius": before["radius"][-1] + part * (after["radius"][0] - before["radius"][-1]),
                "strength": np.full(steps, strength), "bridge": np.ones(steps, dtype=bool)}

    def turned(chain):
        return {name: chain[name][::-1] for name in names}

    joined, done = [], [False] * len(chains)
    # start from the chains that have a free end, so that every run of bridged chains is walked from one end
    for number in sorted(range(len(chains)), key=lambda n: ((n, 0) in partner) + ((n, 1) in partner)):
        if done[number]:
            continue
        done[number] = True
        if chains[number]["closed"]:
            joined.append(chains[number])
            continue
        # walk out of this chain's last point as long as bridges lead on; then the same from its first point
        parts, ring = [{name: chains[number][name] for name in names}], None
        for way in (1, 0):
            if ring is not None:
                break
            if way == 0:
                parts = [turned(part) for part in parts[::-1]]
            at = (number, way)
            while at in partner:
                following, entered, strength = partner[at]
                if following == number:  # the bridges lead back to where the walk began
                    ring = strength
                    break
                if done[following]:
                    break
                done[following] = True
                piece = {name: chains[following][name] for name in names}
                if entered == 1:
                    piece = turned(piece)
                parts += [filling(parts[-1], piece, strength), piece]
                at = (following, 1 - entered)
        if ring is not None:
            parts.append(filling(parts[-1], parts[0], ring))
        chain = {name: np.concatenate([part[name] for part in parts]).astype(chains[number][name].dtype) for name in names}
        chain["closed"] = ring is not None
        joined.append(chain)
    return joined


def junction_candidates(chains, valid):
    """Find where chains meet, and which chain ends stay loose.

    A chain end is a junction candidate when it comes within JUNCTION_MARGIN pixels of the wall of another chain's
    vessel (a round vessel's wall is its half-width / CYLINDER from its centerline), or of its own chain far from
    the end. Two chains whose orientations differ by CROSSING_APART steps or more meet where they pass within
    MEET_WITHIN pixels of each other.

    Returns
      attached   (chain, end, the chain it runs into, the place along that chain), one row per end that runs into one
      meetings   (chain, place along it, other chain, place along the other), one row per meeting
      ends       the ends of the open chains (see chain_ends), with `inside`: clear of the edge of the data
    """
    ends, flat = chain_ends(chains), flat_points(chains)
    points = np.column_stack([flat["x"], flat["y"]])
    tree = scipy.spatial.cKDTree(points)
    # ends against all points
    reach = HYPER_WIDTHS[-1] / CYLINDER + JUNCTION_MARGIN + 1
    near = scipy.spatial.cKDTree(np.column_stack([ends["x"], ends["y"]])).sparse_distance_matrix(
        tree, reach, output_type="ndarray")
    end, point, distance = near["i"], near["j"], near["v"]
    wall = flat["radius"][point] / CYLINDER
    own = flat["chain"][point] == ends["chain"][end]
    tip = np.array([len(chain["x"]) - 1 for chain in chains])[ends["chain"][end]] * ends["end"][end]
    far_along = np.abs(flat["place"][point] - tip) > 4 * (wall + JUNCTION_MARGIN) + END_TANGENT
    touching = (distance <= wall + JUNCTION_MARGIN) & (~own | far_along)
    end, point, distance = end[touching], point[touching], distance[touching]
    # an end runs into the nearest of the vessels it touches
    order = np.lexsort((distance, end))
    end, point = end[order], point[order]
    first = np.ones(len(end), dtype=bool)
    first[1:] = end[1:] != end[:-1]
    end, point = end[first], point[first]
    attached = np.column_stack([ends["chain"][end], ends["end"][end], flat["chain"][point], flat["place"][point]])
    # chains of different orientation that pass close to each other
    pairs = tree.query_pairs(MEET_WITHIN, output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    turned = np.abs(flat["theta"][a] - flat["theta"][b])
    crossing = (flat["chain"][a] != flat["chain"][b]) & (np.minimum(turned, np.pi - turned)
                                                         >= (CROSSING_APART - 0.5) * np.pi / N_ORIENT)
    a, b = a[crossing], b[crossing]
    swap = flat["chain"][a] > flat["chain"][b]
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    # one row per place two chains meet: neighbouring pairs of points of the same two chains count once
    order = np.lexsort((flat["place"][a], flat["chain"][b], flat["chain"][a]))
    a, b = a[order], b[order]
    new = np.ones(len(a), dtype=bool)
    new[1:] = ((flat["chain"][a][1:] != flat["chain"][a][:-1]) | (flat["chain"][b][1:] != flat["chain"][b][:-1])
               | (flat["place"][a][1:] - flat["place"][a][:-1] > 4))
    meetings = np.column_stack([flat["chain"][a], flat["place"][a], flat["chain"][b], flat["place"][b]])[new]
    edge = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    rows, columns = valid.shape
    ends["inside"] = edge[np.clip(np.rint(ends["y"]).astype(int), 0, rows - 1),
                          np.clip(np.rint(ends["x"]).astype(int), 0, columns - 1)] > FIELD_EDGE
    return attached.astype(int).reshape(-1, 4), meetings.astype(int).reshape(-1, 4), ends


def draw_chains(chains, shape, keep=None):
    """Return an image with every kept chain drawn as a line one pixel wide, numbered by its chain (-1 elsewhere)."""
    labels = np.full(shape, -1, dtype=np.int32)
    for number, chain in enumerate(chains):
        if keep is None or keep[number]:
            line = np.rint(np.column_stack([chain["x"], chain["y"]])).astype(np.int32)[None]
            cv2.polylines(labels, line, bool(chain["closed"]), int(number))
    return labels


def chain_numbers(chains, keep, attached, meetings, ends):
    """Return what describes a set of chains after pruning (`keep`, one flag per chain): its length, the number
    of chains and junction candidates, its loose ends, and its networks (chains connected through junction
    candidates)."""
    lengths = np.array([chain_length(chain) for chain in chains])
    attached = attached[keep[attached[:, 0]] & keep[attached[:, 2]]]
    meetings = meetings[keep[meetings[:, 0]] & keep[meetings[:, 2]]]
    held = np.zeros((len(chains), 2), dtype=bool)
    held[attached[:, 0], attached[:, 1]] = True
    loose = keep[ends["chain"]] & ends["inside"] & ~held[ends["chain"], ends["end"]]
    links = np.concatenate([attached[:, [0, 2]], meetings[:, [0, 2]]])
    graph = scipy.sparse.coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(len(chains),) * 2)
    _, network = scipy.sparse.csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(network[keep], weights=lengths[keep])
    # an end that runs into a chain and a meeting of two chains at the same place are one junction candidate
    places = {(int(c), int(p) // 8) for c, p in attached[:, 2:]} | {(int(c), int(p) // 8) for c, p in meetings[:, :2]}
    return {
        "length": float(lengths[keep].sum()),
        "segments": int(keep.sum()),
        "forks": len(places),
        "loose ends": int(loose.sum()),
        "networks": int((sizes > 0).sum()),
        "largest network": float(sizes.max() / max(sizes.sum(), 1)) if len(sizes) else 0.0,
    }


def strung_field(crests, scores, valid):
    """Sections 22's steps for one field: string the crest neurons, bridge the gaps, find the junction candidates.

    Returns a dict with the chains (before pruning), what each chain's pruning is decided on (length, mean and
    largest answer), the junction candidates (see junction_candidates) and a picture of the chains' numbers.
    """
    strings, closed = string_crests(crests, valid.shape)
    pieces = chains_from(crests, strings, closed)
    bridges, evidence, ends, tried = find_bridges(pieces, scores)
    chains = join_chains(pieces, bridges, evidence, ends)
    attached, meetings, ends = junction_candidates(chains, valid)
    return {
        "chains": chains, "attached": attached, "meetings": meetings, "ends": ends,
        "length": np.array([chain_length(chain) for chain in chains]),
        "mean": np.array([chain["strength"].mean() for chain in chains]),
        "top": np.array([chain["strength"].max() for chain in chains]),
        "labels": draw_chains(chains, valid.shape), "pieces": len(pieces), "bridges": len(bridges), "tried": tried,
    }


def pruned(strung, l_min, z_min, t_high):
    """Which chains a pruning keeps: at least l_min long, answering z_min on average, and t_high somewhere."""
    return (strung["length"] >= l_min) & (strung["mean"] >= z_min) & (strung["top"] >= t_high)


def kept_lines(strung, keep):
    """The kept chains as an image: True on their lines."""
    return (strung["labels"] >= 0) & keep[np.maximum(strung["labels"], 0)]


def repeated(lines, valid, other_lines, other_valid, to_other):
    """Return the share of one fixation's lines that the other fixation's pass within REPEAT_WITHIN pixels, the
    share they would pass by chance (the other's lines moved aside by CHANCE_SHIFT pixels), and how far the first
    lies above the second: 0 is chance, 1 is every line found again."""
    aside = np.array([[1, 0, CHANCE_SHIFT], [0, 1, CHANCE_SHIFT], [0, 0, 1.0]])
    found = found_again(lines, valid, other_lines, other_valid, to_other)
    chance = found_again(lines, valid, other_lines, other_valid, aside @ to_other)
    return found, chance, (found - chance) / max(1 - chance, 1e-6)


def chain_picture(chains, under, low, high, attached=None, meetings=None, loose=None):
    """Draw chains in the colors of their orientation over a dimmed gray picture; optionally mark the chain ends
    that run into another chain (yellow), the places where two chains meet (white) and the loose ends (red)."""
    gray = np.uint8(np.clip((np.nan_to_num(under) - low) / (high - low), 0, 1) * 110)
    picture = np.dstack([gray, gray, gray])
    for chain in chains:
        hues = orientation_hues(chain["theta"][None] / (np.pi / N_ORIENT), np.ones((1, len(chain["theta"]))))[0]
        points = np.rint(np.column_stack([chain["x"], chain["y"]])).astype(np.int32)
        for start in range(len(points) - 1):
            cv2.line(picture, tuple(map(int, points[start])), tuple(map(int, points[start + 1])),
                     tuple(map(int, hues[start])), 2)
    for marks, color, size in ((attached, (255, 220, 0), 2), (meetings, (255, 255, 255), 2), (loose, (255, 40, 40), 3)):
        for x, y in (marks if marks is not None else ()):
            cv2.circle(picture, (int(round(x)), int(round(y))), size, color, -1)
    return picture

# %%
L_MIN_SWEEP = (5, 10, 20, 40)  # pixels; a chain shorter than this is pruned
Z_MIN_SWEEP = (2, 3, 4, 5)     # noise units; a chain whose neurons answer less than this on average is pruned
T_HIGH_SWEEP = (3, 4.5, 6)     # noise units; a chain without one neuron above this is pruned
PLATEAU_NUMBERS = ("per 1000 px", "largest network", "above chance")  # what should not move with the pruning
PLATEAU_SLACK = 1.5            # settings that move no more than this many times the least count as on the plateau
CHAIN_ROW = "orientation chains (section 22)"  # the name of these chains in section 18's table


def pruning_numbers(fields, valids, setting):
    """Return the numbers of the anchor's chains at one pruning, with how much of them the other fixations' chains
    at the same pruning trace again (`found again`, `by chance`, `above chance`), and the other way round
    (`and back`)."""
    keeps = [pruned(strung, *setting) for strung in fields]
    here = fields[anchor]
    numbers = chain_numbers(here["chains"], keeps[anchor], here["attached"], here["meetings"], here["ends"])
    numbers["per 1000 px"] = numbers["loose ends"] / max(numbers["length"], 1) * 1000
    lines = kept_lines(here, keeps[anchor])
    again, back = [], []
    for number, there in enumerate(fields):
        if number != anchor:
            to_other = anchor_to_reference(fixations[number])
            other_lines = kept_lines(there, keeps[number])
            again.append(repeated(lines, valids[anchor], other_lines, valids[number], to_other))
            back.append(found_again(other_lines, valids[number], lines, valids[anchor], np.linalg.inv(to_other)))
    numbers["found again"], numbers["by chance"], numbers["above chance"] = (
        np.mean(again, axis=0) if again else (np.nan,) * 3)
    numbers["and back"] = float(np.mean(back)) if back else np.nan
    return numbers


def plateau(table):
    """Return a setting of a pruning table on a plateau: where the numbers move little when the setting is changed
    by one step.

    For every setting, the change of each of PLATEAU_NUMBERS to each neighbouring setting (one step along one of
    the three pruning numbers) is taken as a share of that number's range over the whole table; the smaller the
    mean change, the flatter the table there. Among the settings that move no more than PLATEAU_SLACK times the
    least, the one that prunes most (keeps the least length) is taken: on a plateau, less clutter costs nothing.
    """
    axes = [sorted({setting[place] for setting in table}) for place in range(3)]
    spans = {name: np.ptp([numbers[name] for numbers in table.values()]) or 1.0 for name in PLATEAU_NUMBERS}
    moves = {}
    for setting, numbers in table.items():
        changes = []
        for place in range(3):
            at = axes[place].index(setting[place])
            for step in (-1, 1):
                if 0 <= at + step < len(axes[place]):
                    neighbour = table[setting[:place] + (axes[place][at + step],) + setting[place + 1:]]
                    changes.append(np.mean([abs(numbers[name] - neighbour[name]) / spans[name] for name in PLATEAU_NUMBERS]))
        moves[setting] = float(np.mean(changes))
    flat = [setting for setting in table if moves[setting] <= PLATEAU_SLACK * min(moves.values())]
    return min(flat, key=lambda setting: table[setting]["length"]), moves


# string, bridge and find the junction candidates in every fixation's field
strung_static = []  # per fixation: see strung_field
for number in range(len(fixations)):
    strung = strung_field(static_crests[number], orientation_scores[number], valid_maps[number])
    strung_static.append(strung)
    print(f"Fixation {number}: {len(static_crests[number]['x'])} crest neurons strung into {strung['pieces']} chains; "
          f"{strung['bridges']} gaps bridged (of {strung['tried']} pairs of ends that a smooth curve could join), "
          f"leaving {len(strung['chains'])} chains, {strung['length'].sum():.0f} px in all | "
          f"{len(strung['attached'])} ends run into a chain, {len(strung['meetings'])} places where two chains meet")

# parametric pruning: every setting's numbers for the anchor's fixation
pruning_table = {}  # (L_MIN, Z_MIN, T_HIGH) -> the numbers of the anchor's chains at that pruning
settings = [(l_min, z_min, t_high) for t_high in T_HIGH_SWEEP for z_min in Z_MIN_SWEEP for l_min in L_MIN_SWEEP]
with ThreadPoolExecutor(max_workers=workers) as pool:
    results = pool.map(lambda setting: pruning_numbers(strung_static, valid_maps, setting), settings)
    pruning_table = dict(zip(settings, results))
CHAIN_PRUNING, pruning_moves = plateau(pruning_table)
L_MIN, Z_MIN, T_HIGH_CHOSEN = CHAIN_PRUNING

print(f"\n{'L_MIN':>5s} {'Z_MIN':>5s} {'T_HIGH':>6s} {'length':>7s} {'chains':>6s} {'junctions':>9s} {'loose ends':>10s} "
      f"{'per 1000 px':>11s} {'networks':>8s} {'largest':>7s} {'found again':>11s} {'by chance':>9s} {'above chance':>12s} "
      f"{'and back':>8s} {'moves by':>8s}")
for setting, numbers in pruning_table.items():
    print(f"{setting[0]:5d} {setting[1]:5g} {setting[2]:6g} {numbers['length']:7.0f} {numbers['segments']:6d} "
          f"{numbers['forks']:9d} {numbers['loose ends']:10d} {numbers['per 1000 px']:11.1f} {numbers['networks']:8d} "
          f"{numbers['largest network'] * 100:6.0f}% {numbers['found again'] * 100:10.0f}% {numbers['by chance'] * 100:8.0f}% "
          f"{numbers['above chance'] * 100:11.0f}% {numbers['and back'] * 100:7.0f}% "
          f"{pruning_moves[setting] * 100:7.1f}%" + ("  <- chosen" if setting == CHAIN_PRUNING else ""))

# show each number against the pruning: one panel per number, one line per Z_MIN, one line style per T_HIGH
shown = ("length", "per 1000 px", "largest network", "found again", "above chance")
figure, axes = plt.subplots(1, len(shown), figsize=(16, 3.4))
for axis, name in zip(axes, shown):
    for t_high, style in zip(T_HIGH_SWEEP, (":", "-", "--")):
        for z_min, color in zip(Z_MIN_SWEEP, plt.get_cmap("viridis")(np.linspace(0, 0.9, len(Z_MIN_SWEEP)))):
            axis.plot(L_MIN_SWEEP, [pruning_table[l_min, z_min, t_high][name] for l_min in L_MIN_SWEEP], style,
                      color=color, label=f"Z_MIN {z_min:g}, T_HIGH {t_high:g}")
    axis.plot(L_MIN, pruning_table[CHAIN_PRUNING][name], "o", color="red")
    axis.set_title(name, fontsize=10)
    axis.set_xlabel("L_MIN (px)")
    axis.set_xscale("log")
axes[0].legend(fontsize=5, frameon=False, ncol=2)
figure.tight_layout()
plt.show()

# the chains at the chosen pruning
static_chains = []  # per fixation: the kept chains (polylines with theta, radius and strength at every point)
static_keeps = []   # per fixation: which chains of strung_static are kept
for number, strung in enumerate(strung_static):
    keep = pruned(strung, *CHAIN_PRUNING)
    static_keeps.append(keep)
    static_chains.append([chain for chain, kept in zip(strung["chains"], keep) if kept])

# add them to section 18's table and compare with its graphs
numbers = dict(pruning_table[CHAIN_PRUNING])
numbers["mask"] = np.nan  # no mask is made
graph_table[CHAIN_ROW] = numbers
print(f"\n{'picture':36s} {'length':>7s} {'segments':>8s} {'forks':>6s} {'loose ends':>10s} {'per 1000 px':>11s} "
      f"{'networks':>8s} {'largest':>8s} {'found again':>11s} {'and back':>8s}")
for name, row in graph_table.items():
    print(f"{name:36s} {row['length']:7.0f} {row['segments']:8d} {row['forks']:6d} {row['loose ends']:10d} "
          f"{row['loose ends'] / max(row['length'], 1) * 1000:11.1f} {row['networks']:8d} "
          f"{row['largest network'] * 100:7.0f}% {row['found again'] * 100:10.0f}% {row['and back'] * 100:7.0f}%")
strung, keep = strung_static[anchor], static_keeps[anchor]
crossing_pairs = strung["meetings"][keep[strung["meetings"][:, 0]] & keep[strung["meetings"][:, 2]]]
reference = traced["density, background cleaned", anchor]["centerlines"]
lines = kept_lines(strung, keep)
print(f"At the chosen pruning (L_MIN {L_MIN}, Z_MIN {Z_MIN:g}, T_HIGH {T_HIGH_CHOSEN:g}): {len(crossing_pairs)} places where "
      f"two chains of different orientation meet and stay two chains | "
      f"{found_again(lines, valid_maps[anchor], reference, valid_maps[anchor], None) * 100:.0f}% of the chains lie on "
      f"section 18's centerlines of the density, and "
      f"{found_again(reference, valid_maps[anchor], lines, valid_maps[anchor], None) * 100:.0f}% of those are found")
# found again, with chance taken off: the chains against section 18's graph of the same picture
for number in range(len(fixations)):
    if number != anchor:
        to_other = anchor_to_reference(fixations[number])
        found, chance, above = repeated(reference, valid_maps[anchor], traced["density, background cleaned", number]["centerlines"],
                                        valid_maps[number], to_other)
        print(f"Found again in fixation {number}, with what chance alone finds taken off: section 18's graph of the density "
              f"{found * 100:.0f}% found, {chance * 100:.0f}% by chance, {above * 100:.0f}% above chance | the chains "
              f"{numbers['found again'] * 100:.0f}%, {numbers['by chance'] * 100:.0f}%, {numbers['above chance'] * 100:.0f}%")

# show the anchor's chains at the chosen pruning, with their junction candidates and loose ends
os.makedirs("vessel_pictures/chains", exist_ok=True)
attached = strung["attached"][keep[strung["attached"][:, 0]] & keep[strung["attached"][:, 2]]]
held = np.zeros((len(strung["chains"]), 2), dtype=bool)
held[attached[:, 0], attached[:, 1]] = True
ends = strung["ends"]
loose = keep[ends["chain"]] & ends["inside"] & ~held[ends["chain"], ends["end"]]
picture = chain_picture(
    static_chains[anchor], background_fits[anchor]["OD"], *DENSITY_RANGE,
    attached=[(strung["chains"][c]["x"][p], strung["chains"][c]["y"][p]) for c, p in attached[:, 2:]],
    meetings=[(strung["chains"][c]["x"][p], strung["chains"][c]["y"][p]) for c, p in crossing_pairs[:, :2]],
    loose=zip(ends["x"][loose], ends["y"][loose]))
Image.fromarray(picture).save("vessel_pictures/chains/static_chains.png")
figure, axis = plt.subplots(figsize=(16, 10))
axis.imshow(picture, interpolation="antialiased")
axis.set_title("chains of the orientation field, colored by orientation | yellow: an end runs into a chain, "
               "white: two chains meet, red: loose end", fontsize=10)
axis.axis("off")
figure.tight_layout()
plt.show()
del picture, lines
# %% [markdown]
# ## 23. Is the blood's motion visible at this frame rate?
#
# This is settled before any velocity is built. If the pattern of cells in a vessel is new in every frame, there is nothing to follow, and only "something changes here" can be said.
#
# **Line mass.** For a chain, `chain_signals` reads every frame on a grid of points across the vessel and adds the density up across it, over `PAINT_WIDTH` half-widths to either side. That is the line mass `M(s, t)`: one number per chain point `s` and frame `t`, a kymograph. Added up like this the density hardly changes with defocus, and in absorption imaging it is proportional to the hemoglobin per unit length of vessel.
#
# **Travelling correlation** (`travelling_correlation`). The line mass's fluctuation at chain position `s` in one frame is correlated with that at `s + D` in the next frame, for `D` from -40 to +40 pixels.
#
# - A peak away from `D = 0` is a pattern that travels `D` pixels per frame: flow that can be seen.
# - A peak at 0 is a pattern that stays in place: slow flow, or texture.
# - A flat curve near 0 is a pattern that does not survive one frame: flow too fast for this frame rate.
#
# The question is put to the `TRAVEL_CHAINS` longest chains of each width class in the anchor's fixation. Only pairs of frames one recorded frame apart count; the pairs split by a dropped frame are left out.
#
# **Decision per width class**, recorded in `flow_visible`: where at least half the chains have a clear peak off 0, velocity is read from the kymographs in section 25. Elsewhere flow can only be detected (section 24), not measured. The cell also prints how long the pattern is along a vessel, which sets the fastest flow this frame rate can follow and the frame rate a faster flow would need.

# %%
PROFILE_REACH = 3.0   # half-widths; how far to either side of a chain the frames are read (plus PROFILE_MARGIN)
PROFILE_MARGIN = 2    # pixels
MASS_COVERAGE = 0.8   # a line mass counts where this share of the vessel's cross-section has data
TANGENT_SPAN = 3      # points to either side from which a chain's direction at a point is taken
TRAVEL_REACH = 40     # pixels; the largest shift along a chain at which two frames are compared
TRAVEL_LENGTH = 80    # pixels; a chain has to be this long to be asked whether its pattern travels
WIDTH_CLASSES = ((0, 3, "thin, under 3 px"), (3, 6, "medium, 3 to 6 px"), (6, 1e9, "wide, 6 px and up"))  # by half-width


def chain_grid(chains):
    """Lay a grid of sample points across every chain: at each chain point a row of points one pixel apart along
    the normal, PROFILE_REACH half-widths (of the chain's median) plus PROFILE_MARGIN to either side.

    Returns a dict: x and y of all sample points (one flat array, chain after chain, row after row), the number
    of points and the reach (in samples) of every chain, and for every sample its weight in the line mass (1 inside
    PAINT_WIDTH half-widths of the centerline, 0 outside).
    """
    xs, ys, inner, sizes, reaches = [], [], [], [], []
    for chain in chains:
        points = len(chain["x"])
        ahead = np.minimum(np.arange(points) + TANGENT_SPAN, points - 1)
        behind = np.maximum(np.arange(points) - TANGENT_SPAN, 0)
        tangent = np.stack([chain["x"][ahead] - chain["x"][behind], chain["y"][ahead] - chain["y"][behind]])
        alone = np.hypot(*tangent) < 1e-6  # a single point: its neuron's orientation
        tangent[:, alone] = np.stack([np.cos(chain["theta"][alone]), np.sin(chain["theta"][alone])])
        tangent /= np.hypot(*tangent)
        radius = float(np.median(chain["radius"]))
        reach = int(np.ceil(PROFILE_REACH * radius)) + PROFILE_MARGIN
        offsets = np.arange(-reach, reach + 1, dtype=np.float32)
        xs.append((chain["x"][:, None] - tangent[1][:, None] * offsets[None]).ravel())
        ys.append((chain["y"][:, None] + tangent[0][:, None] * offsets[None]).ravel())
        inner.append(np.tile(np.abs(offsets) <= PAINT_WIDTH * radius, points))
        sizes.append(points)
        reaches.append(reach)
    join = lambda parts, kind: np.concatenate(parts).astype(kind) if parts else np.zeros(0, dtype=kind)
    count = sum(len(part) for part in xs)
    # the sample points as a picture 4096 wide, which is what the resampling wants; the rest of the last row is unused
    maps = np.full((2, -(-max(count, 1) // 4096) * 4096), -10, dtype=np.float32)
    maps[0, :count], maps[1, :count] = join(xs, np.float32), join(ys, np.float32)
    return {"map_x": maps[0].reshape(-1, 4096), "map_y": maps[1].reshape(-1, 4096), "count": count,
            "inner": join(inner, np.float32), "points": np.array(sizes, dtype=int), "reach": np.array(reaches, dtype=int)}


def read_on_grid(frame, grid):
    """Read one frame at a grid's sample points. Returns the values (0 where the frame has no data) and the share
    of data behind each (1 where the four pixels around a sample all have data)."""
    has_data = np.isfinite(frame)
    values = cv2.remap(np.where(has_data, frame, 0).astype(np.float32), grid["map_x"], grid["map_y"], cv2.INTER_LINEAR)
    share = cv2.remap(has_data.astype(np.float32), grid["map_x"], grid["map_y"], cv2.INTER_LINEAR)
    return values.ravel()[:grid["count"]], share.ravel()[:grid["count"]]


def chain_signals(chains, frame_at, positions, threads=None):
    """Read a burst along chains: for every chain its kymograph, the line mass M(s, t), and its mean profile.

    The line mass at a chain point is the frame's density added up across the vessel, over PAINT_WIDTH half-widths
    to either side. Added up like this the density hardly changes with defocus, and in absorption imaging it is
    proportional to the hemoglobin per unit length of vessel: the quantity that is conserved at a junction.

    `frame_at(position)` returns one temporal frame; `positions` are the frames to read. Returns per chain a dict
    with M (points x frames, NaN where a cross-section lacks data) and profile (frames x samples across the vessel,
    the mean over the chain's points).
    """
    grid = chain_grid(chains)
    widths = 2 * grid["reach"] + 1
    rows = np.repeat(widths, grid["points"])  # per chain point: samples in its row
    row_starts = np.cumsum(rows) - rows
    # every sample's place in its chain's profile: all profiles lie one after the other in one long array
    profile_starts = np.cumsum(widths) - widths
    in_profile = (np.repeat(np.repeat(profile_starts, grid["points"]), rows)
                  + np.arange(grid["count"]) - np.repeat(row_starts, rows))
    masses = np.full((len(rows), len(positions)), np.nan, dtype=np.float32)
    profiles = np.zeros((len(positions), int(widths.sum())), dtype=np.float32)
    inner_total = np.add.reduceat(grid["inner"], row_starts) if len(rows) else np.zeros(0)

    def read(place):
        values, share = read_on_grid(frame_at(positions[place]), grid)
        mass = np.add.reduceat(values * grid["inner"], row_starts)
        covered = np.add.reduceat(share * grid["inner"], row_starts) / np.maximum(inner_total, 1)
        masses[:, place] = np.where(covered >= MASS_COVERAGE, mass / np.maximum(covered, 1e-3), np.nan)
        profiles[place] = (np.bincount(in_profile, weights=values, minlength=profiles.shape[1])
                           / np.maximum(np.bincount(in_profile, weights=share, minlength=profiles.shape[1]), 1e-3))

    if len(rows):
        with ThreadPoolExecutor(max_workers=threads or 1) as pool:
            list(pool.map(read, range(len(positions))))
    point_starts = np.cumsum(grid["points"]) - grid["points"]
    return [{"M": masses[start:start + size], "profile": profiles[:, at:at + width]}
            for start, size, at, width in zip(point_starts, grid["points"], profile_starts, widths)]


def mean_of_data(values, axis):
    """The mean over the values that are data (not NaN) along an axis; NaN where there are none."""
    seen = np.isfinite(values)
    count = seen.sum(axis=axis)
    return np.where(count > 0, np.where(seen, values, 0).sum(axis=axis) / np.maximum(count, 1), np.nan)


def fluctuation(mass):
    """Return a kymograph's fluctuation: the line mass minus each point's mean over time and minus each frame's
    mean along the chain (which takes out what the whole piece of vessel does together), 0 where there is no data."""
    centred = mass - mean_of_data(mass, axis=1)[:, None]
    centred = centred - mean_of_data(centred, axis=0)[None]
    return np.nan_to_num(centred), np.isfinite(centred)


def travelling_curves(mass, pairs, reach=TRAVEL_REACH, weights=None):
    """Correlate a chain's line mass in one frame with its line mass in the next, shifted along the chain.

    For every pair of frames (earlier, later) and every shift D from -reach to +reach, the correlation between
    the fluctuation at chain position s in the earlier frame and at s + D in the later one. `weights` (one per
    frame) make some pairs count more; pairs with a weight of 0 are left out.

    Returns the shifts, one curve per pair (pairs x shifts), and every pair's share in their mean (None without
    weights: all pairs count the same). The curves are None when the chain is too short or has under three pairs.
    """
    centred, seen = fluctuation(mass)
    points = len(centred)
    reach = int(min(reach, points - 8))
    shifts = np.arange(-reach, reach + 1)
    if weights is not None:
        pairs = [pair for pair in pairs if weights[pair[0]] * weights[pair[1]] > 0]
    if reach < 1 or len(pairs) < 3:
        return shifts, None, None
    earlier, later = centred[:, [pair[0] for pair in pairs]], centred[:, [pair[1] for pair in pairs]]
    # all shifts at once: the cross-correlation along the chain, by way of its transform
    size = scipy.fft.next_fast_len(points + reach, real=True)
    cross = scipy.fft.irfft(np.conj(scipy.fft.rfft(earlier, n=size, axis=0)) * scipy.fft.rfft(later, n=size, axis=0),
                            n=size, axis=0)
    curves = np.concatenate([cross[-reach:], cross[:reach + 1]]).T  # (pairs, shifts)
    norm = np.sqrt((earlier**2).sum(axis=0) * (later**2).sum(axis=0))
    curves /= np.maximum(norm, 1e-12)[:, None] * ((points - np.abs(shifts)) / points)[None]  # fewer points overlap at a shift
    if weights is None:
        return shifts, curves, None
    share = np.array([weights[pair[0]] * weights[pair[1]] for pair in pairs])
    return shifts, curves, share / share.sum()


def curves_mean(curves, share):
    """The mean of the pairs' curves and its standard error (see travelling_curves for `share`)."""
    if share is None:
        return curves.mean(axis=0), curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    mean = (curves * share[:, None]).sum(axis=0)
    return mean, np.sqrt((((curves - mean) ** 2) * share[:, None] ** 2).sum(axis=0))


def travelling_correlation(mass, pairs, reach=TRAVEL_REACH, weights=None):
    """The travelling correlation of a chain: the mean over the pairs of frames of travelling_curves. A peak away
    from a shift of 0 is a pattern that travels that many pixels per frame; a flat curve near 0 is a pattern that
    is new in every frame. In a kymograph this is the slope of the streaks.

    Returns the shifts, the mean correlation over the pairs, and its standard error.
    """
    shifts, curves, share = travelling_curves(mass, pairs, reach, weights)
    if curves is None:
        return shifts, np.full(len(shifts), np.nan), np.full(len(shifts), np.nan)
    return (shifts, *curves_mean(curves, share))


def pattern_length(masses):
    """How long the pattern along a vessel is, in pixels: the distance over which a kymograph's own correlation
    along the chain falls to half; the median over the kymographs given. It sets how fast a flow a frame rate can
    follow: the pattern is found again while it moves by up to about half this length between two frames."""
    lengths = []
    for mass in masses:
        centred, _ = fluctuation(mass)
        spectrum = np.abs(scipy.fft.rfft(centred, axis=0)) ** 2
        along = scipy.fft.irfft(spectrum.mean(axis=1), axis=0)
        below = np.flatnonzero(along[:len(along) // 2] < 0.5 * along[0])
        lengths.append(below[0] if len(below) else np.nan)
    return float(np.nanmedian(lengths)) if lengths else np.nan


def travel_reading(shifts, mean, error):
    """Read a travelling correlation: where it peaks (pixels per frame, to a fraction of a pixel), how many of
    its standard errors the peak stands above 0, and the correlation at the peak and in place (shift 0)."""
    if not np.isfinite(mean).any():
        return {"shift": np.nan, "z": np.nan, "peak": np.nan, "in place": np.nan}
    best = int(np.nanargmax(mean))
    shift = float(shifts[best])
    if 0 < best < len(shifts) - 1:
        below, top, above = mean[best - 1], mean[best], mean[best + 1]
        if below - 2 * top + above < 0:
            shift += float(np.clip(0.5 * (below - above) / (below - 2 * top + above), -0.5, 0.5))
    return {"shift": shift, "z": float(mean[best] / max(error[best], 1e-9)), "peak": float(mean[best]),
            "in place": float(mean[len(shifts) // 2])}


def width_class(radius):
    """The number of the width class (see WIDTH_CLASSES) a half-width falls in."""
    return next(number for number, (low, high, _) in enumerate(WIDTH_CLASSES) if low <= radius < high)

# %% [markdown]
# The cell below puts the question and shows every chain's travelling correlation, one panel per width class.

# %%
TRAVEL_CHAINS = 15     # chains per width class on which the question is put: the longest ones
TRAVEL_LENGTH = 80     # pixels; a chain has to be this long to be asked
TRAVEL_CLEAR = 4.0     # standard errors; a peak of the travelling correlation this far above 0 is clear
TRAVEL_SHARE = 0.5     # a width class shows visible flow when this share of its chains have a clear peak off 0

# the longest chains of every width class in the anchor's fixation
chains = static_chains[anchor]
lengths = np.array([chain_length(chain) for chain in chains])
radii = np.array([np.median(chain["radius"]) for chain in chains])
asked = []
for number, (low, high, name) in enumerate(WIDTH_CLASSES):
    members = np.flatnonzero((radii >= low) & (radii < high) & (lengths >= TRAVEL_LENGTH))
    asked += list(members[np.argsort(-lengths[members])][:TRAVEL_CHAINS])
pairs = frame_pairs[anchor]
positions = list(range(len(fixations[anchor]["frames"])))
read = chain_signals([chains[number] for number in asked], lambda position: temporal_frame(anchor, position),
                     positions, threads=workers)

travel_curves = []  # per asked chain: its width class, its half-width, the shifts, the mean correlation and its error
for number, signals in zip(asked, read):
    shifts, mean, error = travelling_correlation(signals["M"], pairs)
    travel_curves.append((width_class(radii[number]), radii[number], shifts, mean, error))

flow_visible = {}  # per width class: what the travelling correlation says, and the decision
dropped = len(fixations[anchor]["frames"]) - 1 - len(pairs)
print(f"Fixation {anchor}: {len(pairs)} pairs of frames one frame apart are compared; {dropped} pairs are split by "
      f"dropped frames and left out")
print(f"{'width class':20s} {'chains':>6s} {'in place (r1)':>13s} {'peak':>6s} {'clear peaks off 0':>17s} "
      f"{'px per frame':>12s} {'px per s':>9s}  decision")
for number, (low, high, name) in enumerate(WIDTH_CLASSES):
    readings = [travel_reading(shifts, mean, error) for kind, _, shifts, mean, error in travel_curves if kind == number]
    if not readings:
        continue
    travelling = [reading for reading in readings if reading["z"] >= TRAVEL_CLEAR and abs(reading["shift"]) >= 1]
    staying = [reading for reading in readings if reading["z"] >= TRAVEL_CLEAR and abs(reading["shift"]) < 1]
    speed = float(np.median([abs(reading["shift"]) for reading in travelling])) if travelling else np.nan
    visible = len(travelling) >= TRAVEL_SHARE * len(readings)
    flow_visible[name] = {
        "chains": len(readings), "travelling": len(travelling), "in place": len(staying), "visible": visible,
        "r1": float(np.median([reading["in place"] for reading in readings])),
        "peak": float(np.median([reading["peak"] for reading in readings])), "px per frame": speed,
    }
    decision = ("flow is visible: velocity from the kymographs" if visible else
                "a pattern that stays in place: slow flow or texture" if len(staying) >= TRAVEL_SHARE * len(readings) else
                "no pattern survives one frame: flow from decorrelation only")
    print(f"{name:20s} {len(readings):6d} {flow_visible[name]['r1']:13.2f} {flow_visible[name]['peak']:6.2f} "
          f"{len(travelling):9d} of {len(readings):<4d} {speed:12.1f} {speed * fps:9.0f}  {decision}")

# how long the pattern is along a vessel, which sets how fast a flow this frame rate can follow
PATTERN_LENGTH = pattern_length([signals["M"] for signals in read])  # pixels; where the pattern's own correlation along a chain halves
print(f"The pattern along a vessel keeps half its correlation over {PATTERN_LENGTH:.1f} px, so it can be followed while it "
      f"moves by up to about {PATTERN_LENGTH / 2:.1f} px between frames: {PATTERN_LENGTH / 2 * fps:.0f} px per second at "
      f"{fps:.1f} frames per second. A flow of v px per second needs v / {PATTERN_LENGTH / 2:.1f} frames per second.")

# show the travelling correlation of every chain asked, by width class
figure, axes = plt.subplots(1, len(WIDTH_CLASSES), figsize=(16, 3.6), sharey=True)
for number, (axis, (low, high, name)) in enumerate(zip(axes, WIDTH_CLASSES)):
    for kind, radius, shifts, mean, error in travel_curves:
        if kind == number:
            axis.plot(shifts, mean, linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.5)
    axis.axhline(0, color="gray", linewidth=0.5)
    axis.set_title(f"{name}: line mass in one frame against the next", fontsize=10)
    axis.set_xlabel("shift along the chain (px per frame)")
axes[0].set_ylabel("correlation")
figure.tight_layout()
plt.show()
del read

# %% [markdown]
# ## 24. The flow-sensitive field
#
# The second field sees change, not density. It works on frames reduced by `FLOW_REDUCTION` = 2.
#
# **First stage: the hypercolumns are given the change itself** (`flow_energy_field`). For every pair of frames one frame apart, `D = (later - earlier) / seconds` is put in multiples of its own noise and given to the even and odd fields of every orientation. `D` changes sign as clusters of cells pass, so a neuron's answer is the energy `even^2 + odd^2`, which does not depend on the sign. One pair gives 16 pictures per width, far too many to keep for a burst, so they are added up as they come.
#
# **What the data says about the first stage.** The plan expected these neurons to prefer the vessel's own orientation. They do not: the change inside a vessel is a pattern of cells arriving and leaving, with bands that lie across the vessel as often as along it. The cell prints how the strongest orientation lies to the vessel's. So a centerline cannot be read off these neurons directly.
#
# **Second stage: where things keep changing.** Each hypercolumn's energy, added over its neurons and over the burst, is a picture with a bright ridge along every vessel that flows (`flow_pictures`). It is section 15's smear seen through differences of frames. That picture goes through the layers of sections 19 to 22: contrast, hypercolumns, interactions, crests, chains, and the pruning chosen in section 22.
#
# **Result:**
#
# - `flow_pictures[n]`: where things change (1 is what the quiet pixels show).
# - `flow_energy[n]`: the flow field's orientation score, `(16, rows / 2, columns / 2)`, in noise units; `flow_radius[n]` its half-widths.
# - `flow_fields[n]`: the same after the interactions.
# - `flow_chains[n]`: its chains, at the size of a frame.
# - `flow_pair_activity[n]`: per pair of frames, how strongly the field answered.
#
# **Checks:** the picture against the cleaned smear of section 16; the share of quiet pixels where a neuron answers above the thresholds; and the orientation of the flow field against the static field's where both are strong.

# %%
FLOW_REDUCTION = 2                       # the flow-sensitive field works on frames reduced by this factor
FLOW_CHANGE_WIDTHS = (1.5, 3)               # reduced pixels; half-widths of the receptive fields that are given the change
FLOW_RIDGE_WIDTHS = HYPER_WIDTHS[:6]  # reduced pixels; half-widths of the lines looked for in the picture of the change
FLOW_EDGE = 6                        # reduced pixels; a hypercolumn this close to where a pair's data ends is not counted


def flow_energy_field(frame_at, pairs, valid, quiet, threads=None):
    """Section 24, first stage: pass the change between frames through the hypercolumns and add up their answers.

    For every pair (earlier position, later position, seconds between them), D = (later - earlier) / seconds is
    reduced by FLOW_REDUCTION, put in multiples of its own noise (its spread over the quiet pixels), and given to the
    even and odd fields of every orientation, at the widths FLOW_CHANGE_WIDTHS. D changes sign as clusters of cells pass,
    so a neuron's answer is the energy even^2 + odd^2, which does not depend on the sign (a complex cell). The
    answers to one pair are N_ORIENT pictures per width, far too many to keep for a burst, so they are added up as
    they come: per orientation for the finest width, and over all neurons of a hypercolumn for every width.

    Returns, on the reduced grid and in multiples of what the quiet pixels' hypercolumns answer (1 is noise):
      column    each hypercolumn's mean energy over the burst, its neurons and widths added up: where things change
      oriented  (N_ORIENT, ...) the mean energy of each neuron of the finest width
    and the data area, and per pair how strongly the whole field answered (its mean energy at the finest width).
    """
    rows, columns = valid.shape
    size = (columns // FLOW_REDUCTION, rows // FLOW_REDUCTION)
    small_quiet = cv2.resize(quiet.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    # the receptive fields are the same for every pair, so their transfer functions are made once
    shape, inner = padded_spectrum(np.zeros(size[::-1], dtype=np.float32), 1)[1:]
    fields = {(width, k): line_pair(shape, width, k)[:2] for width in FLOW_CHANGE_WIDTHS for k in range(N_ORIENT)}
    oriented = np.zeros((N_ORIENT, *size[::-1]), dtype=np.float32)
    column = {width: np.zeros(size[::-1], dtype=np.float32) for width in FLOW_CHANGE_WIDTHS}
    counted = np.zeros(size[::-1], dtype=np.float32)
    activity = np.zeros(len(pairs))
    adding = threading.Lock()
    margin = np.ones((2 * FLOW_EDGE + 1,) * 2, np.uint8)

    def answer_pair(place):
        earlier, later, seconds = pairs[place]
        change = (frame_at(later) - frame_at(earlier)) / seconds
        has_data = np.isfinite(change)
        share = cv2.resize(has_data.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        change = cv2.resize(np.where(has_data, change, 0), size, interpolation=cv2.INTER_AREA) / np.maximum(share, 1e-3)
        seen = share > 0.99
        change = np.where(seen, change / robust_sd(change[small_quiet & seen]), 0).astype(np.float32)
        spectrum = padded_spectrum(change, 1)[0]
        inside = cv2.erode(seen.astype(np.uint8), margin, borderValue=0) > 0
        for width in FLOW_CHANGE_WIDTHS:
            whole = np.zeros(size[::-1], dtype=np.float32)
            for k in range(N_ORIENT):
                even, odd = fields[width, k]
                energy = (scipy.fft.irfft2(spectrum * even, s=shape)[inner] ** 2
                          + scipy.fft.irfft2(spectrum * (1j * odd), s=shape)[inner] ** 2) * inside
                whole += energy
                if width == FLOW_CHANGE_WIDTHS[0]:
                    with adding:
                        oriented[k] += energy
            with adding:
                column[width] += whole
            if width == FLOW_CHANGE_WIDTHS[0]:
                activity[place] = whole.sum() / max(inside.sum() * N_ORIENT, 1)
        with adding:
            np.add(counted, inside, out=counted)

    with ThreadPoolExecutor(max_workers=threads or 1) as pool:
        list(pool.map(answer_pair, range(len(pairs))))

    data = counted >= 0.5 * len(pairs)
    floor_at = small_quiet & data
    picture = np.zeros(size[::-1], dtype=np.float32)
    for width in FLOW_CHANGE_WIDTHS:
        mean = column[width] / np.maximum(counted, 1)
        picture += mean / np.median(mean[floor_at]) / len(FLOW_CHANGE_WIDTHS)
    oriented /= np.maximum(counted, 1)
    oriented /= np.median(oriented[:, floor_at])
    return picture * data, oriented * data, data, activity


def flow_field(frame_at, pairs, valid, quiet, threads=None):
    """Section 24, both stages: the hypercolumns are given the change between frames and their answers added up
    (flow_energy_field); the picture of where things keep changing then goes through the contrast layer, the
    hypercolumns and the interactions of sections 19 to 21, on the reduced grid.

    Returns a dict: `picture` (where things change; 1 is what the quiet pixels show), `oriented` (the mean energy
    of each neuron that was given the change itself), `scores` and `radius` (the flow field's orientation score in
    noise units and its half-widths in reduced pixels), `field` (the score after the interactions), `valid` and
    `quiet` (the data area and the quiet pixels on the reduced grid) and `activity` (per pair of frames).
    """
    size = (valid.shape[1] // FLOW_REDUCTION, valid.shape[0] // FLOW_REDUCTION)
    picture, oriented, data, activity = flow_energy_field(frame_at, pairs, valid, quiet, threads)
    small_quiet = (cv2.resize(quiet.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0) & data
    contrast = contrast_layer(picture - 1, robust_sd(picture[small_quiet]), data)
    scores, radius, _, _ = orientation_hypercolumns(contrast, data, small_quiet, widths=FLOW_RIDGE_WIDTHS,
                                                    threads=max(1, (threads or 1) // 2))
    field = lateral_interaction(scores, radius, threads=threads)
    return {"picture": picture, "oriented": oriented, "scores": scores, "radius": radius, "field": field,
            "valid": data, "quiet": small_quiet, "activity": activity}


def at_size(layer, shape, kind=cv2.INTER_LINEAR):
    """Bring a picture from the flow field's reduced grid to the shape of a frame."""
    return cv2.resize(layer.astype(np.float32), (shape[1], shape[0]), interpolation=kind)

# %%
def to_frame_size(strung):
    """Bring the chains of a field that was worked out on the reduced grid to the size of a frame."""
    for chain in strung["chains"]:
        chain["x"], chain["y"] = chain["x"] * FLOW_REDUCTION + (FLOW_REDUCTION - 1) / 2, chain["y"] * FLOW_REDUCTION + (FLOW_REDUCTION - 1) / 2
        chain["radius"] = chain["radius"] * FLOW_REDUCTION
    strung["length"] = strung["length"] * FLOW_REDUCTION
    for name in ("x", "y", "radius"):
        strung["ends"][name] = strung["ends"][name] * FLOW_REDUCTION
    return strung


small_size = (frame_width // FLOW_REDUCTION, frame_height // FLOW_REDUCTION)
flow_pictures = []  # per fixation: where things change from frame to frame (1 is what the quiet pixels show)
flow_oriented = []  # per fixation: (N_ORIENT, ...) the mean energy of each neuron that was given the change itself
flow_energy = []    # per fixation: (N_ORIENT, rows / FLOW_REDUCTION, columns / FLOW_REDUCTION) the flow field's orientation score
flow_radius = []    # per fixation: the half-width of the line each of its neurons sees, in reduced pixels
flow_valid = []     # per fixation: the data area on the reduced grid
flow_pair_activity = []  # per fixation: per pair of frames, how strongly the whole field answered
flow_fields = []    # per fixation: flow_energy after the interactions of section 21
strung_flow = []    # per fixation: the flow field's chains before pruning, at the size of a frame (see strung_field)
flow_chains = []    # per fixation: the chains kept at the pruning chosen in section 22
for number in range(len(fixations)):
    # both stages: the hypercolumns are given the change between frames, and the picture of where things keep
    # changing goes through the layers of sections 19 to 21
    flow = flow_field(lambda position: temporal_frame(number, position), frame_pairs[number], valid_maps[number],
                      quiet_maps[number], threads=workers)
    # then the crests, the chains and the pruning of section 22
    crests = crest_neurons(flow["field"], flow["radius"], flow["valid"], scores=flow["scores"])
    strung = to_frame_size(strung_field(crests, flow["scores"], flow["valid"]))
    keep = pruned(strung, *CHAIN_PRUNING)
    flow_pictures.append(flow["picture"])
    flow_oriented.append(flow["oriented"])
    flow_energy.append(flow["scores"].astype(np.float16))
    flow_radius.append(flow["radius"])
    flow_valid.append(flow["valid"])
    flow_pair_activity.append(flow["activity"])
    flow_fields.append(flow["field"].astype(np.float16))
    strung_flow.append(strung)
    flow_chains.append([chain for chain, kept in zip(strung["chains"], keep) if kept])
    strongest, activity = flow["scores"].max(axis=0)[flow["quiet"]], flow["activity"]
    print(f"Fixation {number}: {len(frame_pairs[number])} pairs of frames | the field's answer per pair varies by "
          f"{activity.std() / activity.mean() * 100:.0f}% around its mean | the strongest neuron of a hypercolumn "
          f"answers above {T_LOW:g} on {(strongest > T_LOW).mean() * 100:.1f}% of the quiet pixels and "
          f"above {T_HIGH:g} on {(strongest > T_HIGH).mean() * 100:.2f}% | {len(crests['x'])} crest "
          f"neurons, {len(strung['chains'])} chains, {int(keep.sum())} kept at the chosen pruning with "
          f"{strung['length'][keep].sum():.0f} px")
    del flow, strongest

# checks, on the anchor's fixation
picture, data = flow_pictures[anchor], flow_valid[anchor]
# 1. the picture of where things change is section 15's smear, seen through differences of frames
smear = cv2.resize(np.nan_to_num(cleaned_smears[anchor]), small_size, interpolation=cv2.INTER_AREA)
both = data & np.isfinite(cv2.resize(cleaned_smears[anchor], small_size, interpolation=cv2.INTER_NEAREST))
print(f"Where things change, against the cleaned smear of section 16: correlation "
      f"{np.corrcoef(picture[both], smear[both])[0, 1]:.2f}")
# 2. orientation: the static field's, at the same place
static_strongest, static_orientation, _ = winning_neurons(orientation_scores[anchor], radius_maps[anchor])
static_strongest = cv2.resize(static_strongest.astype(np.float32), small_size, interpolation=cv2.INTER_AREA)
static_orientation = cv2.resize(static_orientation.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST)


def against_static(orientation, strong):
    """How a field's winning orientations lie to the static field's, where `strong`: the shares that are the same,
    within one step, and at right angles (7 or 8 steps off)."""
    turned = np.abs(orientation[strong].astype(int) - static_orientation[strong])
    turned = np.minimum(turned, N_ORIENT - turned)
    return (turned == 0).mean() * 100, (turned <= 1).mean() * 100, (turned >= 7).mean() * 100


energy = flow_energy[anchor].astype(np.float32)
strongest = energy.max(axis=0)
strong = data & (strongest > T_HIGH) & (static_strongest > T_HIGH)
same, near, across = against_static(np.argmax(energy, axis=0), strong)
print(f"Orientation where the flow field and the static field are both strong ({strong.sum()} reduced pixels): the same "
      f"on {same:.0f}%, within one step on {near:.0f}%, at right angles on {across:.0f}% "
      f"(chance: {100 / N_ORIENT:.0f}%, {300 / N_ORIENT:.0f}%, {300 / N_ORIENT:.0f}%)")
excess = flow_oriented[anchor] - np.median(flow_oriented[anchor], axis=0)  # a neuron's energy above its hypercolumn's median
changing = data & (picture > 1 + T_HIGH * robust_sd(picture[data & (picture < np.median(picture[data]))])) & (static_strongest > T_HIGH)
same, near, across = against_static(np.argmax(excess, axis=0), changing)
print(f"The neurons that are given the change itself, on the {changing.sum()} reduced pixels where things change on a "
      f"vessel: their strongest orientation is the vessel's on {same:.0f}%, within one step on {near:.0f}%, at right "
      f"angles on {across:.0f}%. The change inside a vessel is not a line along the vessel, which is why the centerlines "
      f"are read from the picture of where things change")
lines = draw_chains(flow_chains[anchor], frame_shape) >= 0
static_lines = kept_lines(strung_static[anchor], static_keeps[anchor])
print(f"Flow chains: {found_again(lines, valid_maps[anchor], static_lines, valid_maps[anchor], None) * 100:.0f}% of their "
      f"length lies on the static chains, and they find "
      f"{found_again(static_lines, valid_maps[anchor], lines, valid_maps[anchor], None) * 100:.0f}% of the static chains")

# show the flow field: where things change, its orientation score, and its chains over the mean density
top = np.percentile(picture[data], 99.5)
flow_score_picture = orientation_hues(np.argmax(energy, axis=0), strongest / 20)
chains_picture = chain_picture(flow_chains[anchor], background_fits[anchor]["OD"], *DENSITY_RANGE)
Image.fromarray(chains_picture).save("vessel_pictures/chains/flow_chains.png")
figure, axes = plt.subplots(1, 3, figsize=(16, 3.6))
axes[0].imshow(picture, cmap="gray", vmin=1, vmax=top)
axes[0].set_title("where things change between frames (black: noise level)", fontsize=10)
axes[1].imshow(flow_score_picture, interpolation="antialiased")
axes[1].set_title("flow field: hue is the winning orientation", fontsize=10)
axes[2].imshow(chains_picture, interpolation="antialiased")
axes[2].set_title("chains of the flow field over the mean density", fontsize=10)
for axis in axes:
    axis.axis("off")
figure.tight_layout()
plt.show()
del flow_score_picture, chains_picture, lines, static_lines, strongest, static_strongest, energy, excess

# %% [markdown]
# ## 25. Temporal signals of every chain
#
# `segment_readings` reads four things for every static chain of a fixation.
#
# **a. Line mass.** The kymograph `M(s, t)` of section 23, for every chain.
#
# **b. Velocity**, for the width classes in which section 23 found the flow visible, on chains of at least `VELOCITY_LENGTH` pixels. Cells that travel along a vessel draw slanted streaks in the kymograph, and the slope of a streak is the velocity. The slope is read as the shift at which a frame's line mass correlates best with the next frame's (`travelling_correlation`), weighted towards the chain's lucky frames. A velocity is *clear* when that peak stands `VELOCITY_CLEAR` standard errors above 0; only then is a direction given. As a cross-check the 1-D continuity equation, `v = -sum(dM/dt dM/ds) / sum((dM/ds)^2)`, is evaluated on the same kymograph (`continuity_velocity`).
#
# **Not as in the plan:** the plan reads the streaks by running the hypercolumns of section 20 on each kymograph. The correlation measures the same slope and costs a fraction of the time for thousands of chains. The hypercolumns are not run on the kymographs here.
#
# **c. Slow signals.**
#
# - *Heart rate*: the peak, between 0.7 and 2.5 Hz, of the periodogram of all vessels' line mass together (`heart_rate`). It counts as found when frames in shuffled order reach such a peak in fewer than 1% of tries.
# - *Pulse of each chain*: the amplitude and phase of its line mass at the heart rate and at twice the heart rate (`pulse_fit`).
# - *Vasomotion* is slower than 0.2 Hz and needs a burst longer than 5 seconds. The cell says when the burst is too short.
#
# **d. Focus and lucky frames.** A vessel's width does not change, so a change of its apparent width is a change of focus.
#
# - *Apparent half-width* in every frame (`apparent_width`), from the chain's mean profile across, the median of three frames.
# - *Blur*: `sqrt(width^2 - sharpest width^2)`.
# - *Lucky frames* (`lucky_weights`): frames blurred by no more than `LUCKY_BLUR` of the sharpest width, in runs of at least `LUCKY_RUN` frames that follow each other. Their weights are used for the velocity.
# - *Depth order*: only if focus drifts one way over the burst does each chain's time of best focus order the chains in depth. The cell says whether it does.
#
# **Result:** `segment_signals[n][k]`, one dictionary per chain `k` of `static_chains[n]`:
#
# | Key | Content |
# |---|---|
# | `M`, `times` | the kymograph (points x frames) and the frames' times |
# | `velocity`, `velocity_confidence`, `direction` | pixels per second along the chain's points; standard errors; +1, -1, or 0 when not clear |
# | `velocity_check` | the continuity equation's velocity |
# | `pulse_amplitude`, `pulse_phase` | at the heart rate (NaN when none was found) |
# | `width_trace`, `blur_trace`, `sharpest`, `lucky` | the apparent half-width and blur per frame, the sharpest half-width, the lucky-frame weights |
# | `best_focus_time` | when the chain was sharpest |
#
# `burst_signals[n]` holds what was found for the fixation as a whole: the heart rate and the drift of focus.

# %%
VELOCITY_LENGTH = 40      # pixels; a chain has to be this long for a velocity
VELOCITY_CLEAR = 4.0      # standard errors; a velocity is clear when its correlation peak stands this far above 0
HEART_BAND = (0.7, 2.5)   # Hz; where the heart rate is looked for
HEART_CERTAIN = 0.01      # the heart rate counts as found when shuffled frames reach its peak less often than this
VASOMOTION_BELOW = 0.2    # Hz; vasomotion is slower than this, and needs a burst longer than one such cycle
FOCUS_WIDTHS = 16         # widths tried for a vessel's apparent half-width, from 0.5 to 2.5 x its half-width
LUCKY_BLUR = 0.5          # of a vessel's sharpest half-width; a frame blurred by no more than this is a lucky frame
LUCKY_RUN = 3             # frames; lucky frames count only in runs of at least this many that follow each other
SPEED_ROUNDS = 200        # resamplings of the pairs of frames from which the spread of a streak's angle is taken


def heart_rate(times, signal):
    """Find the heart rate in a signal sampled at uneven times: the peak of its periodogram within HEART_BAND.

    Returns the rate (Hz), how much of the signal's variance the peak explains, and how often frames in shuffled
    order reach a peak as high (the chance that it is nothing).
    """
    signal = scipy.signal.detrend(signal - signal.mean())
    rates = np.arange(HEART_BAND[0], HEART_BAND[1] + 1e-9, 0.02)
    power = scipy.signal.lombscargle(times, signal, 2 * np.pi * rates, normalize=True)
    rng = np.random.default_rng(0)
    shuffled = [scipy.signal.lombscargle(times, rng.permutation(signal), 2 * np.pi * rates, normalize=True).max()
                for _ in range(400)]
    return float(rates[np.argmax(power)]), float(power.max()), float(np.mean(np.array(shuffled) >= power.max()))


def pulse_fit(times, signals, rate):
    """Fit every signal (one row each) with a line plus the heart rate and its first harmonic. Returns the
    amplitude and phase at the heart rate and at the harmonic: the fit is A cos(2 pi rate t - phase)."""
    design = np.column_stack([np.ones_like(times), times - times.mean()]
                             + [wave(2 * np.pi * rate * harmonic * times) for harmonic in (1, 2) for wave in (np.cos, np.sin)])
    weights = np.linalg.lstsq(design, signals.T, rcond=None)[0]
    return (np.hypot(weights[2], weights[3]), np.arctan2(weights[3], weights[2]),
            np.hypot(weights[4], weights[5]), np.arctan2(weights[5], weights[4]))


def apparent_width(profiles, radius):
    """Return a vessel's apparent half-width in every frame, from its profile across (frames x samples).

    Each frame's profile is the median of three frames, which averages the passing cells out. The half-width is
    found by scale selection as in section 18: the width at which the profile's curvature at the centerline,
    times width^1.5, is largest, with a parabola through that width's answer and its neighbours'.
    """
    frames, samples = profiles.shape
    padded = np.concatenate([profiles[:1], profiles, profiles[-1:]])
    smooth = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    offsets = np.arange(samples) - samples // 2
    widths = radius * np.geomspace(0.5, 2.5, FOCUS_WIDTHS)
    # minus the second derivative of a Gaussian of each width, at the centerline, times width^1.5
    kernels = np.stack([(1 - (offsets / width) ** 2) * np.exp(-0.5 * (offsets / width) ** 2) * width**-1.5 for width in widths])
    answers = smooth @ kernels.T  # (frames, widths)
    best = np.clip(np.argmax(answers, axis=1), 1, FOCUS_WIDTHS - 2)
    rows = np.arange(frames)
    below, top, above = answers[rows, best - 1], answers[rows, best], answers[rows, best + 1]
    curve = below - 2 * top + above
    shift = np.where(curve < 0, np.clip(0.5 * (below - above) / np.where(curve < 0, curve, -1), -1, 1), 0)
    return widths[best] * (widths[1] / widths[0]) ** shift


def lucky_weights(blur, sharpest, times, fps):
    """Return a weight per frame from a vessel's blur: near 1 in frames as sharp as its sharpest, falling with the
    blur, and 0 outside runs of at least LUCKY_RUN lucky frames that follow each other without a dropped frame."""
    lucky = blur <= LUCKY_BLUR * sharpest
    follows = np.concatenate([[False], np.diff(times) <= 1.5 / fps])
    # a run starts at a lucky frame that does not follow a lucky frame directly; number the runs and count their frames
    starts = lucky & ~(follows & np.concatenate([[False], lucky[:-1]]))
    run = np.cumsum(starts)
    in_run = lucky & (np.bincount(run[lucky], minlength=run[-1] + 1)[run] >= LUCKY_RUN)
    return in_run / (1 + (blur / (LUCKY_BLUR * sharpest)) ** 2)


def continuity_velocity(mass, pairs, guess):
    """Cross-check a velocity with the 1-D continuity equation, v = -sum(dM/dt dM/ds) / sum((dM/ds)^2), on the
    kymograph reduced along the chain until the guessed shift per frame is under one sample."""
    centred, _ = fluctuation(mass)
    step = int(max(1, np.ceil(1.5 * abs(guess))))
    points = len(centred) // step
    if points < 6 or not pairs:
        return np.nan
    reduced = centred[:points * step].reshape(points, step, -1).mean(axis=1)
    earlier, later = reduced[:, [pair[0] for pair in pairs]], reduced[:, [pair[1] for pair in pairs]]
    middle = 0.5 * (earlier + later)
    along = 0.5 * (middle[2:] - middle[:-2])
    over_time = (later - earlier)[1:-1]
    return float(-step * (over_time * along).sum() / max((along**2).sum(), 1e-12))


def speed_of(mass, pairs, fps, weights=None):
    """The speed of the pattern along a kymograph, and how well it is known.

    The pattern's shift per frame is where the travelling correlation peaks (travel_reading). In the kymograph the
    streaks then lie at the angle phi = arctan(shift) from the time axis, and the speed is fps x tan(phi). How well
    the angle is known, sigma_phi, comes from reading the peak again on SPEED_ROUNDS resamplings of the pairs of
    frames. As a spread of the speed that is fps x sigma_phi / cos^2(phi): the steeper the streaks, the less the
    speed is worth.

    Returns a dict: velocity (pixels per second along the chain's points; NaN without a peak), sigma (of the
    velocity, the same units), z (the peak in its standard errors), shift (pixels per frame), phi and sigma_phi
    (radians).
    """
    speed = {"velocity": np.nan, "sigma": np.nan, "z": 0.0, "shift": np.nan, "phi": np.nan, "sigma_phi": np.nan}
    shifts, curves, share = travelling_curves(mass, pairs, weights=weights)
    if curves is None:
        return speed
    travel = travel_reading(shifts, *curves_mean(curves, share))
    if not np.isfinite(travel["shift"]):
        return speed
    picks = np.random.default_rng(0).choice(len(curves), size=(SPEED_ROUNDS, len(curves)), p=share)
    flat = np.ones(len(shifts))
    angles = [np.arctan(travel_reading(shifts, curves[pick].mean(axis=0), flat)["shift"]) for pick in picks]
    phi, sigma_phi = float(np.arctan(travel["shift"])), float(np.std(angles))
    speed.update({"velocity": travel["shift"] * fps, "sigma": fps * sigma_phi / np.cos(phi) ** 2, "z": travel["z"],
                  "shift": travel["shift"], "phi": phi, "sigma_phi": sigma_phi})
    return speed


def clear_direction(speed, limit=np.inf):
    """The direction a speed (speed_of) gives: +1 along the chain's points, -1 against them, and 0 when it is not
    clear: the correlation peak stands under VELOCITY_CLEAR standard errors above 0, the pattern moves less than
    half a pixel per frame, or the speed is above `limit` (pixels per second)."""
    clear = (np.isfinite(speed["shift"]) and speed["z"] >= VELOCITY_CLEAR and abs(speed["shift"]) >= 0.5
             and abs(speed["velocity"]) <= limit)
    return int(np.sign(speed["shift"])) if clear else 0


def segment_readings(chains, frame_at, times, pairs, fps, visible=None, threads=None):
    """Section 25: read the temporal signals of every chain of one fixation.

    Returns one dict per chain (see the section's text for the keys), and what was found for the fixation as a
    whole: the heart rate and whether focus drifted one way over the burst.
    """
    positions = list(range(len(times)))
    read = chain_signals(chains, frame_at, positions, threads)
    radii = np.array([np.median(chain["radius"]) for chain in chains])
    lengths = np.array([chain_length(chain) for chain in chains])
    # the mean line mass of every chain in every frame, as a share of its mean over time
    means = np.array([mean_of_data(signals["M"], axis=0) for signals in read])
    level = mean_of_data(means, axis=1)[:, None]
    relative = np.nan_to_num(means / np.where(np.abs(level) > 1e-9, level, np.nan) - 1)
    # all vessels together: the line mass of every chain times its length, added up, against its mean over time
    weight = lengths * (np.nan_to_num(level[:, 0]) > 0) * np.isfinite(means).all(axis=1)
    whole = ((np.nan_to_num(means) * weight[:, None]).sum(axis=0)
             / max((np.nan_to_num(level[:, 0]) * weight).sum(), 1e-9) - 1)
    burst = {"seconds": float(times[-1] - times[0]), "global": whole}
    # c. slow signals: the heart rate from all vessels together, then every chain's own pulse
    burst["heart rate"], burst["heart power"], burst["heart chance"] = (np.nan, np.nan, 1.0)
    if burst["seconds"] >= 1.5 / HEART_BAND[0] and len(times) >= 16:
        burst["heart rate"], burst["heart power"], burst["heart chance"] = heart_rate(times, whole)
    found = burst["heart chance"] < HEART_CERTAIN
    if found:
        amplitude, phase, amplitude_2, phase_2 = pulse_fit(times, relative, burst["heart rate"])
    readings = []
    for number, (chain, signals) in enumerate(zip(chains, read)):
        reading = {"M": signals["M"], "times": times, "radius": float(radii[number]), "length": float(lengths[number]),
                   "mass": float(level[number, 0]), "relative": relative[number]}
        reading["pulse_amplitude"], reading["pulse_phase"], reading["pulse_amplitude_2"], reading["pulse_phase_2"] = (
            (float(amplitude[number]), float(phase[number]), float(amplitude_2[number]), float(phase_2[number]))
            if found else (np.nan,) * 4)
        # d. focus: the apparent half-width in every frame, the blur it implies, and the lucky frames
        width = apparent_width(signals["profile"], radii[number])
        sharpest = float(np.percentile(width, 5))
        blur = np.sqrt(np.maximum(width**2 - sharpest**2, 0))
        reading["width_trace"], reading["blur_trace"], reading["sharpest"] = width, blur, sharpest
        reading["lucky"] = lucky_weights(blur, sharpest, times, fps)
        reading["best_focus_time"] = float(times[np.argmin(scipy.ndimage.uniform_filter1d(width, 5, mode="nearest"))])
        # b. velocity: the slope of the streaks in the kymograph, from the correlation of each frame with the next
        reading["velocity"], reading["velocity_confidence"], reading["direction"], reading["velocity_check"] = (
            np.nan, 0.0, 0, np.nan)
        reading["velocity_sigma"] = np.nan
        if lengths[number] >= VELOCITY_LENGTH and (visible is None or visible[width_class(radii[number])]):
            lucky = reading["lucky"] if (reading["lucky"] > 0).sum() >= 0.3 * len(times) else None
            speed = speed_of(signals["M"], pairs, fps, lucky)
            if np.isfinite(speed["shift"]):
                reading["velocity"] = speed["velocity"]  # pixels per second, along the chain's points
                reading["velocity_sigma"], reading["velocity_confidence"] = speed["sigma"], speed["z"]
                reading["direction"] = clear_direction(speed)
                reading["velocity_check"] = continuity_velocity(signals["M"], pairs, speed["shift"]) * fps
        readings.append(reading)
    # does focus drift one way over the burst? then each chain's time of best focus orders the chains in depth
    widths = np.array([reading["width_trace"] / reading["sharpest"] for reading in readings if reading["length"] >= VELOCITY_LENGTH])
    burst["focus drift"] = float(scipy.stats.spearmanr(times, np.median(widths, axis=0))[0]) if len(widths) else np.nan
    return readings, burst

# %% [markdown]
# The cell below reads the signals of every fixation's chains. The figure shows, for the anchor's fixation, the kymograph of the chain with the clearest velocity in each width class (the red line has the slope that was found), the line mass of all vessels together with its periodogram, and the apparent half-width of the six longest chains over the burst.

# %%
visible_classes = [flow_visible.get(name, {}).get("visible", False) for _, _, name in WIDTH_CLASSES]

segment_signals = []  # per fixation: one dict of signals per chain of static_chains (see the text above)
burst_signals = []    # per fixation: what was found for the fixation as a whole (heart rate, focus drift)
for number, fixation in enumerate(fixations):
    readings, burst = segment_readings(
        static_chains[number], lambda position: temporal_frame(number, position), fixation["times"],
        frame_pairs[number], fps, visible=visible_classes, threads=workers)
    segment_signals.append(readings)
    burst_signals.append(burst)
    print(f"Fixation {number}: {len(readings)} chains read over {burst['seconds']:.2f} s")
    # c. slow signals
    if np.isfinite(burst["heart rate"]):
        found = burst["heart chance"] < HEART_CERTAIN
        print(f"   heart rate: the periodogram of all vessels' line mass peaks at {burst['heart rate']:.2f} Hz "
              f"({burst['heart rate'] * 60:.0f} per minute) with {burst['heart power'] * 100:.0f}% of the variance; frames in "
              f"shuffled order reach that in {burst['heart chance'] * 100:.1f}% of tries: "
              + ("found" if found else "not certain, so no pulse is fitted"))
    else:
        print(f"   heart rate: the fixation is too short for one ({burst['seconds']:.2f} s)")
    if burst["seconds"] < 1 / VASOMOTION_BELOW:
        print(f"   vasomotion (below {VASOMOTION_BELOW:g} Hz) needs more than {1 / VASOMOTION_BELOW:.0f} s of burst: not measured")
    # b. velocities
    for kind, (low, high, name) in enumerate(WIDTH_CLASSES):
        members = [reading for reading in readings if low <= reading["radius"] < high and reading["length"] >= VELOCITY_LENGTH]
        clear = [reading for reading in members if reading["direction"] != 0]
        if not members:
            continue
        speeds = np.array([abs(reading["velocity"]) for reading in clear])
        checks = np.array([reading["velocity_check"] for reading in clear])
        agree = np.mean(np.sign(checks) == np.sign([reading["velocity"] for reading in clear])) if len(clear) else np.nan
        print(f"   velocity, {name}: {len(clear)} of {len(members)} chains of {VELOCITY_LENGTH} px or more have a clear one"
              + (f"; median {np.median(speeds):.0f} px/s ({np.median(speeds) / fps:.1f} px per frame), quartiles "
                 f"{np.percentile(speeds, 25):.0f} and {np.percentile(speeds, 75):.0f}; the continuity equation gives the "
                 f"same direction on {agree * 100:.0f}% and {np.nanmedian(np.abs(checks) / speeds):.2f} x the speed (median)"
                 if len(clear) else ""))
    # d. focus
    long = [reading for reading in readings if reading["length"] >= VELOCITY_LENGTH]
    if long:
        blurred = np.median([np.median(reading["blur_trace"]) / reading["sharpest"] for reading in long])
        lucky = np.mean([(reading["lucky"] > 0).mean() for reading in long])
        print(f"   focus: over the burst a vessel's apparent half-width is blurred by {blurred:.2f} of its sharpest (median); "
              f"{lucky * 100:.0f}% of the frames are lucky (in runs of {LUCKY_RUN} or more) | all vessels' half-width "
              f"against time: rank correlation {burst['focus drift']:+.2f}"
              + (", focus drifts one way, so the time of best focus orders the chains in depth"
                 if abs(burst["focus drift"]) >= 0.5 else ", no steady drift, so no depth order is read from it"))

# show, for the anchor's fixation: three kymographs with the velocity found, the signal of all vessels, and blur traces
readings, burst = segment_signals[anchor], burst_signals[anchor]
times = fixations[anchor]["times"]
shown = []
for kind in range(len(WIDTH_CLASSES)):
    members = [reading for reading in readings if width_class(reading["radius"]) == kind and reading["direction"] != 0]
    if members:
        shown.append(max(members, key=lambda reading: reading["velocity_confidence"]))
figure, axes = plt.subplots(2, 3, figsize=(16, 7))
for axis, reading in zip(axes[0], shown):
    centred, _ = fluctuation(reading["M"])
    limit = 3 * robust_sd(centred)
    axis.imshow(centred, cmap="gray", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    middle = len(centred) / 2
    frames = np.array([0, centred.shape[1] - 1])
    slope = reading["velocity"] / fps
    span = min(len(centred) / max(abs(slope), 1e-6), centred.shape[1]) / 2  # frames the line stays inside the picture
    line = np.array([centred.shape[1] / 2 - span, centred.shape[1] / 2 + span])
    axis.plot(line, middle + slope * (line - centred.shape[1] / 2), color="red", linewidth=1)
    axis.set_title(f"half-width {reading['radius']:.1f} px: {reading['velocity']:+.0f} px/s "
                   f"({reading['velocity_confidence']:.0f} standard errors)", fontsize=10)
    axis.set_xlabel("frame of the fixation")
    axis.set_ylabel("pixels along the chain")
axes[1][0].plot(times, burst["global"] * 100, color="black")
axes[1][0].set_title("line mass of all vessels together, change from its mean (%)", fontsize=10)
axes[1][0].set_xlabel("seconds")
rates = np.arange(0.3, 6, 0.02)
power = scipy.signal.lombscargle(times, scipy.signal.detrend(burst["global"] - burst["global"].mean()), 2 * np.pi * rates,
                                 normalize=True)
axes[1][1].plot(rates, power, color="black")
axes[1][1].axvspan(*HEART_BAND, color="gray", alpha=0.2)
axes[1][1].set_title("its periodogram (gray: where the heart rate is looked for)", fontsize=10)
axes[1][1].set_xlabel("Hz")
for reading in sorted(readings, key=lambda reading: -reading["length"])[:6]:
    axes[1][2].plot(times, reading["width_trace"] / reading["sharpest"], linewidth=0.8)
axes[1][2].set_title("apparent half-width of the six longest chains / their sharpest", fontsize=10)
axes[1][2].set_xlabel("seconds")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## 26. One field and one graph
#
# **Agreement of the two fields** (`field_agreement`). The static and the flow orientation scores are compared pixel by pixel, orientation by orientation:
#
# | Where | Meaning |
# |---|---|
# | both answer | a vessel with flow |
# | only the static field | no flow seen: not perfused, or flow too fast or too slow to see |
# | only the flow field | a faint or blurred vessel that the static field missed |
#
# **One field** (`combined_field`): at every place and orientation the larger of the static and the flow field, both in noise units. Where the static field already has a line of that orientation the flow field adds nothing; its crest of the same vessel lies a pixel or two beside the static one and would become a second chain alongside. The combined field is strung and pruned as in section 22: `vessel_chains`.
#
# **The graph** (`graph_from_chains`). Nodes are chain ends and the places where chains meet. Edges are the pieces of chain between two nodes.
#
# - **One junction, one node** (`contract_cores`). A junction's core reaches half the width of its thickest vessel plus `RING_MARGIN`. Two nodes joined by an edge that never leaves the core are one junction; an edge that short which ends loose is a spur inside the vessel.
# - **Split crossings are fused** (`fuse_split_crossings`). A crossing often comes out as two 3-arm nodes joined by a short piece. They are fused when the piece is shorter than `max(3, 2.5 x the thick vessel's width / sin(angle))` and the outer arms pair up straight across it.
#
# - **A vessel traced twice is one arm.** Two chains along one vessel leave a node side by side, within `SAME_ARM` degrees of each other. They count as one arm, read from the longer of the two. A node left with fewer than three arms is no junction.
#
# **Arms are read on a ring outside the core** (`read_arms`), from half the thick vessel's width plus 2 pixels out to several widths. Measuring inside the core, where the vessels overlap, roughly doubles the error of the angles. Each arm records its direction, half-width and width, mean density, and from the chain it belongs to: line mass over time, velocity and direction of flow (as leaving the junction or entering it), pulse phase and blur trace. An edge is often too short for a velocity of its own, so its velocity is read from the chain's kymograph around it (`edge_reading`).
#
# **Result:**
#
# - `agreements[n]`: the three pictures (both, static only, flow only).
# - `vessel_chains[n]`: the chains of the combined field.
# - `vessel_graphs[n]`: `nodes` (with `arms` for every junction), `edges`, the edges' `readings`, and how many crossings were `fused`.

# %%
NODE_MERGE = 3        # points along a chain; junction places closer than this on one chain are one node
RING_MARGIN = 2.0     # pixels; an arm is read from half the thickest vessel's width plus this, outwards
RING_WIDTHS = 3.0     # widths of the thickest vessel; how far out the ring reaches (at least RING_LEAST pixels)
RING_LEAST = 10.0     # pixels
SAME_ARM = 20.0       # degrees; two arms that leave within this of each other, side by side, are one vessel traced twice
FUSE_FACTOR = 2.5     # two 3-way nodes closer than this many widths of the thick vessel / sin(angle) may be one crossing
FUSE_STRAIGHT = 30.0  # degrees; the outer arms of such a pair have to run straight across within this
STATIC_NEAR = 3  # pixels; the flow field adds nothing this close to a static line of the same orientation


def graph_from_chains(chains, attached, meetings):
    """Turn chains and their junction candidates into a graph.

    Nodes are chain ends and the places where chains meet; edges are the pieces of chain between two nodes.
    Returns (nodes, edges): a node is a dict with its place (x, y) and `ends`, the edge ends that meet in it as
    (edge number, 0 for the edge's first point or 1 for its last); an edge is a chain (the same keys, its points
    in order from its first node to its second) with `nodes` and the `chain` it was cut from.
    """
    sizes = np.array([len(chain["x"]) for chain in chains])
    # an "incidence" is a place on a chain where a node sits: first every chain's two ends, then the meetings
    on_chain = list(np.repeat(np.arange(len(chains)), 2))
    at_place = list(np.column_stack([np.zeros(len(chains), dtype=int), sizes - 1]).ravel())
    links = [(2 * number, 2 * number + 1) for number, chain in enumerate(chains) if chain["closed"] or sizes[number] <= NODE_MERGE]
    for chain, end, target, place in attached:
        on_chain.append(target), at_place.append(place)
        links.append((2 * chain + end, len(on_chain) - 1))
    for chain, place, other, other_place in meetings:
        on_chain += [chain, other]
        at_place += [place, other_place]
        links.append((len(on_chain) - 2, len(on_chain) - 1))
    on_chain, at_place = np.array(on_chain), np.array(at_place)
    # places close to each other on one chain are the same node
    order = np.lexsort((at_place, on_chain))
    close = (on_chain[order][1:] == on_chain[order][:-1]) & (at_place[order][1:] - at_place[order][:-1] <= NODE_MERGE)
    links = np.concatenate([np.array(links, dtype=int).reshape(-1, 2), np.column_stack([order[:-1][close], order[1:][close]])])
    graph = scipy.sparse.coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(len(on_chain),) * 2)
    count, node_of = scipy.sparse.csgraph.connected_components(graph, directed=False)
    x = np.array([chains[chain]["x"][place] for chain, place in zip(on_chain, at_place)])
    y = np.array([chains[chain]["y"][place] for chain, place in zip(on_chain, at_place)])
    members = np.bincount(node_of, minlength=count)
    nodes = [{"x": float(px), "y": float(py), "ends": []}
             for px, py in zip(np.bincount(node_of, weights=x) / members, np.bincount(node_of, weights=y) / members)]
    # cut every chain at its nodes
    edges = []
    names = ("x", "y", "theta", "radius", "strength", "bridge")
    starts = np.flatnonzero(np.concatenate([[True], on_chain[order][1:] != on_chain[order][:-1]]))
    for start, stop in zip(starts, np.concatenate([starts[1:], [len(order)]])):
        chain, places, here = on_chain[order[start]], at_place[order[start:stop]], node_of[order[start:stop]]
        for first in range(len(places) - 1):
            if here[first] == here[first + 1] and places[first + 1] - places[first] <= NODE_MERGE:
                continue  # two places of one node: nothing lies between them
            piece = slice(places[first], places[first + 1] + 1)
            edge = {name: chains[chain][name][piece] for name in names}
            edge.update({"closed": False, "nodes": (int(here[first]), int(here[first + 1])), "chain": int(chain),
                         "span": (int(places[first]), int(places[first + 1]))})
            nodes[here[first]]["ends"].append((len(edges), 0))
            nodes[here[first + 1]]["ends"].append((len(edges), 1))
            edges.append(edge)
    return nodes, edges


def contract_cores(nodes, edges):
    """Merge the nodes that belong to one junction, and drop the spurs inside it.

    A junction's core reaches half the width of its thickest vessel plus RING_MARGIN from the node. An edge between
    two nodes that is no longer than that never leaves the core: the two nodes are one junction, and the edge is
    marked `core`. An edge that short which ends loose is a spur inside the vessel, and is marked `spur`. Nodes
    are merged shortest edge first, and only while the merged junction stays within two core sizes.

    Returns the new list of nodes; the edges' `nodes` are renumbered to it.
    """
    lengths = np.array([chain_length(edge) for edge in edges])
    widths = np.array([2 * float(np.median(edge["radius"])) / CYLINDER for edge in edges])
    core = np.array([max([widths[edge] for edge, _ in node["ends"]], default=0.0) / 2 + RING_MARGIN for node in nodes])
    degree = np.array([len(node["ends"]) for node in nodes])
    parent = list(range(len(nodes)))
    box = [[node["x"], node["y"], node["x"], node["y"]] for node in nodes]  # of each merged junction

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for number in np.argsort(lengths):
        first, second = edges[number]["nodes"]
        reach = max(core[first], core[second])
        if lengths[number] > reach:
            break
        if min(degree[first], degree[second]) < 2:
            edges[number]["spur"] = first != second
            continue
        a, b = root(first), root(second)
        merged = [min(box[a][0], box[b][0]), min(box[a][1], box[b][1]), max(box[a][2], box[b][2]), max(box[a][3], box[b][3])]
        if a != b and max(merged[2] - merged[0], merged[3] - merged[1]) > 2 * reach:
            continue
        edges[number]["core"] = True
        parent[b], box[a] = a, merged
    roots = np.array([root(node) for node in range(len(nodes))])
    kept, number_of = np.unique(roots, return_inverse=True)
    members = np.bincount(number_of)
    merged_nodes = [{"x": float(x), "y": float(y), "ends": []} for x, y in zip(
        np.bincount(number_of, weights=[node["x"] for node in nodes]) / members,
        np.bincount(number_of, weights=[node["y"] for node in nodes]) / members)]
    for number, edge in enumerate(edges):
        edge["nodes"] = tuple(int(number_of[node]) for node in edge["nodes"])
        if not edge.get("core") and not edge.get("spur"):
            merged_nodes[edge["nodes"][0]]["ends"].append((number, 0))
            merged_nodes[edge["nodes"][1]]["ends"].append((number, 1))
    return merged_nodes


def read_arm(node, edge, end, reading, density, inner, outer):
    """Read one arm of a junction on a ring around it: the part of an edge between `inner` and `outer` pixels from
    the node. Measuring inside the junction's core, where the vessels overlap, roughly doubles the error of the
    angles. Returns a dict (see the section's text)."""
    order = np.arange(len(edge["x"])) if end == 0 else np.arange(len(edge["x"]))[::-1]  # away from the node
    distance = np.hypot(edge["x"][order] - node["x"], edge["y"][order] - node["y"])
    beyond = np.flatnonzero(distance >= inner)
    if len(beyond) >= 3:
        last = beyond[0] + np.argmax(np.concatenate([distance[beyond[0]:] > outer, [True]]))
        ring = order[beyond[0]:max(last, beyond[0] + 3)]
        short = False
    else:  # the edge never leaves the core: its outer half has to do
        ring = order[len(order) // 2:]
        short = True
    x, y = edge["x"][ring], edge["y"][ring]
    along = np.array([x[-1] - x[0], y[-1] - y[0]]) if len(ring) > 1 else np.array([x[0] - node["x"], y[0] - node["y"]])
    if np.hypot(*along) < 1e-6:
        along = np.array([np.cos(edge["theta"][ring[0]]), np.sin(edge["theta"][ring[0]])])
    along = along / np.hypot(*along)
    rows = np.clip(np.rint(y).astype(int), 0, density.shape[0] - 1)
    columns = np.clip(np.rint(x).astype(int), 0, density.shape[1] - 1)
    out = 1 if end == 0 else -1  # the edge's points run away from the node at its first end, towards it at its last
    arm = {
        "edge": None, "end": end, "short": short, "points": ring, "x": float(x.mean()), "y": float(y.mean()),
        "direction": along, "angle": float(np.degrees(np.arctan2(along[1], along[0])) % 360),
        "radius": float(np.median(edge["radius"][ring])), "width": float(2 * np.median(edge["radius"][ring]) / CYLINDER),
        "density": float(density[rows, columns].mean()), "strength": float(np.median(edge["strength"][ring])),
        "length": chain_length(edge),
    }
    if reading is not None:
        arm.update({
            "mass_trace": mean_of_data(reading["M"][ring], axis=0), "mass": reading["mass"],
            "velocity_out": out * reading["velocity"], "direction_out": out * reading["direction"],
            "velocity_confidence": reading["velocity_confidence"], "pulse_phase": reading["pulse_phase"],
            "pulse_amplitude": reading["pulse_amplitude"], "blur_trace": reading["blur_trace"],
            "width_trace": reading["width_trace"], "sharpest": reading["sharpest"], "lucky": reading["lucky"],
            "velocity_sigma": reading.get("velocity_sigma", np.nan), "trackable": reading.get("trackable", True),
            "vessel": reading.get("vessel", -1), "flagged_out": out * reading.get("direction flagged", 0),
        })
    return arm


def read_arms(node, edges, readings, density):
    """Read all arms of one node on a ring outside its core. The ring starts at half the width of the node's
    thickest vessel plus RING_MARGIN and reaches RING_WIDTHS of that width further out."""
    thick = max([2 * float(np.median(edges[edge]["radius"])) / CYLINDER for edge, _ in node["ends"]], default=0.0)
    inner = thick / 2 + RING_MARGIN
    outer = inner + max(RING_WIDTHS * thick, RING_LEAST)
    arms = []
    for edge, end in node["ends"]:
        arm = read_arm(node, edges[edge], end, readings[edge] if readings is not None else None, density, inner, outer)
        arm["edge"] = edge
        arms.append(arm)
    # two chains that trace one vessel leave the node side by side: they are one arm, read from the longer of the two
    kept = []
    for arm in sorted(arms, key=lambda arm: -arm["length"]):
        twin = any(np.degrees(np.arccos(np.clip(np.dot(arm["direction"], other["direction"]), -1, 1))) < SAME_ARM
                   and np.hypot(arm["x"] - other["x"], arm["y"] - other["y"]) <= max(arm["width"], other["width"]) + RING_MARGIN
                   for other in kept)
        if not twin:
            kept.append(arm)
    node["arms"], node["thick"], node["ring"] = sorted(kept, key=lambda arm: arm["angle"]), thick, (inner, outer)
    return node


def turn_between(first, second):
    """How far two arms are from running straight through a junction, in degrees: 0 when they leave in exactly
    opposite directions."""
    return float(np.degrees(np.arccos(np.clip(-np.dot(first["direction"], second["direction"]), -1, 1))))


def fuse_split_crossings(nodes, edges, readings, density):
    """Fuse the crossings that came out as two 3-way nodes joined by a short piece.

    Two such nodes are fused when the piece between them is shorter than max(3, FUSE_FACTOR x the thick vessel's
    width / sin(angle of the crossing)) and their four outer arms pair up straight across the piece, each pair
    within FUSE_STRAIGHT of a straight line. The fused node gets the four outer arms; the piece becomes its core.

    Returns the number of crossings fused; the nodes that were fused away are marked `gone`.
    """
    candidates = sorted((chain_length(edge), number) for number, edge in enumerate(edges)
                        if edge["nodes"][0] != edge["nodes"][1]
                        and all(len(nodes[node].get("arms", [])) == 3 for node in edge["nodes"]))
    fused = 0
    for length, number in candidates:
        first, second = (nodes[node] for node in edges[number]["nodes"])
        if first.get("gone") or second.get("gone"):
            continue
        ours = [arm for arm in first["arms"] if arm["edge"] != number]
        theirs = [arm for arm in second["arms"] if arm["edge"] != number]
        if len(ours) != 2 or len(theirs) != 2:
            continue
        # the two ways of pairing the outer arms across the piece; the straighter one counts
        pairings = [((ours[0], theirs[0]), (ours[1], theirs[1])), ((ours[0], theirs[1]), (ours[1], theirs[0]))]
        pairing = min(pairings, key=lambda pairs: sum(turn_between(*pair) for pair in pairs))
        if max(turn_between(*pair) for pair in pairing) > FUSE_STRAIGHT:
            continue
        lines = [pair[0]["direction"] - pair[1]["direction"] for pair in pairing]
        cosine = abs(np.dot(lines[0], lines[1])) / max(np.hypot(*lines[0]) * np.hypot(*lines[1]), 1e-9)
        sine = max(np.sqrt(max(1 - cosine**2, 0)), 0.2)
        thick = max(arm["width"] for arm in ours + theirs)
        if length > max(3, FUSE_FACTOR * thick / sine):
            continue
        # one node in the middle takes the four outer arms
        new = {"x": (first["x"] + second["x"]) / 2, "y": (first["y"] + second["y"]) / 2, "fused": True, "core": number,
               "ends": [end for end in first["ends"] + second["ends"] if end[0] != number]}
        first["gone"] = second["gone"] = True
        edges[number]["core"] = True
        for edge, end in new["ends"]:
            ends = list(edges[edge]["nodes"])
            ends[end] = len(nodes)
            edges[edge]["nodes"] = tuple(ends)
        nodes.append(read_arms(new, edges, readings, density))
        fused += 1
    return fused


def edge_reading(edge, reading, pairs, fps, visible=None):
    """Cut an edge's temporal signals out of those of the chain it is a piece of.

    Its line mass is the chain's kymograph between the edge's two nodes; pulse and focus are the chain's. An edge
    is often too short for a velocity of its own, so its velocity is read from the chain's kymograph around it,
    widened to VELOCITY_LENGTH points: the flow of the vessel as it passes the edge.
    """
    first, last = edge["span"]
    points = len(reading["M"])
    cut = {name: reading[name] for name in ("times", "radius", "pulse_amplitude", "pulse_phase", "blur_trace",
                                            "width_trace", "sharpest", "lucky")}
    cut["M"] = reading["M"][first:last + 1]
    level = mean_of_data(mean_of_data(cut["M"], axis=0), axis=0)
    cut.update({"mass": float(level) if np.isfinite(level) else reading["mass"], "velocity": np.nan,
                "velocity_confidence": 0.0, "direction": 0})
    if points >= VELOCITY_LENGTH and (visible is None or visible[width_class(reading["radius"])]):
        wider = max(0, VELOCITY_LENGTH - (last - first + 1) + 1) // 2
        low = max(0, min(first - wider, points - VELOCITY_LENGTH))
        high = min(points, max(last + 1 + wider, VELOCITY_LENGTH))
        lucky = reading["lucky"] if (reading["lucky"] > 0).sum() >= 0.3 * len(reading["lucky"]) else None
        travel = travel_reading(*travelling_correlation(reading["M"][low:high], pairs, weights=lucky))
        if np.isfinite(travel["shift"]):
            cut["velocity"], cut["velocity_confidence"] = travel["shift"] * fps, travel["z"]
            cut["direction"] = int(np.sign(travel["shift"])) if (
                travel["z"] >= VELOCITY_CLEAR and abs(travel["shift"]) >= 0.5) else 0
    return cut


def vessel_graph(chains, valid, density, readings=None, pairs=None, fps=None, visible=None):
    """Section 26, from chains to a graph with its junction arms read: nodes, edges, the edges' temporal signals
    (cut from those of the chains, as section 25 reads them), every junction's arms on a ring outside its core, and
    the split crossings fused. `readings` are the chains' signals from segment_readings, or None for a graph
    without temporal signals."""
    attached, meetings, _ = junction_candidates(chains, valid)
    nodes, edges = graph_from_chains(chains, attached, meetings)
    nodes = contract_cores(nodes, edges)
    cut = None
    if readings is not None:
        cut = [edge_reading(edge, readings[edge["chain"]], pairs, fps, visible) for edge in edges]
    for node in nodes:
        if len(node["ends"]) >= 3:  # a junction; chain ends and nodes that a chain merely passes are not read
            read_arms(node, edges, cut, density)
    fused = fuse_split_crossings(nodes, edges, cut, density)
    if cut is not None:
        for reading in cut:
            del reading["M"]  # every arm has its own line mass over time; the kymographs of all edges are not kept
    return {"nodes": nodes, "edges": edges, "readings": cut, "fused": fused}


def field_agreement(static_scores, flow_scores, valid):
    """Compare a static and a flow orientation score pixel by pixel, orientation by orientation. Returns three
    pictures: where a neuron of both fields answers above T_LOW (same place, same orientation), where only a
    static one does, and where only a flow one does."""
    both, static_any, flow_any = (np.zeros(valid.shape, dtype=bool) for _ in range(3))
    for k in range(N_ORIENT):
        static = static_scores[k] >= T_LOW
        flow = at_size(flow_scores[k], valid.shape) >= T_LOW
        both |= static & flow
        static_any |= static
        flow_any |= flow
    return both & valid, static_any & ~both & valid, flow_any & ~both & valid


def static_lines_near(static_field, static_radius, k):
    """Where the static field already has a line of about orientation k: within a vessel's wall distance (and at
    least STATIC_NEAR pixels) of a static neuron of orientation k - 1, k or k + 1 that answers above T_LOW."""
    near = np.zeros(static_field.shape[1:], dtype=bool)
    answers = np.maximum.reduce([static_field[(k + turn) % N_ORIENT] for turn in (-1, 0, 1)]) >= T_LOW
    widths = np.maximum.reduce([static_radius[(k + turn) % N_ORIENT] for turn in (-1, 0, 1)])
    for low, high in zip((0,) + HYPER_WIDTHS[1::2], HYPER_WIDTHS[1::2] + (np.inf,)):
        lines = answers & (widths >= low) & (widths < high)
        if lines.any():
            reach = int(max(STATIC_NEAR, np.ceil(min(high, HYPER_WIDTHS[-1]) / CYLINDER)))
            near |= cv2.dilate(lines.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8)) > 0
    return near


def combined_field(static_field, static_radius, static_scores, flow):
    """One field from two: at every place and orientation the larger of the static and the flow field (both after
    their interactions, both in noise units), with the half-width of whichever is larger, and the larger of the
    two orientation scores as the evidence for bridges. Where the static field already has a line of that
    orientation the flow field adds nothing: its crest of the same vessel lies a pixel or two beside the static
    one, and would be strung into a second chain alongside the first.

    `flow` holds the flow field's `field`, `radius` and `scores` on the reduced grid (see flow_field)."""
    shape = static_field.shape[1:]
    field = np.empty((N_ORIENT, *shape), dtype=np.float32)
    radius = np.empty((N_ORIENT, *shape), dtype=np.float16)
    scores = np.empty((N_ORIENT, *shape), dtype=np.float16)
    for k in range(N_ORIENT):
        static, moving = static_field[k].astype(np.float32), at_size(flow["field"][k], shape)
        moving[static_lines_near(static_field, static_radius, k)] = 0
        field[k] = np.maximum(static, moving)
        radius[k] = np.where(static >= moving, static_radius[k], at_size(flow["radius"][k], shape, cv2.INTER_NEAREST) * FLOW_REDUCTION)
        scores[k] = np.maximum(static_scores[k], at_size(flow["scores"][k], shape))
    return field, radius, scores

# %%
CHAIN_NEAR = 2   # pixels; a static chain and a flow chain that pass this close are the same line


agreements = []     # per fixation: (both, static only, flow only), boolean pictures
strung_vessels = []  # per fixation: the combined field's chains before pruning (see strung_field)
vessel_chains = []  # per fixation: the combined field's chains at the pruning chosen in section 22
vessel_graphs = []  # per fixation: dict(nodes, edges, readings, fused), see vessel_graph
for number, fixation in enumerate(fixations):
    agreements.append(field_agreement(orientation_scores[number], flow_energy[number], valid_maps[number]))
    both, static_only, flow_only = agreements[number]
    field, radius, scores = combined_field(static_fields[number], radius_maps[number], orientation_scores[number],
                                           {"field": flow_fields[number], "radius": flow_radius[number],
                                            "scores": flow_energy[number]})
    crests = crest_neurons(field, radius, valid_maps[number], scores=scores)
    strung = strung_field(crests, scores, valid_maps[number])
    del field, radius, scores
    keep = pruned(strung, *CHAIN_PRUNING)
    chains = [chain for chain, kept in zip(strung["chains"], keep) if kept]
    readings, _ = segment_readings(chains, lambda position: temporal_frame(number, position), fixation["times"],
                                   frame_pairs[number], fps, visible=visible_classes, threads=workers)
    graph = vessel_graph(chains, valid_maps[number], background_fits[number]["OD"], readings, frame_pairs[number], fps,
                         visible=visible_classes)
    del readings
    strung_vessels.append(strung)
    vessel_chains.append(chains)
    vessel_graphs.append(graph)
    area = valid_maps[number].sum()
    degrees = np.array([len(node.get("arms", node["ends"])) for node in graph["nodes"] if not node.get("gone")])
    lengths = [sum(chain_length(chain) for chain in some) for some in (static_chains[number], flow_chains[number], chains)]
    print(f"Fixation {number}: neurons of both fields answer on {both.sum() / area * 100:.0f}% of the field (a vessel with "
          f"flow), only static ones on {static_only.sum() / area * 100:.0f}% (no flow seen), only flow ones on "
          f"{flow_only.sum() / area * 100:.0f}% (faint or blurred vessels the static field missed)")
    print(f"   chains: static {lengths[0]:.0f} px, flow {lengths[1]:.0f} px, combined field {lengths[2]:.0f} px in "
          f"{len(chains)} chains | graph: {len(graph['edges'])} edges, {(degrees == 1).sum()} ends, "
          f"{(degrees == 3).sum()} nodes with 3 arms, {(degrees >= 4).sum()} with 4 or more; "
          f"{graph['fused']} crossings were fused from two 3-arm nodes")

# show the anchor's fixation: the agreement of the two fields with all chains, and the graph
both, static_only, flow_only = agreements[anchor]
under = np.uint8(np.clip((np.nan_to_num(background_fits[anchor]["OD"]) - DENSITY_RANGE[0])
                        / (DENSITY_RANGE[1] - DENSITY_RANGE[0]), 0, 1) * 60)
agreement_picture = np.dstack([under, under, under])
for where, color in ((both, (150, 150, 150)), (static_only, (40, 80, 170)), (flow_only, (170, 90, 30))):
    agreement_picture[where] = color
static_lines = draw_chains(static_chains[anchor], frame_shape) >= 0
flow_lines = draw_chains(flow_chains[anchor], frame_shape) >= 0
near_flow = cv2.dilate(flow_lines.astype(np.uint8), np.ones((2 * CHAIN_NEAR + 1,) * 2, np.uint8)) > 0
agreement_picture[flow_lines] = (255, 200, 0)
agreement_picture[static_lines] = (0, 220, 255)
agreement_picture[static_lines & near_flow] = (255, 255, 255)
shared = (static_lines & near_flow).sum() / max(static_lines.sum(), 1)
print(f"Fixation {anchor}: {shared * 100:.0f}% of the static chains' length has a flow chain within {CHAIN_NEAR} px "
      f"(the intersection, white in the picture)")

graph = vessel_graphs[anchor]
graph_picture = np.dstack([under, under, under]) * 2
for edge in graph["edges"]:
    points = np.rint(np.column_stack([edge["x"], edge["y"]])).astype(np.int32)[None]
    cv2.polylines(graph_picture, points, False, (255, 120, 0) if edge.get("core") else (200, 200, 200), 1)
for node in graph["nodes"]:
    arms = len(node.get("arms", []))
    if node.get("gone") or arms < 3:
        continue
    color = (255, 0, 255) if node.get("fused") else (255, 220, 0) if arms == 3 else (0, 255, 120)
    cv2.circle(graph_picture, (int(round(node["x"])), int(round(node["y"]))), 4, color, 1)
os.makedirs("vessel_pictures/chains", exist_ok=True)
Image.fromarray(agreement_picture).save("vessel_pictures/chains/agreement.png")
Image.fromarray(graph_picture).save("vessel_pictures/chains/vessel_graph.png")
figure, axes = plt.subplots(1, 2, figsize=(16, 5.3))
axes[0].imshow(agreement_picture, interpolation="antialiased")
axes[0].set_title("gray: both fields, blue: static only, brown: flow only | chains: static cyan, flow yellow, both white",
                  fontsize=9)
axes[1].imshow(graph_picture, interpolation="antialiased")
axes[1].set_title("the graph | nodes: 3 arms yellow, 4 or more green, fused crossings magenta (their core orange)",
                  fontsize=9)
for axis in axes:
    axis.axis("off")
figure.tight_layout()
plt.show()
del agreement_picture, graph_picture, static_lines, flow_lines, near_flow
# %% [markdown]
# ## 27. Reading the junctions
#
# **The baseline: fixed rules on the geometry** (`geometric_reading`).
#
# - *3 arms*: a bifurcation. Unless it is a T, two arms straight through and the stem clearly thinner, whose stem is seen again on the far side of the through vessel: then the stem's vessel passes under or over, a crossing.
# - *4 arms*: a crossing when the arms pair up straight across and that pairing beats the next best by `PAIR_MARGIN`.
# - *more*: unresolved.
# - The junction is moved to where its arms' axes meet (`where_axes_meet`).
#
# **Hypotheses are partitions of the arms into connected groups** (`set_partitions`; after Estrada and others, 2015). A group of two arms is one vessel passing through; a group of three is a bifurcation. Every group has at least two arms. With measured flow directions every group also needs blood coming in and going out.
#
# **Every partition is scored** (`partition_reading`) as a sum of `-z^2 / 2`, where each `z` is how many sigmas a test is off:
#
# | Test | For | What should hold |
# |---|---|---|
# | turn | a pair | the two arms run straight through |
# | width, density | a pair | both arms are as wide and as dark |
# | Murray | a triple | radius of the parent cubed equals the sum of the daughters' cubed |
# | branch | a triple | a daughter leaves near the parent's line |
# | flux | pair or triple | what flows in equals what flows out (velocity times line mass) |
# | pulse | pair or triple | the arms pulse in phase |
# | blur | pair or triple | the arms sharpen and blur together (they share a depth) |
# | fingerprint | pair or triple | the arms' line masses move together, within a lag of 2 frames |
# | overlap | the junction | where two vessels lie over each other their densities add; within one vessel they do not |
#
# **The rule for choosing** (`read_junction`):
#
# - The most specific partition that fits is taken: the one with the most groups among those in which no test is off by more than `Z_PASS` sigmas. One group holding all arms can never fail, so it only wins when nothing finer fits.
# - Two partitions equally specific whose scores differ by less than `TIE` are a tie: the junction is marked ambiguous.
#
# **Flow is evidence, not a condition.** Every junction is first read from the tests of its geometry alone (turn, width, density, Murray, branch, overlap). The tests that need the burst (in and out, flux, pulse, blur, fingerprint) then do one of four things:
#
# - *no flow data*: the geometric reading stands, with its own confidence;
# - *confirms*: the geometric reading passes them too, and its confidence rises by `FLOW_TRUST` times how well they fit, of what was missing to certainty;
# - *contradicts*: it fails them and nothing else fits; the reading is kept and its confidence falls;
# - *changes*: it fails them and another partition fits geometry and flow together; that one is taken.
#
# The tests that use the frames are made on the even frames, and a reading that passes them has to pass again on the odd frames. A fixation needs `SPLIT_FRAMES` frames for that. Every junction keeps both readings: `type` and `confidence` with flow, `type_geometry` and `confidence_geometry` without.
#
# **Network consistency** (`propagate_directions`). Where all arms of a chosen group but one have a measured direction and they all point the same way, the last arm has to point the other way. An edge that gets its direction like this passes it on to the junction at its other end. The junctions are then read again.
#
# **Not as in the plan:** the plan searches small clusters of edges without a direction by trying both directions of each. Here only the one rule above is applied, repeatedly. The flux and pulse tests use velocities and phases measured on all frames, not on the even ones alone.
#
# **Result:**
#
# - `junctions[n]`: one dictionary per junction with `type`, `partition`, `confidence`, `ambiguous`, the `baseline`'s type, and the `tests`.
# - `vessel_pictures/junctions/junctions_fixation_<n>.json`: the same with every arm's angle, width, density, group and flow. (The plan asks for the shape of `vascnet`'s `Result.annotations()`, which is not available here; this is a plain list of junctions.)
# - `vessel_pictures/junctions/gallery.png`: the most and the least confident junctions of the anchor's fixation, for looking through by eye.

# %%
MOST_ARMS = 6            # junctions with more arms than this are left unresolved
STRAIGHT = 30.0          # degrees; two arms within this of a straight line can be one vessel (baseline rule)
PAIR_MARGIN = 15.0       # degrees; the best pairing of four arms has to beat the next by this (baseline rule)
THIN_STEM = 0.8          # a T's stem counts as thinner when its width is under this share of the through vessel's
FAR_SIDE = 8             # pixels; how far beyond the through vessel's far wall a T's stem is looked for
SIGMA_TURN = 15.0        # degrees; how far a vessel that passes through a junction turns, typically
SIGMA_BRANCH = 40.0      # degrees; how far a daughter leaves its parent's line, typically
SIGMA_WIDTH = 0.25       # of log width; the two arms of one vessel
SIGMA_MURRAY = 0.5       # of log(r_parent^3 / sum of r_daughter^3); Murray's law at a bifurcation
SIGMA_DENSITY = 0.35     # of log density; the two arms of one vessel, and the overlap of two vessels
SIGMA_PHASE = 0.6        # radians; pulse phase along one vessel
SIGMA_FLUX = 0.45        # of the flux through the junction; the spread measured at this burst's bifurcations (printed below)
SIGMA_TOGETHER = 0.5     # of 1 - correlation; arms of one vessel blur together, and their line mass moves together
Z_PASS = 3.0             # a hypothesis fits when none of its tests is off by more than this many sigmas
TIE = 1.0                # log units; two hypotheses whose scores differ by less are a tie
FLOW_TRUST = 0.8         # flow tests that agree perfectly close this share of what a reading lacks to certainty
FLOW_TESTS = ("in and out", "flux", "pulse", "blur", "fingerprint")  # the tests that need the burst, not the still
GALLERY_WINDOW = 40  # pixels to either side of a junction in its gallery picture
GALLERY_ZOOM = 4
SPLIT_FRAMES = 24    # a fixation needs this many frames for choosing on one half of them and confirming on the other
GROUP_COLORS = ((255, 220, 0), (0, 220, 255), (255, 90, 200), (120, 255, 120))


def set_partitions(items, least=2):
    """All ways of dividing `items` into groups of at least `least` members."""
    if not items:
        return [[]]
    first, rest = items[0], items[1:]
    ways = []
    for size in range(least - 1, len(rest) + 1):
        for others in itertools.combinations(rest, size):
            left = [item for item in rest if item not in others]
            if 0 < len(left) < least:
                continue
            ways += [[(first,) + others] + way for way in set_partitions(left, least)]
    return ways


def moving_together(node, frames):
    """Measure, for every two arms of a junction, how much they do together over some frames.

    Returns two matrices (arms x arms): the correlation of the arms' blur traces (arms of one vessel lie at one
    depth, so they sharpen and blur together), and the largest correlation of their line masses' changes from
    frame to frame at lags of up to 2 frames (what passes through one arm of a vessel passes through the other).
    NaN where a trace does not vary.
    """
    def standard(traces):
        traces = np.nan_to_num(traces - mean_of_data(traces, axis=1)[:, None])
        size = np.sqrt((traces**2).sum(axis=1, keepdims=True))
        return np.where(size > 1e-9, traces / np.maximum(size, 1e-9), np.nan)

    arms = node["arms"]
    blur = standard(np.array([arm["blur_trace"][frames] for arm in arms]))
    change = standard(np.diff(np.array([arm["mass_trace"][frames] for arm in arms]), axis=1))
    count = change.shape[1]
    lagged = [change[:, lag:] @ change[:, :count - lag].T for lag in range(3) if count - lag > 3]
    together = np.maximum.reduce(lagged + [matrix.T for matrix in lagged]) if lagged else np.full((len(arms),) * 2, np.nan)
    return blur @ blur.T, together


def flux(arm):
    """What an arm carries away from the junction: velocity along it times its line mass (NaN without a clear velocity)."""
    return arm["velocity_out"] * arm["mass"] if arm.get("direction_out", 0) != 0 else np.nan


def group_tests(node, group, together):
    """The tests of one group of arms that is said to be one connected vessel: each a name and how many sigmas the
    group is off (0 fits perfectly). A group of two is a vessel passing through, a group of three a bifurcation;
    larger groups are not tested (they can never fail). `together` are the matrices of moving_together, or None."""
    arms = [node["arms"][arm] for arm in group]
    tests = {}
    pairs = list(itertools.combinations(group, 2))
    if len(arms) == 2:
        a, b = arms
        tests["turn"] = turn_between(a, b) / SIGMA_TURN
        tests["width"] = np.log(a["width"] / b["width"]) / SIGMA_WIDTH
        if a["density"] > 0 and b["density"] > 0:
            tests["density"] = np.log(a["density"] / b["density"]) / SIGMA_DENSITY
    elif len(arms) == 3:
        parent = max(range(3), key=lambda number: arms[number]["width"])
        daughters = [arms[number] for number in range(3) if number != parent]
        tests["murray"] = np.log(arms[parent]["radius"] ** 3 / sum(arm["radius"] ** 3 for arm in daughters)) / SIGMA_MURRAY
        tests["branch"] = min(turn_between(arms[parent], daughter) for daughter in daughters) / SIGMA_BRANCH
    else:
        return tests
    if together is not None:
        flows = np.array([flux(arm) for arm in arms])
        if np.isfinite(flows).all():
            tests["flux"] = flows.sum() / (SIGMA_FLUX * np.abs(flows).sum() + 1e-9)
        phases = [arm["pulse_phase"] for arm in arms]
        if np.isfinite(phases).all():
            tests["pulse"] = max(abs(np.angle(np.exp(1j * (arms[i]["pulse_phase"] - arms[j]["pulse_phase"]))))
                                 for i, j in itertools.combinations(range(len(arms)), 2)) / SIGMA_PHASE
        for name, matrix in zip(("blur", "fingerprint"), together):
            values = [matrix[i, j] for i, j in pairs]
            if np.isfinite(values).all():
                tests[name] = (1 - min(values)) / SIGMA_TOGETHER / 2  # correlations run from -1 to 1
    return tests


def partition_reading(node, groups, centre_density, together, flow=True):
    """Score one hypothesis: a partition of a node's arms into connected groups. Returns the score (the sum of
    -z^2 / 2 over all tests: 0 is a perfect fit), whether it fits (no test off by more than Z_PASS), and the tests.
    Without `flow` only the tests of the geometry are made (`together` is then None as well)."""
    tests = {}
    for number, group in enumerate(groups):
        arms = [node["arms"][arm] for arm in group]
        if flow and "direction_out" in arms[0]:
            known = [arm["direction_out"] for arm in arms if arm["direction_out"] != 0]
            if len(known) == len(arms) and len(set(known)) == 1:
                tests[f"{number}:in and out"] = np.inf  # blood has to come in and go out of every vessel
        tests.update({f"{number}:{name}": value for name, value in group_tests(node, group, together).items()})
    # at the junction itself: where two vessels lie over each other their densities add, within one vessel they do not
    densities = [np.mean([max(node["arms"][arm]["density"], 1e-6) for arm in group]) for group in groups]
    if centre_density > 0 and all(len(group) <= 3 for group in groups):
        expected = sum(densities) if len(groups) > 1 else max(max(node["arms"][arm]["density"], 1e-6) for arm in groups[0])
        tests["overlap"] = np.log(centre_density / expected) / SIGMA_DENSITY
    values = np.array(list(tests.values()), dtype=float)
    return float(-0.5 * np.sum(np.minimum(values, 1e3) ** 2)), bool((np.abs(values) <= Z_PASS).all()), tests


def group_name(group):
    return {2: "through", 3: "bifurcation"}.get(len(group), "unresolved")


def partition_name(groups):
    """What a partition of a junction's arms is called."""
    names = sorted(group_name(group) for group in groups)
    if names == ["through", "through"]:
        return "crossing"
    return " + ".join(names)


def geometric_reading(node, scores):
    """The baseline: read a junction from its geometry alone, by fixed rules.

      3 arms  a bifurcation, unless it is a T (two arms straight through, the stem clearly thinner) whose stem is
              seen again on the far side of the through vessel: then the stem's vessel passes under or over (crossing)
      4 arms  a crossing when the arms pair up straight across and that pairing beats the next best by PAIR_MARGIN
      more    unresolved
    Returns the type, and the measurements the rules used.
    """
    arms = node["arms"]
    measured = {}
    if len(arms) == 3:
        pairs = sorted(itertools.combinations(range(3), 2), key=lambda pair: turn_between(arms[pair[0]], arms[pair[1]]))
        through, stem = pairs[0], arms[3 - sum(pairs[0])]
        parent = max(arms, key=lambda arm: arm["width"])
        measured["turn"] = turn_between(arms[through[0]], arms[through[1]])
        measured["murray"] = float(parent["radius"] ** 3 / sum(arm["radius"] ** 3 for arm in arms if arm is not parent))
        kind = "bifurcation"
        if measured["turn"] <= STRAIGHT and stem["width"] <= THIN_STEM * min(arms[through[0]]["width"], arms[through[1]]["width"]):
            # amodal completion: is the stem's line there again beyond the far wall of the through vessel?
            start = node["thick"] / 2 + RING_MARGIN
            x0, y0 = node["x"] - stem["direction"][0] * start, node["y"] - stem["direction"][1] * start
            x1, y1 = x0 - stem["direction"][0] * FAR_SIDE, y0 - stem["direction"][1] * FAR_SIDE
            rows, columns = scores.shape[1:]
            if 1 <= x1 < columns - 1 and 1 <= y1 < rows - 1 and 1 <= x0 < columns - 1 and 1 <= y0 < rows - 1:
                measured["far side"] = float(evidence_along(scores, np.array([x0]), np.array([y0]), np.array([x1]), np.array([y1]))[0])
                if measured["far side"] >= T_LOW:
                    kind = "crossing"
        return kind, measured
    if len(arms) == 4:
        pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        turns = sorted((max(turn_between(arms[a], arms[b]) for a, b in pairing), pairing) for pairing in pairings)
        measured["turn"], measured["next turn"] = turns[0][0], turns[1][0]
        if turns[0][0] <= STRAIGHT and turns[1][0] - turns[0][0] >= PAIR_MARGIN:
            return "crossing", measured
    return "unresolved", measured


def where_axes_meet(node):
    """Relocate a junction to where its arms' axes meet: the point with the least squared distance to the lines
    through each arm's ring part along its direction. It is not moved farther than the junction's core."""
    lhs, rhs = np.zeros((2, 2)), np.zeros(2)
    for arm in node["arms"]:
        across = np.eye(2) - np.outer(arm["direction"], arm["direction"])
        lhs += across
        rhs += across @ np.array([arm["x"], arm["y"]])
    if abs(np.linalg.det(lhs)) < 1e-6:
        return node["x"], node["y"]
    x, y = np.linalg.solve(lhs, rhs)
    far = np.hypot(x - node["x"], y - node["y"])
    limit = node["ring"][0]
    if far > limit:
        x, y = node["x"] + (x - node["x"]) * limit / far, node["y"] + (y - node["y"]) * limit / far
    return float(x), float(y)


def choose_partition(node, ways, centre, together, flow):
    """Score every partition of a junction's arms and choose: the most specific one that fits (more groups first,
    then the better score); the one group of all arms when nothing finer fits. Returns the choice, its margin over
    the next best equally specific partition that fits (infinite when there is none), how many fit, and the
    second-best reading: the next one in the same order, the fitting ones first (None when there is no other)."""
    read = []
    for groups in ways:
        score, fits, tests = partition_reading(node, groups, centre, together, flow)
        read.append({"groups": groups, "score": score, "fits": fits, "tests": tests})
    fitting = sorted((way for way in read if way["fits"]), key=lambda way: (-len(way["groups"]), -way["score"]))
    chosen = fitting[0] if fitting else next(way for way in read if len(way["groups"]) == 1)
    rivals = [way for way in fitting[1:] if len(way["groups"]) == len(chosen["groups"])]
    others = sorted((way for way in read if way is not chosen),
                    key=lambda way: (not way["fits"], -len(way["groups"]), -way["score"]))
    return (chosen, (chosen["score"] - rivals[0]["score"] if rivals else np.inf), len(fitting),
            others[0] if others else None)


def measured_out(arm, flagged=False):
    """An arm's measured direction of flow: +1 out of the junction, -1 into it, 0 when it has none. A direction
    that was passed on from another junction (propagate_directions) is not a measurement. With `flagged`, a
    direction read from a speed above the trackable limit is given too."""
    out = 0 if arm.get("inferred") else int(arm.get("direction_out", 0))
    return int(arm.get("flagged_out", 0)) if flagged and out == 0 else out


def direction_likelihood(out, groups, p):
    """The probability of the directions measured at a junction's arms (`out`: +1, -1 or 0 per arm) if a reading
    (`groups`, a partition of the arms) is true. Within every group the blood has to come in through at least one
    arm and leave through at least one; every pattern that does is taken to be as likely as any other; a measured
    direction is the true one with probability p; the groups are independent of each other."""
    likelihood = 1.0
    for group in groups:
        total = 0.0
        for true in itertools.product((-1, 1), repeat=len(group)):
            if abs(sum(true)) < len(group):  # not all in, not all out
                total += float(np.prod([p if out[arm] == way else 1 - p for arm, way in zip(group, true) if out[arm] != 0]))
        likelihood *= total / (2 ** len(group) - 2)
    return likelihood


def reading_confidence(chosen, margin):
    """How sure a reading is, from 0 to 1: how well its tests fit, times how clearly it beats its rival."""
    return float(np.exp(chosen["score"] / max(len(chosen["tests"]), 1)) * (1 - np.exp(-min(margin, 50))))


def read_junction(node, density, scores, frames=None, flow=True, evidence="tests", p=None):
    """Section 27 for one junction: the baseline's reading, then every partition of the arms into connected groups,
    scored; the most specific partition that fits is chosen.

    The junction is first read from its geometry alone. Flow is evidence on top of that, never a condition:
      no flow data   the geometric reading stands, with its own confidence
      confirms       the geometric reading passes the tests that need the burst (FLOW_TESTS) as well: its confidence
                     rises by FLOW_TRUST times how well those tests fit, of what was missing to certainty
      changes        it fails them, and another partition fits geometry and flow together: that one is taken
      contradicts    it fails them and nothing else fits either: the geometric reading is kept, its confidence
                     falls by FLOW_TRUST times how badly the flow tests fit

    With `frames` (two sets of frame numbers) the tests that use the frames are made on the first set, and a
    reading that passes them has to pass again on the second set. Without `flow` only the geometric reading is made.

    That is `evidence="tests"`. With `evidence="directions"` flow is weighed as a likelihood ratio instead, and only
    the measured directions of flow count. The ratio is between the best and the second-best geometric reading: how
    much more likely the directions measured are under the one than under the other (direction_likelihood), given
    that a measured direction is right with probability `p`. The geometric confidence is taken as the probability
    of the best reading against the second-best and updated by the ratio:

      fits both        the directions are as likely under both readings: nothing changes
      confirms         they are likelier under the best reading: its confidence rises
      speaks against   they are likelier under the second-best: the confidence falls
      changes          it falls under one half and the second-best is as specific a reading and fits the geometry:
                       that one is taken
      no other reading there is no second-best reading (three arms can only be one bifurcation): nothing changes
    """
    arms = node["arms"]
    x, y = where_axes_meet(node)
    rows, columns = density.shape
    row, column = int(np.clip(round(y), 2, rows - 3)), int(np.clip(round(x), 2, columns - 3))
    centre = float(np.nanmax(density[row - 2:row + 3, column - 2:column + 3]))
    baseline, measured = geometric_reading(node, scores)
    ways = set_partitions(list(range(len(arms)))) if len(arms) <= MOST_ARMS else [[tuple(range(len(arms)))]]

    def named(reading, fitting):
        """What a chosen partition is called; unresolved when nothing fits."""
        resolved = (fitting or len(arms) == 3) and not (len(arms) == 3 and not reading["fits"])
        return partition_name(reading["groups"]) if resolved else "unresolved"

    plain, plain_margin, plain_fitting, second = choose_partition(node, ways, centre, None, False)
    plain_kind = named(plain, plain_fitting)
    geometric = reading_confidence(plain, plain_margin)
    chosen, margin, kind, confidence = plain, plain_margin, plain_kind, geometric
    flow_says, support, confirmed = "no flow data", np.nan, True
    out = [measured_out(arm) for arm in arms]
    # the directions measured cannot be under the geometric reading: a group with all its arms in, or all out
    impossible = any(all(out[arm] != 0 for arm in group) and abs(sum(out[arm] for arm in group)) == len(group)
                     for group in plain["groups"])
    if flow and evidence == "directions":
        if sum(way != 0 for way in out) >= 2 and p is not None and np.isfinite(p):
            flow_says, support = "no other reading", 1.0
            if second is not None:
                support = direction_likelihood(out, plain["groups"], p) / direction_likelihood(out, second["groups"], p)
                prior = float(np.clip(geometric, 1e-6, 1 - 1e-6))
                confidence = prior * support / (prior * support + 1 - prior)
                flow_says = "fits both" if abs(support - 1) < 1e-9 else "confirms" if support > 1 else "speaks against"
                if confidence < 0.5 and second["fits"] and len(second["groups"]) >= len(plain["groups"]):
                    flow_says, chosen, kind, confidence = "changes", second, named(second, True), 1 - confidence
                    margin = np.inf
    elif flow and "blur_trace" in arms[0]:
        choose, confirm = frames if frames is not None else (slice(None), None)
        together = moving_together(node, choose)
        _, agrees, tests = partition_reading(node, plain["groups"], centre, together, True)
        with_flow = [min(abs(value), 1e3) for name, value in tests.items() if name.split(":")[-1] in FLOW_TESTS]
        if with_flow:
            support = float(np.exp(np.mean(-0.5 * np.square(with_flow))))
            if agrees and confirm is not None:
                agrees = confirmed = partition_reading(node, plain["groups"], centre, moving_together(node, confirm), True)[1]
            other, other_margin, other_fitting = (None, 0.0, 0) if agrees else choose_partition(node, ways, centre, together, True)[:3]
            if agrees:
                flow_says, confidence = "confirms", 1 - (1 - geometric) * (1 - FLOW_TRUST * support)
                chosen = {**plain, "tests": tests}
            elif named(other, other_fitting) != "unresolved" and other["groups"] != plain["groups"]:
                flow_says, chosen, margin, kind = "changes", other, other_margin, named(other, other_fitting)
                confidence = reading_confidence(other, other_margin)
            else:
                flow_says, confidence = "contradicts", geometric * (1 - FLOW_TRUST * (1 - support))
                chosen = {**plain, "tests": tests}
    return {
        "x": x, "y": y, "arms": len(arms), "type": kind, "type_geometry": plain_kind,
        "partition": [list(group) for group in chosen["groups"]],
        "confidence": float(confidence), "confidence_geometry": geometric, "flow": flow_says, "flow_support": support,
        "ambiguous": bool(margin < TIE or not confirmed), "confirmed": bool(confirmed), "margin": float(margin),
        "baseline": baseline, "measured": measured, "tests": chosen["tests"], "centre density": centre,
        "hypotheses": len(ways), "fitting": plain_fitting, "directions": out, "impossible": bool(impossible),
        "second": partition_name(second["groups"]) if second is not None else "",
    }


def propagate_directions(graph):
    """Network consistency: give a direction to edges that have none, where the junctions leave only one.

    Blood has to enter and leave every connected group of arms. So where all arms of a chosen group but one have
    a measured direction and they all point the same way (all in, or all out), the last arm points the other way.
    An edge that gets its direction like this passes it on to the junction at its other end, so the rule is
    applied until nothing changes. Returns how many edges were given a direction and how many could not be
    (their two ends disagreed).
    """
    nodes, inferred, conflicts = graph["nodes"], 0, 0
    arm_at = {(arm["edge"], arm["end"]): arm for node in nodes if not node.get("gone") for arm in node.get("arms", [])}
    for _ in range(10):
        changed = False
        for node in nodes:
            for group in node.get("partition", []):
                arms = [node["arms"][arm] for arm in group]
                unknown = [arm for arm in arms if arm["direction_out"] == 0]
                known = {arm["direction_out"] for arm in arms if arm["direction_out"] != 0}
                if len(unknown) != 1 or len(known) != 1 or len(arms) > 3:
                    continue
                arm, way = unknown[0], -known.pop()
                other = arm_at.get((arm["edge"], 1 - arm["end"]))
                if other is not None and other["direction_out"] == way:  # both ends would send it the same way
                    conflicts += 1
                    continue
                arm["direction_out"], arm["inferred"] = way, True
                if other is not None:
                    other["direction_out"], other["inferred"] = -way, True
                inferred += 1
                changed = True
        if not changed:
            break
    return inferred, conflicts


def read_junctions(graph, density, scores, halves=None, flow=True, evidence="tests", p=None):
    """Read every junction of a graph: each on its own (read_junction, which `evidence` and `p` are for); then the
    directions the junctions imply are passed along the edges (propagate_directions), and the junctions are read
    again with them. Without `flow` the junctions are read from their geometry alone and no direction is passed on.

    Returns the junctions (each with `node`, its number among the graph's nodes), and how many edges have a
    measured direction, how many were given one by the junctions, and how many could not be given one.
    """
    places = [index for index, node in enumerate(graph["nodes"]) if not node.get("gone") and len(node.get("arms", [])) >= 3]

    def read_all():
        found = []
        for index in places:
            junction = read_junction(graph["nodes"][index], density, scores, halves, flow, evidence, p)
            junction["node"] = index
            graph["nodes"][index]["partition"] = junction["partition"]
            found.append(junction)
        return found

    found = read_all()
    if graph["readings"] is None or not flow:
        return found, 0, 0, 0
    measured = sum(reading["direction"] != 0 for reading in graph["readings"])
    inferred, conflicts = propagate_directions(graph)
    return (read_all() if inferred else found), measured, inferred, conflicts


def junction_tile(junction, node, density, shown):
    """A close-up of one junction for the gallery: the density around it, enlarged, with its arms drawn from the
    junction to their ring parts in the colors of the groups they were put in."""
    size = 2 * GALLERY_WINDOW + 1
    to_tile = np.float32([[GALLERY_ZOOM, 0, (GALLERY_WINDOW - junction["x"]) * GALLERY_ZOOM],
                          [0, GALLERY_ZOOM, (GALLERY_WINDOW - junction["y"]) * GALLERY_ZOOM]])
    tile = cv2.warpAffine(np.nan_to_num(density).astype(np.float32), to_tile, (size * GALLERY_ZOOM,) * 2, flags=cv2.INTER_CUBIC)
    gray = np.uint8(np.clip((tile - shown[0]) / (shown[1] - shown[0]), 0, 1) * 255)
    tile = np.dstack([gray, gray, gray])
    centre = (GALLERY_WINDOW * GALLERY_ZOOM,) * 2
    for number, group in enumerate(junction["partition"]):
        color = GROUP_COLORS[number % len(GROUP_COLORS)] if len(junction["partition"]) > 1 or len(group) <= 3 else (200, 200, 200)
        for arm in group:
            arm = node["arms"][arm]
            tip = (int((arm["x"] - junction["x"] + GALLERY_WINDOW) * GALLERY_ZOOM),
                   int((arm["y"] - junction["y"] + GALLERY_WINDOW) * GALLERY_ZOOM))
            cv2.line(tile, centre, tip, color, 2, cv2.LINE_AA)
            cv2.circle(tile, tip, 4, color, -1)
    return tile

# %%
GALLERY = 16         # junctions shown from either end of the ordering by confidence


os.makedirs("vessel_pictures/junctions", exist_ok=True)
junctions = []         # per fixation: one dict per junction (a node with 3 or more arms), see read_junction
junction_numbers = []  # per fixation: what the reading came to
for number, fixation in enumerate(fixations):
    graph, density, scores = vessel_graphs[number], background_fits[number]["OD"], orientation_scores[number]
    count = len(fixation["frames"])
    halves = (np.arange(0, count, 2), np.arange(1, count, 2)) if count >= SPLIT_FRAMES else None
    # every junction is read on its own; then the directions the junctions imply are passed along the edges, and
    # the junctions are read again with them
    found, measured, inferred, conflicts = read_junctions(graph, density, scores, halves)
    junctions.append(found)

    kinds = {}
    for junction in found:
        kinds[junction["type"]] = kinds.get(junction["type"], 0) + 1
    resolved = [junction for junction in found if junction["type"] != "unresolved" and not junction["ambiguous"]]
    ambiguous = [junction for junction in found if junction["type"] != "unresolved" and junction["ambiguous"]]
    crossings = [junction for junction in found if junction["type"] == "crossing"]
    baseline_crossings = [junction for junction in found if junction["baseline"] == "crossing"]
    # flux balance at clean Y junctions: bifurcations that fit, with a clear velocity in all three arms
    residuals = [junction["tests"]["0:flux"] * SIGMA_FLUX for junction in found
                 if junction["type"] == "bifurcation" and "0:flux" in junction["tests"]]
    junction_numbers.append({"junctions": len(found), "types": kinds, "resolved": len(resolved), "ambiguous": len(ambiguous),
                             "flux residuals": np.array(residuals), "directions": (measured, inferred, conflicts)})
    print(f"Fixation {number}: {len(found)} junctions ({sum(j['arms'] == 3 for j in found)} with 3 arms, "
          f"{sum(j['arms'] == 4 for j in found)} with 4, {sum(j['arms'] > 4 for j in found)} with more)")
    print("   read as: " + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items(), key=lambda item: -item[1])))
    print(f"   resolved {len(resolved)} ({len(resolved) / max(len(found), 1) * 100:.0f}%), ambiguous {len(ambiguous)} "
          f"({len(ambiguous) / max(len(found), 1) * 100:.0f}%), unresolved {len(found) - len(resolved) - len(ambiguous)} | "
          f"crossings: {len(crossings)} by partitions, {len(baseline_crossings)} by the baseline's rules, "
          f"{sum(j['baseline'] == 'crossing' for j in crossings)} by both")
    print(f"   directions: {measured} of {len(graph['edges'])} edges have a measured one; the junctions give one to "
          f"{inferred} more; {conflicts} could not be given one because their two ends disagree")
    if len(residuals) >= 5:
        print(f"   flux balance at the {len(residuals)} bifurcations with a clear velocity in all three arms: what flows in "
              f"minus what flows out is {np.median(residuals) * 100:+.0f}% of the flux through the junction (median), "
              f"robust sd {robust_sd(np.array(residuals)) * 100:.0f}% (SIGMA_FLUX is {SIGMA_FLUX * 100:.0f}%)")
    else:
        print(f"   flux balance: only {len(residuals)} bifurcations have a clear velocity in all three arms")
    # export: one record per junction, with its arms
    export = []
    for junction in found:
        node = graph["nodes"][junction["node"]]
        group_of = {arm: group for group, arms in enumerate(junction["partition"]) for arm in arms}
        export.append({
            "id": junction["node"], "x": round(junction["x"], 2), "y": round(junction["y"], 2), "type": junction["type"],
            "confidence": round(junction["confidence"], 4), "ambiguous": junction["ambiguous"],
            "baseline": junction["baseline"], "partition": junction["partition"],
            "arms": [{"angle": round(arm["angle"], 1), "width": round(arm["width"], 2), "density": round(arm["density"], 4),
                      "group": group_of[index], "direction_out": int(arm.get("direction_out", 0)),
                      "velocity_out": None if not np.isfinite(arm.get("velocity_out", np.nan)) else round(float(arm["velocity_out"]), 1)}
                     for index, arm in enumerate(node["arms"])],
        })
    with open(f"vessel_pictures/junctions/junctions_fixation_{number}.json", "w") as file:
        json.dump(export, file, indent=1)

# the gallery, for the anchor's fixation: the most and the least confident junctions that were given a type
graph, density = vessel_graphs[anchor], background_fits[anchor]["OD"]
typed = sorted((junction for junction in junctions[anchor] if junction["type"] != "unresolved"),
               key=lambda junction: -junction["confidence"])
shown = typed[:GALLERY] + typed[-GALLERY:] if len(typed) > 2 * GALLERY else typed
figure, axes = plt.subplots(4, 8, figsize=(16, 8.6))
for axis in axes.ravel():
    axis.axis("off")
for axis, junction in zip(axes.ravel(), shown):
    axis.imshow(junction_tile(junction, graph["nodes"][junction["node"]], density, DENSITY_RANGE))
    axis.set_title(f"{junction['type']}\nconfidence {junction['confidence']:.2f}" + (" (tie)" if junction["ambiguous"] else ""),
                   fontsize=7)
figure.suptitle("junctions of the anchor's fixation: the most confident (top two rows) and the least confident (bottom two); "
                "arms of one color were put in one group", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/junctions/gallery.png", dpi=120)
plt.show()
print(f"Saved the junctions of every fixation as vessel_pictures/junctions/junctions_fixation_<n>.json and the gallery "
      f"as gallery.png")

# %% [markdown]
# ## 28. Check on synthetic bursts, where everything is known
#
# The real burst has no truth to compare with. So bursts are made in which the truth is known, and read with the same functions, the same constants and the pruning chosen for the real burst.
#
# **What a synthetic burst holds** (`synthetic_burst`):
#
# - `SYNTH_TREES` vessel trees, each at its own depth. Each enters at the frame's edge and splits once; the daughters' radii follow Murray's law. Where two trees pass over each other is a crossing, and their densities add.
# - Clusters of red cells carried through each tree. At a split a cluster goes down one daughter, with the odds of that daughter's share of the flux, so flux is conserved.
# - A focal plane that drifts during the burst, so each tree is blurred by a different, changing amount.
# - A 12-bit camera with shot and read noise, at the real burst's frame rate, with two recorded frames dropped.
#
# - The tissue's own texture, as strong as on the real burst's quiet pixels, so that a vessel stands out from its surroundings about as much as a real one.
#
# The frames are handed over as densities against the true background. Registration and background fitting (sections 1 to 14) are not part of this check.
#
# **Read these numbers before trusting section 27.** They say how much of what the sections above find is true when the truth is known: which chains are vessels, which junctions are real, and whether the partitions read a junction's type better than the baseline's rules do.
#
# **What is compared** (`score_reading`), over `SYNTH_SEEDS` bursts. The scoring is `vascnet.evaluate`'s, so `vascnet` has to be installed (`pip install -e ./attempt5000`; the cell below says which commit).
#
# - *Centerlines* (`centerline_metrics`): the share of the chains within 2 px of a true centerline, and of the true centerlines with a chain within 2 px.
# - *Junctions* (`junction_metrics`): how many junctions read are true (precision), and how many true ones are found (recall). Before matching, true junctions that lie close together are fused into one "complex" junction (`cluster_truth`), as someone marking them by hand would.
# - *Nobody is asked for some junctions.* A true junction is left out of both counts when a vessel in it is under 3 times the noise, or when it is a crossing whose vessel stops too soon after it or runs alongside the other one (`mark_visible`, which applies vascnet's own rules).
# - *Type*: of the junctions found, how many are typed right; and of all true junctions, how many are found and typed right. Three readings are scored: the partitions with flow, the partitions from geometry alone, and the baseline's rules. A junction left unresolved counts as found and as typed wrong.
# - *Flow at the junctions found*: at every found junction that is a single true bifurcation or crossing, each arm is matched to the true limb it leaves along. Printed: how often a measured direction is the true one, and the error of the speed.
#
# Every share has a 95% interval, from resampling the bursts (`with_interval`). Section 30 runs the same scoring on more bursts and on every way of reading them.

# %%
# The scoring is vascnet's (github.com/karimghabra/attempt5000, branch claude/beautiful-ritchie-zc7xu0, commit 0e53e10;
# installed with `pip install -e ./attempt5000`). Only this section and section 30 need it.
import vascnet.evaluate
import vascnet.synth

SYNTH_SHAPE = (384, 512)    # rows, columns of a synthetic frame
SYNTH_FPS = 32.236          # frames per second, the real burst's
SYNTH_FRAMES = 64           # frames recorded
SYNTH_DROPPED = (17, 40)    # recorded frames the gate drops, to leave gaps as in a real burst
SYNTH_TREES = 3             # vessel trees, each at its own depth, each with one bifurcation
SYNTH_APART = 10            # pixels; true crossings closer than this to each other are one crossing
SYNTH_LIGHT = 2500.0        # DN; brightness of the background
SYNTH_GAIN = 1.0            # electrons per DN: shot noise of 50 DN at the background's brightness, as in the real burst
SYNTH_READ = 5.0            # DN; read noise
SYNTH_SATURATION = 4095     # DN; a 12-bit camera
SYNTH_TEXTURE = 0.005       # density; spread of the tissue's own texture, as on the real burst's quiet pixels
SYNTH_GRAIN = 2.0           # pixels; how fine that texture is
SYNTH_ABSORB = 0.025        # density per pixel of blood along the line of sight
SYNTH_SHARP = 0.8           # pixels; blur of a tree in focus
SYNTH_DEFOCUS = 3.0         # pixels of blur per unit of depth away from the focal plane
SYNTH_FOCUS = (-0.3, 0.3)   # where the focal plane is at the start and at the end of the burst
CLUSTER_SPACING = 18.0      # pixels; mean distance between clusters of red cells along a vessel
CLUSTER_LENGTH = 3.0        # pixels; half-length of a cluster
CLUSTER_DEPTH = 0.6         # how much of a vessel's density the clusters modulate
LIMB_REACH = 20             # points along a true vessel from a junction; a true limb's direction is measured over them
LIMB_WITHIN = 40.0          # degrees; an arm read at a junction is a true limb's when it leaves within this of it
BOOTSTRAP = 2000            # resamplings of the bursts for a 95% interval
VASCNET_WORDS = {"bifurcation": "bifurcation", "crossing": "crossover"}  # our junction types in vascnet's words
OUR_WORDS = {"bifurcation": "bifurcation", "crossover": "crossing", "complex": "complex"}  # and back
TRUE_TYPES = ("bifurcation", "crossover", "complex")  # what a true junction can be after close ones are fused
ABLATION_COLUMNS = ("centerline precision", "centerline recall", "junction precision", "junction recall",
                    "type right, of found", "found and typed right")


def wandering_path(start, heading, length, rng, wander=0.012):
    """A gently curving path: points one pixel apart from `start`, the heading doing a small random walk."""
    angles = heading + np.cumsum(rng.normal(0, wander, int(length)))
    return np.column_stack([start[0] + np.cumsum(np.cos(angles)), start[1] + np.cumsum(np.sin(angles))]), float(angles[-1])


def synthetic_network(rng):
    """Draw SYNTH_TREES vessel trees. Each enters at the frame's edge, runs inwards and splits once into two
    daughters whose radii follow Murray's law; flux is conserved at the split. A tree's flow runs either from its
    root to its leaves or the other way. Returns the segments: dicts with points (x, y; from the root outwards),
    radius, speed (pixels per frame along the points; negative towards the root), tree, depth, and parent."""
    rows, columns = SYNTH_SHAPE
    segments = []
    for tree in range(SYNTH_TREES):
        side = rng.integers(4)
        along = rng.uniform(0.25, 0.75)
        start, heading = (((along * columns, 0), np.pi / 2), ((along * columns, rows - 1), -np.pi / 2),
                          ((0, along * rows), 0.0), ((columns - 1, along * rows), np.pi))[side]
        heading += rng.uniform(-0.35, 0.35)
        radius, speed = rng.uniform(3.5, 6.0), rng.uniform(1.5, 3.5) * rng.choice([-1, 1])
        depth = rng.uniform(-0.6, 0.6)
        trunk, heading = wandering_path(start, heading, rng.uniform(0.3, 0.45) * np.hypot(rows, columns), rng)
        segments.append({"points": trunk, "radius": radius, "speed": speed, "tree": tree, "depth": depth, "parent": None})
        share = rng.uniform(0.35, 0.65)  # of the flux, and by Murray's law of radius cubed, that the first daughter takes
        for part, turn in ((share, rng.uniform(0.4, 0.75)), (1 - share, -rng.uniform(0.4, 0.75))):
            branch, _ = wandering_path(trunk[-1], heading + turn, 0.8 * np.hypot(rows, columns), rng)
            inside = ((branch[:, 0] >= 0) & (branch[:, 0] < columns) & (branch[:, 1] >= 0) & (branch[:, 1] < rows))
            branch = branch[:np.argmin(inside) if not inside.all() else len(branch)]
            segments.append({"points": branch, "radius": radius * part ** (1 / 3), "speed": speed * part ** (1 / 3),
                             "tree": tree, "depth": depth, "parent": len(segments) - 1 - (turn < 0)})
    return segments


def network_truth(segments):
    """Return what is true of a network: its centerlines as a picture, and its junctions. A bifurcation is where a
    trunk splits; a crossing is where segments of two trees (two depths) pass over each other. Each junction has its
    place, type, `radius` (of the trunk, or of the thicker of two crossing vessels), the `segments` that meet in it
    (a bifurcation's trunk first) and, for a crossing, the `places` along the two segments where they cross."""
    lines = np.zeros(SYNTH_SHAPE, dtype=np.uint8)
    for segment in segments:
        cv2.polylines(lines, np.rint(segment["points"]).astype(np.int32)[None], False, 1)
    junctions = [{"x": float(segment["points"][-1, 0]), "y": float(segment["points"][-1, 1]), "type": "bifurcation",
                  "radius": float(segment["radius"]),
                  "segments": [number] + [other for other, child in enumerate(segments) if child["parent"] == number]}
                 for number, segment in enumerate(segments) if segment["parent"] is None]
    for first, second in itertools.combinations(range(len(segments)), 2):
        if segments[first]["tree"] == segments[second]["tree"]:
            continue
        distance, nearest = scipy.spatial.cKDTree(segments[second]["points"]).query(segments[first]["points"])
        close = np.flatnonzero(distance < 1.0)
        for run in np.split(close, np.flatnonzero(np.diff(close) > 5) + 1):
            if len(run):
                at = run[np.argmin(distance[run])]
                x, y = segments[first]["points"][at]
                # a crossing counts once, also where a tree's segments meet and all of them cross the other tree's
                known = any(np.hypot(x - junction["x"], y - junction["y"]) < SYNTH_APART
                            for junction in junctions if junction["type"] == "crossing")
                if 8 <= x < SYNTH_SHAPE[1] - 8 and 8 <= y < SYNTH_SHAPE[0] - 8 and not known:
                    junctions.append({"x": float(x), "y": float(y), "type": "crossing",
                                      "radius": float(max(segments[first]["radius"], segments[second]["radius"])),
                                      "segments": [first, second], "places": [int(at), int(nearest[at])]})
    return lines > 0, junctions


def synthetic_burst(seed):
    """Make a synthetic burst with everything known: vessel trees at different depths, clusters of red cells carried
    through them (at a split a cluster goes down one daughter, so flux is conserved), a focal plane that drifts
    during the burst, and a 12-bit camera with shot and read noise, at the real burst's frame rate with gaps
    where frames are dropped.

    Returns a dict: the density frames as the pipeline's sections 19 to 27 take them (each frame's density against
    the true background), their times and pairs, the mean density, the data area, the quiet pixels, and the truth
    (segments, centerlines, junctions; each junction flagged if nobody can be asked to find it, see mark_visible).
    """
    rng = np.random.default_rng(seed)
    rows, columns = SYNTH_SHAPE
    segments = synthetic_network(rng)
    lines, junctions = network_truth(segments)
    # where each segment's blood lies: its pixels, their place along it and the length of the line of sight through it
    y, x = np.mgrid[:rows, :columns]
    pixels = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    for segment in segments:
        distance, along = scipy.spatial.cKDTree(segment["points"]).query(pixels, distance_upper_bound=segment["radius"] + 1)
        inside = np.flatnonzero(np.isfinite(distance))
        segment["pixels"], segment["along"] = inside, along[inside]
        segment["chord"] = 2 * np.sqrt(np.maximum(segment["radius"] ** 2 - distance[inside] ** 2, 0)) * SYNTH_ABSORB
    # clusters of red cells: (segment, place along it, size), spread over the network to begin with
    clusters = [[number, place, rng.lognormal(0, 0.4)] for number, segment in enumerate(segments)
                for place in rng.uniform(0, len(segment["points"]), rng.poisson(len(segment["points"]) / CLUSTER_SPACING))]
    children = {number: [other for other, child in enumerate(segments) if child["parent"] == number]
                for number in range(len(segments))}
    illumination = SYNTH_LIGHT * (1 + 0.15 * np.cos(2 * np.pi * x / columns / 1.7) * np.cos(2 * np.pi * y / rows / 2.3))
    # the tissue's own texture: it stays in the mean frame, where it is the noise the vessels have to stand out from
    texture = cv2.GaussianBlur(rng.normal(0, 1, (rows, columns)).astype(np.float32), (0, 0), SYNTH_GRAIN)
    texture *= SYNTH_TEXTURE / texture.std()
    kept = [frame for frame in range(SYNTH_FRAMES) if frame not in SYNTH_DROPPED]
    frames = np.zeros((len(kept), rows, columns), dtype=np.float32)
    layers = np.zeros((SYNTH_TREES, rows, columns), dtype=np.float32)  # each tree's density without noise, mean over the frames
    for frame in range(SYNTH_FRAMES):
        # move the clusters; at a split one daughter is taken with the odds of its share of the flux
        moved = []
        for number, place, size in clusters:
            segment = segments[number]
            place += segment["speed"]
            if place >= len(segment["points"]) and children[number]:
                odds = np.array([abs(segments[child]["speed"]) * segments[child]["radius"] ** 2 for child in children[number]])
                number = children[number][rng.choice(len(odds), p=odds / odds.sum())]
                place = 0.0
            elif place < 0 and segment["parent"] is not None:
                number, place = segment["parent"], len(segments[segment["parent"]]["points"]) - 1.0
            if 0 <= place < len(segments[number]["points"]):
                moved.append([number, place, size])
        clusters = moved
        # new clusters come in where the blood comes in: at the root, or at the leaves
        for number, segment in enumerate(segments):
            enters = (segment["parent"] is None) if segment["speed"] > 0 else (not children[number])
            if enters and rng.random() < abs(segment["speed"]) / CLUSTER_SPACING:
                clusters.append([number, 0.0 if segment["speed"] > 0 else len(segment["points"]) - 1.0, rng.lognormal(0, 0.4)])
        if frame not in kept:
            continue
        focus = SYNTH_FOCUS[0] + (SYNTH_FOCUS[1] - SYNTH_FOCUS[0]) * frame / (SYNTH_FRAMES - 1)
        density = texture.copy()
        for tree in range(SYNTH_TREES):
            layer = np.zeros(rows * columns, dtype=np.float32)
            for number, segment in enumerate(segments):
                if segment["tree"] != tree:
                    continue
                here = np.array([[place, size] for at, place, size in clusters if at == number]).reshape(-1, 2)
                profile = np.bincount(here[:, 0].astype(int), weights=here[:, 1], minlength=len(segment["points"]))
                profile = scipy.ndimage.gaussian_filter1d(profile, CLUSTER_LENGTH, mode="constant") * CLUSTER_SPACING
                blood = segment["chord"] * (1 + CLUSTER_DEPTH * (profile[segment["along"]] - 1))
                layer[segment["pixels"]] = np.maximum(layer[segment["pixels"]], blood)  # one lumen where a tree's segments meet
            blur = SYNTH_SHARP + SYNTH_DEFOCUS * abs(segments[[s["tree"] for s in segments].index(tree)]["depth"] - focus)
            layer = cv2.GaussianBlur(layer.reshape(rows, columns), (0, 0), blur)
            layers[tree] += layer / len(kept)
            density += layer  # densities of two depths add
        light = illumination * np.exp(-density)
        seen = rng.poisson(light * SYNTH_GAIN) / SYNTH_GAIN + rng.normal(0, SYNTH_READ, light.shape)
        seen = np.clip(np.rint(seen), 1, SYNTH_SATURATION)
        frames[kept.index(frame)] = np.log(illumination) - np.log(seen)
    times = np.array(kept) / SYNTH_FPS
    pairs = [(place, place + 1, float(times[place + 1] - times[place])) for place in range(len(kept) - 1)
             if kept[place + 1] - kept[place] <= MAX_GAP_FRAMES]
    wall = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 5)
    mean, quiet = frames.mean(axis=0), wall > 3 * max(s["radius"] for s in segments)
    mark_visible(junctions, segments, layers, robust_sd(mean[quiet]))
    return {"frames": frames, "times": times, "pairs": pairs, "fps": SYNTH_FPS, "mean": mean,
            "valid": np.ones((rows, columns), dtype=bool), "quiet": quiet,
            "segments": segments, "lines": lines, "junctions": junctions}


def read_burst(burst, pruning, threads=None):
    """Run sections 19 to 27 on a burst given as density frames (see synthetic_burst): the same functions, the same
    constants and the pruning that was chosen for the real burst (`pruning`: L_MIN, Z_MIN, T_HIGH). Returns the
    chains, the graph and the junctions."""
    mean, valid, quiet = burst["mean"], burst["valid"], burst["quiet"]
    frame_at = lambda position: burst["frames"][position]
    contrast = contrast_layer(mean, robust_sd(mean[quiet]), valid)
    scores, radius, _, _ = orientation_hypercolumns(contrast / robust_sd(contrast[quiet]), valid, quiet, threads=threads)
    field = lateral_interaction(scores, radius, threads=threads)
    flow = flow_field(frame_at, burst["pairs"], valid, quiet, threads=threads)
    field, radius, scores = combined_field(field, radius, scores, flow)
    strung = strung_field(crest_neurons(field, radius, valid, scores=scores), scores, valid)
    chains = [chain for chain, kept in zip(strung["chains"], pruned(strung, *pruning)) if kept]
    readings, whole = segment_readings(chains, frame_at, burst["times"], burst["pairs"], burst["fps"], threads=threads)
    graph = vessel_graph(chains, valid, mean, readings, burst["pairs"], burst["fps"])
    count = len(burst["times"])
    halves = (np.arange(0, count, 2), np.arange(1, count, 2)) if count >= SPLIT_FRAMES else None
    found, _, _, _ = read_junctions(graph, mean, scores, halves)
    return {"chains": chains, "readings": readings, "graph": graph, "junctions": found, "burst": whole}


def mark_visible(junctions, segments, layers, noise):
    """Flag the true junctions that nobody can be asked to find, by vascnet's own rules for its synthetic images
    (vascnet.synth._annotate_visibility; the segments are handed to it as its branches): a junction whose faintest
    vessel is under 3 x noise, and a crossing where a vessel is faint beside it ("faint"), stops too soon after it
    ("stub") or runs alongside the other vessel instead of across it ("tangential"). Each junction gets `contrast`,
    `visible` and, where one of the three struck, `ambiguous`. `layers` holds each tree's density without noise;
    `noise` is that of the picture the vessels are found in."""
    children = [[other for other, child in enumerate(segments) if child["parent"] == number] for number in range(len(segments))]
    branches = [vascnet.synth.Branch(segment["points"], np.full(len(segment["points"]), segment["radius"]), segment["tree"],
                                     segment["parent"], children[number]) for number, segment in enumerate(segments)]
    described = [dict(junction, type="crossover", branches=tuple(junction["segments"]), pos=tuple(junction["places"]))
                 if junction["type"] == "crossing" else dict(junction, branch=junction["segments"][0])
                 for junction in junctions]
    vascnet.synth._annotate_visibility(vascnet.synth.Network(SYNTH_SHAPE, branches, [], described), layers, noise)
    for junction, seen in zip(junctions, described):
        junction.update({name: seen[name] for name in ("contrast", "visible", "ambiguous") if name in seen})


def as_vascnet(kind):
    """One of our junction types in the words of vascnet.evaluate. A junction left unresolved is passed on as
    vascnet's "contact", a type no true junction has: it counts as found, and never as typed right. One vessel
    passing through a node is not a junction."""
    if kind in VASCNET_WORDS:
        return VASCNET_WORDS[kind]
    if kind == "through":
        return "none"
    return "complex" if " + " in kind and "unresolved" not in kind else "contact"


def true_limbs(junction, segments, fps):
    """The limbs of a true junction: a bifurcation's trunk and two daughters, a crossing's two vessels on either
    side. For each: its direction away from the junction (over LIMB_REACH points), its width, which way its blood
    runs (+1 away from the junction, -1 towards it) and its speed in pixels per second."""
    if "places" in junction:
        ends = [(f"vessel {which}", number, place, way)
                for which, (number, place) in enumerate(zip(junction["segments"], junction["places"])) for way in (-1, 1)]
    else:
        trunk, *daughters = junction["segments"]
        ends = [("trunk", trunk, len(segments[trunk]["points"]) - 1, -1)] + [("daughter", number, 0, 1) for number in daughters]
    limbs = []
    for name, number, place, way in ends:
        points = segments[number]["points"]
        far = int(np.clip(place + way * LIMB_REACH, 0, len(points) - 1))
        if far == place:
            continue
        along = points[far] - points[place]
        limbs.append({"limb": name, "segment": number, "direction": along / np.hypot(*along),
                      "width": 2 * segments[number]["radius"], "out": int(way * np.sign(segments[number]["speed"])),
                      "speed": abs(segments[number]["speed"]) * fps})
    return limbs


def score_reading(burst, read, kinds="type"):
    """Compare what was read from a synthetic burst with what is true of it. The scoring is vascnet.evaluate's:

      centerlines  centerline_metrics: the share of the lines read that lie within 2 px of a true centerline
                   (precision), and of the true centerlines that have a line read within 2 px (recall)
      junctions    junction_metrics: true junctions that lie close together are first fused into one "complex"
                   junction (cluster_truth); each true junction is matched to at most one junction read, within
                   max(6, 2 x radius + 2) px; a junction read on a true one that nobody can be asked to find
                   (mark_visible) is neither a hit nor a false alarm

    `kinds` names which reading of the junctions is scored: `type` (with flow), `type_geometry` or `baseline`.
    Returns a dict: `centerline` and `junctions` (what the two functions return), `rows` (one per junction read
    and one per true junction missed: place, how it was read, what is true, the outcome), `arms` (one per arm
    of a junction found at a single true bifurcation or crossing, matched to the true limb it leaves along within
    LIMB_WITHIN degrees: what was read of it and what is true of it) and `vessels` (one per whole vessel with a
    clear speed, where the flow was read along whole vessels: see vessel_rows)."""
    lines = draw_chains(read["chains"], SYNTH_SHAPE) >= 0
    centerline = vascnet.evaluate.centerline_metrics(lines, burst["lines"])
    truth = [dict(junction, type=as_vascnet(junction["type"])) for junction in burst["junctions"]]
    found = read["junctions"]
    claimed = [number for number, junction in enumerate(found) if as_vascnet(junction[kinds]) in vascnet.evaluate.JUNCTION_TYPES]
    predicted = [{"x": found[number]["x"], "y": found[number]["y"], "type": as_vascnet(found[number][kinds])} for number in claimed]
    metrics = vascnet.evaluate.junction_metrics(predicted, truth)
    # the lists junction_metrics matched: the fused truth, the part of it that is asked for, and the part that is not
    fused = vascnet.evaluate.cluster_truth(truth)
    asked = [junction for junction in fused if junction.get("visible", True)]
    excused = [junction for junction in fused if not junction.get("visible", True)]
    hit = dict(metrics.pop("matches"))  # place among the junctions claimed -> place among the true ones asked for
    rest = [place for place in range(len(predicted)) if place not in hit]
    spared = set()
    if rest and excused:
        gate = np.array([max(6.0, 2.0 * junction.get("radius", 0) + 2) for junction in excused])  # junction_metrics' own
        spared = {rest[place] for place, _ in vascnet.evaluate.match_points(
            np.array([[predicted[place]["x"], predicted[place]["y"]] for place in rest]),
            np.array([[junction["x"], junction["y"]] for junction in excused]), gate)}
    assert len(predicted) - len(spared) == metrics["n_pred"]
    rows, arms = [], []
    for number, junction in enumerate(found):
        row = {"x": junction["x"], "y": junction["y"], "arms": junction["arms"], "read": junction[kinds],
               "by geometry": junction["type_geometry"], "confidence": junction["confidence"],
               "confidence by geometry": junction["confidence_geometry"], "flow": junction["flow"],
               "flow support": junction["flow_support"], "impossible": junction["impossible"],
               "outcome": "not a junction", "true": "", "type right": False, "geometry right": False}
        if number in claimed:
            place = claimed.index(number)
            row["outcome"] = "found" if place in hit else "nobody is asked for it" if place in spared else "false alarm"
            if place in hit:
                true = asked[hit[place]]
                row.update({"true": OUR_WORDS[true["type"]], "type right": true["type"] == predicted[place]["type"],
                            "geometry right": true["type"] == as_vascnet(junction["type_geometry"])})
                if "segments" in true:  # a single true junction: its limbs are known
                    node = read["graph"]["nodes"][junction["node"]]
                    limbs = true_limbs(true, burst["segments"], burst["fps"])
                    off = np.array([[np.degrees(np.arccos(np.clip(np.dot(arm["direction"], limb["direction"]), -1, 1)))
                                     for limb in limbs] for arm in node["arms"]])
                    taken_arms, taken_limbs = set(), set()
                    for flat in np.argsort(off, axis=None):  # the best-fitting arm and limb first
                        arm, limb = np.unravel_index(flat, off.shape)
                        if off[arm, limb] > LIMB_WITHIN:
                            break
                        if arm in taken_arms or limb in taken_limbs:
                            continue
                        taken_arms.add(arm), taken_limbs.add(limb)
                        mine, theirs = node["arms"][arm], limbs[limb]
                        clear = mine.get("direction_out", 0) != 0
                        arms.append({
                            "junction": len(rows), "true": row["true"], "arm": int(arm), "limb": theirs["limb"],
                            "segment": theirs["segment"], "degrees off": float(off[arm, limb]),
                            "group": next(index for index, group in enumerate(junction["partition"]) if arm in group),
                            "width": mine["width"], "true width": theirs["width"],
                            "out": int(mine.get("direction_out", 0)), "true out": theirs["out"],
                            "inferred": bool(mine.get("inferred", False)),
                            "speed": abs(float(mine["velocity_out"])) if clear and not mine.get("inferred") else np.nan,
                            "true speed": theirs["speed"], "velocity confidence": float(mine.get("velocity_confidence", 0.0)),
                            "speed sigma": float(mine.get("velocity_sigma", np.nan)), "vessel": int(mine.get("vessel", -1)),
                        })
        rows.append(row)
    for place, true in enumerate(asked):
        if place not in hit.values():
            rows.append({"x": true["x"], "y": true["y"], "arms": 0, "read": "", "by geometry": "", "confidence": np.nan,
                         "confidence by geometry": np.nan, "flow": "", "flow support": np.nan, "impossible": False,
                         "outcome": "missed", "true": OUR_WORDS[true["type"]], "type right": False, "geometry right": False})
    vessels = vessel_rows(burst, read["whole"], read["readings"]) if "whole" in read else []
    return {"centerline": centerline, "junctions": metrics, "rows": rows, "arms": arms, "vessels": vessels}


def with_interval(numerators, denominators=None):
    """A share over several bursts with its 95% interval: the sum of the numerators over the sum of the denominators
    (without denominators: the mean of the numerators). The interval is from BOOTSTRAP resamplings of the bursts
    with replacement, the 2.5th and the 97.5th percentile of the share. Bursts are resampled, not junctions: the
    junctions of one burst are not independent of each other. Returns (share, low, high)."""
    numerators = np.asarray(numerators, dtype=float)
    denominators = np.ones(len(numerators)) if denominators is None else np.asarray(denominators, dtype=float)
    picks = np.random.default_rng(0).integers(len(numerators), size=(BOOTSTRAP, len(numerators)))
    shares = numerators[picks].sum(axis=1) / np.maximum(denominators[picks].sum(axis=1), 1e-9)
    low, high = np.percentile(shares, [2.5, 97.5])
    return float(numerators.sum() / max(denominators.sum(), 1e-9)), float(low), float(high)


def ablation_row(scores, junctions="junctions"):
    """One row of the ablation table from the scores of several bursts (score_reading), every share with its 95%
    interval over the bursts (with_interval). `junctions` names which junction scores of each burst are used."""
    found = [score[junctions] for score in scores]
    count = lambda name: [burst[name] for burst in found]
    typed = lambda kinds: [sum(burst["per_type"][kind]["type_correct"] for kind in kinds) for burst in found]
    seen = lambda kind: [burst["per_type"][kind]["detected"] for burst in found]
    return {
        "centerline precision": with_interval([score["centerline"]["precision"] for score in scores]),
        "centerline recall": with_interval([score["centerline"]["recall"] for score in scores]),
        "junction precision": with_interval(count("tp"), count("n_pred")),
        "junction recall": with_interval(count("tp"), count("n_gt")),
        "type right, of found": with_interval(typed(TRUE_TYPES), count("tp")),
        "found and typed right": with_interval(typed(TRUE_TYPES), count("n_gt")),
        "bifurcations typed right": with_interval(typed(("bifurcation",)), seen("bifurcation")),
        "crossings typed right": with_interval(typed(("crossover",)), seen("crossover")),
        "true junctions": int(sum(count("n_gt"))), "junctions read": int(sum(count("n_pred"))),
    }


def print_ablation(rows, columns=ABLATION_COLUMNS):
    """Print rows of the ablation table ({name: ablation_row}): every share in percent with its 95% interval over
    bursts in brackets."""
    width = max(len(name) for name in rows) + 2
    print(" " * width + "".join(f"{column:>25s}" for column in columns))
    for name, row in rows.items():
        cells = [f"{row[column][0] * 100:.0f} ({row[column][1] * 100:.0f} to {row[column][2] * 100:.0f})" for column in columns]
        print(f"{name:{width}s}" + "".join(f"{cell:>25s}" for cell in cells))

# %%
SYNTH_SEEDS = (1, 2, 3, 4, 5, 6)  # one synthetic burst is made and read for each

synthetic_scores = []  # per burst: how sections 19 to 27 read it, against the truth (see score_reading)
for seed in SYNTH_SEEDS:
    burst = synthetic_burst(seed)
    read = read_burst(burst, CHAIN_PRUNING, threads=workers)
    synthetic_scores.append({**score_reading(burst, read),
                             "geometry": score_reading(burst, read, kinds="type_geometry")["junctions"],
                             "baseline": score_reading(burst, read, kinds="baseline")["junctions"]})
    if seed == SYNTH_SEEDS[0]:
        shown = (burst, read)
print(f"{len(SYNTH_SEEDS)} synthetic bursts of {SYNTH_SHAPE[1]} x {SYNTH_SHAPE[0]} px and {SYNTH_FRAMES} frames, read with the "
      f"real burst's constants and pruning. Shares in percent, with their 95% interval over bursts:")
print_ablation({"partitions, with flow": ablation_row(synthetic_scores),
                "partitions, geometry alone": ablation_row(synthetic_scores, "geometry"),
                "the baseline's rules": ablation_row(synthetic_scores, "baseline")})
row = ablation_row(synthetic_scores)
print(f"   {row['true junctions']} true junctions are asked for, {row['junctions read']} junctions were read | of the found "
      f"bifurcations {row['bifurcations typed right'][0] * 100:.0f}% and of the found crossings "
      f"{row['crossings typed right'][0] * 100:.0f}% are typed right")
arms = [arm for score in synthetic_scores for arm in score["arms"]]
measured = [arm for arm in arms if arm["out"] != 0 and not arm["inferred"]]
if measured:
    errors = [abs(arm["speed"] - arm["true speed"]) / arm["true speed"] for arm in measured]
    print(f"   flow at the junctions found: {len(measured)} of {len(arms)} arms have a measured direction; it is the true one "
          f"on {np.mean([arm['out'] == arm['true out'] for arm in measured]) * 100:.0f}%; median error of the speed "
          f"{np.median(errors) * 100:.0f}% of the true speed")

# show the first synthetic burst: its mean density with what is true and what was read
burst, read = shown
picture = chain_picture(read["chains"], burst["mean"], 0, np.percentile(burst["mean"], 99.9))
for junction in burst["junctions"]:
    color = (255, 220, 0) if junction["type"] == "bifurcation" else (255, 255, 255)
    cv2.circle(picture, (int(junction["x"]), int(junction["y"])), 9, color if junction["visible"] else (120, 120, 120), 1)
for junction in read["junctions"]:
    color = {"bifurcation": (255, 220, 0), "crossing": (255, 255, 255)}.get(junction["type"], (255, 60, 60))
    cv2.circle(picture, (int(junction["x"]), int(junction["y"])), 3, color, -1)
figure, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(burst["frames"][0], cmap="gray", vmin=-0.05, vmax=np.percentile(burst["mean"], 99.9))
axes[0].set_title("one frame of a synthetic burst (density)", fontsize=10)
axes[1].imshow(picture, interpolation="antialiased")
axes[1].set_title("chains read from it | circles: true junctions (gray: nobody is asked for it), dots: read "
                  "(yellow bifurcation, white crossing, red other)", fontsize=10)
for axis in axes:
    axis.axis("off")
figure.tight_layout()
plt.show()
del shown, burst, read
# %% [markdown]
# ## 29. Vessels read straight from the static score, with flow as evidence
#
# Section 20's orientation score is the cleanest picture of the vessels so far. This section starts from it directly and leaves out sections 21 and 22: no interactions, no stringing, no pruning sweep.
#
# **No vessel is dropped for want of flow data.** Flow is carried along as what is known about a vessel, and as evidence at the junctions.
#
# **This section runs from files.** It needs five things from the sections above: the static score, its winning orientation and half-width, the flow score, and the temporal frames. They are saved once, and the section then works on the saved files alone. It can be run in a kernel that holds nothing else, which is how to run it when memory is short:
#
# 1. Run the notebook to section 26 once and run the cell *Save the inputs*.
# 2. Restart the kernel.
# 3. Run from the cell *Open the inputs* to the end of the section. Nothing above it has to be run.
#
# How memory is kept low:
#
# - The pictures are read as `float32`. The two stacks (the 16 layers of the static score, the temporal frames) are opened with `np.load(mmap_mode="r")`: a layer or a frame is read from disk when it is used.
# - The kymographs are read off the frames one frame at a time (`chain_signals`). No stack of frames is built.
# - Every cell ends by printing what the kernel holds and the most it has held (`memory_note`).
# - This section runs no filter over the field: the filtering was done in sections 20 and 24. So there is nothing to cut into tiles here.
#
# The cell below defines the section's functions and constants.

# %%
SAVED_FOLDER = "section_29_inputs"  # where this section's inputs are saved, next to the notebook (about 1.5 GB)
SAVED_MAPS = ("static_strength", "static_orientation", "half_width", "flow_strength", "flow_orientation",
              "valid", "density")  # one picture per fixation each, read whole
SAVED_STACKS = ("static_scores", "frames")  # per fixation: the static score's 16 layers and the temporal frames, mapped
SCORE_FULL = 20.0  # noise units; a score this strong is drawn at full brightness, as in section 20's picture
FLOW_SHARE = 0.5  # a centerline counts as having flow data when this share of its points has it
LINE_LEAST = 20   # pixels; a shorter piece of ridge that ends freely is dropped
AROUND = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))  # a pixel's neighbours, the nearest first
RIDGE_JOIN = 2     # pixels; pieces of ridge this close to each other are joined before they are thinned
RIDGE_CHILD = 12   # pixels; a side branch shorter than this is removed
RIDGE_NETWORK = 40  # pixels; lines connected to each other and to nothing else are removed when shorter than this in all
RIDGE_REACH = 16   # pixels; a free end is extended straight ahead to a line it meets within this distance
KNOT_SPACING = 12.0  # pixels between the knots of a centerline's spline
RADIUS_KNOTS = 3     # the half-width along a vessel is fitted with knots this many times farther apart
SPLINE_LEAST = 5     # pixels; a shorter piece of centerline has too few points for a cubic and is left out
BLUR_SHARE = 0.05    # the blur of a picture is read off its thinnest lines: this share of the ridge pixels is thinner
TUBE_BLURS = 2.0     # a vessel's tube reaches its radius plus this many sigmas of the blur from its centerline
STUB_FIELDS = 2.0    # a piece with a free end is kept when this many lengths of its neuron's field lie outside the tube
ARROW_EVERY = 70     # pixels along a directed vessel between two arrowheads in the picture


def load_saved(folder=SAVED_FOLDER):
    """Open what section 29 works on, from the files its first cell saved. The pictures are read; the two stacks
    (the 16 layers of the static score, the temporal frames) are only mapped, so that a layer or a frame is read
    from disk when it is used and nothing more. Returns a dict: per name a list with one entry per fixation, and
    the small things (times, pairs of frames, section 22's chains, ...) as they were saved."""
    with open(os.path.join(folder, "section_29.pkl"), "rb") as file:
        saved = pickle.load(file)
    count = len(saved["times"])
    for name in SAVED_MAPS:
        saved[name] = [np.load(os.path.join(folder, f"{name}_{number}.npy")) for number in range(count)]
    for name in SAVED_STACKS:
        saved[name] = [np.load(os.path.join(folder, f"{name}_{number}.npy"), mmap_mode="r") for number in range(count)]
    saved["threads"] = max(1, (os.cpu_count() or 2) // 2)
    return saved


def flow_winners(flow_scores, shape):
    """The flow field's strongest neuron at every pixel of a frame: its answer (noise units) and its orientation
    number. The field lives on the reduced grid, so each of its layers is brought to the size of a frame first."""
    strongest = np.zeros(shape, dtype=np.float32)
    orientation = np.zeros(shape, dtype=np.uint8)
    for k in range(N_ORIENT):
        layer = at_size(flow_scores[k], shape)
        better = layer > strongest
        strongest[better], orientation[better] = layer[better], k
    return strongest, orientation


def score_colors(orientation, strength):
    """A score as section 20 draws it: the hue is the winning orientation, full brightness at SCORE_FULL."""
    return orientation_hues(orientation, strength / SCORE_FULL)


def without_short_branches(skeleton, shorter_than):
    """A skeleton without its short side branches: pieces shorter than `shorter_than` pixels that are free at one
    end and at a fork at the other."""
    for _ in range(3):  # removing a side branch can leave a shorter one behind it
        ends, forks, pieces, count = skeleton_parts(skeleton)
        length = np.bincount(pieces.ravel(), minlength=count)
        has_end = np.bincount(pieces[ends], minlength=count) > 0
        near_fork = cv2.dilate(forks.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        at_fork = np.bincount(pieces[near_fork & (pieces > 0)], minlength=count) > 0
        drop = has_end & at_fork & (length < shorter_than)
        drop[0] = False
        if not drop.any():
            break
        skeleton = skeletonize(skeleton & ~drop[pieces])
    return skeleton


def extend_ends(lines, strength, reach=RIDGE_REACH):
    """Extend every free end of a picture of thin lines straight ahead to the first other line it meets within
    `reach` pixels, if the score along the extension is T_LOW / 2 or more (its median). A branch that stops just
    short of the vessel it comes from, or of the vessel it passes under, then meets it. Returns the lines with the
    extensions drawn in."""
    ends, forks, pieces, count = skeleton_parts(lines)
    rows, columns = lines.shape
    added = np.zeros(lines.shape, dtype=np.uint8)
    for y, x in zip(*np.nonzero(ends)):
        top, left = max(y - 10, 0), max(x - 10, 0)
        own_y, own_x = np.nonzero(pieces[top:y + 11, left:x + 11] == pieces[y, x])  # the last stretch of its line
        outward = np.array([x - (own_x.mean() + left), y - (own_y.mean() + top)])   # the end points away from it
        if len(own_y) < 4 or np.hypot(*outward) < 1:
            continue  # a speck: no direction
        outward /= np.hypot(*outward)
        for step in np.arange(2, reach + 0.5, 0.5):
            px, py = int(round(x + step * outward[0])), int(round(y + step * outward[1]))
            if not (0 <= px < columns and 0 <= py < rows):
                break
            if lines[py, px] and pieces[py, px] != pieces[y, x]:
                along = np.linspace(0, 1, int(step) + 2)
                under = strength[np.rint(y + along * (py - y)).astype(int), np.rint(x + along * (px - x)).astype(int)]
                if np.median(under) >= T_LOW / 2:
                    cv2.line(added, (int(x), int(y)), (px, py), 1)
                break
    return lines | (added > 0)


def consolidate_lines(lines, strength):
    """Tidy a picture of ridge lines with the steps of `development copy.ipynb`'s consolidation that work on a
    picture of lines. They take out what one vessel has too much of: a second line a pixel or two beside the
    first, stubs, specks, and ends that stop just short of the line they belong to.

      1. pieces within RIDGE_JOIN pixels of each other are joined, and the result thinned to one pixel
      2. side branches shorter than RIDGE_CHILD are removed, then whole networks shorter than RIDGE_NETWORK
      3. free ends are extended straight ahead into a line they meet within RIDGE_REACH (extend_ends)
    """
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RIDGE_JOIN + 1,) * 2)
    joined = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, disc) > 0
    skeleton = without_short_branches(skeletonize(joined), RIDGE_CHILD)
    count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    keep = np.bincount(labels.ravel(), minlength=count) >= RIDGE_NETWORK
    keep[0] = False
    return without_short_branches(skeletonize(extend_ends(keep[labels], strength)), RIDGE_CHILD)


def picture_blur(half_width, lines):
    """The blur of the picture the ridges are read from, in pixels (the sigma of a Gaussian). No line can look
    thinner than the blur, so it is read off the thinnest lines: the half-width that only BLUR_SHARE of the ridge
    pixels are thinner than."""
    return float(np.quantile(half_width[lines], BLUR_SHARE)) if lines.any() else float(HYPER_WIDTHS[0])


def tube_reach(half_width, blur):
    """How far a vessel's tube reaches from its centerline, at every pixel: the vessel's radius plus TUBE_BLURS
    sigmas of the blur. A line's half-width holds the blur already (their squares add), so the blur is taken out
    of it before the half-width is turned into a radius."""
    sharp = np.sqrt(np.maximum(half_width.astype(np.float32) ** 2 - np.float32(blur) ** 2, 0))
    return sharp / np.float32(CYLINDER) + np.float32(TUBE_BLURS * blur)


def without_stubs(skeleton, half_width, blur, valid, fields=STUB_FIELDS):
    """The rule for dropping pieces of centerline. Only a piece with a free end can be dropped; a piece with a
    junction at both ends never is.

    A piece with a free end is judged by how much of it lies outside the tubes of its parents (tube_reach). Its
    parents are the forks, the pieces with a junction at both ends, and the pieces with a free end that are longer
    than it and were kept. What lies inside a parent's tube is the parent's own image, seen a second time. The
    piece is kept when at least `fields` lengths of its own neuron's receptive field (ELONGATION x its half-width
    each) lie outside: a shorter response is no longer than the field that gave it, and says nothing about a line.
    An end where the data ends is not a free end. Returns the skeleton without the dropped pieces, thinned again.
    """
    reach = tube_reach(half_width, blur)
    edge = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    for _ in range(3):  # dropping a stub joins the two pieces it stood between: they are judged again as one
        ends, forks, pieces, count = skeleton_parts(skeleton)
        free = np.bincount(pieces[ends & (edge > FIELD_EDGE)], minlength=count) > 0
        free[0] = False
        if not free.any():
            break
        rows, columns = np.nonzero(skeleton)
        label = pieces[rows, columns]  # 0 on the forks
        size = np.bincount(label, minlength=count)
        width = np.bincount(label, weights=half_width[rows, columns], minlength=count) / np.maximum(size, 1)
        everything = scipy.spatial.cKDTree(np.column_stack([columns, rows]))
        reach_at = reach[rows, columns]
        dropped = np.zeros(count, dtype=bool)
        for piece in sorted(np.flatnonzero(free), key=lambda piece: -size[piece]):  # the longest first
            members = np.flatnonzero(label == piece)
            near = scipy.spatial.cKDTree(np.column_stack([columns[members], rows[members]])).sparse_distance_matrix(
                everything, float(reach_at.max()), output_type="ndarray")
            other = label[near["j"]]
            parent = (other != piece) & ~dropped[other] & (~free[other] | (size[other] > size[piece]))
            inside = np.zeros(len(members), dtype=bool)
            inside[near["i"][parent & (near["v"] <= reach_at[near["j"]])]] = True
            dropped[piece] = (~inside).sum() < fields * ELONGATION * width[piece]
        if not dropped.any():
            break
        skeleton = skeletonize(skeleton & ~dropped[pieces])
    return skeleton


def ridge_crests(strength, orientation, region):
    """The ridges of a score: a pixel is on a ridge when its strength is the largest across its own winning
    orientation (one pixel to either side). Ridge pixels above T_LOW that are joined to one above T_HIGH are kept."""
    rows, columns = strength.shape
    theta = orientation.astype(np.float32) * np.float32(np.pi / N_ORIENT)
    x, y = np.meshgrid(np.arange(columns, dtype=np.float32), np.arange(rows, dtype=np.float32))
    across_x, across_y = -np.sin(theta), np.cos(theta)
    one_side = cv2.remap(strength, x + across_x, y + across_y, cv2.INTER_LINEAR)
    other_side = cv2.remap(strength, x - across_x, y - across_y, cv2.INTER_LINEAR)
    crest = (strength >= one_side) & (strength >= other_side) & region
    return apply_hysteresis_threshold(np.where(crest, strength, 0), T_LOW, T_HIGH)


def ridge_skeleton(strength, orientation, region, consolidate=True):
    """Read the ridges of a score (ridge_crests) as lines one pixel wide, dropping pieces by their length: the
    ridges are either only thinned to one pixel with side branches under SPUR_LENGTH pruned (section 18's
    `centerlines`), or consolidated (consolidate_lines)."""
    lines = ridge_crests(strength, orientation, region)
    return consolidate_lines(lines, strength) if consolidate else centerlines(lines)


def ridge_skeleton_by_tube(strength, orientation, half_width, region, valid, consolidate=True, fields=STUB_FIELDS):
    """Read the ridges of a score as lines one pixel wide, dropping pieces by the tube rule (without_stubs) and by
    nothing else. With `consolidate` the two steps of consolidate_lines that drop nothing are made as well: pieces
    within RIDGE_JOIN pixels are joined first, and free ends are extended into a line they meet (extend_ends), after
    which the rule is applied once more."""
    lines = ridge_crests(strength, orientation, region)
    blur = picture_blur(half_width, lines)
    if not consolidate:
        return without_stubs(skeletonize(lines), half_width, blur, valid, fields)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RIDGE_JOIN + 1,) * 2)
    joined = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, disc) > 0
    skeleton = without_stubs(skeletonize(joined), half_width, blur, valid, fields)
    return without_stubs(skeletonize(extend_ends(skeleton, strength)), half_width, blur, valid, fields)


def ridge_lines(skeleton, strength, orientation, half_width, least=LINE_LEAST, to_forks=False):
    """Cut a skeleton at its forks and return every piece as a chain (the same keys as section 22's chains: x, y,
    theta, radius, strength, ..., its points in order from one end to the other), with `loose`: which of its two
    ends stops freely instead of at a fork. Pieces shorter than `least` that end freely are dropped. With `to_forks`
    a piece is drawn up to the fork pixel at each of its ends, so that the shortest piece between two forks has
    points to be a chain with and no piece between two junctions is lost."""
    ends, forks, pieces, count = skeleton_parts(skeleton)
    near_fork = cv2.dilate(forks.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    which_fork = cv2.connectedComponents(forks.astype(np.uint8), connectivity=8)[1]

    def forks_beside(pixel):
        """The fork pixels next to a pixel: {fork: its nearest pixel}."""
        beside = {}
        for dy, dx in AROUND:
            y, x = pixel[0] + dy, pixel[1] + dx
            if 0 <= y < forks.shape[0] and 0 <= x < forks.shape[1] and forks[y, x]:
                beside.setdefault(int(which_fork[y, x]), (y, x))
        return beside

    rows, columns = np.nonzero(pieces)
    label = pieces[rows, columns]
    order = np.argsort(label, kind="stable")
    rows, columns, label = rows[order].tolist(), columns[order].tolist(), label[order]
    starts = np.flatnonzero(np.r_[True, label[1:] != label[:-1]])
    lines = []
    for start, stop in zip(starts, np.r_[starts[1:], len(label)]):
        pixels = list(zip(rows[start:stop], columns[start:stop]))
        inside = set(pixels)
        beside = lambda pixel: [(pixel[0] + dy, pixel[1] + dx) for dy, dx in AROUND if (pixel[0] + dy, pixel[1] + dx) in inside]
        # walk from a pixel with one neighbour (an end of the piece) to the other end
        path = [next((pixel for pixel in pixels if len(beside(pixel)) <= 1), pixels[0])]
        seen = {path[0]}
        while True:
            following = next((pixel for pixel in beside(path[-1]) if pixel not in seen), None)
            if following is None:
                break
            path.append(following)
            seen.add(following)
        if to_forks:
            # the fork pixels next to the piece's two ends, by the fork they belong to; the two ends go to two
            # different forks where there are two (a piece of one or two pixels has both next to either end)
            first, last = forks_beside(path[0]), forks_beside(path[-1])
            start = next(iter(first), None)
            stop = next((fork for fork in last if fork != start), None)
            if stop is None and last and len(path) > 2:
                stop = next(iter(last))  # a loop that comes back to the fork it left
            if start is not None:
                path.insert(0, first[start])
            if stop is not None:
                path.append(last[stop])
        row, column = np.array(path).T
        loose = (not near_fork[row[0], column[0]], not near_fork[row[-1], column[-1]])
        if len(path) < 2 or (any(loose) and len(path) < least):
            continue
        lines.append({"x": column.astype(np.float32), "y": row.astype(np.float32),
                      "theta": orientation[row, column].astype(np.float32) * np.float32(np.pi / N_ORIENT),
                      "radius": half_width[row, column].astype(np.float32), "strength": strength[row, column],
                      "bridge": np.zeros(len(path), dtype=bool), "closed": False, "loose": loose})
    return lines


def flow_share(chain, has_flow):
    """The share of a chain's points that have flow data."""
    rows, columns = has_flow.shape
    return float(has_flow[np.clip(np.rint(chain["y"]).astype(int), 0, rows - 1),
                          np.clip(np.rint(chain["x"]).astype(int), 0, columns - 1)].mean())


def loose_ends_of(lines, valid):
    """How many ends of ridge lines stop freely, away from the edge of the data."""
    edge = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    return sum(free and edge[int(line["y"][end]), int(line["x"][end])] > FIELD_EDGE
               for line in lines for end, free in zip((0, -1), line["loose"]))


def fit_spline(values, along, spacing):
    """A cubic B-spline fitted by least squares to values given at increasing distances `along` a line, with a knot
    every `spacing` pixels. With far fewer knots than points it smooths; how much is set by the spacing alone."""
    inner = np.linspace(0, along[-1], max(int(along[-1] // spacing), 1) + 1)[1:-1]
    degree = min(3, len(along) - 1)  # a piece of two or three points is fitted with a line or a parabola
    return scipy.interpolate.make_lsq_spline(along, values, np.r_[[0.0] * (degree + 1), inner, [along[-1]] * (degree + 1)],
                                             k=degree)


def spline_vessel(chain, flow_strength):
    """Fit one centerline with splines over the distance along it: one for its place (x, y) and one for its
    half-width. Returns the vessel as a chain whose points are read off the spline a pixel apart, with the splines
    themselves (`place`, `width`), the rms distance of the centerline's points from the spline (`residual`), the
    flow score along it (`flow`) and the share of it that has flow data (`flow share`).

    "A pixel apart" is measured along the spline itself. The centerline's own points are the pixels of a staircase,
    whose length overstates the distance along an oblique line by up to 8%; a speed read along points spaced by
    that length would be too high by as much."""
    points = np.column_stack([chain["x"], chain["y"]]).astype(np.float64)
    along = np.r_[0, np.cumsum(np.maximum(np.hypot(*np.diff(points, axis=0).T), 1e-3))]
    place = fit_spline(points, along, KNOT_SPACING)
    width = fit_spline(chain["radius"].astype(np.float64), along, RADIUS_KNOTS * KNOT_SPACING)
    fine = np.linspace(0, along[-1], 4 * int(np.ceil(along[-1])) + 1)
    travelled = np.r_[0, np.cumsum(np.hypot(*np.diff(place(fine), axis=0).T))]  # the distance along the spline itself
    at = np.interp(np.linspace(0, travelled[-1], int(round(travelled[-1])) + 1), travelled, fine)
    x, y = place(at).T
    dx, dy = place(at, 1).T
    rows, columns = flow_strength.shape
    row, column = np.clip(np.rint(y).astype(int), 0, rows - 1), np.clip(np.rint(x).astype(int), 0, columns - 1)
    return {
        "x": x.astype(np.float32), "y": y.astype(np.float32), "tangent": np.arctan2(dy, dx).astype(np.float32),
        "theta": (np.arctan2(dy, dx) % np.pi).astype(np.float32),
        "radius": np.clip(width(at), HYPER_WIDTHS[0] / 2, None).astype(np.float32),
        "strength": np.interp(at, along, chain["strength"]).astype(np.float32),
        "flow": flow_strength[row, column], "flow share": chain["flow share"],
        "bridge": np.zeros(len(at), dtype=bool), "closed": False,
        "length": float(travelled[-1]), "place": place, "width": width,
        "residual": float(np.sqrt(np.mean(np.sum((place(along) - points) ** 2, axis=1)))),
    }


def downstream(vessel, reading):
    """Give a vessel its flow: the velocity read from its kymograph (section 25). Where the direction is clear the
    vessel's points are put in the order the blood passes them, so that its tangent points downstream; the reading
    is turned with it. A vessel without a clear direction keeps the order it was traced in."""
    vessel.update({"velocity": np.nan, "velocity_confidence": float(reading["velocity_confidence"]),
                   "directed": reading["direction"] != 0, "turned": False})
    if vessel["directed"]:
        vessel["velocity"] = abs(float(reading["velocity"]))
        if reading["direction"] < 0:  # the blood runs against the order of the points: turn the vessel round
            for name in ("x", "y", "theta", "radius", "strength", "flow"):
                vessel[name] = vessel[name][::-1]
            vessel["tangent"] = np.angle(-np.exp(1j * vessel["tangent"][::-1])).astype(np.float32)
            vessel["turned"] = True
            reading.update({"M": reading["M"][::-1], "velocity": -reading["velocity"], "direction": 1})
    return vessel


def direction_colors(angle):
    """Colors for directions of flow: a full turn of hue for a full turn of direction (red runs to the right)."""
    hsv = np.zeros((1, len(angle), 3), dtype=np.uint8)
    hsv[0, :, 0] = (np.degrees(angle) % 360 / 2).astype(np.uint8)
    hsv[0, :, 1:] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def draw_flow(picture, vessel, label=False):
    """Draw a vessel with velocity data on a picture: its spline in the color of the direction the blood runs in,
    arrowheads pointing downstream, and with `label` its speed in pixels per second next to its middle (with its
    sigma, where the vessel has a `velocity_sigma`)."""
    points = np.rint(np.column_stack([vessel["x"], vessel["y"]])).astype(np.int32)
    colors = direction_colors(vessel["tangent"])
    for place in range(len(points) - 1):
        cv2.line(picture, tuple(map(int, points[place])), tuple(map(int, points[place + 1])), tuple(map(int, colors[place])), 2)
    for place in range(min(ARROW_EVERY // 2, len(points) // 2), len(points) - 4, ARROW_EVERY):
        ahead = (float(np.cos(vessel["tangent"][place])), float(np.sin(vessel["tangent"][place])))
        tip = (int(points[place][0] + 5 * ahead[0]), int(points[place][1] + 5 * ahead[1]))
        tail = (int(points[place][0] - 5 * ahead[0]), int(points[place][1] - 5 * ahead[1]))
        cv2.arrowedLine(picture, tail, tip, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.6)
    if label:
        middle = len(points) // 2
        aside = (-np.sin(vessel["tangent"][middle]), np.cos(vessel["tangent"][middle]))  # next to the line, not on it
        text = f"{vessel['velocity']:.0f}" + (f"+-{vessel['velocity_sigma']:.0f}" if "velocity_sigma" in vessel else "")
        cv2.putText(picture, text, (int(points[middle][0] + 7 * aside[0] - 8),
                                                              int(points[middle][1] + 7 * aside[1] + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def direction_key(picture):
    """A ring in the picture's corner: which color runs which way."""
    for degrees in range(0, 360, 4):
        angle = np.radians(degrees)
        color = tuple(map(int, direction_colors(np.array([angle]))[0]))
        cv2.line(picture, (int(40 + 14 * np.cos(angle)), int(40 + 14 * np.sin(angle))),
                 (int(40 + 28 * np.cos(angle)), int(40 + 28 * np.sin(angle))), color, 3)


def ridge_front(burst, threads=None):
    """What every way of reading a burst's ridges starts from, made once per burst (see synthetic_burst for the
    burst): the static score, its winners, the mask, and the flow score with where there is flow data."""
    mean, valid, quiet = burst["mean"], burst["valid"], burst["quiet"]
    contrast = contrast_layer(mean, robust_sd(mean[quiet]), valid)
    scores, radius, _, _ = orientation_hypercolumns(contrast / robust_sd(contrast[quiet]), valid, quiet, threads=threads)
    strength, orientation, half_width = winning_neurons(scores, radius)
    flow = flow_field(lambda position: burst["frames"][position], burst["pairs"], valid, quiet, threads=threads)
    flow_strength, _ = flow_winners(flow["scores"], valid.shape)
    return {"scores": scores, "strength": strength, "orientation": orientation, "half_width": half_width,
            "mask": apply_hysteresis_threshold(strength, T_LOW, T_HIGH) & valid, "flow_strength": flow_strength,
            "has_flow": apply_hysteresis_threshold(flow_strength, T_LOW, T_HIGH)}


def read_burst_ridges(burst, front, consolidate=True, least=SPLINE_LEAST, stubs="by length", fields=STUB_FIELDS,
                      speeds="per piece", evidence="tests", p=None, threads=None):
    """Section 29 on a burst given as density frames: the vessels from the ridges of the static score (`front`, see
    ridge_front), as splines, each with its flow; their graph; and its junctions, read from the geometry and with
    flow. What the ways of reading differ in:

      consolidate  whether the ridge lines are consolidated
      stubs        how pieces of centerline are dropped: "by length" (side branches under SPUR_LENGTH or RIDGE_CHILD,
                   loose pieces under LINE_LEAST, networks under RIDGE_NETWORK, and every piece under `least`), or
                   "by tube" (only pieces with a free end, by their length outside their parent's tube: without_stubs;
                   `least` is then not used; `fields` is that rule's one setting)
      speeds       where the flow is read: "per piece" (every piece of centerline has a kymograph of its own), or
                   "whole vessels" (the pieces are joined through crossings and simple joins first, and one
                   kymograph is read per whole vessel: whole_vessel_flow)
      evidence, p  how flow counts at a junction (read_junction): by the "tests", or by the "directions" as a
                   likelihood ratio, with p the share of measured directions that are right
    The reading also holds `counts` (direction_counts) where the flow was read along whole vessels.
    """
    mean, valid = burst["mean"], burst["valid"]
    frame_at = lambda position: burst["frames"][position]
    if stubs == "by tube":
        skeleton = ridge_skeleton_by_tube(front["strength"], front["orientation"], front["half_width"], front["mask"],
                                          valid, consolidate, fields)
        lines = ridge_lines(skeleton, front["strength"], front["orientation"], front["half_width"], least=0, to_forks=True)
        least = 0
    else:
        lines = ridge_lines(ridge_skeleton(front["strength"], front["orientation"], front["mask"], consolidate),
                            front["strength"], front["orientation"], front["half_width"])
    for line in lines:
        line["flow share"] = flow_share(line, front["has_flow"])
    vessels = [spline_vessel(line, front["flow_strength"]) for line in lines if chain_length(line) >= least]
    count = len(burst["times"])
    halves = (np.arange(0, count, 2), np.arange(1, count, 2)) if count >= SPLIT_FRAMES else None
    if speeds == "whole vessels":
        graph = vessel_graph(vessels, valid, mean)
        read_junctions(graph, mean, front["scores"], flow=False)  # from the geometry: which arms pass through
        whole, readings, limit = whole_vessel_flow(graph, mean, frame_at, burst["times"], burst["pairs"], burst["fps"],
                                                   threads=threads)
        found = read_junctions(graph, mean, front["scores"], halves, evidence=evidence, p=p)[0]
        return {"chains": vessels, "whole": whole, "readings": readings, "graph": graph, "speed limit": limit,
                "junctions": found, "counts": direction_counts(graph, found, readings)}
    readings, _ = segment_readings(vessels, frame_at, burst["times"], burst["pairs"], burst["fps"], threads=threads)
    vessels = [downstream(vessel, reading) for vessel, reading in zip(vessels, readings)]
    graph = vessel_graph(vessels, valid, mean, readings, burst["pairs"], burst["fps"])
    return {"chains": vessels, "readings": readings, "graph": graph,
            "junctions": read_junctions(graph, mean, front["scores"], halves)[0]}

# %% [markdown]
# **Save the inputs.** Run once, in a kernel that has run sections 1 to 26. It writes the folder `section_29_inputs` next to the notebook (about 1.5 GB):
#
# | File, per fixation `n` | What |
# |---|---|
# | `static_strength_n.npy`, `static_orientation_n.npy`, `half_width_n.npy` | the winning simple cell of section 20's score at every pixel: its answer in multiples of noise, its orientation (0 to 15), its half-width in pixels |
# | `static_scores_n.npy` | all 16 orientation layers of that score (the junction reading looks at them on a ring) |
# | `flow_strength_n.npy`, `flow_orientation_n.npy` | the same winner of section 24's flow field, at the size of a frame |
# | `valid_n.npy`, `density_n.npy` | where there is data, and the mean density |
# | `frames_n.npy` | the temporal frames (`temporal_frame`), written one frame at a time |
# | `section_29.pkl` (one file) | the small things: frame times, pairs of frames, the maps between fixations, section 22's chains for comparison |

# %%
# save what section 29 works on, so that it can be run in a kernel that holds nothing else
os.makedirs(SAVED_FOLDER, exist_ok=True)
small = {
    "fps": fps, "anchor": anchor, "shape": frame_shape, "density range": DENSITY_RANGE, "visible": visible_classes,
    "times": [fixation["times"] for fixation in fixations], "pairs": frame_pairs,
    "to anchor": [anchor_to_reference(fixation) for fixation in fixations],  # anchor (x, y) -> that fixation's frame
    "pattern length": PATTERN_LENGTH, "chain pruning": CHAIN_PRUNING, "chain numbers": dict(pruning_table[CHAIN_PRUNING]), "chains": [], "chain loose ends": [],
}
for number, fixation in enumerate(fixations):
    strength, orientation, half_width = winning_neurons(orientation_scores[number], radius_maps[number])
    flow_strength, flow_orientation = flow_winners(flow_energy[number], frame_shape)
    pictures = (strength.astype(np.float32), orientation.astype(np.uint8), half_width.astype(np.float32), flow_strength,
                flow_orientation, valid_maps[number], background_fits[number]["OD"].astype(np.float32))
    for name, picture in zip(SAVED_MAPS, pictures):
        np.save(os.path.join(SAVED_FOLDER, f"{name}_{number}.npy"), picture)
    # the two stacks are written a layer and a frame at a time: neither is ever whole in memory
    layers = np.lib.format.open_memmap(os.path.join(SAVED_FOLDER, f"static_scores_{number}.npy"), mode="w+",
                                       dtype=np.float32, shape=orientation_scores[number].shape)
    for k in range(N_ORIENT):
        layers[k] = orientation_scores[number][k]
    frames = np.lib.format.open_memmap(os.path.join(SAVED_FOLDER, f"frames_{number}.npy"), mode="w+", dtype=np.float32,
                                       shape=(len(fixation["frames"]), *frame_shape))
    for position in range(len(fixation["frames"])):
        frames[position] = temporal_frame(number, position)
    del layers, frames, pictures
    strung, keep = strung_static[number], static_keeps[number]
    small["chains"].append([chain for chain, kept in zip(strung["chains"], keep) if kept])
    small["chain loose ends"].append(chain_numbers(strung["chains"], keep, strung["attached"], strung["meetings"],
                                                   strung["ends"])["loose ends"])
with open(os.path.join(SAVED_FOLDER, "section_29.pkl"), "wb") as file:
    pickle.dump(small, file)
size = sum(os.path.getsize(os.path.join(SAVED_FOLDER, name)) for name in os.listdir(SAVED_FOLDER))
print(f"Saved in {SAVED_FOLDER}: {len(os.listdir(SAVED_FOLDER))} files, {size / 2**30:.2f} GB")
memory_note("inputs saved")
del small

# %% [markdown]
# **Open the inputs.** In a fresh kernel, start here.
#
# - If the kernel has not run the cells above, this cell runs the ones that only define functions and constants, by their ids, from the saved notebook file. Save the notebook first. No data is loaded by them.
# - Then the saved inputs are opened: `saved`, a dictionary with one entry per name in the table above, each a list with one item per fixation.

# %%
# In a fresh kernel, start here: nothing above this cell has to be run. The cells that only define functions and
# constants, above this cell and below it, are run by their ids from the saved notebook file; nothing of the burst
# is loaded. Then the saved inputs are opened. In a kernel that has run the sections above, only the inputs are opened.
DEFINITION_CELLS = ("setup", "background-functions", "graph-functions", "shared-functions", "contrast-functions",
                    "hypercolumn-functions", "lateral-functions", "chain-functions", "signal-functions", "flow-functions",
                    "segment-functions", "graph-builder", "junction-functions", "synthetic-functions",
                    "flow-vessel-functions", "harness-functions", "whole-vessel-functions", "flow-evidence-functions",
                    "generator-functions", "cyclic-graph-functions", "flux-network-functions",
                    "junction-cell-functions", "render-functions", "core-functions", "recursion-functions")
fresh_kernel = "load_saved" not in globals()
if fresh_kernel:
    import contextlib
    import io
    import json
    with open(globals().get("__vsc_ipynb_file__", "development copy 2.ipynb"), encoding="utf-8") as file:
        sources = {cell.get("id"): "".join(cell["source"]) for cell in json.load(file)["cells"]}
    with contextlib.redirect_stdout(io.StringIO()):  # what those cells print belongs where they stand
        for cell_id in DEFINITION_CELLS:
            exec(sources[cell_id], globals())
    del sources
saved = load_saved()
if fresh_kernel:  # the values the sections above would have set
    anchor, frame_shape, fps, workers, DENSITY_RANGE = (saved["anchor"], saved["shape"], saved["fps"], saved["threads"],
                                                        saved["density range"])
print(f"{'A fresh kernel: the definitions were run from the notebook file. ' if fresh_kernel else ''}"
      f"Opened {len(saved['times'])} fixations of {frame_shape[1]} x {frame_shape[0]} px: "
      + ", ".join(f"{len(times)} frames" for times in saved["times"]))
memory_note("inputs opened")

# %% [markdown]
# **The static score as a mask, the flow score inside it.**
#
# - *The mask*: where the strongest simple cell of the static field answers above `T_LOW`, joined to a place above `T_HIGH`.
# - *Flow data*: the same test on the flow field of section 24.
# - At every pixel of the mask the flow field's winning orientation is looked up and compared with the static one. The cell prints how much of the mask has flow data, and how the two orientations lie to each other where both answer.
# - The third picture is the flow score on the intersection: same positions as the static score, the flow field's strength and orientation. Full brightness is at `SCORE_FULL` times the noise.
#
# The flow field has an orientation (0 to 180 degrees), not a direction. The direction the blood runs in comes from the kymographs, two cells further down.
#
# **Result:** `flow_masks[n]`, a dictionary of three boolean pictures at the size of a frame: `mask`, `has flow`, `both`.

# %%
flow_masks = []  # per fixation: the mask, where there is flow data, and where both are (boolean pictures)
for number in range(len(saved["times"])):
    valid = saved["valid"][number]
    static_orientation, flow_orientation = saved["static_orientation"][number], saved["flow_orientation"][number]
    # the mask: where a simple cell of the static field answers (above T_LOW, joined to a place above T_HIGH)
    mask = apply_hysteresis_threshold(saved["static_strength"][number], T_LOW, T_HIGH) & valid
    # where there is flow data: the same test on the flow field
    has_flow = apply_hysteresis_threshold(saved["flow_strength"][number], T_LOW, T_HIGH) & valid
    both = mask & has_flow
    turned = np.abs(flow_orientation[both].astype(int) - static_orientation[both].astype(int))
    turned = np.minimum(turned, N_ORIENT - turned)
    flow_masks.append({"mask": mask, "has flow": has_flow, "both": both})
    print(f"Fixation {number}: the mask (static score) covers {mask.sum() / valid.sum() * 100:.0f}% of the field; "
          f"{both.sum() / mask.sum() * 100:.0f}% of it has flow data, {(mask & ~has_flow).sum() / mask.sum() * 100:.0f}% "
          f"has none | flow data outside the mask: {(has_flow & ~mask).sum() / valid.sum() * 100:.0f}% of the field")
    print(f"   inside the intersection the flow field's winning orientation is the static one's on "
          f"{(turned == 0).mean() * 100:.0f}% of the pixels, within one step on {(turned <= 1).mean() * 100:.0f}%, within two "
          f"on {(turned <= 2).mean() * 100:.0f}%, at right angles (7 or 8 steps) on {(turned >= 7).mean() * 100:.1f}% "
          f"(chance: {100 / N_ORIENT:.0f}%, {300 / N_ORIENT:.0f}%, {500 / N_ORIENT:.0f}%, {300 / N_ORIENT:.0f}%)")

# show, for the anchor's fixation: the static score, the flow score, and the flow score on the intersection
here = flow_masks[anchor]
os.makedirs("vessel_pictures/flow_vessels", exist_ok=True)
pictures = (
    ("static score (the mask): hue is the winning orientation", "static_score",
     score_colors(saved["static_orientation"][anchor], saved["static_strength"][anchor] * here["mask"])),
    ("flow score: hue is the flow field's winning orientation", "flow_score",
     score_colors(saved["flow_orientation"][anchor], saved["flow_strength"][anchor])),
    ("the flow score inside the mask: where both fields answer", "flow_score_in_mask",
     score_colors(saved["flow_orientation"][anchor], saved["flow_strength"][anchor] * here["both"])),
)
figure, axes = plt.subplots(1, 3, figsize=(16, 3.7))
for axis, (title, name, picture) in zip(axes, pictures):
    Image.fromarray(picture).save(f"vessel_pictures/flow_vessels/{name}.png")
    axis.imshow(picture, interpolation="antialiased")
    axis.set_title(title, fontsize=10)
    axis.axis("off")
figure.tight_layout()
plt.show()
del pictures, here
memory_note("mask and flow")

# %% [markdown]
# **Centerlines.** Four readings are made and compared, because which one is best is settled by numbers, not by argument:
#
# | Reading | How |
# |---|---|
# | ridges of the static score | A pixel is on a ridge when its strength is the largest across its own winning orientation. Ridge pixels above `T_LOW` joined to one above `T_HIGH` are thinned to one pixel. Nothing else. |
# | the same, consolidated | The steps of `development copy.ipynb`'s consolidation that work on a picture of lines (`consolidate_lines`): pieces within `RIDGE_JOIN` pixels are joined, side branches under `RIDGE_CHILD` and networks under `RIDGE_NETWORK` pixels are removed, and a free end is extended straight ahead into a line it meets within `RIDGE_REACH` pixels, if the score under the extension supports it. |
# | section 22's chains | For comparison. |
# | ridges of the flow score inside the mask | Only to see what the flow score's own ridges would give. |
#
# For each: the length, the loose ends per 1000 px, how much is found again in the other fixation above chance, and the share of the length that has flow data. The reading that is found again best is taken; among readings within 3 points of it, the one with the fewest loose ends.
#
# Ridge lines are cut at their forks, so every piece is one stretch of vessel between two junctions (`ridge_lines`). Every piece knows the share of its points with flow data (`flow share`).
#
# **Result:** `lines_by` (every reading's centerlines, per fixation), `line_table`, `LINE_READING` and `vessel_lines[n]`, the centerlines that are fitted next.

# %%
# four ways to centerlines, in every fixation. No vessel is dropped for want of flow data: every centerline is
# told how much of it has flow data, and the flow score's own ridges are read only to see what they would give
READINGS = ("ridges of the static score", "the same, consolidated", "section 22's chains",
            "ridges of the flow score inside the mask")
lines_by = {reading: [] for reading in READINGS}  # per reading and fixation: the centerlines, as chains
loose_by = {reading: [] for reading in READINGS}  # per reading and fixation: their loose ends
for number in range(len(saved["times"])):
    here, valid = flow_masks[number], saved["valid"][number]
    strength, orientation, half_width = (saved["static_strength"][number], saved["static_orientation"][number],
                                         saved["half_width"][number])
    for reading, consolidate in zip(READINGS[:2], (False, True)):
        lines_by[reading].append(ridge_lines(ridge_skeleton(strength, orientation, here["mask"], consolidate),
                                             strength, orientation, half_width))
        loose_by[reading].append(loose_ends_of(lines_by[reading][number], valid))
    lines_by[READINGS[2]].append(saved["chains"][number])
    loose_by[READINGS[2]].append(saved["chain loose ends"][number])
    lines_by[READINGS[3]].append(ridge_lines(
        ridge_skeleton(saved["flow_strength"][number], saved["flow_orientation"][number], here["both"], consolidate=False),
        saved["flow_strength"][number], saved["flow_orientation"][number], half_width))
    loose_by[READINGS[3]].append(loose_ends_of(lines_by[READINGS[3]][number], valid))
    for reading in READINGS:
        for line in lines_by[reading][number]:
            line["flow share"] = flow_share(line, here["has flow"])

# how the four compare in the anchor's fixation
line_table = {}
print(f"{'centerlines':42s} {'length':>7s} {'lines':>6s} {'loose ends':>10s} {'per 1000 px':>11s} {'found again':>11s} "
      f"{'by chance':>9s} {'above chance':>12s} {'with flow data':>14s}")
for reading in READINGS:
    lines = lines_by[reading][anchor]
    picture = draw_chains(lines, frame_shape) >= 0
    again = [repeated(picture, saved["valid"][anchor], draw_chains(lines_by[reading][number], frame_shape) >= 0,
                      saved["valid"][number], saved["to anchor"][number])
             for number in range(len(saved["times"])) if number != anchor]
    lengths = np.array([chain_length(line) for line in lines])
    flowing = np.array([line["flow share"] >= FLOW_SHARE for line in lines])
    numbers = {"length": float(lengths.sum()), "lines": len(lines), "loose ends": loose_by[reading][anchor],
               "per 1000 px": loose_by[reading][anchor] / max(lengths.sum(), 1) * 1000,
               "with flow data": float(lengths[flowing].sum() / max(lengths.sum(), 1))}
    numbers["found again"], numbers["by chance"], numbers["above chance"] = np.mean(again, axis=0) if again else (np.nan,) * 3
    line_table[reading] = numbers
    print(f"{reading:42s} {numbers['length']:7.0f} {numbers['lines']:6d} {numbers['loose ends']:10d} "
          f"{numbers['per 1000 px']:11.1f} {numbers['found again'] * 100:10.0f}% {numbers['by chance'] * 100:8.0f}% "
          f"{numbers['above chance'] * 100:11.0f}% {numbers['with flow data'] * 100:13.0f}%")
# the reading to go on with, of the three that hold all vessels: the one found again best above chance; of the
# readings within 3 points of it, the one with the fewest loose ends per 1000 px
best = max(line_table[reading]["above chance"] for reading in READINGS[:3])
LINE_READING = min((reading for reading in READINGS[:3] if line_table[reading]["above chance"] >= best - 0.03),
                   key=lambda reading: line_table[reading]["per 1000 px"])
vessel_lines = lines_by[LINE_READING]  # per fixation: the centerlines of all vessels, as chains
print(f"Going on with: {LINE_READING}")

# show the four readings over the mean density, colored by orientation
shade = np.uint8(np.clip((np.nan_to_num(saved["density"][anchor]) - DENSITY_RANGE[0])
                         / (DENSITY_RANGE[1] - DENSITY_RANGE[0]), 0, 1) * 255)
figure, axes = plt.subplots(2, 2, figsize=(16, 10.4))
for axis, reading, name in zip(axes.ravel(), READINGS, ("static_ridges", "static_ridges_consolidated", "chains", "flow_ridges")):
    picture = chain_picture(lines_by[reading][anchor], shade, 0, 255)
    Image.fromarray(picture).save(f"vessel_pictures/flow_vessels/centerlines_{name}.png")
    axis.imshow(picture, interpolation="antialiased")
    axis.set_title(f"{reading}: {line_table[reading]['length'] / 1000:.0f} thousand px", fontsize=9)
    axis.axis("off")
figure.tight_layout()
plt.show()
del picture, shade
memory_note("centerlines")

# %% [markdown]
# **Splines, and the vessels with velocity data.** Every centerline is fitted with two cubic B-splines over the distance along it (`spline_vessel`): one for its place `(x, y)` with a knot every `KNOT_SPACING` pixels, one for its half-width with knots three times farther apart. The knot spacing alone sets how smooth the fit is. The cell prints how many numbers describe how many pixels of centerline, and how far the centerline's points lie from their spline.
#
# Each vessel is then read along its spline as in section 25 (`segment_readings`), and annotated (`downstream`):
#
# - **Velocity data**: where the kymograph shows a clear travelling pattern, the vessel has a speed in pixels per second and a direction. Its points are put in the order the blood passes them, so its tangent points downstream.
# - **Flow data**: the share of its points where the flow field answers.
# - A vessel with neither is kept all the same.
#
# **Two pictures** of the anchor's fixation, saved in `vessel_pictures/flow_vessels`:
#
# - `directed_vessels.png`: all vessels. Those with velocity data are drawn in the color of the direction the blood runs in (the ring in the corner is the key) with arrows pointing downstream; white ones have flow data but no clear direction; pale blue ones have no flow data.
# - `velocity_annotated.png`: only the vessels with velocity data, each with its speed written next to it.
#
# **Result:** `spline_vessels[n]`, one dictionary per vessel: its points a pixel apart (`x`, `y`, `tangent`, `radius`, `strength`, `flow`), its splines (`place`, `width`), `length`, `residual`, `flow share`, `directed`, `velocity`, `velocity_confidence`. The same is written to `vessel_pictures/flow_vessels/vessels_fixation_<n>.json` with every vessel's middle, half-width, speed and direction.

# %%
spline_vessels = []   # per fixation: every vessel as splines, with its flow (see spline_vessel and downstream)
vessel_readings = []  # per fixation: the vessels' temporal signals as section 25 reads them; the next cell uses them up
for number, times in enumerate(saved["times"]):
    lines = [line for line in vessel_lines[number] if chain_length(line) >= SPLINE_LEAST]
    vessels = [spline_vessel(line, saved["flow_strength"][number]) for line in lines]
    # the kymographs are read off the mapped frames one frame at a time; no stack of frames is ever in memory
    readings, _ = segment_readings(vessels, lambda position: saved["frames"][number][position], times,
                                   saved["pairs"][number], fps, visible=saved["visible"], threads=workers)
    vessels = [downstream(vessel, reading) for vessel, reading in zip(vessels, readings)]
    spline_vessels.append(vessels)
    vessel_readings.append(readings)
    lengths = np.array([vessel["length"] for vessel in vessels])
    flowing = np.array([vessel["flow share"] >= FLOW_SHARE for vessel in vessels])
    directed = np.array([vessel["directed"] for vessel in vessels])
    numbers = sum(len(vessel["place"].c) * 2 + len(vessel["width"].c) for vessel in vessels)
    print(f"Fixation {number}: {len(vessels)} vessels fitted, {lengths.sum():.0f} px of centerline described by {numbers} "
          f"numbers | the centerlines' points lie {np.median([vessel['residual'] for vessel in vessels]):.2f} px from "
          f"their spline (median of the rms)")
    print(f"   flow data on {flowing.sum()} vessels with {lengths[flowing].sum() / lengths.sum() * 100:.0f}% of the length; "
          f"a clear direction of flow on {directed.sum()} with {lengths[directed].sum() / lengths.sum() * 100:.0f}% of the "
          f"length (median speed {np.median([vessel['velocity'] for vessel in vessels if vessel['directed']]):.0f} px/s) | "
          f"a clear direction on {lengths[directed & flowing].sum() / max(lengths[flowing].sum(), 1) * 100:.0f}% of the "
          f"length with flow data and on {lengths[directed & ~flowing].sum() / max(lengths[~flowing].sum(), 1) * 100:.0f}% "
          f"of the length without")
    # every vessel's annotation and its splines, to be read without this notebook. A vessel with velocity data has
    # its speed and the direction the blood runs in (degrees, clockwise from pointing right, as on the screen)
    export = []
    for identity, vessel in enumerate(vessels):
        middle = len(vessel["x"]) // 2
        export.append({
            "id": identity, "length": round(vessel["length"], 1), "x": round(float(vessel["x"][middle]), 1),
            "y": round(float(vessel["y"][middle]), 1), "half_width": round(float(np.median(vessel["radius"])), 2),
            "flow_share": round(vessel["flow share"], 2), "has_velocity": bool(vessel["directed"]),
            "speed_px_per_s": None if not vessel["directed"] else round(vessel["velocity"], 1),
            "direction_degrees": None if not vessel["directed"] else round(float(np.degrees(np.angle(
                np.exp(1j * vessel["tangent"]).mean())) % 360), 1),
            "velocity_confidence": round(vessel["velocity_confidence"], 1), "turned": bool(vessel["turned"]),
            "place_knots": np.round(vessel["place"].t, 2).tolist(), "place_xy": np.round(vessel["place"].c, 2).tolist(),
            "width_knots": np.round(vessel["width"].t, 2).tolist(), "width": np.round(vessel["width"].c, 2).tolist(),
        })
    with open(f"vessel_pictures/flow_vessels/vessels_fixation_{number}.json", "w") as file:
        json.dump(export, file)

# two pictures of the anchor's fixation over the mean density:
#   all vessels: those with velocity data in the color of their direction, the others white or pale blue
#   only the vessels with velocity data, each with its speed (pixels per second) written next to it
shade = np.uint8(np.clip((np.nan_to_num(saved["density"][anchor]) - DENSITY_RANGE[0])
                         / (DENSITY_RANGE[1] - DENSITY_RANGE[0]), 0, 1) * 110)
everything, annotated = np.dstack([shade, shade, shade]), np.dstack([shade, shade, shade])
for vessel in spline_vessels[anchor]:
    if vessel["directed"]:
        draw_flow(everything, vessel)
        draw_flow(annotated, vessel, label=vessel["length"] >= ARROW_EVERY)
    else:
        points = np.rint(np.column_stack([vessel["x"], vessel["y"]])).astype(np.int32)
        flowing = vessel["flow share"] >= FLOW_SHARE
        cv2.polylines(everything, points[None], False, (230, 230, 230) if flowing else (120, 150, 200), 1, cv2.LINE_AA)
direction_key(everything)
direction_key(annotated)
Image.fromarray(everything).save("vessel_pictures/flow_vessels/directed_vessels.png")
Image.fromarray(annotated).save("vessel_pictures/flow_vessels/velocity_annotated.png")
figure, axes = plt.subplots(2, 1, figsize=(16, 20))
axes[0].imshow(everything, interpolation="antialiased")
axes[0].set_title("all vessels, as splines | color: the direction of flow (ring: which color runs which way), arrows point "
                  "downstream | white: flow data but no clear direction | pale blue: no flow data", fontsize=10)
axes[1].imshow(annotated, interpolation="antialiased")
axes[1].set_title("the vessels with velocity data: direction (color and arrows) and speed in pixels per second (numbers, on "
                  f"vessels of {ARROW_EVERY} px and more)", fontsize=10)
for axis in axes:
    axis.axis("off")
figure.tight_layout()
plt.show()
del everything, annotated, shade
memory_note("splines and kymographs")

# %% [markdown]
# **Junctions, with flow as evidence.** The spline vessels are made into a graph as in section 26 (`vessel_graph`) and its junctions are read as in section 27 (`read_junctions`). Every junction is read from its geometry first. What flow then does to the reading (`read_junction`):
#
# | Flow | What happens |
# |---|---|
# | no flow data | The geometric reading stands, with its own confidence. |
# | confirms | The geometric reading also passes the tests that need the burst (blood comes in and goes out of every vessel, flux, pulse phase, blur, fingerprint). Its confidence rises. |
# | contradicts | It fails them, and no other reading fits. The geometric reading is kept and its confidence falls. |
# | changes | It fails them, and another reading fits geometry and flow together. That one is taken. |
#
# Flow never takes a junction's type away.
#
# **Does flow help?** The second half of the cell reads synthetic bursts (section 28) this way and prints, for the geometry alone and for geometry with flow: how many junction types are read right, and the mean confidence of the right and of the wrong readings. It also prints how the ridge vessels compare with section 22's chains on the same bursts: how much of them lies on true centerlines, and how many of the junctions found are true.
#
# **Result:** `ridge_graphs[n]` and `ridge_junctions[n]`. Every junction has `type` and `confidence` (with flow), `type_geometry` and `confidence_geometry` (without), and `flow`: one of the four words above. The figure shows the junctions whose confidence flow raised most, and those it lowered most.

# %%
# the graph of the spline vessels, and its junctions with flow as evidence
ridge_graphs = []     # per fixation: the graph of spline_vessels (see vessel_graph)
ridge_junctions = []  # per fixation: its junctions (see read_junction): type and confidence with flow, and by geometry alone
for number, times in enumerate(saved["times"]):
    density = saved["density"][number]
    graph = vessel_graph(spline_vessels[number], saved["valid"][number], density, vessel_readings[number],
                         saved["pairs"][number], fps, visible=saved["visible"])
    halves = (np.arange(0, len(times), 2), np.arange(1, len(times), 2)) if len(times) >= SPLIT_FRAMES else None
    found, measured, inferred, conflicts = read_junctions(graph, density, saved["static_scores"][number], halves)
    ridge_graphs.append(graph)
    ridge_junctions.append(found)
    kinds = {}
    for junction in found:
        kinds[junction["type"]] = kinds.get(junction["type"], 0) + 1
    print(f"Fixation {number}: {len(found)} junctions on {len(graph['edges'])} edges; {measured} edges have a measured "
          f"direction of flow, the junctions give one to {inferred} more")
    print("   read as: " + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items(), key=lambda item: -item[1])))
    for word, meaning in (("no flow data", "the geometric reading stands"), ("confirms", "flow confirms the geometric reading"),
                          ("contradicts", "flow speaks against it and offers no other"), ("changes", "flow changes the reading")):
        some = [junction for junction in found if junction["flow"] == word]
        if some:
            print(f"   {len(some):5d} junctions: {meaning}; mean confidence "
                  f"{np.mean([junction['confidence_geometry'] for junction in some]):.2f} by geometry -> "
                  f"{np.mean([junction['confidence'] for junction in some]):.2f}")
vessel_readings.clear()  # the kymographs of all vessels are not kept

# show, for the anchor's fixation, the junctions whose confidence flow raised most, and those it lowered most
graph, density = ridge_graphs[anchor], saved["density"][anchor]
typed = [junction for junction in ridge_junctions[anchor] if junction["type"] != "unresolved" and junction["flow"] != "no flow data"]
change = lambda junction: junction["confidence"] - junction["confidence_geometry"]
figure, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for axis in axes.ravel():
    axis.axis("off")
for row, some in zip(axes, (sorted(typed, key=change)[::-1][:8], sorted(typed, key=change)[:8])):
    for axis, junction in zip(row, some):
        axis.imshow(junction_tile(junction, graph["nodes"][junction["node"]], density, DENSITY_RANGE))
        axis.set_title(f"{junction['type']}\n{junction['confidence_geometry']:.2f} -> {junction['confidence']:.2f} "
                       f"({junction['flow']})", fontsize=7)
figure.suptitle("top: junctions whose confidence flow raised most | bottom: those it lowered most "
                "(confidence by geometry -> with flow)", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/flow_vessels/junctions_with_flow.png", dpi=120)
plt.show()
memory_note("junctions")

# %% [markdown]
# ## 30. The harness: every change is measured on synthetic bursts
#
# From here on nothing is changed on the strength of a picture alone. A change is a row in a table: the same synthetic bursts are read the old way and the new way, and both are scored against the truth with `score_reading` (section 28).
#
# **Rules.**
#
# - **Nothing is learned.** No setting is fitted to the synthetic truth. The truth is used for scoring only.
# - **Two sets of bursts.** `HARNESS_SEEDS` are the bursts for developing: they are looked at while choosing between ways. `HELD_OUT_SEEDS` are read only for the numbers that are reported, and nothing is chosen by them.
# - **Every share has a 95% interval**, from resampling the bursts (`with_interval`). Two rows whose intervals overlap widely are not told apart by these bursts.
#
# **The rows** (`HARNESS_WAYS`). The first is section 22's chains, for reference; all others read the ridges of the static score (section 29).
#
# | Row | What it is |
# |---|---|
# | chains of sections 19 to 27 | interactions, stringing, pruning |
# | plain ridges | the ridges, thinned; nothing else |
# | consolidated, keeping short pieces | joined, short side branches and small networks removed, free ends extended |
# | consolidated, dropping short pieces | the same, and every piece of centerline under `SHORT_PIECE` pixels left out |
# | tube rule on plain ridges; consolidated, tube rule | pieces dropped by section 31's rule instead of by length |
# | ..., whole vessels | the flow read along whole vessels (section 32) instead of per piece |
# | ..., whole vessels, likelihood ratio | the same junctions read again with flow as a likelihood ratio (section 33); p is measured on the same set of bursts, without truth |
#
# **Tables kept** for the work that follows, in memory and as files in `vessel_pictures/harness`:
#
# - `harness_junctions` (`junctions.csv`): one row per junction read and per true junction missed, for every row of the table and every burst.
# - `harness_arms` (`arms.csv`): one row per arm of a junction found at a single true bifurcation or crossing: the direction, speed and width read, and the true ones.
#
# The harness needs no data of the real burst except the pruning chosen for it. The cell below only defines; the table is made in section 34, after every way of reading has been defined. In a fresh kernel, run the cell *Open the inputs* of section 29 first, then section 34's cell.

# %%
HARNESS_SEEDS = tuple(range(1, 17))        # bursts for developing: looked at while choosing between ways
HELD_OUT_SEEDS = tuple(range(1001, 1017))  # bursts for the numbers that are reported: nothing is chosen by them
SHORT_PIECE = 24  # pixels; "dropping short pieces" leaves out every piece of centerline shorter than this
# The rows of the ablation table: a name, and how read_burst_ridges is called. None: section 22's chains.
# {"again": row}: the junctions of that row read again with flow as a likelihood ratio.
HARNESS_WAYS = {
    "chains of sections 19 to 27": None,
    "plain ridges": {"consolidate": False},
    "consolidated, keeping short pieces": {"consolidate": True},
    "consolidated, dropping short pieces": {"consolidate": True, "least": SHORT_PIECE},
    "tube rule on plain ridges": {"consolidate": False, "stubs": "by tube"},
    "consolidated, tube rule": {"consolidate": True, "stubs": "by tube"},
    "plain ridges, whole vessels": {"consolidate": False, "speeds": "whole vessels"},
    "consolidated, whole vessels": {"consolidate": True, "speeds": "whole vessels"},
    "plain ridges, whole vessels, likelihood ratio": {"again": "plain ridges, whole vessels"},
    "consolidated, whole vessels, likelihood ratio": {"again": "consolidated, whole vessels"},
}
OUTCOME_COLORS = {"found": (90, 255, 90), "false alarm": (255, 60, 60), "nobody is asked for it": (150, 150, 150),
                  "not a junction": (90, 90, 90)}


def read_way(burst, front, options, pruning, threads=None):
    """Read a synthetic burst one way: `options` are a row of HARNESS_WAYS (not one that reads another row again)."""
    if options is None:
        return read_burst(burst, pruning, threads=threads)
    return read_burst_ridges(burst, front, threads=threads, **options)


def scored_way(burst, read, seed):
    """A reading's score (score_reading) with its seed, and with `geometry`: the junction scores of the same
    junctions read without flow."""
    return {**score_reading(burst, read), "seed": seed,
            "geometry": score_reading(burst, read, kinds="type_geometry")["junctions"]}


def harness_scores(seeds, ways, pruning, threads=None):
    """Read every burst every way and score it against the truth. Returns {way: [one score per burst]} (see
    scored_way), and for every row that reads another again: the counts it took p from, and p from the three-arm
    junctions and from the sides (see reliability). For such a row p is measured over all the bursts given, without
    truth; the junctions of the other row are then read again with it."""
    scores = {way: [] for way in ways}
    again = {way: options["again"] for way, options in ways.items() if options is not None and "again" in options}
    kept = {way: [] for way in again.values()}  # what the second reading needs of the first, without the frames
    for seed in seeds:
        burst = synthetic_burst(seed)
        front = ridge_front(burst, threads)
        for way, options in ways.items():
            if way in again:
                continue
            read = read_way(burst, front, options, pruning, threads)
            scores[way].append(scored_way(burst, read, seed))
            if way in kept:
                kept[way].append(({name: value for name, value in burst.items() if name != "frames"}, front["scores"], read))
    measured = {}
    for way, first in again.items():
        counts = add_counts([read["counts"] for _, _, read in kept[first]])
        p_three, p_sides = reliability(counts)
        measured[way] = (counts, p_three, p_sides)
        p = p_three if np.isfinite(p_three) else p_sides
        for seed, (burst, layers, read) in zip(seeds, kept[first]):
            found = read_junctions(read["graph"], burst["mean"], layers, evidence="directions", p=p)[0]
            scores[way].append(scored_way(burst, {**read, "junctions": found}, seed))
    return scores, measured


def table_of(scores, name):
    """All bursts' and all ways' rows of one of the tables score_reading keeps (`rows`, `arms` or `vessels`) as one
    list, each row with the way and the seed it comes from."""
    return [{"way": way, "seed": score["seed"], **row} for way, bursts in scores.items() for score in bursts
            for row in score[name]]


def save_table(rows, path):
    """Write a list of dicts with the same keys as a csv file."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def share_of(bursts, test, among=lambda row: True):
    """A share over the rows of several bursts (a list of rows per burst) with its 95% interval over bursts: the
    rows that pass `test` among those that pass `among`. Returns (share, low, high, rows counted)."""
    counted = [[row for row in rows if among(row)] for rows in bursts]
    return (*with_interval([sum(bool(test(row)) for row in rows) for rows in counted], [len(rows) for rows in counted]),
            sum(len(rows) for rows in counted))


def told_apart(right, wrong):
    """The chance that one of the values in `right` is larger than one of the values in `wrong` (ties count half):
    1 when they are fully apart, 0.5 when they are not told apart at all."""
    right, wrong = np.asarray(right, dtype=float), np.asarray(wrong, dtype=float)
    if not len(right) or not len(wrong):
        return np.nan
    return float((right[:, None] > wrong[None]).mean() + 0.5 * (right[:, None] == wrong[None]).mean())


def outcome_picture(burst, read):
    """A synthetic burst's mean density with what was read from it: the lines; every true junction as a circle
    (yellow bifurcation, white crossing, gray: nobody is asked for it); every junction read as a dot in the color
    of its outcome (green found, red false alarm, gray not counted), ringed in orange when it is found but typed
    wrong."""
    picture = chain_picture(read["chains"], burst["mean"], 0, np.percentile(burst["mean"], 99.9))
    for junction in burst["junctions"]:
        color = (255, 220, 0) if junction["type"] == "bifurcation" else (255, 255, 255)
        cv2.circle(picture, (int(junction["x"]), int(junction["y"])), 10, color if junction["visible"] else (120, 120, 120), 1)
    for row in score_reading(burst, read)["rows"]:
        if row["outcome"] == "missed":
            continue
        cv2.circle(picture, (int(row["x"]), int(row["y"])), 4, OUTCOME_COLORS[row["outcome"]], -1)
        if row["outcome"] == "found" and not row["type right"]:
            cv2.circle(picture, (int(row["x"]), int(row["y"])), 6, (255, 150, 0), 1)
    return picture
# %% [markdown]
# ## 31. A rule for dropping pieces by the parent's tube (tried, not adopted)
#
# The consolidation of section 29 drops pieces of centerline by their length: side branches under `RIDGE_CHILD` pixels, loose pieces under `LINE_LEAST`, networks under `RIDGE_NETWORK`. A crossing's short pieces can go with them. This section tries a rule that cannot do that (`without_stubs`):
#
# - **Only a piece with a free end can be dropped.** A piece with a junction at both ends never is.
# - **It is judged by its length outside its parent's tube.** A vessel's tube reaches its radius plus `TUBE_BLURS` sigmas of the picture's blur from its centerline (`tube_reach`). What lies inside a parent's tube is the parent's own image seen a second time. Parents are the forks, the pieces with a junction at both ends, and longer pieces with a free end that were kept.
# - **The blur is measured on the picture** (`picture_blur`): no line looks thinner than the blur, so it is the half-width that only `BLUR_SHARE` of the ridge pixels are thinner than.
# - **How much has to lie outside**: `STUB_FIELDS` lengths of the piece's own receptive field (`ELONGATION` x its half-width each). A response no longer than the field that gave it says nothing about a line. This is the rule's one setting. It was set to 2 from that argument, before any sweep.
# - An end where the data ends is not a free end.
#
# **The sweep below is on the real burst, where nothing is known.** For every setting: the length of centerline, its loose ends, and how much of it is found again in the other fixation above chance. The setting is not chosen on the synthetic bursts; those only score it (section 34).
#
# **Result.** Repeatability is flat from 0.25 to 2 field lengths, so 2 is on the flat part and is kept. But the rule does not earn its place: on the real burst it leaves about twice the loose ends of the consolidation by length at slightly lower repeatability, and on the held-out synthetic bursts it finds no more junctions (section 34). Section 29 therefore keeps dropping by length. The rule stays in the code as `stubs="by tube"`.

# %%
# the tube rule's one setting, swept on the real burst: nothing here knows any truth
TUBE_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # field lengths that have to lie outside the parent's tube

def line_numbers(skeletons, least, to_forks):
    """What describes one way of making centerlines, from its skeleton in every fixation: the anchor's length, its
    pieces, its loose ends per 1000 px, and how much of it is found again in the other fixations above chance."""
    lines = [ridge_lines(skeleton, saved["static_strength"][number], saved["static_orientation"][number],
                         saved["half_width"][number], least=least, to_forks=to_forks) for number, skeleton in enumerate(skeletons)]
    picture = draw_chains(lines[anchor], frame_shape) >= 0
    again = [repeated(picture, saved["valid"][anchor], draw_chains(lines[number], frame_shape) >= 0, saved["valid"][number],
                      saved["to anchor"][number]) for number in range(len(lines)) if number != anchor]
    length = sum(chain_length(line) for line in lines[anchor])
    return {"length": length, "pieces": len(lines[anchor]),
            "loose ends per 1000 px": loose_ends_of(lines[anchor], saved["valid"][anchor]) / max(length, 1) * 1000,
            "above chance": float(np.mean(again, axis=0)[2]) if again else np.nan}

fixation_numbers = range(len(saved["times"]))
crests = [ridge_crests(saved["static_strength"][number], saved["static_orientation"][number], flow_masks[number]["mask"])
          for number in fixation_numbers]
print("The blur of the picture, per fixation: "
      + ", ".join(f"{picture_blur(saved['half_width'][number], crests[number]):.2f} px" for number in fixation_numbers))
tube_sweep = {}  # per way of making centerlines: its numbers (see line_numbers)
for name, consolidate in (("plain ridges", False), ("consolidated, by length", True)):
    tube_sweep[name] = line_numbers([ridge_skeleton(saved["static_strength"][number], saved["static_orientation"][number],
                                                    flow_masks[number]["mask"], consolidate) for number in fixation_numbers],
                                    LINE_LEAST, False)
for fields in TUBE_SWEEP:
    tube_sweep[fields] = line_numbers([ridge_skeleton_by_tube(
        saved["static_strength"][number], saved["static_orientation"][number], saved["half_width"][number],
        flow_masks[number]["mask"], saved["valid"][number], True, fields) for number in fixation_numbers], 0, True)
print(f"{'centerlines':44s} {'length':>8s} {'pieces':>7s} {'loose ends per 1000 px':>23s} {'found again above chance':>25s}")
for name, numbers in tube_sweep.items():
    label = name if isinstance(name, str) else f"consolidated, tube rule, {name:.2f} field lengths"
    print(f"{label:44s} {numbers['length']:8.0f} {numbers['pieces']:7d} {numbers['loose ends per 1000 px']:23.1f} "
          f"{numbers['above chance'] * 100:24.0f}%")

figure, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for axis, column, unit in zip(axes, ("loose ends per 1000 px", "above chance"), (1, 100)):
    axis.plot(TUBE_SWEEP, [tube_sweep[fields][column] * unit for fields in TUBE_SWEEP], "o-", color="tab:blue",
              label="consolidated, tube rule")
    for name, color in (("plain ridges", "gray"), ("consolidated, by length", "tab:orange")):
        axis.axhline(tube_sweep[name][column] * unit, color=color, linestyle="--", label=name)
    axis.axvline(STUB_FIELDS, color="tab:blue", linewidth=0.6)
    axis.set_xlabel("field lengths that have to lie outside the parent's tube")
    axis.set_title(column + (" (%)" if unit == 100 else ""), fontsize=10)
axes[0].legend(fontsize=8)
figure.suptitle("The tube rule's setting on the real burst (anchor's fixation); the thin line is the setting used", fontsize=10)
figure.tight_layout()
os.makedirs("vessel_pictures/harness", exist_ok=True)
figure.savefig("vessel_pictures/harness/tube_rule_sweep.png", dpi=110)
plt.show()
del crests
memory_note("tube rule sweep")

# %% [markdown]
# ## 32. Speed along whole vessels
#
# Section 29 cut every centerline at every fork and read a speed per piece. Pieces are short, and neighbouring pieces of one vessel came out with speeds like 96, 119, 588, 140, 482 px/s. Here one kymograph is read per whole vessel (`whole_vessel_flow`).
#
# **Joining** (`join_pieces`). Two pieces are one vessel:
#
# - where the geometric junction reading pairs their arms as a vessel passing through (at a crossing, both pairs);
# - where just two pieces meet end to end.
#
# A bifurcation joins nothing: its three arms are three vessels with three speeds.
#
# **One vessel runs one way.** At every join the direction is read on either side, each from its own window. Where the two run against each other the join is undone.
#
# **A speed with its interval** (`speed_of`). In the kymograph a pattern that moves `shift` pixels per frame draws streaks at the angle phi = arctan(shift) from the time axis; the speed is fps x tan(phi). The angle's spread sigma_phi comes from reading it again on resamplings of the pairs of frames. As a spread of the speed it is
#
# sigma_v = fps x sigma_phi / cos^2(phi),
#
# so the steeper the streaks, the less a speed is worth.
#
# **The sigmas are widened by a factor measured on the burst.** The resampling takes the pairs of frames to be independent. They are not: the same cells are followed through many frames, and on synthetic bursts the sigmas came out about 2.4 times too narrow. So every whole vessel long enough is read again in windows of `SPEED_WINDOW` pixels, and the robust spread of the differences between neighbouring windows, in sigmas of those differences, is the factor (`widen`). No truth goes into it.
#
# **The trackable limit.** A pattern can be found again while it moves by up to half its own length between two frames (section 23): `SPEED_LIMIT` = fps x `PATTERN_LENGTH` / 2. A speed above it is flagged, and neither it nor its direction counts.
#
# **A correction to section 29.** Its spline vessels had their points spaced by the length of the pixel staircase they were fitted to, which overstates distance along an oblique line by up to 8%, and every speed with it. `spline_vessel` now spaces its points by the distance along the spline itself. The speeds of sections 23 and 25 are read along raw chains and still carry that error.
#
# **Result:** `whole_graphs[n]`, `whole_vessels[n]` (chains with `pieces`) and `whole_readings[n]` (per whole vessel: `velocity`, `velocity_sigma`, `direction`, `trackable`, `windows`). The picture and `whole_vessels_fixation_<n>.json` are in `vessel_pictures/flow_vessels`.

# %%
SPEED_WINDOW = TRAVEL_LENGTH  # pixels; a whole vessel is read again in windows of at least this length, to compare speeds
SPEED_AGREE = 2.0             # two speeds agree when they differ by no more than this many sigmas of their difference
SAME_SEGMENT = 0.7            # a whole vessel is a true segment's when this share of its points lies within REPEAT_WITHIN of it
WIDEN_LEAST = 8               # pairs of neighbouring windows a burst needs for the widening of its speeds' intervals

def join_pieces(graph, undone=()):
    """Join the pieces (edges) of a graph into whole vessels.

    Two pieces are one vessel where the junction reading says so: at every junction, the two arms of a group of two
    in its partition are a vessel passing through (at a crossing both pairs). And where just two pieces meet end
    to end, they are one vessel. A bifurcation joins nothing: its three arms are three vessels with three speeds.

    `undone` are joins that are not to be made: pairs of (edge, end), as the paths name them.

    Returns the whole vessels as paths: lists of (edge, way, joined, junction), `way` +1 where the edge's own points
    run the vessel's way and -1 where they run against it, `joined` the join the vessel came into this edge through
    (a pair of (edge, end); None for a path's first edge), `junction` whether that join passes a junction (and is
    not a simple join). Every edge is in exactly one path.
    """
    partner = {}  # (edge, end) -> the (edge, end) it runs on into
    through = set()  # the ends that are joined through a junction

    def join(first, second, junction=False):
        if first[0] != second[0] and frozenset((first, second)) not in undone:
            partner[first], partner[second] = second, first
            if junction:
                through.update((first, second))

    for node in graph["nodes"]:
        if node.get("gone"):
            continue
        if "arms" in node:
            arms = node["arms"]
            groups = [tuple(range(len(arms)))] if len(arms) == 2 else node.get("partition", [])
            for group in groups:
                if len(group) == 2:
                    join(*((arms[arm]["edge"], arms[arm]["end"]) for arm in group), junction=len(arms) > 2)
        elif len(node["ends"]) == 2:
            join(*node["ends"])
    paths, seen = [], set()
    for number in range(len(graph["edges"])):
        if number in seen:
            continue
        # back from this edge's first end to where its vessel starts (or once round, if the vessel is a ring)
        edge, end = number, 0
        for _ in range(len(graph["edges"])):
            if (edge, end) not in partner:
                break
            other, other_end = partner[(edge, end)]
            edge, end = other, 1 - other_end
            if edge == number:
                break
        path = []
        while edge not in seen:
            seen.add(edge)
            joined = frozenset(((edge, end), partner[(edge, end)])) if path else None
            path.append((edge, 1 if end == 0 else -1, joined, bool(path) and (edge, end) in through))
            if (edge, 1 - end) not in partner:
                break
            edge, end = partner[(edge, 1 - end)]
        paths.append(path)
    return paths

def whole_vessel(path, edges):
    """The pieces of a path (join_pieces) as one chain, its points in the order the path runs. Where two pieces do
    not touch (a junction's core lies between them) the gap is filled with points a pixel apart on a straight line,
    marked `bridge`. `pieces` says where each edge lies in it: (edge, first point, last point, way). `passes` are its
    joins: (last point before, first point after, the join, whether it passes a junction there)."""
    names = ("x", "y", "theta", "radius", "strength")
    parts = {name: [] for name in names}
    bridge, pieces, passes, count = [], [], [], 0
    for edge_number, way, joined, junction in path:
        edge = edges[edge_number]
        piece = {name: edge[name][::way] for name in names}
        if count:
            dx, dy = piece["x"][0] - parts["x"][-1][-1], piece["y"][0] - parts["y"][-1][-1]
            steps = int(np.ceil(np.hypot(dx, dy))) - 1
            if steps > 0:
                along = np.arange(1, steps + 1, dtype=np.float32) / (steps + 1)
                fill = {"x": parts["x"][-1][-1] + along * dx, "y": parts["y"][-1][-1] + along * dy,
                        "theta": np.full(steps, np.arctan2(dy, dx) % np.pi),
                        "radius": np.full(steps, (parts["radius"][-1][-1] + piece["radius"][0]) / 2),
                        "strength": np.full(steps, min(parts["strength"][-1][-1], piece["strength"][0]))}
                for name in names:
                    parts[name].append(fill[name].astype(np.float32))
                bridge.append(np.ones(steps, dtype=bool))
                count += steps
        if joined is not None:
            passes.append((pieces[-1][2], count, joined, junction))
        pieces.append((edge_number, count, count + len(piece["x"]) - 1, way))
        for name in names:
            parts[name].append(np.asarray(piece[name], dtype=np.float32))
        bridge.append(np.zeros(len(piece["x"]), dtype=bool))
        count += len(piece["x"])
    vessel = {name: np.concatenate(parts[name]) for name in names}
    vessel.update({"bridge": np.concatenate(bridge), "closed": False, "pieces": pieces, "passes": passes})
    return vessel

def speeds_along(mass, pairs, fps, weights=None):
    """Read a whole vessel's speed again in windows along it, to see whether one vessel has one speed: as many
    windows of equal length as fit, each at least SPEED_WINDOW points. Returns one speed_of per window (none when
    the vessel is shorter than two windows)."""
    count = len(mass) // SPEED_WINDOW
    if count < 2:
        return []
    bounds = np.linspace(0, len(mass), count + 1).astype(int)
    return [speed_of(mass[first:last], pairs, fps, weights) for first, last in zip(bounds[:-1], bounds[1:])]

def whole_vessel_flow(graph, density, frame_at, times, pairs, fps, limit=None, visible=None, threads=None):
    """Task 3: read the flow of a graph along whole vessels instead of piece by piece.

    The graph's junctions have to be read from their geometry already (read_junctions without flow), so that it is
    known which arms pass through. The pieces are joined (join_pieces), one kymograph is read per whole vessel
    (segment_readings), and every speed gets its interval (speed_of). A speed above `limit` (pixels per second; by
    default this frame rate's limit for the pattern found here, fps x half the pattern's length) is flagged: the
    pattern moves too far between two frames to be followed, so neither the speed nor the direction counts (the
    direction is kept apart as `direction flagged`).
    Every edge is then given its vessel's signals, and the junctions' arms are read again with them.

    One vessel runs one way. At every join of a vessel the direction is therefore read on either side of it, each
    from its own window of up to SPEED_WINDOW points. Where the two run against each other the join is undone, and
    all whole vessels are made and read once more (the graph's `undone` counts these joins).

    The speeds' sigmas are widened by one factor for the whole burst (the graph's `widen`): the robust spread of
    the differences between neighbouring windows of one vessel, in sigmas of those differences, where that is
    above 1 and the burst has at least WIDEN_LEAST such pairs.

    Returns the whole vessels (chains with `pieces`), their readings (with `velocity_sigma`, `trackable`,
    `windows`: the speeds read again along the vessel, `sides`: the directions read on either side of every
    junction the vessel passes (`sides flagged`: the same without the limit), and without their kymographs), and the
    limit used.
    The graph gets `readings`, one per edge.
    """
    edges = graph["edges"]
    undone, given = set(), limit
    for attempt in range(2):
        vessels = [whole_vessel(path, edges) for path in join_pieces(graph, undone)]
        readings, _ = segment_readings(vessels, frame_at, times, pairs, fps, visible=visible, threads=threads)
        if given is None:
            long_ones = [reading["M"] for reading in readings if reading["length"] >= TRAVEL_LENGTH]
            limit = fps * pattern_length(long_ones) / 2 if long_ones else np.inf
        against = set()
        for vessel, reading in zip(vessels, readings):
            lucky = reading["lucky"] if (reading["lucky"] > 0).sum() >= 0.3 * len(times) else None
            # at every join of the vessel: the direction read on either side of it, each from its own window
            reading["sides"], reading["sides flagged"] = [], []
            for before, after, joined, junction in vessel["passes"]:
                sides = (reading["M"][max(0, before + 1 - SPEED_WINDOW):before + 1], reading["M"][after:after + SPEED_WINDOW])
                speeds = [speed_of(side, pairs, fps, lucky) if len(side) >= VELOCITY_LENGTH else None for side in sides]
                ways = tuple(clear_direction(speed, limit) if speed is not None else 0 for speed in speeds)
                if junction:
                    reading["sides"].append(ways)
                    reading["sides flagged"].append(tuple(clear_direction(speed) if speed is not None else 0 for speed in speeds))
                if ways[0] * ways[1] < 0:
                    against.add(joined)
        if attempt or not against:
            break
        undone |= against  # one vessel runs one way: these joins are not made, and everything is read again
    graph["readings"], graph["undone"] = [None] * len(edges), len(undone)
    apart = []  # neighbouring windows of one vessel: how far their speeds are apart, in sigmas of their difference
    for vessel, reading in zip(vessels, readings):
        reading["trackable"] = bool(not np.isfinite(reading["velocity"]) or abs(reading["velocity"]) <= limit)
        reading["direction flagged"] = 0
        if not reading["trackable"]:  # the direction is kept apart: it does not count, but its reliability can be measured
            reading["direction flagged"], reading["direction"] = reading["direction"], 0
        lucky = reading["lucky"] if (reading["lucky"] > 0).sum() >= 0.3 * len(times) else None
        reading["windows"] = speeds_along(reading["M"], pairs, fps, lucky) if reading["direction"] != 0 else []
        windows = [window for window in reading["windows"] if np.isfinite(window["velocity"]) and window["sigma"] > 0]
        apart += [abs(a["velocity"] - b["velocity"]) / np.hypot(a["sigma"], b["sigma"]) for a, b in zip(windows[:-1], windows[1:])]
    # The resampling behind a speed's sigma takes the pairs of frames to be independent of each other. They are not:
    # the same cells are followed through many frames. How much too narrow the sigmas come out is measured on the
    # burst itself: two neighbouring windows of one vessel should differ by about one sigma of their difference.
    widen = max(1.0, 1.4826 * float(np.median(apart))) if len(apart) >= WIDEN_LEAST else 1.0
    graph["widen"], graph["window pairs"] = widen, len(apart)
    for number, (vessel, reading) in enumerate(zip(vessels, readings)):
        reading["velocity_sigma"] *= widen
        for window in reading["windows"]:
            window["sigma"] *= widen
        for edge, first, last, way in vessel["pieces"]:
            cut = {name: reading[name] for name in ("times", "radius", "pulse_amplitude", "pulse_phase", "blur_trace",
                                                    "width_trace", "sharpest", "lucky", "velocity_confidence",
                                                    "velocity_sigma", "trackable")}
            cut["M"] = reading["M"][first:last + 1][::way]
            level = mean_of_data(mean_of_data(cut["M"], axis=0), axis=0)
            cut.update({"mass": float(level) if np.isfinite(level) else reading["mass"], "vessel": number,
                        "velocity": way * reading["velocity"], "direction": way * reading["direction"],
                        "direction flagged": way * reading["direction flagged"]})
            graph["readings"][edge] = cut
    for node in graph["nodes"]:
        if "arms" in node and not node.get("gone"):
            read_arms(node, edges, graph["readings"], density)
    for reading in readings + graph["readings"]:
        del reading["M"]  # every arm has its own line mass over time; the kymographs are not kept
    return vessels, readings, float(limit)

def downstream_whole(vessel, reading):
    """A whole vessel with a clear speed as draw_flow takes it: its points in the order the blood passes them, their
    tangent pointing downstream, its speed and the speed's sigma."""
    way = 1 if reading["direction"] > 0 else -1
    x, y = vessel["x"][::way], vessel["y"][::way]
    return {"x": x, "y": y, "tangent": np.arctan2(np.gradient(y), np.gradient(x)) if len(x) > 1 else np.zeros(len(x)),
            "velocity": abs(reading["velocity"]), "velocity_sigma": reading["velocity_sigma"]}

def vessel_rows(burst, vessels, readings):
    """For scoring on a synthetic burst: one row per whole vessel with a clear speed that lies on one true segment
    (SAME_SEGMENT of its points within REPEAT_WITHIN of it): the speed read with its sigma, the true speed, whether
    the direction is right, and how its windows compare (pairs of neighbouring windows, and how many of them agree
    within SPEED_AGREE sigmas of their difference)."""
    trees = [scipy.spatial.cKDTree(segment["points"]) for segment in burst["segments"]]
    rows = []
    for vessel, reading in zip(vessels, readings):
        if reading["direction"] == 0:
            continue
        points = np.column_stack([vessel["x"], vessel["y"]])
        hits = [tree.query(points, distance_upper_bound=REPEAT_WITHIN) for tree in trees]
        share = [np.isfinite(distance).mean() for distance, _ in hits]
        best = int(np.argmax(share))
        if share[best] < SAME_SEGMENT:
            continue
        on = np.flatnonzero(np.isfinite(hits[best][0]))
        same_way = np.sign(hits[best][1][on[-1]] - hits[best][1][on[0]])  # does the vessel run the way the segment's points do?
        true = burst["segments"][best]["speed"] * burst["fps"]
        windows = [window for window in reading["windows"] if np.isfinite(window["velocity"])]
        close = [abs(a["velocity"] - b["velocity"]) <= SPEED_AGREE * np.hypot(a["sigma"], b["sigma"])
                 for a, b in zip(windows[:-1], windows[1:])]
        rows.append({"segment": best, "length": reading["length"], "speed": abs(reading["velocity"]),
                     "sigma": reading["velocity_sigma"], "true speed": abs(true),
                     "direction right": bool(np.sign(reading["velocity"]) * same_way == np.sign(true)),
                     "window pairs": len(close), "window pairs that agree": int(sum(close))})
    return rows

# %%
SPEED_LIMIT = fps * saved["pattern length"] / 2  # pixels per second; section 23's limit for a pattern of that length
whole_graphs = []    # per fixation: the graph of spline_vessels, its arms carrying the flow of their whole vessels
whole_vessels = []   # per fixation: the whole vessels (see whole_vessel)
whole_readings = []  # per fixation: their temporal signals (see whole_vessel_flow)
print(f"The pattern along a vessel is {saved['pattern length']:.1f} px long (section 23): speeds above {SPEED_LIMIT:.0f} px/s "
      f"cannot be followed at {fps:.1f} frames per second")
for number, times in enumerate(saved["times"]):
    density = saved["density"][number]
    graph = vessel_graph(spline_vessels[number], saved["valid"][number], density)
    read_junctions(graph, density, saved["static_scores"][number], flow=False)  # from the geometry: which arms pass through
    vessels, readings, _ = whole_vessel_flow(graph, density, lambda position: saved["frames"][number][position], times,
                                             saved["pairs"][number], fps, limit=SPEED_LIMIT, visible=saved["visible"],
                                             threads=workers)
    whole_graphs.append(graph)
    whole_vessels.append(vessels)
    whole_readings.append(readings)
    lengths = np.array([reading["length"] for reading in readings])
    clear = np.array([reading["direction"] != 0 for reading in readings])
    flagged = np.array([not reading["trackable"] for reading in readings])
    speeds = np.array([abs(reading["velocity"]) for reading in readings])
    sigmas = np.array([reading["velocity_sigma"] for reading in readings])
    windows = [[window for window in reading["windows"] if np.isfinite(window["velocity"])] for reading in readings]
    close = [abs(a["velocity"] - b["velocity"]) <= SPEED_AGREE * np.hypot(a["sigma"], b["sigma"])
             for some in windows for a, b in zip(some[:-1], some[1:])]
    spread = [np.ptp([abs(window["velocity"]) for window in some]) / np.median([abs(window["velocity"]) for window in some])
              for some in windows if len(some) >= 2]
    print(f"Fixation {number}: {len(graph['edges'])} pieces are {len(vessels)} whole vessels; {graph['undone']} joins were "
          f"undone because their two sides ran against each other")
    print(f"   a clear speed on {clear.sum()} whole vessels with {lengths[clear].sum() / lengths.sum() * 100:.0f}% of the "
          f"length: median {np.median(speeds[clear]):.0f} px/s, median sigma {np.median(sigmas[clear]):.0f} px/s "
          f"({np.median(sigmas[clear] / speeds[clear]) * 100:.0f}% of the speed) | {flagged.sum()} more with "
          f"{lengths[flagged].sum() / lengths.sum() * 100:.0f}% of the length are above the limit and flagged")
    print(f"   the sigmas were widened {graph['widen']:.1f} times, from {graph['window pairs']} pairs of neighbouring windows | "
          f"after that {np.mean(close) * 100 if close else np.nan:.0f}% of the {len(close)} pairs agree within {SPEED_AGREE:.0f} "
          f"sigmas | along one vessel the windows' speeds span {np.median(spread) * 100 if spread else np.nan:.0f}% of their "
          f"median (median over {len(spread)} vessels)")
    export = [{"id": identity, "length": round(reading["length"], 1), "pieces": len(vessel["pieces"]),
               "x": np.round(vessel["x"][::5], 1).tolist(), "y": np.round(vessel["y"][::5], 1).tolist(),
               "has_speed": bool(reading["direction"] != 0), "above_limit": bool(not reading["trackable"]),
               "speed_px_per_s": None if reading["direction"] == 0 else round(abs(reading["velocity"]), 1),
               "sigma_px_per_s": None if reading["direction"] == 0 else round(reading["velocity_sigma"], 1),
               "runs_along_points": int(reading["direction"])}
              for identity, (vessel, reading) in enumerate(zip(vessels, readings))]
    with open(f"vessel_pictures/flow_vessels/whole_vessels_fixation_{number}.json", "w") as file:
        json.dump(export, file)

# the anchor's fixation: whole vessels with a clear speed in the color of their direction, each with speed +- sigma;
# flagged ones in white, the others in pale blue
shade = np.uint8(np.clip((np.nan_to_num(saved["density"][anchor]) - DENSITY_RANGE[0])
                         / (DENSITY_RANGE[1] - DENSITY_RANGE[0]), 0, 1) * 110)
picture = np.dstack([shade, shade, shade])
for vessel, reading in zip(whole_vessels[anchor], whole_readings[anchor]):
    if reading["direction"] != 0:
        draw_flow(picture, downstream_whole(vessel, reading), label=reading["length"] >= ARROW_EVERY)
    else:
        points = np.rint(np.column_stack([vessel["x"], vessel["y"]])).astype(np.int32)
        cv2.polylines(picture, points[None], False, (120, 150, 200) if reading["trackable"] else (235, 235, 235), 1, cv2.LINE_AA)
direction_key(picture)
Image.fromarray(picture).save("vessel_pictures/flow_vessels/whole_vessels.png")
figure, axis = plt.subplots(figsize=(16, 10))
axis.imshow(picture, interpolation="antialiased")
axis.set_title("whole vessels | color: the direction of flow, arrows point downstream, numbers: speed +- sigma in px/s (on vessels "
               f"of {ARROW_EVERY} px and more) | white: a speed above the trackable limit | pale blue: no clear speed", fontsize=10)
axis.axis("off")
figure.tight_layout()
plt.show()
del picture, shade
memory_note("whole vessels")

# %% [markdown]
# ## 33. What the directions of flow are worth: reliability, evidence, parents
#
# **How reliable is a measured direction, without knowing any truth?** (`direction_counts`, `reliability`). Let p be the share of measured directions that are right, each independently of the others.
#
# - At a bifurcation one arm runs against the other two. All three in, or all three out, cannot be; it is read when exactly the wrong arms are wrong, with probability p(1 - p). From the share f of three-arm junctions with all arms measured that come out so: p = (1 + sqrt(1 - 4f)) / 2.
# - One vessel runs one way. The directions read on the two sides of a junction it passes, each from its own window, agree with probability p^2 + (1 - p)^2. From the share a that agree: p = (1 + sqrt(2a - 1)) / 2.
#
# The two p are not of the same thing: the first is of whole vessels, the second of windows of `SPEED_WINDOW` pixels. Section 34 checks both on synthetic bursts, where the truth gives p directly.
#
# **Flow as a likelihood ratio** (`read_junction` with `evidence="directions"`). Section 27 let flow confirm or contradict a reading through five tests. Here only the measured directions count, and they count as evidence between two readings:
#
# - The two readings are the best and the second-best that the geometry gives.
# - Under a reading, the blood has to come in and go out of every group of arms; every pattern that does is taken to be as likely as any other; a measured direction is right with probability p (`direction_likelihood`).
# - The ratio of the two likelihoods updates the geometric confidence, taken as the probability of the best reading against the second-best.
# - **A pattern that fits both readings changes nothing.** Three arms can only be one bifurcation, so there flow changes nothing at all; that all three run in is then put down to a wrong direction, not to a wrong junction.
#
# **The parent of a bifurcation** (`parent_arm`). Three rules are compared: the widest arm; the arm whose blood runs against that of the other two; and the second where it speaks, else the first. On the real burst only their agreement can be printed; section 34 scores them.
#
# **Result:** `evidence_junctions[n]`: every junction with `type`, `confidence`, `flow` (`confirms`, `speaks against`, `fits both`, `changes`, `no other reading`, `no flow data`), `flow_support` (the likelihood ratio), `impossible` (its measured directions cannot be under its reading) and `parent` (the trunk's arm by direction, where that rule speaks).

# %%
def direction_counts(graph, junctions, readings, flagged=False):
    """Count what says how reliable a measured direction of flow is, without knowing any truth.

      three arms  junctions with three arms whose directions are all measured, and how many of them have all three
                  in or all three out. That cannot be, so at least one direction is wrong there.
      sides       junctions that a whole vessel passes (`readings` of whole_vessel_flow) with a direction measured
                  on either side, each from its own window, and how many of them agree. One vessel runs one way.
    With `flagged` only what involves a direction read from a speed above the trackable limit is counted: the
    three-arm junctions with at least one such arm, and the sides of which at least one is such.
    Returns a dict of the four counts.
    """
    counts = {"three arms": 0, "all in or all out": 0, "sides": 0, "sides that agree": 0}
    for junction in junctions:
        arms = graph["nodes"][junction["node"]]["arms"]
        out = [measured_out(arm, flagged) for arm in arms]
        if len(out) == 3 and 0 not in out and (not flagged or 0 in [measured_out(arm) for arm in arms]):
            counts["three arms"] += 1
            counts["all in or all out"] += abs(sum(out)) == 3
    for reading in readings:
        within = reading.get("sides", [])
        for place, (before, after) in enumerate(reading.get("sides flagged", []) if flagged else within):
            if flagged and 0 not in within[place]:
                continue
            if before != 0 and after != 0:
                counts["sides"] += 1
                counts["sides that agree"] += before == after
    return counts

def reliability(counts):
    """Turn direction_counts into p, the share of measured directions that are right, in two independent ways.

    If each direction is right with probability p, independently: at a bifurcation (one arm against two) all three
    come out in, or all three out, with probability p(1 - p), so p = (1 + sqrt(1 - 4 f)) / 2 from that share f;
    and the two sides of one vessel agree with probability p^2 + (1 - p)^2, so p = (1 + sqrt(2 a - 1)) / 2 from
    that share a. Each share is taken with half a count added to either outcome, so that none of a few junctions
    being wrong does not make p exactly 1. Returns (p from the three-arm junctions, p from the sides); NaN where
    there is nothing to count.
    """
    wrong = (counts["all in or all out"] + 0.5) / (counts["three arms"] + 1) if counts["three arms"] else np.nan
    agree = (counts["sides that agree"] + 0.5) / (counts["sides"] + 1) if counts["sides"] else np.nan
    return (float((1 + np.sqrt(max(1 - 4 * wrong, 0))) / 2) if np.isfinite(wrong) else np.nan,
            float((1 + np.sqrt(max(2 * agree - 1, 0))) / 2) if np.isfinite(agree) else np.nan)

def add_counts(many):
    """Add up the counts of several bursts."""
    return {name: sum(counts[name] for counts in many) for name in many[0]}

def parent_arm(widths, out, rule):
    """Which of a bifurcation's three arms is the trunk (its number), by one of three rules:

      widest     the widest arm
      direction  the arm whose blood runs against that of the other two; None unless all three directions are
                 measured and one runs against two
      both       by direction where that rule speaks, else the widest
    """
    widest = int(np.argmax(widths))
    against = None
    if 0 not in out and abs(sum(out)) == 1:
        against = int(np.flatnonzero(np.array(out) == -np.sign(sum(out)))[0])
    return {"widest": widest, "direction": against, "both": widest if against is None else against}[rule]

def parent_scores(arms):
    """Score the three rules for the trunk on the arms table of synthetic bursts (score_reading): for every junction
    found at a true bifurcation with three arms, all matched to their true limbs, whether the rule names the true
    trunk. Returns {rule: [(seed, 1 if it named the trunk else 0), ...]}, one entry per junction the rule speaks at."""
    by_junction = {}
    for arm in arms:
        if arm["true"] == "bifurcation":
            by_junction.setdefault((arm.get("way", ""), arm.get("seed", 0), arm["junction"]), []).append(arm)
    scores = {rule: [] for rule in ("widest", "direction", "both")}
    for members in by_junction.values():
        if len(members) != 3 or sorted(arm["limb"] for arm in members) != ["daughter", "daughter", "trunk"]:
            continue
        members = sorted(members, key=lambda arm: arm["arm"])
        trunk = next(index for index, arm in enumerate(members) if arm["limb"] == "trunk")
        out = [0 if arm["inferred"] else arm["out"] for arm in members]
        for rule in scores:
            named = parent_arm([arm["width"] for arm in members], out, rule)
            if named is not None:
                scores[rule].append((members[0].get("seed", 0), int(named == trunk)))
    return scores

# %%
evidence_junctions = []    # per fixation: the junctions of whole_graphs, read with flow as a likelihood ratio
direction_reliability = []  # per fixation: the counts, and p from the three-arm junctions and from the sides
for number, times in enumerate(saved["times"]):
    graph, density, scores = whole_graphs[number], saved["density"][number], saved["static_scores"][number]
    halves = (np.arange(0, len(times), 2), np.arange(1, len(times), 2)) if len(times) >= SPLIT_FRAMES else None
    by_tests = read_junctions(graph, density, scores, halves)[0]  # section 27's tests, on the flow of whole vessels
    counts = direction_counts(graph, by_tests, whole_readings[number])
    p_three, p_sides = reliability(counts)
    direction_reliability.append((counts, p_three, p_sides))
    p = p_three if np.isfinite(p_three) else p_sides
    found = read_junctions(graph, density, scores, halves, evidence="directions", p=p)[0]
    evidence_junctions.append(found)
    print(f"Fixation {number}: {counts['all in or all out']} of {counts['three arms']} three-arm junctions with all arms "
          f"measured are all in or all out: p = {p_three:.2f} | {counts['sides that agree']} of {counts['sides']} junctions "
          f"that a vessel passes have the same direction on both sides: p = {p_sides:.2f}")
    above = direction_counts(graph, by_tests, whole_readings[number], flagged=True)
    print(f"   directions from speeds above the limit, which do not count: {above['all in or all out']} of {above['three arms']} "
          f"three-arm junctions with such an arm are all in or all out: p = {reliability(above)[0]:.2f} | "
          f"{above['sides that agree']} of {above['sides']} pairs of sides with such a side agree: p = {reliability(above)[1]:.2f}")
    if not np.isfinite(p):
        print("   too few junctions with measured directions to say how reliable a direction is: flow changes no reading here")
    words = {}
    for junction in found:
        words[junction["flow"]] = words.get(junction["flow"], 0) + 1
    print(f"   {len(found)} junctions, flow as a likelihood ratio with p = {p:.2f}: "
          + ", ".join(f"{count} {word}" for word, count in sorted(words.items(), key=lambda item: -item[1])))
    moved = [junction for junction in found if junction["flow"] in ("confirms", "speaks against", "changes")]
    if moved:
        print(f"   where flow speaks for or against, the mean confidence goes from "
              f"{np.mean([junction['confidence_geometry'] for junction in moved]):.2f} to "
              f"{np.mean([junction['confidence'] for junction in moved]):.2f}; "
              f"{sum(junction['impossible'] for junction in found)} junctions have directions that cannot be under their reading")
    # what became of the contradictions: section 29 read flow per piece and by the tests
    old = {junction["node"]: junction for junction in ridge_junctions[number]}
    tests = {junction["node"]: junction for junction in by_tests}
    new = {junction["node"]: junction for junction in found}
    were = [node for node, junction in old.items() if junction["flow"] == "contradicts" and node in new]
    print(f"   of the {sum(junction['flow'] == 'contradicts' for junction in old.values())} junctions that flow contradicted "
          f"in section 29 (per piece, by the tests): with the flow of whole vessels the tests still contradict "
          f"{sum(tests[node]['flow'] == 'contradicts' for node in were)}; as a likelihood ratio flow speaks against "
          f"{sum(new[node]['flow'] == 'speaks against' for node in were)}, changes {sum(new[node]['flow'] == 'changes' for node in were)}, "
          f"and {sum(new[node]['impossible'] for node in were)} have directions that cannot be under their reading")
    # the trunk of every junction read as a bifurcation: by width, and by direction where that rule speaks
    spoken = same = bifurcations = 0
    for junction in found:
        arms = graph["nodes"][junction["node"]]["arms"]
        if junction["type"] != "bifurcation" or len(arms) != 3:
            continue
        bifurcations += 1
        out = [measured_out(arm) for arm in arms]
        junction["parent"] = parent_arm([arm["width"] for arm in arms], out, "direction")
        if junction["parent"] is not None:
            spoken += 1
            same += junction["parent"] == parent_arm([arm["width"] for arm in arms], out, "widest")
    print(f"   the trunk of {bifurcations} bifurcations: the direction rule speaks at {spoken}; at {same} of them "
          f"({same / max(spoken, 1) * 100:.0f}%) it names the widest arm (three arms alike would agree at 33%)")
    os.makedirs("vessel_pictures/junctions", exist_ok=True)
    with open(f"vessel_pictures/junctions/whole_vessel_junctions_fixation_{number}.json", "w") as file:
        json.dump([{name: (junction[name] if not isinstance(junction[name], float) or np.isfinite(junction[name]) else None)
                    for name in ("x", "y", "arms", "type", "type_geometry", "confidence", "confidence_geometry", "flow",
                                 "flow_support", "impossible", "directions", "partition")} for junction in found], file)

# show, for the anchor's fixation, the junctions the likelihood ratio raised most and those it lowered most
graph, density = whole_graphs[anchor], saved["density"][anchor]
moved = [junction for junction in evidence_junctions[anchor] if junction["flow"] in ("confirms", "speaks against", "changes")]
change = lambda junction: junction["confidence"] - junction["confidence_geometry"]
figure, axes = plt.subplots(2, 8, figsize=(16, 4.6))
for axis in axes.ravel():
    axis.axis("off")
for row, some in zip(axes, (sorted(moved, key=change)[::-1][:8], sorted(moved, key=change)[:8]) if moved else ()):
    for axis, junction in zip(row, some):
        axis.imshow(junction_tile(junction, graph["nodes"][junction["node"]], density, DENSITY_RANGE))
        axis.set_title(f"{junction['type']}\n{junction['confidence_geometry']:.2f} -> {junction['confidence']:.2f} "
                       f"(ratio {junction['flow_support']:.2f})", fontsize=7)
figure.suptitle("top: junctions whose confidence the directions raised most | bottom: those they lowered most "
                "(confidence by geometry -> with flow, and the likelihood ratio)", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/flow_vessels/junctions_likelihood_ratio.png", dpi=120)
plt.show()
memory_note("flow evidence")

# %% [markdown]
# ## 34. The ablation table
#
# Every row of `HARNESS_WAYS` on the bursts for developing and on the bursts held out (section 30 has the rules). The numbers to quote are the held-out ones.
#
# After the table, what each of sections 32 and 33 has to show on synthetic bursts, where the truth is known:
#
# - **Speeds** (section 32): how far the speed of a whole vessel is from the true one; how often the true speed lies within 2 sigma; how often neighbouring windows of one vessel agree.
# - **Reliability** (section 33): p as measured without truth, next to the share of directions that are in fact right.
# - **Evidence** (section 33): the confidence of the junctions found whose type is right against those whose type is wrong, by geometry alone and with flow, and how well each tells the two apart (1: fully, 0.5: not at all).
# - **Parents** (section 33): how often each of the three rules names the true trunk.
#
# **Tables kept**, in memory and as files in `vessel_pictures/harness`: `harness_junctions` (`junctions.csv`), `harness_arms` (`arms.csv`), `harness_vessels` (`vessels.csv`) and the table itself (`ablation.csv`).

# %%
harness = {}     # per set of bursts ("development", "held out"): {way: [one score per burst]} (see harness_scores)
harness_p = {}   # per set of bursts: {likelihood-ratio row: (counts, p from three-arm junctions, p from the sides)}
for name, seeds in (("development", HARNESS_SEEDS), ("held out", HELD_OUT_SEEDS)):
    harness[name], harness_p[name] = harness_scores(seeds, HARNESS_WAYS, saved["chain pruning"], threads=workers)
    memory_note(f"{name}: {len(seeds)} bursts read {len(HARNESS_WAYS)} ways")
ablation = {name: {way: ablation_row(scores) for way, scores in harness[name].items()} for name in harness}
first = next(iter(HARNESS_WAYS))
for name in harness:
    print(f"\n{name}: {len(harness[name][first])} bursts with {ablation[name][first]['true junctions']} true junctions that "
          f"are asked for. Shares in percent, with their 95% interval over bursts:")
    print_ablation(ablation[name])
    print("   junctions read, and the share of the found bifurcations and crossings that are typed right:")
    for way, row in ablation[name].items():
        print(f"      {way:48s} {row['junctions read']:4d} read | bifurcations {row['bifurcations typed right'][0] * 100:3.0f}%, "
              f"crossings {row['crossings typed right'][0] * 100:3.0f}%")

interval = lambda value: f"{value[0] * 100:.0f}% ({value[1] * 100:.0f} to {value[2] * 100:.0f}; n={value[3]})"
for name in harness:
    print(f"\n{name}, section 32: the speed of whole vessels with a clear speed that lie on one true segment")
    for way, scores in harness[name].items():
        vessels = [score["vessels"] for score in scores]
        every = [row for rows in vessels for row in rows]
        if not every:
            continue
        errors = [abs(row["speed"] - row["true speed"]) / row["true speed"] for row in every]
        pairs_all = sum(row["window pairs"] for row in every)
        print(f"   {way:48s} median error {np.median(errors) * 100:.0f}% of the true speed, 90th percentile "
              f"{np.percentile(errors, 90) * 100:.0f}% | median sigma {np.median([row['sigma'] / row['speed'] for row in every]) * 100:.0f}% "
              f"| true speed within 2 sigma: {interval(share_of(vessels, lambda row: abs(row['speed'] - row['true speed']) <= 2 * row['sigma']))} "
              f"| direction right: {interval(share_of(vessels, lambda row: row['direction right']))} | neighbouring windows "
              f"that agree: {sum(row['window pairs that agree'] for row in every)} of {pairs_all}")
    print(f"{name}, the flow at the arms of the junctions found (measured directions): per piece against whole vessels")
    for way, scores in harness[name].items():
        arms = [[arm for arm in score["arms"] if arm["out"] != 0 and not arm["inferred"]] for score in scores]
        every = [arm for rows in arms for arm in rows]
        if every and HARNESS_WAYS[way] is not None and "again" not in HARNESS_WAYS[way]:
            errors = [abs(arm["speed"] - arm["true speed"]) / arm["true speed"] for arm in every]
            print(f"   {way:48s} direction right: {interval(share_of(arms, lambda arm: arm['out'] == arm['true out']))} | "
                  f"speed error: median {np.median(errors) * 100:.0f}%, 90th percentile {np.percentile(errors, 90) * 100:.0f}%")
    print(f"{name}, section 33: how reliable a measured direction is")
    for way, (counts, p_three, p_sides) in harness_p[name].items():
        base = HARNESS_WAYS[way]["again"]
        arms = [arm for score in harness[name][base] for arm in score["arms"] if arm["out"] != 0 and not arm["inferred"]]
        vessels = [row for score in harness[name][base] for row in score["vessels"]]
        print(f"   {base:34s} without truth: {counts['all in or all out']} of {counts['three arms']} three-arm junctions are all "
              f"in or all out, p = {p_three:.2f}; {counts['sides that agree']} of {counts['sides']} pairs of sides agree, p = "
              f"{p_sides:.2f} | with truth: {np.mean([arm['out'] == arm['true out'] for arm in arms]):.2f} of {len(arms)} "
              f"arms, {np.mean([row['direction right'] for row in vessels]):.2f} of {len(vessels)} whole vessels")
    print(f"{name}, section 33: the confidence of the junctions found, type right against type wrong")
    for way, scores in harness[name].items():
        rows = [row for score in scores for row in score["rows"] if row["outcome"] == "found"]
        right, wrong = [row for row in rows if row["type right"]], [row for row in rows if not row["type right"]]
        if not right or not wrong or HARNESS_WAYS[way] is None:
            continue
        words = {}
        for row in rows:
            words[row["flow"]] = words.get(row["flow"], 0) + 1
        print(f"   {way:48s} {len(right)} right, {len(wrong)} wrong | by geometry {np.mean([r['confidence by geometry'] for r in right]):.2f} "
              f"against {np.mean([r['confidence by geometry'] for r in wrong]):.2f}, told apart "
              f"{told_apart([r['confidence by geometry'] for r in right], [r['confidence by geometry'] for r in wrong]):.2f} | with "
              f"flow {np.mean([r['confidence'] for r in right]):.2f} against {np.mean([r['confidence'] for r in wrong]):.2f}, told "
              f"apart {told_apart([r['confidence'] for r in right], [r['confidence'] for r in wrong]):.2f} | "
              + ", ".join(f"{count} {word}" for word, count in sorted(words.items(), key=lambda item: -item[1])))
    print(f"{name}, section 33: the trunk of a bifurcation (found at a true bifurcation, three arms, all matched to true limbs)")
    for way, scores in harness[name].items():
        if HARNESS_WAYS[way] is None or "again" in HARNESS_WAYS[way]:
            continue
        rules = parent_scores([{**arm, "seed": score["seed"]} for score in scores for arm in score["arms"]])
        told = []
        for rule, results in rules.items():
            by_seed = {}
            for seed, named in results:
                by_seed.setdefault(seed, []).append(named)
            if by_seed:
                value = with_interval([sum(some) for some in by_seed.values()], [len(some) for some in by_seed.values()])
                told.append(f"{rule}: {interval((*value, len(results)))}")
        print(f"   {way:48s} " + " | ".join(told))

# the tables kept for the work that follows
harness_junctions = table_of(harness["development"], "rows") + table_of(harness["held out"], "rows")
harness_arms = table_of(harness["development"], "arms") + table_of(harness["held out"], "arms")
harness_vessels = table_of(harness["development"], "vessels") + table_of(harness["held out"], "vessels")
os.makedirs("vessel_pictures/harness", exist_ok=True)
save_table(harness_junctions, "vessel_pictures/harness/junctions.csv")
save_table(harness_arms, "vessel_pictures/harness/arms.csv")
save_table(harness_vessels, "vessel_pictures/harness/vessels.csv")
save_table([{"bursts": name, "way": way, **{column: f"{row[column][0]:.3f} ({row[column][1]:.3f} to {row[column][2]:.3f})"
                                             for column in ABLATION_COLUMNS},
             "true junctions": row["true junctions"], "junctions read": row["junctions read"]}
            for name in ablation for way, row in ablation[name].items()], "vessel_pictures/harness/ablation.csv")
print(f"\nKept: {len(harness_junctions)} rows of junctions, {len(harness_arms)} rows of arms and {len(harness_vessels)} rows of "
      f"whole vessels, in vessel_pictures/harness")

# figure 1: the first development burst read four ways, and the junction scores of every row with intervals
burst = synthetic_burst(HARNESS_SEEDS[0])
front = ridge_front(burst, workers)
pictured = [way for way in ("chains of sections 19 to 27", "plain ridges", "consolidated, keeping short pieces",
                         "consolidated, tube rule") if way in HARNESS_WAYS]
figure = plt.figure(figsize=(16, 13))
grid = figure.add_gridspec(2, 4, height_ratios=(1, 1.5))
for place, way in enumerate(pictured):
    axis = figure.add_subplot(grid[0, place])
    axis.imshow(outcome_picture(burst, read_way(burst, front, HARNESS_WAYS[way], saved["chain pruning"], workers)),
                interpolation="antialiased")
    axis.set_title(way, fontsize=9)
    axis.axis("off")
shown = ("junction precision", "junction recall", "type right, of found", "found and typed right")
for place, column in enumerate(shown):
    axis = figure.add_subplot(grid[1, place])
    for offset, (name, color) in enumerate((("development", "tab:blue"), ("held out", "tab:orange"))):
        for line, way in enumerate(HARNESS_WAYS):
            share, low, high = ablation[name][way][column]
            axis.errorbar(share * 100, line + 0.18 * (2 * offset - 1), xerr=[[(share - low) * 100], [(high - share) * 100]],
                          fmt="o", color=color, capsize=3, label=name if line == 0 else None)
    axis.set_yticks(range(len(HARNESS_WAYS)))
    axis.set_yticklabels(list(HARNESS_WAYS) if place == 0 else [], fontsize=8)
    axis.set_xlim(0, 100)
    axis.invert_yaxis()
    axis.set_title(f"{column} (%)", fontsize=9)
    axis.grid(axis="x", alpha=0.3)
    if place == 0:
        axis.legend(fontsize=8, loc="lower left")
figure.suptitle("top: one development burst read four ways | circles: true junctions (yellow bifurcation, white crossing, gray "
                "nobody is asked for it) | dots: junctions read (green found, orange ring typed wrong, red false alarm)\n"
                "bottom: junction scores of every row with 95% intervals over bursts", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/harness/harness.png", dpi=110)
plt.show()

# figure 2, held-out bursts: speeds against the truth, confidence of right and wrong readings, and the trunk rules
whole = "plain ridges, whole vessels"
figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for way, color in (("plain ridges", "tab:gray"), (whole, "tab:blue")):
    arms = [arm for score in harness["held out"][way] for arm in score["arms"] if arm["out"] != 0 and not arm["inferred"]]
    sigma = [arm["speed sigma"] if np.isfinite(arm["speed sigma"]) else 0 for arm in arms]
    axes[0].errorbar([arm["true speed"] for arm in arms], [arm["speed"] for arm in arms], yerr=2 * np.array(sigma), fmt=".",
                     color=color, alpha=0.6, linewidth=0.6, label="per piece" if way == "plain ridges" else "whole vessels, +- 2 sigma")
limit = max(arm["true speed"] for score in harness["held out"][whole] for arm in score["arms"]) * 1.3
axes[0].plot([0, limit], [0, limit], color="black", linewidth=0.6)
axes[0].set_xlim(0, limit)
axes[0].set_ylim(0, limit)
axes[0].set_xlabel("true speed (px/s)")
axes[0].set_ylabel("speed read at a junction's arm (px/s)")
axes[0].set_title("section 32: speeds at the arms of the junctions found", fontsize=10)
axes[0].legend(fontsize=8)
rows = [row for score in harness["held out"][whole + ", likelihood ratio"] for row in score["rows"] if row["outcome"] == "found"]
bins = np.linspace(0, 1, 11)
for place, (column, title) in enumerate((("confidence by geometry", "by geometry"), ("confidence", "with flow"))):
    for kind, color in ((True, "tab:green"), (False, "tab:red")):
        values = [row[column] for row in rows if row["type right"] == kind]
        axes[1].hist(values, bins=bins, histtype="step", color=color, linestyle="--" if place == 0 else "-",
                     label=f"type {'right' if kind else 'wrong'}, {title}")
axes[1].set_xlabel("confidence")
axes[1].set_title("section 33: junctions found, by confidence", fontsize=10)
axes[1].legend(fontsize=7)
rules = parent_scores([{**arm, "seed": score["seed"]} for score in harness["held out"][whole] for arm in score["arms"]])
for place, (rule, results) in enumerate(rules.items()):
    by_seed = {}
    for seed, named in results:
        by_seed.setdefault(seed, []).append(named)
    if by_seed:
        share, low, high = with_interval([sum(some) for some in by_seed.values()], [len(some) for some in by_seed.values()])
        axes[2].errorbar(place, share * 100, yerr=[[(share - low) * 100], [(high - share) * 100]], fmt="o", capsize=4, color="tab:blue")
        axes[2].annotate(f"n={len(results)}", (place, 5), ha="center", fontsize=8)
axes[2].set_xticks(range(len(rules)))
axes[2].set_xticklabels(list(rules))
axes[2].set_xlim(-0.5, len(rules) - 0.5)
axes[2].set_ylim(0, 105)
axes[2].set_ylabel("names the true trunk (%)")
axes[2].set_title("section 33: the trunk of a bifurcation, by rule", fontsize=10)
figure.suptitle("held-out bursts, plain ridges", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/harness/flow_on_synthetic.png", dpi=110)
plt.show()
del burst, front
memory_note("harness")

# %% [markdown]
# # Plan for a clean cyclic graph (`docs/cyclic_graph_recursion_plan.md`)
#
# Sections 35 to 42 carry out the plan in `docs/cyclic_graph_recursion_plan.md`, in the order of its acceptance steps (Part III). Each step ends with a number and a gate; its report is in `reports/step_<n>_<date>.md`. Like section 29 these sections run from the saved inputs: in a fresh kernel, run section 29's cell *Open the inputs* first. Every cell that only defines has an id listed in that cell's `DEFINITION_CELLS`.
#
# ## 35. Synthetic bursts with the things this plan is about
#
# Section 28's generator makes three trees, each splitting once, with nothing thinner than about 2.5 px, no loops, no gaps, no vessel that stops inside the field, the density blurred instead of the light, and the true background handed over. Every claim of the plan about capillaries, loops, gaps, plexus junctions, overlaps or the second fixation needs a burst that has them. `synthetic_burst_35(seed, kind, **things)` adds them, each one switched on by its name:
#
# | thing | what it adds |
# |---|---|
# | `capillaries` | trees with a trunk of radius 1.5 to 2 px (daughters of 1 to 1.7 px), half of whose daughters split a second time |
# | `loops` | a daughter that curves round and runs into the other daughter: a loop, closed where both fluxes join (Murray's radius, the flux conserved) |
# | `gaps` | a stretch of `SYNTH_GAP_LENGTH` px along which a segment fades to `SYNTH_GAP_DEPTH` of its density |
# | `ends` | a segment that stops inside the field instead of leaving it |
# | `halos` | around every tree with a vessel of 3 px or more, its own density blurred over three radii, of either sign |
# | `overlaps` | a hidden vessel: a vessel of another depth lying along a stretch of a vessel, nothing emerging at either end |
# | `profile` | the cells' speed changes across a vessel, `v_max (1 - (r / R)^SYNTH_BLUNT)`; capillaries in single file |
# | `transmittance` | the camera blurs each tree's transmittance, not its density (as `vascnet.synth.render`) |
# | `background` | the frames measured against section 12's `fit_background` of the mean frame instead of the true background |
# | `second` | a second fixation: the same network and tissue moved by a known shift, with fresh clusters and camera noise |
# | `fine` | a pixel's density as a mean over 7 x 7 points, so that a vessel of radius 1 keeps its mass within 2% anywhere on the grid |
#
# and `kind` is `"trees"` (section 28's three trees and what the things add to them) or one of the two kinds of plexus made from the same lines of one seed: `"crossing plexus"`, every line a vessel at its own depth, so that every meeting is a projected crossing; and `"anastomotic plexus"`, every line in one lumen, so that every meeting is a mixing node of four arms, with the fluxes of a network of resistors.
#
# **The old seeds are kept.** With no thing switched on, `synthetic_burst_35(seed)` draws the same random numbers in the same order and does the same arithmetic as `synthetic_burst(seed)`: its burst is the same to the bit, with new keys beside the old ones. What a thing needs on top is drawn from a random stream of its own (`stream`), so that switching one thing on leaves what the others draw as it was. Section 28's code is not changed, so sections 30 and 34 read the same bursts.
#
# **The truth** has, per junction, its number of arms `n` (3 at a bifurcation, 4 at a crossing or an anastomosis; `junction_arms` sums them over a fused "complex" junction), its `lumens` (1 or 2), its `kind` (split, loop, anastomosis, projected crossing) and its `limbs`; per segment its nodes and flux; and the gaps, terminations, hidden vessels and halos that were put in. `mark_visible_35` applies vascnet's visibility rules to any network.
#
# **New bursts.** `TREE_SEEDS` and `TREE_HELD_OUT` (16 each) are trees with every thing on (`SYNTH_THINGS`); `PLEXUS_SEEDS` and `PLEXUS_HELD_OUT` (8 each) are plexus bursts, every seed read as both kinds. The cell after the definitions checks the old seeds and counts what every new seed holds.
#
# **Result:** the generator, the check of the 32 old seeds, and per new seed the counts of loops, gaps, terminations, plexus junctions of each kind and collinear overlaps (`synthetic_counts`; `vessel_pictures/synthetic_35/counts.csv`, and a picture of one burst of each kind in `kinds.png`).

# %%
SYNTH_LOOP = 0.5             # chance that one daughter of a tree runs back into the other one, which makes a loop
SYNTH_GAP = 0.25             # chance that a segment has a gap: a stretch along which its density fades
SYNTH_GAP_LENGTH = 24        # pixels; the length of a gap
SYNTH_GAP_DEPTH = 0.15       # the share of its density a segment keeps at the middle of a gap
SYNTH_END = 0.25             # chance that a segment that would leave the frame stops inside it instead (a termination)
SYNTH_BLUNT = 3.0            # the speed across a vessel is v_max (1 - (r / R)^SYNTH_BLUNT): 2 is a parabola, more is blunter
SYNTH_FILE = 2.0             # pixels; in a vessel of this radius or less the cells go in single file, all at one speed
SYNTH_LAMINA = 1.2           # pixels; a vessel's clusters are drawn in strips across it about this wide (5 at the most)
SYNTH_CAPILLARY_TREES = 2    # capillary trees added to the three trees of section 28
SYNTH_CAPILLARY = (1.5, 2.0)        # pixels; the radius of a capillary tree's trunk (its daughters come out at 1 to 1.7)
SYNTH_CAPILLARY_SPEED = (0.8, 2.5)  # pixels per frame; the speed in a capillary tree's trunk, and in a thin line of a plexus
SYNTH_SPLIT_AGAIN = 0.5      # chance that a daughter of a capillary tree splits a second time
SYNTH_HALO = 0.5             # a thick tree's halo is its own density blurred over SYNTH_HALO_REACH radii, times this, of
SYNTH_HALO_REACH = 3.0       # either sign: about a tenth of the vessel's density on its centre, reaching three radii out
SYNTH_HALO_LEAST = 3.0       # pixels; a tree has a halo when its thickest vessel has at least this radius
SYNTH_OVERLAPS = 1           # hidden vessels per burst: a vessel of another depth lying along a vessel, nothing emerging
SYNTH_PLEXUS = (3, 4)        # lines of a plexus that enter from the left and from the top
SYNTH_PLEXUS_THICK = 0.4     # the share of a plexus' lines that are thick (radius 3 to 4.5); the others are capillaries
SYNTH_FINE = 7               # a pixel's density is the mean over this many points a side: a vessel of radius 1 keeps its
                             # mass within 2% wherever it lies on the pixel grid (with 3 a side it loses up to 7%)
SYNTH_SHIFT = (8.0, 30.0)    # pixels; how far the second fixation is moved along each axis, either way
SYNTH_BACKGROUND = 8         # pixels; the width of the background fit's local mean (section 12's BACKGROUND_SIGMA)
# The things section 35 adds, each of which can be switched on by its name. With every one off, synthetic_burst_35
# makes the burst of section 28 for the same seed, bit for bit.
SYNTH_PLAIN = {"capillaries": 0, "loops": 0.0, "gaps": 0.0, "ends": 0.0, "halos": False, "overlaps": 0, "profile": False,
               "transmittance": False, "background": None, "second": False, "fine": False}
SYNTH_THINGS = {"capillaries": SYNTH_CAPILLARY_TREES, "loops": SYNTH_LOOP, "gaps": SYNTH_GAP, "ends": SYNTH_END,
                "halos": True, "overlaps": SYNTH_OVERLAPS, "profile": True, "transmittance": True, "background": "fit",
                "second": True, "fine": True}
SYNTH_KINDS = ("trees", "crossing plexus", "anastomotic plexus")
TREE_SEEDS = tuple(range(2001, 2017))          # bursts of trees with the things of section 35, for developing
TREE_HELD_OUT = tuple(range(3001, 3017))       # and held out
PLEXUS_SEEDS = tuple(range(4001, 4009))        # plexus bursts for developing: every seed is read as both kinds
PLEXUS_HELD_OUT = tuple(range(5001, 5009))     # and held out
COUNTED = ("trees", "segments", "capillary segments", "bifurcations", "loops", "gaps", "terminations",
           "projected crossings", "anastomoses", "collinear overlaps", "halos", "junctions asked for")


def stream(seed, name):
    """A random stream of its own for one of the things section 35 adds, so that switching one thing on or off
    leaves what the others draw as it was."""
    return np.random.default_rng([seed, int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")])


def inside_frame(points, margin=0.0):
    """Which of some (x, y) points lie inside a synthetic frame, at least `margin` pixels from its edge."""
    rows, columns = SYNTH_SHAPE
    return ((points[..., 0] >= margin) & (points[..., 0] < columns - margin)
            & (points[..., 1] >= margin) & (points[..., 1] < rows - margin))


def cut_at_frame(points):
    """A path up to where it first leaves the frame (as section 28 cuts a daughter)."""
    inside = inside_frame(points)
    return points[:np.argmin(inside) if not inside.all() else len(points)]


def tree_nodes(segments, nodes=None, first=0):
    """Give the segments of trees (each with `parent`, a parent before its children in the list) their nodes: `start`
    and `end`, numbers into `nodes`, a list of (x, y). A trunk starts at a node of its own, a daughter at the node
    its parent ends in, and every segment ends in a node of its own. Returns the nodes."""
    nodes = [] if nodes is None else nodes
    for segment in segments[first:]:
        if segment["parent"] is None:
            segment["start"] = len(nodes)
            nodes.append(segment["points"][0].copy())
        else:
            segment["start"] = segments[segment["parent"]]["end"]
        segment["end"] = len(nodes)
        nodes.append(segment["points"][-1].copy())
    return nodes


def apart_within_tree(segments, members, margin=2.0, near=15):
    """Whether the segments of one tree keep clear of each other: no two of them come closer than the sum of their
    radii plus `margin`, except within `near` points of a node they share. Two segments of one tree that touch
    anywhere else would be drawn as one lumen with no junction in the truth."""
    for first, second in itertools.combinations(members, 2):
        a, b = segments[first], segments[second]
        points_a, points_b = a["points"], b["points"]
        shared = {a["start"], a["end"]} & {b["start"], b["end"]}
        keep_a, keep_b = np.ones(len(points_a), dtype=bool), np.ones(len(points_b), dtype=bool)
        for node in shared:
            for segment, keep in ((a, keep_a), (b, keep_b)):
                if segment["start"] == node:
                    keep[:near] = False
                if segment["end"] == node:
                    keep[-near:] = False
        if not keep_a.any() or not keep_b.any():
            continue
        distance, _ = scipy.spatial.cKDTree(points_b[keep_b]).query(points_a[keep_a])
        if distance.min() < a["radius"] + b["radius"] + margin:
            return False
    return True


def capillary_trees(rng, segments, nodes, count):
    """Add `count` capillary trees to a network: drawn as section 28 draws a tree, with a trunk of radius
    SYNTH_CAPILLARY, shorter, and with daughters that split a second time with the chance SYNTH_SPLIT_AGAIN (where
    they are still thick enough to have daughters of a pixel's radius). A tree whose segments would touch each other
    away from their junctions is drawn again."""
    rows, columns = SYNTH_SHAPE
    diagonal = np.hypot(rows, columns)
    tree = max(segment["tree"] for segment in segments) + 1 if segments else 0
    for _ in range(count):
        for attempt in range(30):
            grown, base = [], len(segments)
            side, along = rng.integers(4), rng.uniform(0.2, 0.8)
            start, heading = (((along * columns, 0), np.pi / 2), ((along * columns, rows - 1), -np.pi / 2),
                              ((0, along * rows), 0.0), ((columns - 1, along * rows), np.pi))[side]
            heading += rng.uniform(-0.35, 0.35)
            radius = rng.uniform(*SYNTH_CAPILLARY)
            speed = rng.uniform(*SYNTH_CAPILLARY_SPEED) * rng.choice([-1, 1])
            depth = rng.uniform(-0.6, 0.6)
            trunk, heading = wandering_path(start, heading, rng.uniform(0.2, 0.35) * diagonal, rng)
            grown.append({"points": trunk, "radius": radius, "speed": speed, "tree": tree, "depth": depth, "parent": None})
            waiting = [(0, heading)]  # segments that split at their end: (place among `grown`, heading there)
            while waiting:
                parent, heading = waiting.pop(0)
                share = rng.uniform(0.35, 0.65)
                for part, turn in ((share, rng.uniform(0.4, 0.75)), (1 - share, -rng.uniform(0.4, 0.75))):
                    daughter = {"radius": grown[parent]["radius"] * part ** (1 / 3),
                                "speed": grown[parent]["speed"] * part ** (1 / 3), "tree": tree, "depth": depth,
                                "parent": base + parent}
                    again = parent == 0 and daughter["radius"] >= 1.3 and rng.random() < SYNTH_SPLIT_AGAIN
                    length = rng.uniform(60, 120) if again else 0.8 * diagonal
                    path, ahead = wandering_path(grown[parent]["points"][-1], heading + turn, length, rng)
                    daughter["points"] = cut_at_frame(path)
                    grown.append(daughter)
                    if again and len(daughter["points"]) == len(path) and inside_frame(path[-1], 30):
                        waiting.append((len(grown) - 1, ahead))
            if any(len(segment["points"]) < 20 for segment in grown) or not inside_frame(trunk[-1], 20):
                continue
            segments.extend(grown)
            marked = len(nodes)
            tree_nodes(segments, nodes, base)
            if apart_within_tree(segments, range(base, len(segments))):
                break
            del segments[base:], nodes[marked:]
        tree += 1


def close_loop(rng, segments, nodes, trunk):
    """Make a loop in a tree: one daughter of `trunk` leaves as it did, curves round and runs into the other
    daughter some way along it. The other daughter is cut there in two; its far part carries both fluxes, with the
    radius Murray's law gives and the speed that conserves the flux. Returns whether a loop that keeps inside the
    frame and clear of the rest of its tree was found (the network is left as it was otherwise)."""
    daughters = [number for number, segment in enumerate(segments) if segment["parent"] == trunk]
    leaves = [number for number in daughters if not any(other["parent"] == number for other in segments)]
    if len(daughters) != 2 or len(leaves) != 2:
        return False
    split = segments[trunk]["points"][-1]
    for attempt in range(8):
        a, b = daughters if rng.random() < 0.5 else daughters[::-1]  # a comes back into b
        coming, taking = segments[a], segments[b]
        reach, angle, stretch = int(rng.uniform(70, 130)), rng.uniform(0.5, 0.9), rng.uniform(1.0, 1.4)
        if reach > len(taking["points"]) - 40 or len(coming["points"]) < 12:
            continue
        target = taking["points"][reach]
        if not inside_frame(target, 20):
            continue
        ahead = taking["points"][reach + 5] - taking["points"][reach - 5]
        ahead /= np.hypot(*ahead)
        leave = coming["points"][8] - split
        leave /= np.hypot(*leave)
        beside = coming["points"][min(30, len(coming["points"]) - 1)] - target
        side = np.sign(ahead[0] * beside[1] - ahead[1] * beside[0]) or 1.0  # which side of the other daughter it is on
        turn = -side * angle  # it arrives from its own side, at this angle to the daughter it runs into
        arrive = np.array([ahead[0] * np.cos(turn) - ahead[1] * np.sin(turn), ahead[0] * np.sin(turn) + ahead[1] * np.cos(turn)])
        # a cubic through the split and the target, leaving and arriving in those directions, a pixel between points
        t = np.linspace(0, 1, 2000)[:, None]
        pull = stretch * np.hypot(*(target - split))
        curve = ((2 * t ** 3 - 3 * t ** 2 + 1) * split + (t ** 3 - 2 * t ** 2 + t) * pull * leave
                 + (-2 * t ** 3 + 3 * t ** 2) * target + (t ** 3 - t ** 2) * pull * arrive)
        travelled = np.r_[0, np.cumsum(np.hypot(*np.diff(curve, axis=0).T))]
        steps = np.arange(1, int(travelled[-1]) + 1)
        path = np.column_stack([np.interp(steps, travelled, curve[:, 0]), np.interp(steps, travelled, curve[:, 1])])
        path[-1] = target
        if len(path) < 40 or len(path) > 400 or not inside_frame(path, 6).all():
            continue
        middle = path[15:-15]
        if scipy.spatial.cKDTree(taking["points"]).query(middle)[0].min() < coming["radius"] + taking["radius"] + 2:
            continue
        if scipy.spatial.cKDTree(segments[trunk]["points"][:-15]).query(middle)[0].min() < coming["radius"] + segments[trunk]["radius"] + 2:
            continue
        flux = abs(coming["speed"]) * coming["radius"] ** 2 + abs(taking["speed"]) * taking["radius"] ** 2
        radius = (coming["radius"] ** 3 + taking["radius"] ** 3) ** (1 / 3)
        far = {"points": taking["points"][reach + 1:], "radius": radius, "speed": np.sign(taking["speed"]) * flux / radius ** 2,
               "tree": taking["tree"], "depth": taking["depth"], "parent": b, "start": len(nodes), "end": taking["end"]}
        nodes.append(target.copy())
        coming["points"], coming["end"] = path, far["start"]
        taking["points"], taking["end"] = taking["points"][:reach + 1], far["start"]
        segments.append(far)
        coming["loop"] = far["loop"] = taking["loop"] = True
        return True
    return False


def plexus_network(seed, anastomoses):
    """Draw a plexus: SYNTH_PLEXUS lines that enter from the left and from the top and cross the frame, wandering
    as section 28's vessels do, each a capillary or (SYNTH_PLEXUS_THICK of them) a thick vessel.

    Two kinds are made from the same lines of one seed. Without `anastomoses` every line is a vessel at a depth of
    its own, with a speed of its own: where two lines pass over each other is a projected crossing. With
    `anastomoses` all lines lie in one lumen at one depth: where two meet is a mixing node of four arms, the lines
    are cut there into pieces, and the flux of every piece is that of a network of resistors (conductance radius^4
    over length) between pressures given where the lines enter and leave the frame, so that the flux is conserved
    at every node. Returns the segments and the nodes."""
    rng = stream(seed, "plexus")
    rows, columns = SYNTH_SHAPE
    diagonal = np.hypot(rows, columns)
    lines = []  # (points, radius)
    cuts, kept = [], []  # per line the places where other lines meet it; the meetings: ((line, place), (line, place))
    plans = [("left", place) for place in range(SYNTH_PLEXUS[0])] + [("top", place) for place in range(SYNTH_PLEXUS[1])]
    for side, place in plans:
        for attempt in range(60):
            count = SYNTH_PLEXUS[0] if side == "left" else SYNTH_PLEXUS[1]
            along = (place + rng.uniform(0.2, 0.8)) / count
            start, heading = ((0, along * rows), 0.0) if side == "left" else ((along * columns, 0), np.pi / 2)
            heading += rng.uniform(-0.3, 0.3)
            radius = rng.uniform(3.0, 4.5) if rng.random() < SYNTH_PLEXUS_THICK else rng.uniform(1.2, 2.0)
            points = cut_at_frame(wandering_path(start, heading, 2 * diagonal, rng)[0])
            if len(points) < 0.5 * min(rows, columns):
                continue
            # where it meets the lines drawn before it, at the place where they come closest. Every contact has to
            # be a crossing, not a run alongside or a near miss, at least 12 points from the lines' ends and from
            # every other meeting on either line, so that no piece between two nodes is too short to be a vessel.
            fits, meetings, mine = True, [], []
            for number, (other, other_radius) in enumerate(lines):
                distance, nearest = scipy.spatial.cKDTree(other).query(points)
                touching = np.flatnonzero(distance < radius + other_radius + 1)
                for run in np.split(touching, np.flatnonzero(np.diff(touching) > 5) + 1):
                    if not len(run):
                        continue
                    at = int(run[np.argmin(distance[run])])
                    there = int(nearest[at])
                    if (len(run) > 6 * (radius + other_radius + 1) or distance[at] >= 1.0
                            or min(at, len(points) - 1 - at, there, len(other) - 1 - there) < 12
                            or any(abs(there - cut) < 12 for cut in cuts[number])
                            or any(abs(at - cut) < 12 for cut in mine)):
                        fits = False
                    meetings.append(((len(lines), at), (number, there)))
                    mine.append(at)
            if fits:
                lines.append((points, radius))
                cuts.append(mine)
                for meeting in meetings:
                    cuts[meeting[1][0]].append(meeting[1][1])
                kept += meetings
                break
    cuts = dict(enumerate(cuts))
    segments, nodes = [], []
    if not anastomoses:
        for number, (points, radius) in enumerate(lines):
            limits = SYNTH_CAPILLARY_SPEED if radius <= SYNTH_FILE else (1.5, 3.5)
            segments.append({"points": points, "radius": radius, "speed": rng.uniform(*limits) * rng.choice([-1, 1]),
                             "tree": number, "depth": rng.uniform(-0.6, 0.6), "parent": None})
        return segments, tree_nodes(segments)
    depth = rng.uniform(-0.6, 0.6)
    node_at = {}  # (line, place) -> node
    for (first, at), (second, other) in kept:
        lines[second][0][other] = lines[first][0][at]  # both lines pass through one point there
        node_at[(first, at)] = node_at[(second, other)] = len(nodes)
        nodes.append(lines[first][0][at].copy())
    given = []  # the nodes where a line enters or leaves the frame: the pressure is given there
    for number, (points, radius) in enumerate(lines):
        places = [0] + sorted(cuts[number]) + [len(points) - 1]
        for place in (places[0], places[-1]):
            node_at[(number, place)] = len(nodes)
            given.append(len(nodes))
            nodes.append(points[place].copy())
        for before, after in zip(places[:-1], places[1:]):
            # a piece starts a point beyond the node it leaves, as a daughter of section 28 does, and ends on its node
            segments.append({"points": points[before + (before > 0):after + 1], "radius": radius, "tree": 0, "depth": depth,
                             "parent": None, "start": node_at[(number, before)], "end": node_at[(number, after)], "line": number})
    # the pressures: a slope across the frame in some direction, and a little of its own at every end
    slope = rng.uniform(0, 2 * np.pi)
    pressure = np.zeros(len(nodes))
    for node in given:
        pressure[node] = (np.cos(slope) * nodes[node][0] / columns + np.sin(slope) * nodes[node][1] / rows + rng.normal(0, 0.15))
    conductance = np.array([segment["radius"] ** 4 / len(segment["points"]) for segment in segments])
    laplacian = np.zeros((len(nodes), len(nodes)))
    for segment, g in zip(segments, conductance):
        a, b = segment["start"], segment["end"]
        laplacian[a, a] += g
        laplacian[b, b] += g
        laplacian[a, b] -= g
        laplacian[b, a] -= g
    free = np.setdiff1d(np.arange(len(nodes)), given)
    if len(free):
        pressure[free] = np.linalg.solve(laplacian[np.ix_(free, free)], -laplacian[np.ix_(free, given)] @ pressure[given])
    flux = np.array([g * (pressure[segment["start"]] - pressure[segment["end"]]) for segment, g in zip(segments, conductance)])
    speed = flux / np.array([segment["radius"] ** 2 for segment in segments])
    speed *= rng.uniform(2.0, 3.5) / np.abs(speed).max()  # the fastest piece at a speed of section 28's vessels
    for segment, value in zip(segments, speed):
        segment["speed"] = float(value)
    return segments, nodes


def node_ends(segments, nodes):
    """Per node, the ends of the segments that meet in it: (segment, end), end 0 for a segment's first point and 1
    for its last, in the order of the segments."""
    ends = [[] for _ in nodes]
    for number, segment in enumerate(segments):
        ends[segment["start"]].append((number, 0))
        ends[segment["end"]].append((number, 1))
    return ends


def network_truth_35(segments, nodes):
    """What is true of a network, as network_truth (section 28) returns it and with what section 35 adds.

    The centerlines as a picture. The junctions: every node where three or more segments of one lumen meet (three:
    a `bifurcation`, which is a split or the place where a loop closes; four: an `anastomosis`), and every place
    where segments of two trees pass over each other (a `crossing`). Each junction has the keys of section 28 (x, y,
    type, radius, segments, and places for a crossing) and, new: `n`, the number of arms; `lumens`, 1 or 2; `kind`
    ("split", "loop", "anastomosis", "projected crossing"); `limbs`, its arms as (segment, place along it, way: +1
    where the segment runs on from the junction towards its last point, -1 towards its first); and for a crossing
    `overlap`, the length in pixels along which the two vessels lie on each other. A hidden vessel (`host`) makes no
    crossing with the vessel it lies along."""
    lines = np.zeros(SYNTH_SHAPE, dtype=np.uint8)
    for segment in segments:
        cv2.polylines(lines, np.rint(segment["points"]).astype(np.int32)[None], False, 1)
    junctions = []
    for node, ends in enumerate(node_ends(segments, nodes)):
        if len(ends) < 3:
            continue
        members = sorted((number for number, _ in ends), key=lambda number: -segments[number]["radius"])
        thickest = members[0]
        members = [thickest] + sorted(number for number in members if number != thickest)
        place = segments[thickest]["points"][-1 if segments[thickest]["end"] == node else 0]
        looped = len(ends) == 3 and sum(end == 1 for _, end in ends) == 2 and any(segments[number].get("loop") for number, _ in ends)
        junctions.append({"x": float(place[0]), "y": float(place[1]), "type": "bifurcation" if len(ends) == 3 else "anastomosis",
                          "radius": float(segments[thickest]["radius"]), "segments": members, "n": len(ends), "lumens": 1,
                          "kind": "anastomosis" if len(ends) > 3 else "loop" if looped else "split",
                          "limbs": [(number, len(segments[number]["points"]) - 1 if end else 0, -1 if end else 1)
                                    for number, end in sorted(ends, key=lambda item: members.index(item[0]))]})
    for first, second in itertools.combinations(range(len(segments)), 2):
        if segments[first]["tree"] == segments[second]["tree"]:
            continue
        if segments[first].get("host") == second or segments[second].get("host") == first:
            continue
        distance, nearest = scipy.spatial.cKDTree(segments[second]["points"]).query(segments[first]["points"])
        close = np.flatnonzero(distance < 1.0)
        for run in np.split(close, np.flatnonzero(np.diff(close) > 5) + 1):
            if len(run):
                at = run[np.argmin(distance[run])]
                x, y = segments[first]["points"][at]
                # a crossing counts once, also where a tree's segments meet and all of them cross the other tree's
                known = any(np.hypot(x - junction["x"], y - junction["y"]) < SYNTH_APART
                            for junction in junctions if junction["type"] == "crossing")
                if 8 <= x < SYNTH_SHAPE[1] - 8 and 8 <= y < SYNTH_SHAPE[0] - 8 and not known:
                    reach = segments[first]["radius"] + segments[second]["radius"] + 1
                    lying = distance < reach
                    low = high = at
                    while low > 0 and lying[low - 1]:
                        low -= 1
                    while high < len(lying) - 1 and lying[high + 1]:
                        high += 1
                    junctions.append({"x": float(x), "y": float(y), "type": "crossing",
                                      "radius": float(max(segments[first]["radius"], segments[second]["radius"])),
                                      "segments": [first, second], "places": [int(at), int(nearest[at])],
                                      "n": 4, "lumens": 2, "kind": "projected crossing", "overlap": int(high - low + 1),
                                      "limbs": [(number, place, way) for number, place in ((first, int(at)), (second, int(nearest[at])))
                                                for way in (-1, 1)]})
    return lines > 0, junctions


def mark_visible_35(junctions, segments, nodes, layers, noise):
    """mark_visible (section 28) for any network: the true junctions that nobody can be asked to find, by vascnet's
    rules. A crossing is handed to vascnet.synth._annotate_visibility as there. A junction of one lumen (a split,
    the closing of a loop, an anastomosis) is visible when its faintest segment stands at 3 x noise or more, which is
    vascnet's rule for a bifurcation, applied to every segment that meets in it. For the rule that a crossing's
    vessel has to run on beyond it, a segment runs on where its node holds another segment."""
    ends = node_ends(segments, nodes)
    others = lambda number, node: [other for other, _ in ends[node] if other != number]
    branches = []
    for number, segment in enumerate(segments):
        before, after = others(number, segment["start"]), others(number, segment["end"])
        branches.append(vascnet.synth.Branch(segment["points"], np.full(len(segment["points"]), segment["radius"]),
                                             segment["tree"], before[0] if before else None, after))
    crossings = [junction for junction in junctions if junction["type"] == "crossing"]
    described = [dict(junction, type="crossover", branches=tuple(junction["segments"]), pos=tuple(junction["places"]))
                 for junction in crossings]
    vascnet.synth._annotate_visibility(vascnet.synth.Network(SYNTH_SHAPE, branches, [], described), layers, noise)
    for junction, seen in zip(crossings, described):
        junction.update({name: seen[name] for name in ("contrast", "visible", "ambiguous") if name in seen})
    floor = 3.0 * max(noise, 1e-6)
    for junction in junctions:
        if junction["type"] != "crossing":
            junction["contrast"] = float(min(branches[number].contrast for number in junction["segments"]))
            junction["visible"] = bool(junction["contrast"] >= floor)


def junction_arms(junction):
    """The number of arms of a true junction, also after close ones were fused into one "complex" junction
    (vascnet.evaluate.cluster_truth keeps the members' types): 3 for a bifurcation, 4 for a crossing or an
    anastomosis, and their sum for a complex one."""
    if "n" in junction:
        return int(junction["n"])
    return sum(3 if member == "bifurcation" else 4 for member in junction.get("members", ()))


def footprint(segment, pixels, fine):
    """Where a segment's blood lies: its pixels, their place along it, and the length of the line of sight through
    it there (`chord`, times the absorption). As in section 28 the pixels are those within the radius plus one of
    one of the segment's points. With `fine` the chord is the mean over SYNTH_FINE x SYNTH_FINE points of every
    pixel, each measured from the segment's polyline and not from its nearest point, and every pixel also gets
    `across`: its place across the vessel, from -1 at one wall to 1 at the other. A gap is put into the chord."""
    points, radius = segment["points"], segment["radius"]
    distance, along = scipy.spatial.cKDTree(points).query(pixels, distance_upper_bound=radius + 1)
    inside = np.flatnonzero(np.isfinite(distance))
    segment["pixels"], segment["along"] = inside, along[inside]
    nearest = segment["along"]
    if not fine:
        segment["chord"] = 2 * np.sqrt(np.maximum(radius ** 2 - distance[inside] ** 2, 0)) * SYNTH_ABSORB
    else:
        chord = np.zeros(len(inside))
        offsets = (np.arange(SYNTH_FINE) + 0.5) / SYNTH_FINE - 0.5
        for dy, dx in itertools.product(offsets, offsets):
            spot = pixels[inside] + (dx, dy)
            best = np.full(len(inside), np.inf)
            for step in (-1, 0):  # the two pieces of the polyline that end in the nearest point
                first = np.clip(nearest + step, 0, max(len(points) - 2, 0))
                a, b = points[first], points[np.minimum(first + 1, len(points) - 1)]
                ab = b - a
                t = np.clip(((spot - a) * ab).sum(axis=1) / np.maximum((ab ** 2).sum(axis=1), 1e-12), 0, 1)
                best = np.minimum(best, np.hypot(*(spot - (a + t[:, None] * ab)).T))
            chord += 2 * np.sqrt(np.maximum(radius ** 2 - best ** 2, 0))
        segment["chord"] = chord / SYNTH_FINE ** 2 * SYNTH_ABSORB
    ahead = points[np.minimum(nearest + 1, len(points) - 1)] - points[np.maximum(nearest - 1, 0)]
    ahead /= np.maximum(np.hypot(*ahead.T), 1e-12)[:, None]
    aside = pixels[inside] - points[nearest]
    segment["across"] = np.clip((ahead[:, 0] * aside[:, 1] - ahead[:, 1] * aside[:, 0]) / radius, -1, 1)
    gap = segment.get("gap")
    if gap is not None:
        away = np.abs(np.arange(len(points)) - gap["place"]) / (gap["length"] / 2)
        fade = 1 - (1 - gap["depth"]) * np.where(away < 1, 0.5 * (1 + np.cos(np.pi * away)), 0.0)
        segment["chord"] = segment["chord"] * fade[segment["along"]]


def laminae_of(segment):
    """The strips across a segment in which its clusters are drawn when the speed changes across the vessel: how
    many (one for a capillary, where the cells go in single file), and the share of the vessel's cross-section
    that lies in each (a cluster is anywhere in the cross-section with the same chance)."""
    count = 1 if segment["radius"] <= SYNTH_FILE else int(np.clip(round(segment["radius"] / SYNTH_LAMINA), 2, 5))
    edges = np.linspace(-1, 1, count + 1)
    area = lambda a: (a * np.sqrt(1 - a ** 2) + np.arcsin(a)) / np.pi + 0.5  # of the unit disc, the share left of a
    return count, np.diff(area(edges))


def image_network(segments, nodes, rng, things, texture=None, extra=None, lines=None, junctions=None, valid=None):
    """Image a network as section 28's synthetic_burst does: clusters of red cells carried through it, a focal plane
    that drifts, a 12-bit camera with shot and read noise, frames dropped. With every thing of section 35 off the
    random numbers are drawn in the same order and the same arithmetic is done as there.

    `rng` draws what section 28 draws; `extra` draws what the things of section 35 need on top of it. `texture`:
    the tissue's texture where it is given (the second fixation shows the first one's tissue), else it is drawn.
    Returns a dict with the frames, their times and pairs, the mean, the quiet pixels, the texture, and each
    tree's density without noise (`layers`)."""
    rows, columns = SYNTH_SHAPE
    y, x = np.mgrid[:rows, :columns]
    pixels = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    for segment in segments:
        footprint(segment, pixels, things["fine"])
    profile = bool(things["profile"])
    trees = sorted({segment["tree"] for segment in segments})
    # clusters of red cells: (segment, place along it, size), spread over the network to begin with
    clusters = [[number, place, rng.lognormal(0, 0.4)] for number, segment in enumerate(segments)
                for place in rng.uniform(0, len(segment["points"]), rng.poisson(len(segment["points"]) / CLUSTER_SPACING))]
    if profile:
        # every cluster also has a place in the vessel's cross-section, (a, b) in the unit disc: a across the vessel
        # as the camera sees it, b along the line of sight. Its speed depends on how far it is from the axis.
        for cluster in clusters:
            radius, angle = np.sqrt(extra.random()), extra.uniform(0, 2 * np.pi)
            cluster += [radius * np.cos(angle), radius * np.sin(angle)]
        for segment in segments:
            segment["laminae"], segment["lamina share"] = laminae_of(segment)
            segment["lamina"] = np.minimum(((segment["across"] + 1) / 2 * segment["laminae"]).astype(int), segment["laminae"] - 1)
            # the speed on the axis that gives the segment its mean speed: the mean of 1 - r^k over the disc is k / (k + 2)
            segment["speed max"] = segment["speed"] * ((SYNTH_BLUNT + 2) / SYNTH_BLUNT if segment["radius"] > SYNTH_FILE else 1.0)

    def pace(segment, cluster):
        """How far a cluster moves along a segment in one frame."""
        if not profile or segment["radius"] <= SYNTH_FILE:
            return segment["speed"]
        return segment["speed max"] * (1 - np.hypot(cluster[3], cluster[4]) ** SYNTH_BLUNT)

    # where a cluster goes that leaves a segment by one of its ends: the segments that carry blood away from the
    # node there, with the odds of their flux
    ends = node_ends(segments, nodes)
    beyond = {}
    for number, segment in enumerate(segments):
        for end, node in ((0, segment["start"]), (1, segment["end"])):
            onward = [(other, other_end) for other, other_end in ends[node]
                      if other != number and (segments[other]["speed"] > 0) == (other_end == 0) and segments[other]["speed"] != 0]
            odds = np.array([abs(segments[other]["speed"]) * segments[other]["radius"] ** 2 for other, _ in onward])
            beyond[(number, end)] = (onward, odds)
    # where the blood comes in: at the end of a segment that nothing else reaches, when the segment's blood leaves it
    enters = [len(ends[segment["start"] if segment["speed"] > 0 else segment["end"]]) == 1 for segment in segments]
    illumination = SYNTH_LIGHT * (1 + 0.15 * np.cos(2 * np.pi * x / columns / 1.7) * np.cos(2 * np.pi * y / rows / 2.3))
    if texture is None:
        # the tissue's own texture: it stays in the mean frame, where it is the noise the vessels have to stand out from
        texture = cv2.GaussianBlur(rng.normal(0, 1, (rows, columns)).astype(np.float32), (0, 0), SYNTH_GRAIN)
        texture *= SYNTH_TEXTURE / texture.std()
    kept = [frame for frame in range(SYNTH_FRAMES) if frame not in SYNTH_DROPPED]
    frames = np.zeros((len(kept), rows, columns), dtype=np.float32)
    layers = np.zeros((max(trees) + 1, rows, columns), dtype=np.float32)  # each tree's density without noise, mean over the frames
    depth_of = {tree: next(segment["depth"] for segment in segments if segment["tree"] == tree) for tree in trees}
    halos = things.get("halo signs") or {}
    for frame in range(SYNTH_FRAMES):
        # move the clusters; at a split one daughter is taken with the odds of its share of the flux
        moved = []
        for cluster in clusters:
            number, place = cluster[0], cluster[1]
            segment = segments[number]
            step = pace(segment, cluster)
            place += step
            for _ in range(len(segments) if profile else 1):
                count = len(segments[number]["points"])
                if not (place >= count or place < 0):
                    break
                onward, odds = beyond[(number, 1 if place >= count else 0)]
                if not onward:
                    break
                left = place - count if place >= count else -place  # how far beyond the end it came
                other, other_end = onward[rng.choice(len(odds), p=odds / odds.sum())] if len(onward) > 1 else onward[0]
                if profile:  # the rest of the frame is spent in the next segment, at the cluster's speed there
                    left *= abs(pace(segments[other], cluster)) / max(abs(step), 1e-9)
                    step = pace(segments[other], cluster)
                    place = left if other_end == 0 else len(segments[other]["points"]) - 1.0 - left
                else:
                    place = 0.0 if other_end == 0 else len(segments[other]["points"]) - 1.0
                number = other
            if 0 <= place < len(segments[number]["points"]):
                moved.append([number, place] + cluster[2:])
        clusters = moved
        # new clusters come in where the blood comes in: at the root, or at the leaves
        for number, segment in enumerate(segments):
            if not enters[number]:
                continue
            start = 0.0 if segment["speed"] > 0 else len(segment["points"]) - 1.0
            if not profile:
                if rng.random() < abs(segment["speed"]) / CLUSTER_SPACING:
                    clusters.append([number, start, rng.lognormal(0, 0.4)])
            elif rng.random() < abs(segment["speed max" if segment["radius"] > SYNTH_FILE else "speed"]) / CLUSTER_SPACING:
                # more clusters come in near the axis, where the blood is faster: the cross-section stays evenly filled
                radius, angle = np.sqrt(extra.random()), extra.uniform(0, 2 * np.pi)
                if segment["radius"] <= SYNTH_FILE or extra.random() < 1 - radius ** SYNTH_BLUNT:
                    clusters.append([number, start, rng.lognormal(0, 0.4), radius * np.cos(angle), radius * np.sin(angle)])
        if frame not in kept:
            continue
        focus = SYNTH_FOCUS[0] + (SYNTH_FOCUS[1] - SYNTH_FOCUS[0]) * frame / (SYNTH_FRAMES - 1)
        density = texture.copy()
        for tree in trees:
            layer = np.zeros(rows * columns, dtype=np.float32)
            for number, segment in enumerate(segments):
                if segment["tree"] != tree:
                    continue
                if profile and segment["laminae"] > 1:
                    here = np.array([[c[1], c[2], c[3]] for c in clusters if c[0] == number]).reshape(-1, 3)
                    strip = np.minimum(((here[:, 2] + 1) / 2 * segment["laminae"]).astype(int), segment["laminae"] - 1)
                    modulation = np.zeros((segment["laminae"], len(segment["points"])))
                    np.add.at(modulation, (strip, here[:, 0].astype(int)), here[:, 1])
                    modulation = (scipy.ndimage.gaussian_filter1d(modulation, CLUSTER_LENGTH, axis=1, mode="constant")
                                  * CLUSTER_SPACING / segment["lamina share"][:, None])
                    blood = segment["chord"] * (1 + CLUSTER_DEPTH * (modulation[segment["lamina"], segment["along"]] - 1))
                else:
                    here = np.array([[c[1], c[2]] for c in clusters if c[0] == number]).reshape(-1, 2)
                    modulation = np.bincount(here[:, 0].astype(int), weights=here[:, 1], minlength=len(segment["points"]))
                    modulation = scipy.ndimage.gaussian_filter1d(modulation, CLUSTER_LENGTH, mode="constant") * CLUSTER_SPACING
                    blood = segment["chord"] * (1 + CLUSTER_DEPTH * (modulation[segment["along"]] - 1))
                layer[segment["pixels"]] = np.maximum(layer[segment["pixels"]], blood)  # one lumen where a tree's segments meet
            blur = SYNTH_SHARP + SYNTH_DEFOCUS * abs(depth_of[tree] - focus)
            if things["transmittance"]:  # the camera blurs the light that comes through, not the density
                layer = -np.log(cv2.GaussianBlur(np.exp(-layer.reshape(rows, columns)), (0, 0), blur))
            else:
                layer = cv2.GaussianBlur(layer.reshape(rows, columns), (0, 0), blur)
            layers[tree] += layer / len(kept)
            density += layer  # densities of two depths add
            if tree in halos:
                reach = SYNTH_HALO_REACH * max(segment["radius"] for segment in segments if segment["tree"] == tree)
                density += np.float32(halos[tree] * SYNTH_HALO) * cv2.GaussianBlur(layer, (0, 0), reach)
        light = illumination * np.exp(-density)
        seen = rng.poisson(light * SYNTH_GAIN) / SYNTH_GAIN + rng.normal(0, SYNTH_READ, light.shape)
        seen = np.clip(np.rint(seen), 1, SYNTH_SATURATION)
        frames[kept.index(frame)] = np.log(illumination) - np.log(seen)
    times = np.array(kept) / SYNTH_FPS
    pairs = [(place, place + 1, float(times[place + 1] - times[place])) for place in range(len(kept) - 1)
             if kept[place + 1] - kept[place] <= MAX_GAP_FRAMES]
    wall = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, 5)
    mean, quiet = frames.mean(axis=0), wall > 3 * max(s["radius"] for s in segments)
    out = {"times": times, "pairs": pairs, "texture": texture, "layers": layers}
    if things["background"]:
        # The true background is no longer handed over. The frames are measured against a background fitted to the
        # mean of the frames as recorded: section 12's fit_background ("fit"), or the true background with a local
        # mean of the density taken off on the quiet pixels ("local mean": the harness is approximate there).
        valid_here = np.ones((rows, columns), dtype=bool) if valid is None else valid
        if things["background"] == "fit":
            recorded = illumination * np.exp(-frames.astype(np.float64)).mean(axis=0)  # the mean frame, in DN
            fit = fit_background(recorded, valid=valid_here, sigma=SYNTH_BACKGROUND, saturation=SYNTH_SATURATION)
            shift = (fit["B"] - np.log(illumination)).astype(np.float32)
        else:
            shift = (-masked_mean(mean, quiet & valid_here, SYNTH_BACKGROUND)).astype(np.float32)
        frames += shift
        mean = frames.mean(axis=0)
        out["background error"] = shift  # the fitted background minus the true one, in density
    if valid is not None:
        quiet = quiet & valid
    out.update({"frames": frames, "mean": mean, "quiet": quiet})
    if junctions is not None:
        mark_visible_35(junctions, segments, nodes, layers, robust_sd(mean[quiet]))
    return out


def synthetic_burst_35(seed, kind="trees", **things):
    """A synthetic burst with the things of section 35, each switched on by its name (SYNTH_THINGS has the settings
    used for the new bursts; with none given, every one is off and the burst is section 28's synthetic_burst(seed),
    bit for bit, with new keys beside the old ones).

      kind           "trees": the three trees of section 28; "crossing plexus", "anastomotic plexus": plexus_network
      capillaries    the number of capillary trees added (capillary_trees)
      loops          the chance, per tree, that a daughter runs back into the other one (close_loop)
      gaps           the chance, per segment, of a gap SYNTH_GAP_LENGTH long that fades to SYNTH_GAP_DEPTH
      ends           the chance that a segment that leaves the frame stops inside it instead
      halos          a halo of either sign around every tree with a vessel of SYNTH_HALO_LEAST or more
      overlaps       the number of hidden vessels: a vessel of another depth lying along a stretch of a vessel,
                     with nothing emerging at either end
      profile        the speed changes across a vessel, v_max (1 - (r / R)^SYNTH_BLUNT), and a cluster that passes
                     a node spends the rest of the frame in the next segment
      transmittance  each tree's transmittance is blurred, not its density
      background     "fit": the frames are measured against section 12's fit of the mean frame; "local mean": against
                     the true background, with a local mean of the density taken off; None: the true background
      second         a second fixation: the same network and tissue moved by a known shift, imaged again
      fine           the density of a pixel is a mean over SYNTH_FINE x SYNTH_FINE points

    Returns the dict of synthetic_burst, with: `kind`, `things`; `nodes`: per node x, y and the (segment, end) that
    meet in it; `gaps`, `terminations`, `overlaps`, `halos`: what was put in, each recorded; `counts`: how many of
    each (COUNTED); `second`: the second fixation as a burst of its own (frames, times, pairs, mean, valid, quiet,
    lines) with `shift` (dx, dy) and `to second`, the 3x3 map from the first fixation's (x, y) to the second's.
    Segments have, new: `start`, `end` (their nodes), `flux` (speed x radius^2, signed along the points), and
    where it applies `gap`, `host`, `loop`, `speed max`. Junctions: see network_truth_35.
    """
    unknown = set(things) - set(SYNTH_PLAIN)
    if unknown or kind not in SYNTH_KINDS:
        raise ValueError(f"not a thing or a kind of section 35: {sorted(unknown) or kind}")
    things = {**SYNTH_PLAIN, **things}
    rng = np.random.default_rng(seed)
    rows, columns = SYNTH_SHAPE
    gaps, terminations, overlaps, loops = [], [], [], 0
    if kind == "trees":
        segments = synthetic_network(rng)
        nodes = tree_nodes(segments)
        if things["capillaries"]:
            capillary_trees(stream(seed, "capillaries"), segments, nodes, things["capillaries"])
        if things["loops"]:
            draw = stream(seed, "loops")
            for trunk in [number for number, segment in enumerate(segments) if segment["parent"] is None]:
                if draw.random() < things["loops"]:
                    loops += close_loop(draw, segments, nodes, trunk)
        if things["ends"]:
            draw = stream(seed, "ends")
            ends = node_ends(segments, nodes)
            for number, segment in enumerate(segments):
                leaf = len(ends[segment["end"]]) == 1 and segment["parent"] is not None and not segment.get("loop")
                if leaf and len(segment["points"]) >= 90 and draw.random() < things["ends"]:
                    stop = int(draw.uniform(50, min(140, len(segment["points"]) - 30)))
                    if inside_frame(segment["points"][stop], 30):
                        segment["points"] = segment["points"][:stop + 1]
                        nodes[segment["end"]] = segment["points"][-1].copy()
                        terminations.append({"segment": number, "x": float(segment["points"][-1, 0]), "y": float(segment["points"][-1, 1])})
        if things["overlaps"]:
            draw = stream(seed, "overlaps")
            hosts = [number for number, segment in enumerate(segments) if len(segment["points"]) >= 140 and segment["radius"] >= 2.5]
            for _ in range(things["overlaps"] if hosts else 0):
                host = int(draw.choice(hosts))
                length = int(draw.uniform(40, 80))
                first = int(draw.uniform(30, len(segments[host]["points"]) - length - 30))
                hidden = {"points": segments[host]["points"][first:first + length].copy(),
                          "radius": segments[host]["radius"] * draw.uniform(0.6, 1.0), "speed": draw.uniform(1.0, 3.0) * draw.choice([-1, 1]),
                          "tree": max(segment["tree"] for segment in segments) + 1, "depth": draw.uniform(-0.6, 0.6),
                          "parent": None, "host": host}
                segments.append(hidden)
                tree_nodes(segments, nodes, len(segments) - 1)
                overlaps.append({"segment": len(segments) - 1, "host": host, "from": first, "length": length})
    else:
        segments, nodes = plexus_network(seed, kind == "anastomotic plexus")
    if things["gaps"]:
        draw = stream(seed, "gaps")
        for number, segment in enumerate(segments):
            if "host" in segment or len(segment["points"]) < 4 * SYNTH_GAP_LENGTH or draw.random() >= things["gaps"]:
                continue
            place = int(draw.uniform(1.5 * SYNTH_GAP_LENGTH, len(segment["points"]) - 1.5 * SYNTH_GAP_LENGTH))
            if inside_frame(segment["points"][place], 10):
                segment["gap"] = {"place": place, "length": SYNTH_GAP_LENGTH, "depth": SYNTH_GAP_DEPTH}
                gaps.append({"segment": number, "place": place, "x": float(segment["points"][place, 0]),
                             "y": float(segment["points"][place, 1])})
    halos = {}
    if things["halos"]:
        draw = stream(seed, "halos")
        for tree in sorted({segment["tree"] for segment in segments}):
            if max((segment["radius"] for segment in segments if segment["tree"] == tree and "host" not in segment),
                   default=0) >= SYNTH_HALO_LEAST:
                halos[tree] = int(draw.choice([-1, 1]))
    extended = kind != "trees" or any(things[name] != SYNTH_PLAIN[name] for name in things)
    lines, junctions = network_truth_35(segments, nodes) if extended else network_truth(segments)
    imaged = image_network(segments, nodes, rng, {**things, "halo signs": halos}, extra=stream(seed, "profile"),
                           lines=lines, junctions=junctions)
    if not extended:  # the junctions of section 28 have the new keys too; they are added after the old ones were made
        for junction, full in zip(junctions, network_truth_35(segments, nodes)[1]):
            junction.update({name: full[name] for name in full if name not in junction})
    for segment in segments:
        segment["flux"] = float(segment["speed"] * segment["radius"] ** 2)
    ends = node_ends(segments, nodes)
    burst = {"frames": imaged["frames"], "times": imaged["times"], "pairs": imaged["pairs"], "fps": SYNTH_FPS,
             "mean": imaged["mean"], "valid": np.ones((rows, columns), dtype=bool), "quiet": imaged["quiet"],
             "segments": segments, "lines": lines, "junctions": junctions, "kind": kind, "things": things,
             "nodes": [{"x": float(place[0]), "y": float(place[1]), "ends": ends[node]} for node, place in enumerate(nodes)],
             "gaps": gaps, "terminations": terminations, "overlaps": overlaps, "halos": halos, "layers": imaged["layers"]}
    if "background error" in imaged:
        burst["background error"] = imaged["background error"]
    asked = [junction for junction in vascnet.evaluate.cluster_truth(
        [dict(junction, type=as_vascnet(junction["type"]) if junction["type"] != "anastomosis" else "crossover")
         for junction in junctions]) if junction.get("visible", True)]
    burst["counts"] = {
        "trees": len({segment["tree"] for segment in segments}), "segments": len(segments),
        "capillary segments": sum(segment["radius"] <= SYNTH_FILE for segment in segments),
        "bifurcations": sum(junction["type"] == "bifurcation" for junction in junctions), "loops": int(loops),
        "gaps": len(gaps), "terminations": len(terminations),
        "projected crossings": sum(junction["type"] == "crossing" for junction in junctions),
        "anastomoses": sum(junction["type"] == "anastomosis" for junction in junctions),
        "collinear overlaps": len(overlaps), "halos": len(halos), "junctions asked for": len(asked)}
    if things["second"]:
        # the second fixation: the same network and the same tissue, moved by a known shift, imaged again with
        # clusters, camera noise and frames of its own. Where the first fixation's field does not reach, the second
        # has no data: the network was only drawn inside the first one's frame.
        draw = stream(seed, "second")
        shift = np.array([draw.uniform(*SYNTH_SHIFT) * draw.choice([-1, 1]) for _ in range(2)])  # (dx, dy)
        to_second = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]])
        moved = []
        for segment in segments:
            moved.append({name: value for name, value in segment.items() if name not in ("pixels", "along", "chord", "across", "lamina")})
            moved[-1]["points"] = segment["points"] + shift
        moved_nodes = [place + shift for place in nodes]
        size = (columns, rows)
        texture = cv2.warpAffine(imaged["texture"], to_second[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        valid = cv2.warpAffine(np.ones((rows, columns), dtype=np.float32), to_second[:2], size, flags=cv2.INTER_LINEAR) > 0.999
        moved_lines = np.zeros(SYNTH_SHAPE, dtype=np.uint8)
        for segment in moved:
            cv2.polylines(moved_lines, np.rint(segment["points"]).astype(np.int32)[None], False, 1)
        again = image_network(moved, moved_nodes, draw, {**things, "halo signs": halos}, texture=texture,
                              extra=stream(seed, "second profile"), lines=moved_lines > 0, valid=valid)
        burst["second"] = {"frames": again["frames"], "times": again["times"], "pairs": again["pairs"], "fps": SYNTH_FPS,
                           "mean": again["mean"], "valid": valid, "quiet": again["quiet"], "lines": (moved_lines > 0) & valid,
                           "shift": (float(shift[0]), float(shift[1])), "to second": to_second}
    return burst


SECTION_28_SEGMENT_KEYS = ("points", "radius", "speed", "tree", "depth", "parent", "pixels", "along", "chord")
SECTION_28_JUNCTION_KEYS = ("x", "y", "type", "radius", "segments", "places", "contrast", "visible", "ambiguous")


def section_28_part(burst):
    """What synthetic_burst (section 28) returns, taken out of a burst of synthetic_burst_35: its keys, its
    segments' keys and its junctions' keys, without what section 35 adds."""
    old = {key: burst[key] for key in ("frames", "times", "pairs", "fps", "mean", "valid", "quiet", "segments", "lines", "junctions")}
    old["segments"] = [{key: segment[key] for key in SECTION_28_SEGMENT_KEYS} for segment in burst["segments"]]
    old["junctions"] = [{key: junction[key] for key in SECTION_28_JUNCTION_KEYS if key in junction} for junction in burst["junctions"]]
    return old


def same_bits(first, second):
    """Whether two values are the same to the bit: arrays of one dtype and shape with the same bytes, dicts with the
    same keys, lists of the same length, numbers of one type that are equal (NaN equal to NaN)."""
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (isinstance(first, np.ndarray) and isinstance(second, np.ndarray) and first.dtype == second.dtype
                and first.shape == second.shape and first.tobytes() == second.tobytes())
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(same_bits(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(map(same_bits, first, second))
    return type(first) is type(second) and (first == second or (first != first and second != second))

# %%
# the 32 old seeds, made by both generators: with no thing switched on they have to be the same to the bit
old_seeds = HARNESS_SEEDS + HELD_OUT_SEEDS
differ = [seed for seed in old_seeds if not same_bits(synthetic_burst(seed), section_28_part(synthetic_burst_35(seed)))]
print(f"Old seeds: {len(old_seeds) - len(differ)} of {len(old_seeds)} are the same to the bit when made by section 35's "
      f"generator with no thing switched on" + (f"; DIFFERENT: {differ}" if differ else ""))

# what every new seed holds
synthetic_counts = []  # one row per new burst: its set, kind and seed, and its counts (COUNTED)
sets = (("trees, developing", "trees", TREE_SEEDS), ("trees, held out", "trees", TREE_HELD_OUT),
        ("plexus, developing", "crossing plexus", PLEXUS_SEEDS), ("plexus, developing", "anastomotic plexus", PLEXUS_SEEDS),
        ("plexus, held out", "crossing plexus", PLEXUS_HELD_OUT), ("plexus, held out", "anastomotic plexus", PLEXUS_HELD_OUT))
for name, kind, seeds in sets:
    for seed in seeds:
        burst = synthetic_burst_35(seed, kind, **SYNTH_THINGS)
        synthetic_counts.append({"set": name, "kind": kind, "seed": seed, **burst["counts"],
                                 "shift": "{:+.1f} {:+.1f}".format(*burst["second"]["shift"]),
                                 "background error": float(robust_sd(burst["background error"][burst["quiet"]]))})
del burst
short = {"capillary segments": "capil", "bifurcations": "bifur", "terminations": "ends", "projected crossings": "cross",
         "anastomoses": "anast", "collinear overlaps": "overl", "junctions asked for": "asked"}
print(f"\nPer new burst: {', '.join(f'{short.get(name, name)} = {name}' for name in COUNTED if name in short)}")
print(f"{'set':20s} {'kind':19s} {'seed':>5s} " + " ".join(f"{short.get(name, name):>6s}" for name in COUNTED)
      + f" {'shift (x y)':>12s} {'bg error':>8s}")
for row in synthetic_counts:
    print(f"{row['set']:20s} {row['kind']:19s} {row['seed']:5d} " + " ".join(f"{row[name]:6d}" for name in COUNTED)
          + f" {row['shift']:>12s} {row['background error']:8.4f}")
print("\nIn all, per set and kind:")
for name, kind, seeds in sets:
    rows = [row for row in synthetic_counts if row["set"] == name and row["kind"] == kind]
    print(f"{name:20s} {kind:19s} {len(rows):3d} bursts | " + ", ".join(f"{sum(row[count] for row in rows)} {count}" for count in COUNTED
                                                                     if count not in ("trees", "segments")))
os.makedirs("vessel_pictures/synthetic_35", exist_ok=True)
save_table(synthetic_counts, "vessel_pictures/synthetic_35/counts.csv")

# one burst of each kind: a frame, the mean with what is true of it, and the second fixation
cases = ((TREE_SEEDS[1], "trees"), (PLEXUS_SEEDS[0], "crossing plexus"), (PLEXUS_SEEDS[0], "anastomotic plexus"))
colors = {"split": "gold", "loop": "orange", "projected crossing": "cyan", "anastomosis": "magenta"}
figure, axes = plt.subplots(len(cases), 3, figsize=(18, 4.6 * len(cases)))
for row, (seed, kind) in enumerate(cases):
    burst = synthetic_burst_35(seed, kind, **SYNTH_THINGS)
    top = np.percentile(burst["mean"], 99.8)
    axes[row, 0].imshow(burst["frames"][10], cmap="gray_r", vmin=-0.05, vmax=top)
    axes[row, 0].set_title(f"{kind}, seed {seed}: one frame (density)", fontsize=10)
    axes[row, 1].imshow(burst["mean"], cmap="gray_r", vmin=-0.05, vmax=top)
    for junction in burst["junctions"]:
        axes[row, 1].add_patch(plt.Circle((junction["x"], junction["y"]), 9, fill=False, lw=1.2,
                                          color=colors[junction["kind"]] if junction["visible"] else "gray"))
    for gap in burst["gaps"]:
        axes[row, 1].plot(gap["x"], gap["y"], "x", color="red", ms=10, mew=2)
    for end in burst["terminations"]:
        axes[row, 1].plot(end["x"], end["y"], "s", color="lime", ms=7, mfc="none", mew=2)
    for overlap in burst["overlaps"]:
        points = burst["segments"][overlap["segment"]]["points"]
        axes[row, 1].plot(points[:, 0], points[:, 1], "-", color="purple", lw=4, alpha=0.5)
    axes[row, 1].set_title("the mean, with the truth: junctions (gold split, orange loop closed, cyan crossing, magenta "
                           "anastomosis, gray nobody is asked for it), gaps (red x), terminations (green), hidden vessel (purple)", fontsize=7)
    axes[row, 2].imshow(np.where(burst["second"]["valid"], burst["second"]["mean"], np.nan), cmap="gray_r", vmin=-0.05, vmax=top)
    axes[row, 2].set_title("the second fixation, moved by ({:+.1f}, {:+.1f}) px".format(*burst["second"]["shift"]), fontsize=10)
    for axis in axes[row]:
        axis.axis("off")
figure.tight_layout()
figure.savefig("vessel_pictures/synthetic_35/kinds.png", dpi=90)
plt.show()
del burst
memory_note("section 35")

# %% [markdown]
# ## 37. The clean cyclic graph: every piece kept, each with a confidence (acceptance step 2)
#
# Section 29 consolidates its ridges: it joins them, then drops side branches, small networks and short pieces by their length. The plan's first principle is that nothing is dropped below the thinning scale. Every piece is kept and carries a confidence instead, and that has to be worth something. **The gate:** on the held-out bursts, the edge confidence must tell true pieces from false ones with `told_apart` ≥ 0.75 on its five bottom-up terms, and its curve must pass the consolidated row's point. Otherwise section 29's consolidation is kept and the plan stops here for the graph.
#
# **The pieces** (`cyclic_pieces`). They are built in five steps:
#
# 1. `ridge_crests` on the pre-interaction winner (section 29's saved maps; never section 21's field).
# 2. Closed within `RIDGE_JOIN`, and thinned with section 18's `SPUR_LENGTH` pruning (`joined_ridges`).
# 3. Every free end extended into a line it meets (`extend_ends`).
# 4. Cut fork to fork by `ridge_lines`, with nothing dropped (`least = 0`, `to_forks`).
# 5. Each point moved to its crest's place across the line, to a fraction of a pixel (`crest_places`; the pixel's centre where it has no crest).
#
# Each piece is then fitted with `spline_vessel` after a knot refinement (`refine_knots`). That refinement is priced in the ridge's own unit as a BIC, with no λ:
#
# - the χ² is the summed squared distance from the spline over `RIDGE_PRECISION`², counted in independent samples (`RIDGE_CORRELATION` points make one);
# - each added coefficient costs ln(samples);
# - one move halves the knot spacing, down to `KNOT_LEAST`;
# - the other move cuts the piece at its point of largest residual. It is taken only where the two halves meet at a bend of more than `FUSE_STRAIGHT`.
#
# **The five bottom-up terms** (`edge_terms`):
#
# | term | how it is read |
# |---|---|
# | length | in field lengths (`ELONGATION` half-widths), zero at one and full by three |
# | strength | the median pre-interaction score over `T_HIGH` |
# | repeatability | `found_again` per edge against the other fixation's joined ridges, above chance (`edge_repeatability`); on the harness, section 35's second fixation |
# | flow data | `flow_share` |
# | twin or stub | a twin is a piece running alongside a longer one by `read_arms`' rule; a stub is a free-ended piece with less than `STUB_FIELDS` field lengths outside its parents' tubes (`twins_and_stubs`) |
#
# **Section 41's calibration** (`calibrate_terms`). Each term is binned on the bursts for developing, and each bin gets its log-likelihood ratio of a true piece against a false one. The prior is kept apart. The confidence is the prior plus the terms' ratios, squashed (`edge_confidence`). A piece is true when most of its points lie within 2 px of a true centerline, the harness's tolerance.
#
# **The curves.** At each cutoff of the confidence, the pieces kept are read as section 29 reads its pieces (a kymograph per piece, the graph, the junctions with flow as tests) and scored with `score_reading`. The curves are compared with the plain and the consolidated rows, read from the same bursts.
#
# **Two sets of bursts.**
#
# - *Section 30's bursts.* The old seeds are made with a second fixation; their first fixation is section 30's burst to the bit. They are calibrated on `HARNESS_SEEDS` and measured on `HELD_OUT_SEEDS`; the gate is read here.
# - *Section 35's trees.* These have every thing of section 35 on. They are calibrated on `TREE_SEEDS` and measured on `TREE_HELD_OUT`.
#
# **Measured here where the plan guessed** (see `reports/step_2_<date>.md`):
#
# - `RIDGE_PRECISION` is 0.38 px with the sub-pixel places and 0.46 px at the pixels' centres.
# - The points' deviations are correlated over 1.9 points.
# - The sub-pixel places come from the crests of the pre-interaction score, which is the score the ridges come from. Section 29's saved inputs have no interaction field, and these crests give a lower residual on more pixels.
#
# **Result:** per piece its terms and confidence (`vessel_pictures/cyclic/pieces.csv`), the curves (`curves.csv`, `curves.png`), the reliability diagrams (`reliability.png`), the real burst's pieces (`real_confidence.png`), and the knot refinement's film (`vessel_pictures/fit/initialisation.gif`). κ, the twins' line mass and `spline_to_graph` belong to step 5, with the renderer.

# %%
import vascnet.watch

RIDGE_PRECISION = 0.38    # pixels; the rms distance of the ridge points, at the crests' sub-pixel places, from the true
                          # centerline (measured on the generator's vessels, away from junctions; 0.46 at the pixels' centres)
RIDGE_CORRELATION = 1.9   # points; a ridge point's deviation is correlated with its neighbours': this many points are
                          # worth one independent one (measured: the sum of rho^2 over all lags, median over pieces)
KNOT_LEAST = 6.0          # pixels; the closest knots a spline gets
CONFIDENCE_BINS = 6       # bins of a term on the bursts for developing, where its log-likelihood ratio is read
TRUE_WITHIN = 2.0         # pixels; a piece is true when most of its points lie this close to a true centerline (the
                          # harness's centerline tolerance)
CONFIDENCE_CUTOFFS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # the curves over edge confidence
EDGE_TERMS = ("length", "strength", "repeatability", "flow data", "twin or stub")  # the five bottom-up terms


def joined_ridges(strength, orientation, region):
    """The plain ridges joined, and nothing dropped (section 37): `ridge_crests` on the pre-interaction winner,
    closed within RIDGE_JOIN, thinned with section 18's SPUR_LENGTH pruning (`centerlines`; a stub of 1 to 11 px is
    a thinning artefact), and every free end extended into a line it meets (`extend_ends`). None of
    consolidate_lines' dropping steps (RIDGE_CHILD, LINE_LEAST, RIDGE_NETWORK) is run.

    Returns a dict: `ridges` (the crests), `skeleton` (the joined lines, one pixel wide), `added` (the pixels the
    extensions added: extended & ~lines) and `closed` (the skeleton before the extensions)."""
    ridges = ridge_crests(strength, orientation, region)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RIDGE_JOIN + 1,) * 2)
    lines = centerlines(cv2.morphologyEx(ridges.astype(np.uint8), cv2.MORPH_CLOSE, disc) > 0)
    extended = extend_ends(lines, strength)
    return {"ridges": ridges, "skeleton": skeletonize(extended), "added": extended & ~lines, "closed": lines}


def crest_places(scores, orientation, valid):
    """The place of the ridge across its line to a fraction of a pixel, at every pixel that has one: the crest
    neurons (section 21's crest_neurons) of the pre-interaction score, the score the ridges are read from, of the
    orientation nearest to the pixel's winning one. Returns (x, y) pictures, NaN where a pixel has no crest neuron
    within one orientation step of its winner; the ridge then keeps the pixel's centre."""
    stack = np.asarray(scores, dtype=np.float32)
    widths = np.zeros_like(stack, dtype=np.float32)
    crests = crest_neurons(stack, widths, valid)
    turn = np.abs(crests["k"] - orientation[crests["row"], crests["column"]].astype(np.int16)) % N_ORIENT
    turn = np.minimum(turn, N_ORIENT - turn)
    kept = np.flatnonzero(turn <= 1)
    order = kept[np.argsort(-turn[kept], kind="stable")]  # per pixel the nearest orientation is written last
    x, y = (np.full(orientation.shape, np.nan, dtype=np.float32) for _ in range(2))
    x[crests["row"][order], crests["column"][order]] = crests["x"][order]
    y[crests["row"][order], crests["column"][order]] = crests["y"][order]
    return x, y


def fine_lines(skeleton, strength, orientation, half_width, places):
    """`ridge_lines` with least = 0 and to_forks (every piece fork to fork or fork to free end, nothing dropped),
    each point moved to its crest's place across the line where it has one (`crest_places`; `offset` says which
    did). A piece of two points keeps them (fit_spline fits a line to them)."""
    lines = ridge_lines(skeleton, strength, orientation, half_width, least=0, to_forks=True)
    x, y = places
    for line in lines:
        row, column = np.rint(line["y"]).astype(int), np.rint(line["x"]).astype(int)
        crest_x, crest_y = x[row, column], y[row, column]
        line["offset"] = np.isfinite(crest_x)
        line["x"] = np.where(line["offset"], crest_x, line["x"]).astype(np.float32)
        line["y"] = np.where(line["offset"], crest_y, line["y"]).astype(np.float32)
    return lines


def along_of(points):
    """The distance along a polyline at each of its points."""
    return np.r_[0, np.cumsum(np.maximum(np.hypot(*np.diff(points, axis=0).T), 1e-3))]


def priced_fit(points, spacing):
    """A place spline with knots `spacing` apart and its chi-square in the ridge's own unit: the summed squared
    distance of the points from it over RIDGE_PRECISION^2, counted in independent samples (RIDGE_CORRELATION points
    make one). Returns the spline, the chi-square and the number of coefficients."""
    along = along_of(points)
    place = fit_spline(points, along, spacing)
    chi2 = float(np.sum((place(along) - points) ** 2) / (RIDGE_PRECISION ** 2 * RIDGE_CORRELATION))
    return place, chi2, 2 * len(place.c)


def refine_knots(chain, watch=None):
    """The knot refinement of section 37, priced in the ridge's own unit with no lambda. Two moves are priced as a
    BIC with sigma RIDGE_PRECISION over the independent samples (points / RIDGE_CORRELATION), each paying ln(samples)
    per added coefficient:

      halving  the piece's knot spacing halved, from KNOT_SPACING down to KNOT_LEAST at the closest;
      split    the piece cut in two at its point of largest residual, each half fitted at the same spacing; taken
               only where the two halves meet at a bend of more than FUSE_STRAIGHT, so that a ridge pulled aside
               for a few pixels (by a crossing vessel, at the frame's edge) is not cut into two.

    The move that gains most is taken while one gains; the halves of a split are refined again from KNOT_SPACING.
    Returns a list of (first point, last point, spacing): the pieces the chain is fitted as. `watch`, if given, is
    handed every candidate move (a `vascnet.watch.Recorder`)."""
    points = np.column_stack([chain["x"], chain["y"]]).astype(np.float64)
    out, waiting = [], [(0, len(points) - 1)]
    while waiting:
        first, last = waiting.pop()
        part = points[first:last + 1]
        spacing = KNOT_SPACING
        place, chi2, count = priced_fit(part, spacing)
        price = np.log(max(len(part) / RIDGE_CORRELATION, 2.0))
        cut = None
        while len(part) >= 8:
            moves = []
            if spacing / 2 >= KNOT_LEAST:
                finer = priced_fit(part, spacing / 2)
                moves.append({"kind": "knots", "spacing": spacing / 2, "fit": finer,
                              "gain": chi2 - finer[1] - price * max(finer[2] - count, 1)})
            residual = np.hypot(*(place(along_of(part)) - part).T)
            at = int(np.argmax(residual[3:-3])) + 3 if len(part) > 8 else None
            if at is not None:
                halves = [priced_fit(part[:at + 1], spacing), priced_fit(part[at:], spacing)]
                ahead = [halves[0][0](along_of(part[:at + 1])[-1], 1), halves[1][0](0.0, 1)]
                cosine = np.dot(*ahead) / max(np.hypot(*ahead[0]) * np.hypot(*ahead[1]), 1e-12)
                bend = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
                moves.append({"kind": "split", "at": at, "bend": bend, "spacing": spacing,
                              "gain": chi2 - halves[0][1] - halves[1][1] - price * max(halves[0][2] + halves[1][2] - count, 1)
                              if bend > FUSE_STRAIGHT else -np.inf})
            best = max(moves, key=lambda move: move["gain"]) if moves else None
            for move in moves:
                if watch is not None:
                    taken = move is best and move["gain"] > 0
                    curve = (move["fit"][0] if move["kind"] == "knots" else place)(np.linspace(0, along_of(part)[-1], 4 * len(part)))
                    watch({"kind": move["kind"], "taken": bool(taken), "piece": (first, last), "spacing": move["spacing"],
                           "gain": float(move["gain"]) if np.isfinite(move["gain"]) else -1e9, "points": part.copy(),
                           "fit": curve.copy(), "at": first + move.get("at", 0)})
            if best is None or best["gain"] <= 0:
                break
            if best["kind"] == "split":
                cut = first + best["at"]
                break
            spacing, (place, chi2, count) = best["spacing"], best["fit"]
        if cut is not None:
            waiting += [(first, cut), (cut, last)]
        else:
            out.append((first, last, spacing))
    return sorted(out)


def other_lines_here(valid, other_lines, other_valid, to_other):
    """The other fixation's lines brought into this fixation's frame, as found_again (section 18) does: per pixel the
    distance to them, and where both fixations have data (the other's data area eroded by 9 x 9)."""
    size = valid.shape[::-1]
    distance = cv2.warpPerspective(distance_to(other_lines), to_other, size,
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=1e3)
    seen = cv2.warpPerspective(other_valid.astype(np.float32), to_other, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return distance, valid & (cv2.erode(np.uint8(seen > 0.99), np.ones((9, 9), np.uint8)) > 0)


def edge_repeatability(edges, valid, other_lines, other_valid, to_other):
    """Per edge, found_again (section 18) applied to the edge alone: the share of its points (where both fixations
    have data) that the other fixation's lines pass within REPEAT_WITHIN, the share they pass by chance (the
    other's lines moved aside by CHANCE_SHIFT, as `repeated` does), and how far the first lies above the second
    (0 chance, 1 found everywhere). `to_other` maps this fixation's (x, y) to the other's. Returns three arrays,
    NaN for an edge with no point where both have data."""
    aside = np.array([[1, 0, CHANCE_SHIFT], [0, 1, CHANCE_SHIFT], [0, 0, 1.0]])
    maps = [other_lines_here(valid, other_lines, other_valid, mapping) for mapping in (to_other, aside @ to_other)]
    rows, columns = valid.shape
    found, chance = np.full(len(edges), np.nan), np.full(len(edges), np.nan)
    for number, edge in enumerate(edges):
        row = np.clip(np.rint(edge["y"]).astype(int), 0, rows - 1)
        column = np.clip(np.rint(edge["x"]).astype(int), 0, columns - 1)
        for values, (distance, both) in zip((found, chance), maps):
            here = both[row, column]
            if here.any():
                values[number] = np.mean(distance[row[here], column[here]] <= REPEAT_WITHIN)
    with np.errstate(invalid="ignore"):
        above = (found - chance) / np.maximum(1 - chance, 1e-6)
    return found, chance, above


def near_pairs(x, y, reach):
    """Every ordered pair (i, j) of points, i != j, closer than the larger of their two reaches, with its distance.
    The points are queried in classes of reach, so that a few thick vessels do not make every thin one look far."""
    tree = scipy.spatial.cKDTree(np.column_stack([x, y]))
    found = []
    limits = np.unique(np.ceil(reach / 4) * 4)
    for low, high in zip(np.r_[0, limits[:-1]], limits):
        members = np.flatnonzero((reach > low) & (reach <= high))
        if not len(members):
            continue
        near = tree.sparse_distance_matrix(scipy.spatial.cKDTree(np.column_stack([x[members], y[members]])), float(high),
                                           output_type="ndarray")
        a, b, distance = near["i"], members[near["j"]], near["v"]
        keep = (a != b) & (distance <= reach[b])
        found.append(np.column_stack([a[keep], b[keep], distance[keep]]))
    pairs = np.concatenate(found) if found else np.zeros((0, 3))
    pairs = np.concatenate([pairs, pairs[:, [1, 0, 2]]])  # both orders
    _, unique = np.unique(pairs[:, 0] * len(x) + pairs[:, 1], return_index=True)
    pairs = pairs[unique]
    return pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2]


def twins_and_stubs(edges, blur, valid):
    """Per edge, the fifth bottom-up term of section 37: "twin" where for most of its points a longer edge runs
    alongside it by read_arms' rule (within SAME_ARM degrees, closer than the larger width plus RING_MARGIN; a
    width is 2 x half-width / CYLINDER); "stub" where it has a free end (away from where the data ends) and fewer
    than STUB_FIELDS field lengths of it lie outside the tubes of its parents (section 31's rule: the parents are the
    edges without a free end and the longer ones; a tube reaches tube_reach from the parent's centerline); else
    "plain"."""
    rows, columns = valid.shape
    edge_of = np.concatenate([np.full(len(edge["x"]), number) for number, edge in enumerate(edges)])
    x, y = np.concatenate([edge["x"] for edge in edges]), np.concatenate([edge["y"] for edge in edges])
    radius = np.concatenate([edge["radius"] for edge in edges]).astype(np.float64)
    theta = np.concatenate([edge["theta"] for edge in edges]).astype(np.float64)
    width = 2 * radius / CYLINDER
    reach = tube_reach(radius, blur).astype(np.float64)
    lengths = np.array([edge["length"] for edge in edges])
    border = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    at = lambda edge, end: border[int(np.clip(round(float(edge["y"][end])), 0, rows - 1)),
                                  int(np.clip(round(float(edge["x"][end])), 0, columns - 1))]
    free = np.array([any(loose and at(edge, end) > FIELD_EDGE for end, loose in zip((0, -1), edge.get("loose", (False, False))))
                     for edge in edges])
    i, j, distance = near_pairs(x, y, np.maximum(width + RING_MARGIN, reach))
    other = edge_of[i] != edge_of[j]
    i, j, distance = i[other], j[other], distance[other]
    longer = lengths[edge_of[j]] > lengths[edge_of[i]]
    turned = np.abs(theta[i] - theta[j]) % np.pi
    alongside = (longer & (np.degrees(np.minimum(turned, np.pi - turned)) < SAME_ARM)
                 & (distance <= np.maximum(width[i], width[j]) + RING_MARGIN))
    beside = np.zeros(len(x), dtype=bool)
    beside[i[alongside]] = True
    parent = ~free[edge_of[j]] | longer
    inside = np.zeros(len(x), dtype=bool)
    inside[i[parent & (distance <= reach[j])]] = True
    kinds = []
    for number, edge in enumerate(edges):
        mine = edge_of == number
        if beside[mine].mean() > 0.5:
            kinds.append("twin")
        elif free[number] and (~inside[mine]).sum() < STUB_FIELDS * ELONGATION * float(np.median(edge["radius"])):
            kinds.append("stub")
        else:
            kinds.append("plain")
    return kinds


def closing_joins(ridges, lines):
    """Where the thinned lines join pieces of ridge (connected parts of `ridges`, before the closing within
    RIDGE_JOIN) that were apart: per join the pieces, its place, and whether it joins two of them end to end (a join
    of two arms) or at a fork. A line pixel belongs to the ridge piece within 1.5 px of it; one that lies farther
    is in a gap the closing filled. Returns the joins and the pieces' labels."""
    count, labels = cv2.connectedComponents(ridges.astype(np.uint8), connectivity=8)
    distance, (near_y, near_x) = distance_transform_edt(~ridges, return_indices=True)
    piece = np.where(lines & (distance <= 1.5), labels[near_y, near_x], 0)
    gap_count, gap_labels = cv2.connectedComponents((lines & (piece == 0)).astype(np.uint8), connectivity=8)
    unit = np.where(piece > 0, piece, -gap_labels)  # a ridge piece (> 0) or a gap (< 0) per line pixel
    neighbours = cv2.filter2D(lines.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - 1
    rows, columns = lines.shape
    meeting = {}  # (unit, unit) -> the line pixels where they meet
    for y, x in zip(*np.nonzero(lines)):
        for dy, dx in AROUND:
            v, u = y + dy, x + dx
            if 0 <= v < rows and 0 <= u < columns and lines[v, u] and unit[v, u] != unit[y, x]:
                meeting.setdefault(tuple(sorted((int(unit[y, x]), int(unit[v, u])))), []).append((y, x))
    groups = {}  # a gap and the pieces it touches; or two pieces that touch directly
    for (a, b), places in meeting.items():
        if a < 0 < b:
            groups.setdefault(a, set()).add(b)
        elif a > 0:
            groups[(a, b)] = {a, b}
    sizes = np.bincount(labels.ravel(), minlength=count)
    joins = []
    for key, pieces in groups.items():
        if len(pieces) < 2:
            continue
        where = np.argwhere(unit == key) if not isinstance(key, tuple) else np.array(meeting[key])
        y, x = where.mean(axis=0)
        fork = len(pieces) > 2 or bool((neighbours[where[:, 0], where[:, 1]] >= 3).any())
        pieces = tuple(sorted(int(p) for p in pieces))
        # a join of lines: every piece is a line of its own (SPUR_LENGTH pixels of ridge or more), not a speck the
        # closing touched
        joins.append({"pieces": pieces, "x": float(x), "y": float(y), "two arms": not fork,
                      "of lines": bool(min(sizes[p] for p in pieces) >= SPUR_LENGTH)})
    return joins, labels


def true_pieces(edges, lines):
    """Per edge, whether it is true: most of its points lie within TRUE_WITHIN of a true centerline (`lines`, a
    picture), the harness's centerline tolerance."""
    distance = distance_to(lines)
    rows, columns = lines.shape
    return np.array([np.mean(distance[np.clip(np.rint(edge["y"]).astype(int), 0, rows - 1),
                                      np.clip(np.rint(edge["x"]).astype(int), 0, columns - 1)] <= TRUE_WITHIN) > 0.5
                     for edge in edges], dtype=bool)


def edge_terms(edges, valid, blur, repeatability):
    """The five bottom-up terms of section 37 per edge (EDGE_TERMS), as a list of dicts: length in field lengths
    (zero at one, full by three; a field is ELONGATION half-widths long), strength (the median pre-interaction score
    over T_HIGH), repeatability (above chance, edge_repeatability; NaN where the other fixation has no data), flow
    data (flow_share), and twin or stub (twins_and_stubs)."""
    kinds = twins_and_stubs(edges, blur, valid)
    rows = []
    for number, edge in enumerate(edges):
        field = ELONGATION * max(float(np.median(edge["radius"])), 0.5)
        rows.append({"length": float(np.clip((edge["length"] / field - 1) / 2, 0, 1)),
                     "strength": float(np.median(edge["strength"])) / T_HIGH,
                     "repeatability": float(repeatability[number]), "flow data": float(edge["flow share"]),
                     "twin or stub": kinds[number]})
    return rows


def bin_of(value, cuts):
    """The bin a term's value falls in: its place among a word's levels (one more for a level not seen), or among
    the cuts of a number (one more for NaN)."""
    if isinstance(cuts, list):
        return cuts.index(value) if value in cuts else len(cuts)
    if not np.isfinite(value):
        return len(cuts) + 1
    return int(np.searchsorted(cuts, value, side="right"))


def calibrate_terms(rows, truth, terms=EDGE_TERMS, bins=CONFIDENCE_BINS):
    """Section 41 for the edge confidence: each term binned on the bursts for developing (quantile bins of a number,
    NaN a bin of its own; a word's levels), and per bin the log-likelihood ratio of a true edge against a false
    one, with half a count added to every bin; the prior is the log odds of a true edge, kept apart. Returns the
    model: per term its cuts, its ratios, and the share true and count in each bin (the reliability diagram)."""
    truth = np.asarray(truth, dtype=bool)
    model = {"prior": float(np.log(max(truth.sum(), 0.5) / max((~truth).sum(), 0.5))), "terms": {}}
    for term in terms:
        values = [row[term] for row in rows]
        if isinstance(values[0], str):
            cuts = sorted(set(values))
            count = len(cuts) + 1
        else:
            finite = np.array([value for value in values if np.isfinite(value)])
            cuts = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1])) if len(finite) else np.array([])
            count = len(cuts) + 2
        place = np.array([bin_of(value, cuts) for value in values])
        right = np.bincount(place[truth], minlength=count) + 0.5
        wrong = np.bincount(place[~truth], minlength=count) + 0.5
        model["terms"][term] = {"cuts": cuts, "ratio": np.log(right / right.sum()) - np.log(wrong / wrong.sum()),
                                "share true": (right - 0.5) / np.maximum(right + wrong - 1, 1), "count": right + wrong - 1}
    return model


def edge_confidence(rows, model, terms=EDGE_TERMS):
    """The edge confidence: the prior plus the terms' log-likelihood ratios (calibrate_terms), squashed to 0..1.
    Also returns each term's ratio per edge."""
    ratios = np.array([[model["terms"][term]["ratio"][bin_of(row[term], model["terms"][term]["cuts"])] for term in terms]
                       for row in rows], dtype=float).reshape(len(rows), len(terms))
    return 1 / (1 + np.exp(-(model["prior"] + ratios.sum(axis=1)))), ratios


def saved_front(saved, number):
    """What ridge_front gives for a fixation of the real burst, made from section 29's saved inputs."""
    strength, valid, flow_strength = saved["static_strength"][number], saved["valid"][number], saved["flow_strength"][number]
    return {"scores": saved["static_scores"][number], "strength": strength, "orientation": saved["static_orientation"][number],
            "half_width": saved["half_width"][number], "mask": apply_hysteresis_threshold(strength, T_LOW, T_HIGH) & valid,
            "flow_strength": flow_strength, "has_flow": apply_hysteresis_threshold(flow_strength, T_LOW, T_HIGH)}


def cyclic_pieces(front, valid, watch=None):
    """Section 37's pieces of one fixation: joined_ridges, fine_lines (the crests' sub-pixel places), and per piece
    its splines (spline_vessel) after the knot refinement. A piece that refine_knots splits becomes two pieces with a
    bend between them; `knot spacing` is the spacing the refinement settled on (the splines are drawn at 12 px; a
    finer spacing is used once a piece is rendered, section 38). Returns the pieces and joined_ridges' dict."""
    joined = joined_ridges(front["strength"], front["orientation"], front["mask"])
    places = crest_places(front["scores"], front["orientation"], valid)
    vessels = []
    for line in fine_lines(joined["skeleton"], front["strength"], front["orientation"], front["half_width"], places):
        size = len(line["x"])
        for first, last, spacing in refine_knots(line, watch):
            part = {name: value[first:last + 1] if isinstance(value, np.ndarray) and len(value) == size else value
                    for name, value in line.items()}
            part["loose"] = (line["loose"][0] and first == 0, line["loose"][1] and last == size - 1)
            part["flow share"] = flow_share(part, front["has_flow"])
            vessel = spline_vessel(part, front["flow_strength"])
            vessel.update({"loose": part["loose"], "knot spacing": spacing, "split": (first, last) != (0, size - 1),
                           "offset share": float(part["offset"].mean())})
            vessels.append(vessel)
    return vessels, joined


JOIN_LOOK = 10.0  # pixels; a join is judged by the ridge pixels of its pieces this close to it


def join_is_right(join, labels, segments):
    """A join of pieces of ridge is right when, near the join (JOIN_LOOK), every piece lies on a vessel (most of its
    pixels there within TRUE_WITHIN of one true segment) and all lie on one segment, or on segments of one lumen
    that meet in a node."""
    points = np.concatenate([segment["points"] for segment in segments])
    owner = np.concatenate([np.full(len(segment["points"]), number) for number, segment in enumerate(segments)])
    tree = scipy.spatial.cKDTree(points)
    rows, columns = labels.shape
    top, bottom = int(max(join["y"] - JOIN_LOOK, 0)), int(min(join["y"] + JOIN_LOOK + 1, rows))
    left, right = int(max(join["x"] - JOIN_LOOK, 0)), int(min(join["x"] + JOIN_LOOK + 1, columns))
    ids = []
    for piece in join["pieces"]:
        y, x = np.nonzero(labels[top:bottom, left:right] == piece)
        y, x = y + top, x + left
        near = np.hypot(x - join["x"], y - join["y"]) <= JOIN_LOOK
        if not near.any():
            return False
        distance, nearest = tree.query(np.column_stack([x[near], y[near]]))
        votes = np.bincount(np.where(distance <= TRUE_WITHIN, owner[nearest], -1) + 1)
        ids.append(int(np.argmax(votes)) - 1)
    if min(ids) < 0:
        return False
    ends = lambda number: {segments[number]["start"], segments[number]["end"]}
    return all(a == b or (segments[a]["tree"] == segments[b]["tree"] and bool(ends(a) & ends(b)))
               for a, b in itertools.combinations(ids, 2))


def edge_table(burst, threads=None):
    """Section 37 on a synthetic burst made with a second fixation (synthetic_burst_35, second=True): the pieces,
    their five terms (repeatability against the second fixation's joined ridges), which pieces are true, and the
    joins the closing within RIDGE_JOIN made, with whether each is right."""
    front = ridge_front(burst, threads)
    vessels, joined = cyclic_pieces(front, burst["valid"])
    blur = picture_blur(front["half_width"], joined["ridges"])
    second = burst["second"]
    other = ridge_front(second, threads)
    other_lines = joined_ridges(other["strength"], other["orientation"], other["mask"])["skeleton"]
    found, chance, above = edge_repeatability(vessels, burst["valid"], other_lines, second["valid"], second["to second"])
    joins, labels = closing_joins(joined["ridges"], joined["closed"])
    for join in joins:
        join["right"] = join_is_right(join, labels, burst["segments"])
    return {"front": front, "vessels": vessels, "joined": joined, "blur": blur, "found": found, "chance": chance,
            "terms": edge_terms(vessels, burst["valid"], blur, above), "truth": true_pieces(vessels, burst["lines"]),
            "joins": joins, "other lines": other_lines}


def confidence_curve(table, confidence, burst, cutoffs=CONFIDENCE_CUTOFFS, threads=None):
    """The pieces whose edge confidence is at least each cutoff, read as section 29 reads its pieces (a kymograph
    per piece, the graph, the junctions with flow as tests) and scored against the truth (score_reading): per cutoff
    one score. Below two pieces nothing is read."""
    front, vessels = table["front"], table["vessels"]
    readings, _ = segment_readings(vessels, lambda position: burst["frames"][position], burst["times"], burst["pairs"],
                                   burst["fps"], threads=threads)
    vessels = [downstream(vessel, reading) for vessel, reading in zip(vessels, readings)]
    count = len(burst["times"])
    halves = (np.arange(0, count, 2), np.arange(1, count, 2)) if count >= SPLIT_FRAMES else None
    scores = {}
    for cutoff in cutoffs:
        keep = [number for number in range(len(vessels)) if confidence[number] >= cutoff]
        if len(keep) < 2:
            scores[cutoff] = score_reading(burst, {"chains": [], "graph": {"nodes": []}, "junctions": []})
            continue
        kept, kept_readings = [vessels[number] for number in keep], [readings[number] for number in keep]
        graph = vessel_graph(kept, burst["valid"], burst["mean"], kept_readings, burst["pairs"], burst["fps"])
        found = read_junctions(graph, burst["mean"], front["scores"], halves)[0]
        scores[cutoff] = score_reading(burst, {"chains": kept, "readings": kept_readings, "graph": graph, "junctions": found})
    return scores


def passes_point(curve, point, precision, recall):
    """The cutoffs at which a curve (cutoff -> ablation_row) reaches a row's point: its precision and its recall at
    least that row's, both."""
    return [cutoff for cutoff, row in curve.items()
            if row[precision][0] >= point[precision][0] and row[recall][0] >= point[recall][0]]


def knot_frame(state, size=(5.0, 4.0), dpi=80):
    """A frame of the knot refinement's film: a piece's ridge points, the spline the candidate spacing gives (or
    where the piece is split), and a caption with the candidate, its gain in the ridge's unit and whether it was
    taken."""
    import matplotlib
    matplotlib.use("Agg")
    figure, axis = plt.subplots(figsize=size, dpi=dpi)
    points = state["points"]
    axis.plot(points[:, 0], points[:, 1], ".", color="gray", markersize=3)
    if "fit" in state:
        axis.plot(state["fit"][:, 0], state["fit"][:, 1], "-", color="tab:orange" if state["taken"] else "tab:blue", lw=1.5)
    if state.get("kind") == "split":
        at = points[state["at"] - state["piece"][0]]
        axis.plot(at[0], at[1], "o", color="tab:red", markersize=9, mfc="none", mew=2)
    caption = (f"candidate {state.get('candidate', '?')}: " + (f"knots {state['spacing']:.0f} px apart, gain {state['gain']:+.1f}"
               if state.get("kind") == "knots" else "split where it fits worst")
               + (" — taken" if state.get("taken") else " — refused"))
    axis.set_title(caption, fontsize=9)
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.axis("off")
    frame = vascnet.watch.frame_of(figure)
    plt.close(figure)
    return frame

# %%
# section 37 on one synthetic burst first: the pieces, their terms, which are true, and what the consolidation drops
burst = synthetic_burst_35(HARNESS_SEEDS[0], second=True)
table = edge_table(burst, workers)
vessels, terms, truth = table["vessels"], table["terms"], table["truth"]
consolidated = ridge_skeleton(table["front"]["strength"], table["front"]["orientation"], table["front"]["mask"], consolidate=True)
near_consolidated = distance_to(consolidated)
rows, columns = consolidated.shape
dropped = np.array([np.mean(near_consolidated[np.clip(np.rint(v["y"]).astype(int), 0, rows - 1),
                                              np.clip(np.rint(v["x"]).astype(int), 0, columns - 1)] <= TRUE_WITHIN) < 0.5
                    for v in vessels])
kinds = [row["twin or stub"] for row in terms]
print(f"Synthetic burst {HARNESS_SEEDS[0]}: {len(vessels)} pieces, {sum(v['length'] for v in vessels):.0f} px, of which "
      f"{truth.sum()} true ({truth.mean() * 100:.0f}%) | {sum(v['split'] for v in vessels)} pieces come from a split, knots "
      f"closer than 12 px on {sum(v['knot spacing'] < KNOT_SPACING for v in vessels)} | sub-pixel places on "
      f"{np.mean([v['offset share'] for v in vessels]) * 100:.0f}% of the points")
print(f"   twins {kinds.count('twin')} ({sum(t for t, k in zip(truth, kinds) if k == 'twin')} true), stubs {kinds.count('stub')} "
      f"({sum(t for t, k in zip(truth, kinds) if k == 'stub')} true), plain {kinds.count('plain')} | the extensions added "
      f"{table['joined']['added'].sum()} px | the blur of the picture {table['blur']:.2f} px")
print(f"   the consolidation drops {dropped.sum()} of these pieces ({sum(v['length'] for v, d in zip(vessels, dropped) if d):.0f} px), "
      f"{truth[dropped].sum()} of them true | repeatability against the second fixation: found again "
      f"{np.nanmean(table['found']) * 100:.0f}%, by chance {np.nanmean(table['chance']) * 100:.0f}% (means over pieces)")
two_arm = [join for join in table["joins"] if join["two arms"]]
of_lines = [join for join in two_arm if join["of lines"]]
print(f"   the closing within RIDGE_JOIN made {len(table['joins'])} joins, {len(two_arm)} of two arms: {len(of_lines)} join two lines "
      f"({sum(not join['right'] for join in of_lines)} false), {len(two_arm) - len(of_lines)} join a line to a speck of under "
      f"{SPUR_LENGTH} px ({sum(not join['right'] for join in two_arm if not join['of lines'])} of the specks off any vessel)")

picture = np.dstack([np.uint8(np.clip(burst["mean"] / np.percentile(burst["mean"], 99.8), 0, 1) * 110)] * 3)
for vessel, right, kind in zip(vessels, truth, kinds):
    color = (90, 255, 90) if right else (255, 60, 60)
    if kind != "plain":
        color = (255, 200, 0) if kind == "twin" else (0, 200, 255)
    cv2.polylines(picture, np.rint(np.column_stack([vessel["x"], vessel["y"]])).astype(np.int32)[None], False, color, 1)
for join in table["joins"]:
    cv2.circle(picture, (int(join["x"]), int(join["y"])), 5, (255, 255, 255) if join["right"] else (255, 0, 255), 1)
os.makedirs("vessel_pictures/cyclic", exist_ok=True)
Image.fromarray(picture).save("vessel_pictures/cyclic/one_burst.png")
figure, axis = plt.subplots(figsize=(10, 7.5))
axis.imshow(picture, interpolation="antialiased")
axis.set_title("section 37's pieces on one synthetic burst | green true, red false, yellow twin, blue stub | circles: joins the "
               "closing made (white right, magenta false)", fontsize=9)
axis.axis("off")
plt.show()
del burst, table, picture

# %%
# acceptance step 2: the edge confidence calibrated on the bursts for developing, its told_apart on the bursts held out,
# and the curves over the confidence against the consolidated row's point. Two sets: section 30's bursts (the old seeds,
# made with a second fixation; their first fixation is section 30's burst to the bit) and section 35's trees.
CYCLIC_SETS = (("section 30's bursts", HARNESS_SEEDS, HELD_OUT_SEEDS, {}),
               ("section 35's trees", TREE_SEEDS, TREE_HELD_OUT, SYNTH_THINGS))
cyclic_harness = {}  # per set: the model, and per held-out burst its pieces' terms, truth, confidence and curves
for name, developing, held_out, things in CYCLIC_SETS:
    rows, truth = [], []
    for seed in developing:
        table = edge_table(synthetic_burst_35(seed, **{**things, "second": True}), workers)
        rows += table["terms"]
        truth += list(table["truth"])
    model = calibrate_terms(rows, truth)
    held = {"model": model, "developing": len(rows), "truth": [], "confidence": [], "ratios": [], "terms": [], "curves": [],
            "consolidated": [], "plain": [], "joins": [], "seeds": list(held_out)}
    for seed in held_out:
        burst = synthetic_burst_35(seed, **{**things, "second": True})
        table = edge_table(burst, workers)
        confidence, ratios = edge_confidence(table["terms"], model)
        for key, value in (("truth", table["truth"]), ("confidence", confidence), ("ratios", ratios), ("terms", table["terms"]),
                           ("joins", table["joins"])):
            held[key].append(value)
        held["curves"].append(confidence_curve(table, confidence, burst, threads=workers))
        for way, consolidate in (("consolidated", True), ("plain", False)):
            held[way].append(score_reading(burst, read_burst_ridges(burst, table["front"], consolidate=consolidate, threads=workers)))
        del burst, table
    cyclic_harness[name] = held
    memory_note(f"section 37, {name}")

for name, held in cyclic_harness.items():
    truth, confidence = np.concatenate(held["truth"]), np.concatenate(held["confidence"])
    ratios = np.concatenate(held["ratios"])
    told = told_apart(confidence[truth], confidence[~truth])
    picks = np.random.default_rng(0).integers(len(held["truth"]), size=(500, len(held["truth"])))
    resampled = [told_apart(np.concatenate([held["confidence"][b][held["truth"][b]] for b in pick]),
                            np.concatenate([held["confidence"][b][~held["truth"][b]] for b in pick])) for pick in picks]
    low, high = np.percentile(resampled, [2.5, 97.5])
    curve = {cutoff: ablation_row([scores[cutoff] for scores in held["curves"]]) for cutoff in CONFIDENCE_CUTOFFS}
    point, plain = ablation_row(held["consolidated"]), ablation_row(held["plain"])
    centerline_cutoffs = passes_point(curve, point, "centerline precision", "centerline recall")
    junction_cutoffs = passes_point(curve, point, "junction precision", "junction recall")
    held.update({"told apart": (told, low, high), "curve": curve, "point": point, "plain row": plain,
                 "passes": (centerline_cutoffs, junction_cutoffs)})
    joins = [join for joins in held["joins"] for join in joins if join["two arms"] and join["of lines"]]
    specks = [join for joins in held["joins"] for join in joins if join["two arms"] and not join["of lines"]]
    print(f"\n{name}: calibrated on {held['developing']} pieces of the bursts for developing; held out: {len(truth)} pieces of "
          f"{len(held['truth'])} bursts, {truth.sum()} true and {(~truth).sum()} false")
    print(f"   told apart, the edge confidence on the five terms: {told:.3f} ({low:.3f} to {high:.3f}, 95% over bursts) | each term "
          f"alone: " + ", ".join(f"{term} {told_apart(ratios[truth, place], ratios[~truth, place]):.2f}" for place, term in enumerate(EDGE_TERMS)))
    print(f"   skeleton joins of two arms (the closing within RIDGE_JOIN) between two lines of {SPUR_LENGTH} px or more: {len(joins)}, "
          f"of which {sum(not j['right'] for j in joins)} false ({(sum(not j['right'] for j in joins) / max(len(joins), 1)) * 100:.0f}%) | "
          f"{len(specks)} more join a line to a speck, {sum(not j['right'] for j in specks)} of the specks off any vessel")
    shown = ("centerline precision", "centerline recall", "junction precision", "junction recall", "found and typed right")
    print(f"   {'cutoff':>8s} {'pieces':>7s} {'true':>6s} " + "".join(f"{column:>24s}" for column in shown))
    for cutoff, row in list(curve.items()) + [("consolidated", point), ("plain ridges", plain)]:
        if isinstance(cutoff, float):
            kept = confidence >= cutoff
            counts = f"{kept.sum():7d} {truth[kept].sum():6d}"
        else:
            counts = " " * 14
        cells = "".join(f"{row[column][0] * 100:5.1f} ({row[column][1] * 100:3.0f} to {row[column][2] * 100:3.0f})".rjust(24) for column in shown)
        print(f"   {str(cutoff):>8s} {counts} {cells}")
    verdict = told >= 0.75 and centerline_cutoffs and junction_cutoffs
    print(f"   the gate (told apart >= 0.75, the curve passing the consolidated row's point): told apart {told:.3f}; the centerline "
          f"curve passes it at the cutoffs {centerline_cutoffs or 'none'}, the junction curve at {junction_cutoffs or 'none'} -> "
          f"{'PASSED' if verdict else 'FAILED'}")

# the reliability diagram of every term on the bursts for developing, and the confidence of the held-out pieces
figure, axes = plt.subplots(len(cyclic_harness), len(EDGE_TERMS) + 1, figsize=(22, 4.2 * len(cyclic_harness)))
for row, (name, held) in enumerate(cyclic_harness.items()):
    for place, term in enumerate(EDGE_TERMS):
        calibration = held["model"]["terms"][term]
        labels = list(calibration["cuts"]) + ["new"] if isinstance(calibration["cuts"], list) else (
            [f"<{calibration['cuts'][0]:.2f}"] + [f">={cut:.2f}" for cut in calibration["cuts"]] + ["none"])
        axes[row, place].bar(range(len(calibration["ratio"])), calibration["ratio"], color="tab:blue")
        for spot, (share, count) in enumerate(zip(calibration["share true"], calibration["count"])):
            axes[row, place].annotate(f"{share * 100:.0f}%\nn={count:.0f}", (spot, 0), ha="center", fontsize=7,
                                      xytext=(0, -24 if calibration["ratio"][spot] >= 0 else 6), textcoords="offset points")
        axes[row, place].set_xticks(range(len(labels)))
        axes[row, place].set_xticklabels(labels, rotation=45, fontsize=7)
        axes[row, place].axhline(0, color="black", lw=0.6)
        axes[row, place].set_title(f"{name}: {term}", fontsize=9)
    truth, confidence = np.concatenate(held["truth"]), np.concatenate(held["confidence"])
    bins = np.linspace(0, 1, 21)
    axes[row, -1].hist(confidence[truth], bins=bins, histtype="step", color="tab:green", label="true pieces")
    axes[row, -1].hist(confidence[~truth], bins=bins, histtype="step", color="tab:red", label="false pieces")
    axes[row, -1].set_title(f"held out: told apart {held['told apart'][0]:.2f}", fontsize=9)
    axes[row, -1].legend(fontsize=7)
figure.suptitle("bars: each term's log-likelihood ratio per bin on the bursts for developing (share true and count below) | right: "
                "the edge confidence of the held-out pieces", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/cyclic/reliability.png", dpi=90)
plt.show()

# the curves over the confidence, against the plain and the consolidated rows' points
figure, axes = plt.subplots(1, 2 * len(cyclic_harness), figsize=(20, 4.6))
for row, (name, held) in enumerate(cyclic_harness.items()):
    for place, (precision, recall) in enumerate((("centerline precision", "centerline recall"), ("junction precision", "junction recall"))):
        axis = axes[2 * row + place]
        axis.plot([held["curve"][c][recall][0] * 100 for c in CONFIDENCE_CUTOFFS],
                  [held["curve"][c][precision][0] * 100 for c in CONFIDENCE_CUTOFFS], "o-", color="tab:blue", ms=3,
                  label="joined, over the edge confidence")
        for label, point, color in (("consolidated", held["point"], "tab:red"), ("plain ridges", held["plain row"], "tab:gray")):
            axis.errorbar(point[recall][0] * 100, point[precision][0] * 100, color=color, fmt="s",
                          xerr=[[(point[recall][0] - point[recall][1]) * 100], [(point[recall][2] - point[recall][0]) * 100]],
                          yerr=[[(point[precision][0] - point[precision][1]) * 100], [(point[precision][2] - point[precision][0]) * 100]],
                          label=label, capsize=3)
        axis.set_xlabel(f"{recall} (%)")
        axis.set_ylabel(f"{precision} (%)")
        axis.set_title(f"{name}, held out", fontsize=9)
        axis.legend(fontsize=7)
figure.tight_layout()
figure.savefig("vessel_pictures/cyclic/curves.png", dpi=90)
plt.show()

# the held-out pieces with their terms, as a table for what follows
save_table([{"set": name, "seed": seed, "true": bool(right), "confidence": float(value), **terms}
            for name, held in cyclic_harness.items()
            for seed, burst_truth, burst_confidence, burst_terms in zip(held["seeds"], held["truth"], held["confidence"], held["terms"])
            for right, value, terms in zip(burst_truth, burst_confidence, burst_terms)], "vessel_pictures/cyclic/pieces.csv")
save_table([{"set": name, "cutoff": cutoff, **{column: f"{row[column][0]:.3f} ({row[column][1]:.3f} to {row[column][2]:.3f})"
                                              for column in ABLATION_COLUMNS}}
            for name, held in cyclic_harness.items()
            for cutoff, row in list(held["curve"].items()) + [("consolidated", held["point"]), ("plain ridges", held["plain row"])]],
           "vessel_pictures/cyclic/curves.csv")
memory_note("section 37, harness")

# %%
# section 37 on the real burst, in the anchor's fixation: the pieces, their terms (repeatability against the other
# fixations), their confidence under each set's calibration, and what the consolidation of section 29 drops of them.
# The knot refinement is filmed.
front = saved_front(saved, anchor)
valid = saved["valid"][anchor]
knot_recorder = vascnet.watch.Recorder(every=40)
real_pieces, real_joined = cyclic_pieces(front, valid, watch=knot_recorder)
real_blur = picture_blur(front["half_width"], real_joined["ridges"])
others = [number for number in range(len(saved["times"])) if number != anchor]
other_front = saved_front(saved, others[0])
other_lines = joined_ridges(other_front["strength"], other_front["orientation"], other_front["mask"])["skeleton"]
real_found, real_chance, real_above = edge_repeatability(real_pieces, valid, other_lines, saved["valid"][others[0]],
                                                         saved["to anchor"][others[0]])
real_terms = edge_terms(real_pieces, valid, real_blur, real_above)
real_confidence = {name: edge_confidence(real_terms, held["model"])[0] for name, held in cyclic_harness.items()}
drawn_pieces = draw_chains(real_pieces, valid.shape) >= 0
whole = repeated(drawn_pieces, valid, other_lines, saved["valid"][others[0]], saved["to anchor"][others[0]])
consolidated = ridge_skeleton(front["strength"], front["orientation"], front["mask"], consolidate=True)
near_consolidated = distance_to(consolidated)
rows, columns = valid.shape
dropped = np.array([np.mean(near_consolidated[np.clip(np.rint(v["y"]).astype(int), 0, rows - 1),
                                              np.clip(np.rint(v["x"]).astype(int), 0, columns - 1)] <= TRUE_WITHIN) < 0.5
                    for v in real_pieces])
kinds = [row["twin or stub"] for row in real_terms]
lengths = np.array([piece["length"] for piece in real_pieces])
print(f"Real burst, fixation {anchor}: {len(real_pieces)} pieces, {lengths.sum():.0f} px (section 29's consolidated lines: "
      f"{consolidated.sum()} px) | the extensions added {real_joined['added'].sum()} px | the blur {real_blur:.2f} px")
print(f"   knots: {sum(p['split'] for p in real_pieces)} pieces come from a split at a bend, "
      f"{sum(p['knot spacing'] < KNOT_SPACING for p in real_pieces)} took knots closer than 12 px | sub-pixel places on "
      f"{np.average([p['offset share'] for p in real_pieces], weights=lengths) * 100:.0f}% of the length | twins {kinds.count('twin')}, "
      f"stubs {kinds.count('stub')}, plain {kinds.count('plain')}")
print(f"   repeatability against fixation {others[0]}: the pieces are found again on {np.nanmean(real_found) * 100:.0f}% (by chance "
      f"{np.nanmean(real_chance) * 100:.0f}%; mean over pieces) | all pieces as one picture (section 29's `repeated`): found "
      f"{whole[0] * 100:.0f}%, by chance {whole[1] * 100:.0f}%, above chance {whole[2] * 100:.0f}% | "
      f"{np.isnan(real_found).sum()} pieces lie where fixation {others[0]} has no data")
print(f"   the consolidation drops {dropped.sum()} of the pieces ({lengths[dropped].sum():.0f} px); it keeps {(~dropped).sum()} "
      f"({lengths[~dropped].sum():.0f} px)")
for name, confidence in real_confidence.items():
    print(f"   confidence calibrated on {name}: median {np.median(confidence):.2f} | of the pieces the consolidation keeps, "
          f"{np.mean(confidence[~dropped] >= 0.5) * 100:.0f}% at 0.5 or more; of those it drops, {np.mean(confidence[dropped] >= 0.5) * 100:.0f}% "
          f"({lengths[dropped & (confidence >= 0.5)].sum():.0f} px) | told apart, kept against dropped: "
          f"{told_apart(confidence[~dropped], confidence[dropped]):.2f}")

# the pieces coloured by their confidence (section 35's trees' calibration) over the density
density = np.nan_to_num(saved["density"][anchor])
gray = np.uint8(np.clip((density - DENSITY_RANGE[0]) / (DENSITY_RANGE[1] - DENSITY_RANGE[0]), 0, 1) * 90)
picture = np.dstack([gray, gray, gray])
colors = plt.get_cmap("RdYlGn")
for piece, value in zip(real_pieces, real_confidence["section 35's trees"]):
    color = tuple(int(255 * c) for c in colors(float(value))[:3])
    cv2.polylines(picture, np.rint(np.column_stack([piece["x"], piece["y"]])).astype(np.int32)[None], False, color, 2)
Image.fromarray(picture).save("vessel_pictures/cyclic/real_confidence.png")
figure, axis = plt.subplots(figsize=(16, 10))
axis.imshow(picture, interpolation="antialiased")
axis.set_title(f"fixation {anchor}: section 37's pieces coloured by their edge confidence (red 0, green 1; calibrated on section 35's trees)",
               fontsize=10)
axis.axis("off")
plt.show()

# the film of the knot refinement: every candidate taken, and one in 40 of the others
os.makedirs("vessel_pictures/fit", exist_ok=True)
film = [state for state in knot_recorder.states if state.get("taken")][:60] + [state for state in knot_recorder.states if not state.get("taken")][:40]
film.sort(key=lambda state: state["candidate"])
if film:
    vascnet.watch.animate(film, knot_frame, "vessel_pictures/fit/initialisation.gif", fps=2.0)
print(f"   the knot refinement made {knot_recorder.count} candidate moves, {sum(s.get('taken', False) for s in knot_recorder.states)} "
      f"taken; {len(film)} of them are filmed in vessel_pictures/fit/initialisation.gif")
del picture, density, gray, front, other_front
memory_note("section 37, real burst")

# %% [markdown]
# ## 40. Net flux = 0 on the whole-vessel graphs (acceptance step 3)
#
# `vascnet/fluxnet.py` is the flux network of the plan's section 40. Here it runs on the graphs section 32 makes: whole vessels read along the consolidated lines, with one direction and one speed per whole vessel.
#
# **What it takes from a graph:**
#
# - **Fluxes** (`edge_flux`): per edge, line mass × velocity where the speed is clear and trackable, with its sigma from the widened sigma of the speed. An edge's measurement is its whole vessel, so all of a vessel's edges turn round together.
# - **Balances** (`flux_lumens`): every group of a junction's reading. A crossing gives two through-lumens, set aside together. Two pieces joined end to end give one through-lumen. A join that section 32 undid gives none.
# - **Provenance of each join** (`join_provenance`): skeleton, extension, junction or undone. The pixels the consolidation's `extend_ends` added are found by making its steps once more (`consolidated_extensions`).
# - **Softness** (`sigma_flux_by_kind`): `SIGMA_FLUX` per lumen kind, measured as section 27 measures it. It is the robust sd of the flux test at lumens whose every arm is measured and whose directions are possible. Through-lumens of one vessel (with the joins) are one kind, bifurcations another and crossings a third. Where a kind has fewer than `FLUX_LEAST` lumens, the harness's value stands in.
# - **p**: section 33's `reliability`, from the counts of the bursts for developing, without truth (`chosen_p`). It uses the three-arm junctions where there are at least `FLUX_LEAST` of them, else the sides of whole vessels. Section 34 takes the first whenever it is defined; on a plexus that rests on one or two junctions and gives 0.5, at which a flip costs nothing.
# - **Structural confidences** (`lumen_confidences`, section 41), calibrated per set on its own bursts for developing, since the prior differs by tissue (a crossing read as two vessels is always right on trees and never on the anastomotic plexus):
#   - a two-arm join: P(join right | the pair's turn, width ratio, density ratio and provenance), each binned and taken as a log-likelihood ratio (`join_features`, `calibrate_terms`). On the consolidated graph, a two-arm node is often a crossing whose other two arms were absorbed, so it joins two trees' vessels at a turn;
#   - a crossing: P(two vessels, not one lumen), by section 27's overlap z, binned;
#   - a bifurcation whose rival "through pair + free end" has evidence (`bifurcation, T`: section 27's baseline saw the thin stem again beyond the through vessel's far wall): the share right of such readings. The rival is not added to section 27's partitions, because that would change its readings, and section 34's numbers depend on them.
#   - Any other bifurcation, a through group inside a larger reading, or an unresolved junction is `SURE`.
#
# **Scoring** (`direction_scores`), on the held-out bursts against the truth (`edge_truth`):
#
# - among the edges without a measured direction, those the network infers (|q| / σ ≥ `FLUX_INFERRED`) and those `propagate_directions` does, each with how many are right;
# - among the measured ones, the flips right and wrong.
#
# **The gate**, stated before the held-out run: on section 35's held-out trees, where the velocity profile is in, the network gives the right direction to more edges without a measured direction than `propagate_directions` does, at a share right no lower. An edge counts as inferred by the network at |q| / σ ≥ `FLUX_INFERRED` = 2, which was set before any run. On the 16 bursts for developing, 2 is the only cutoff at which the network's share right reaches `propagate_directions`' (3 of 3, against 19 of 21). At 1, the network gives more edges the right direction (26 of 33) but at a lower share. A balance allowed to miss by `SIGMA_FLUX` × the flux through it caps the confidence of a direction inferred through one junction at about 1 / (2 `SIGMA_FLUX`). The held-out curve over cutoffs is printed as description. The plexus bursts (both kinds) are reported; the plan expects no change there. Section 30's held-out bursts are reported as well.
#
# On the real burst, the counts the plan names are printed: sign-constraining lumens, set-asides, impossible lumens, directed cycles and the flux at free ends. So are `SIGMA_FLUX` per kind with counts, and the consolidated graph's shares of impossible junctions and directed cycles. Those shares are the reference of λ's rule. λ prices structural moves only, and none exists before the recursion (step 7), so there is nothing to sweep yet.
#
# **Result:** `flux_setup` (p and `SIGMA_FLUX` per set), `calibration`, `flux_results` (`vessel_pictures/flux/held_out.csv`), `real_runs`, and the film `vessel_pictures/fit/flux.gif`.

# %%
import vascnet.fluxnet as fluxnet

FLUX_INFERRED = 2.0    # |q| / sigma at which the network's direction of an edge without a measured one counts as
                       # inferred (set before any run; see the section's text for how it held on the bursts for developing)
FLUX_CYCLE = 3.0       # |q| / sigma of the edges a directed cycle is looked for over (section 40)
FLUX_LEAST = 10        # lumens of a kind with every arm measured: below this the harness's SIGMA_FLUX stands in
OVERLAP_BINS = 4       # bins of section 27's overlap z at a crossing, for P(two vessels, not one lumen)
LUMEN_KINDS = ("through", "bifurcation", "bifurcation, T", "join, skeleton", "join, extension", "crossing", "unresolved")
SURE = 1 - 1e-6        # the structural confidence of a structure the network is not to set aside
JOIN_REACH = 10        # points; an arm of a two-arm join is read over this much of its edge from the node
JOIN_TERMS = ("turn", "width", "density", "provenance")  # what P(join right) is calibrated on (section 37)
JOIN_BINS = 4          # bins of each of them on the bursts for developing


def chosen_p(counts):
    """p for the sign step, from section 33's counts: from the three-arm junctions where there are FLUX_LEAST of
    them with every arm measured, else from the sides of whole vessels (section 34 takes the first whenever it is
    defined; on a plexus it rests on one or two junctions), else 0.85."""
    p_three, p_sides = reliability(counts)
    if counts["three arms"] >= FLUX_LEAST and np.isfinite(p_three):
        return p_three
    return p_sides if np.isfinite(p_sides) else 0.85


def join_features(graph, join, density):
    """What section 37 calibrates a two-arm join's confidence on: how far the pair turns from running straight
    through (degrees), the ratio of their widths and of their densities (|log|), each arm read over its first
    JOIN_REACH points from the node; and the join's provenance."""
    rows, columns = density.shape
    arms = []
    for edge, end in join["ends"]:
        piece = graph["edges"][edge]
        order = np.arange(len(piece["x"]))[:JOIN_REACH] if end == 0 else np.arange(len(piece["x"]))[::-1][:JOIN_REACH]
        away = np.array([piece["x"][order[-1]] - piece["x"][order[0]], piece["y"][order[-1]] - piece["y"][order[0]]])
        row = np.clip(np.rint(piece["y"][order]).astype(int), 0, rows - 1)
        column = np.clip(np.rint(piece["x"][order]).astype(int), 0, columns - 1)
        arms.append((away / max(np.hypot(*away), 1e-9), float(np.median(piece["radius"][order])),
                     float(np.nanmean(density[row, column]))))
    (a, wa, da), (b, wb, db) = arms
    turn = float(np.degrees(np.arccos(np.clip(-np.dot(a, b), -1, 1))))
    ratio = lambda first, second: float(abs(np.log(max(first, 1e-6) / max(second, 1e-6)))) if first > 0 and second > 0 else np.nan
    return {"turn": turn, "width": ratio(wa, wb), "density": ratio(da, db), "provenance": join["provenance"]}


def consolidated_extensions(strength, orientation, region):
    """The pixels extend_ends added in section 29's consolidation (consolidate_lines), found by making its steps
    once more: the lines before the extension, and extended & ~lines. Returns (added, the consolidated lines)."""
    lines = ridge_crests(strength, orientation, region)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RIDGE_JOIN + 1,) * 2)
    joined = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, disc) > 0
    skeleton = without_short_branches(skeletonize(joined), RIDGE_CHILD)
    count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    keep = np.bincount(labels.ravel(), minlength=count) >= RIDGE_NETWORK
    keep[0] = False
    before = keep[labels]
    extended = extend_ends(before, strength)
    return extended & ~before, without_short_branches(skeletonize(extended), RIDGE_CHILD)


def end_nodes(graph):
    """Per (edge, end) the node it lies in, among the nodes that are not gone."""
    out = {}
    for number, node in enumerate(graph["nodes"]):
        if node.get("gone"):
            continue
        for edge, end in node["ends"]:
            out[(edge, end)] = number
    return out


def join_provenance(graph, added=None):
    """Section 37's provenance of the joins of a whole-vessel graph (section 32): every pair of edge ends that
    join_pieces joins into one vessel, before any join is undone. Each is "undone" where section 32 undid it (the
    two sides carry different vessel numbers in graph["readings"]), "junction" where it is a through group of a
    junction's reading, "extension" where its node lies on pixels extend_ends added (`added`, a picture), and
    "skeleton" otherwise (two pieces of one ridge, meeting end to end). Returns one dict per join: its two (edge,
    end), node, place and provenance."""
    nodes = end_nodes(graph)
    near_added = None if added is None else cv2.dilate(added.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    joins = []
    for path in join_pieces(graph):
        for edge, way, joined, junction in path:
            if joined is None:
                continue
            first, second = sorted(joined)
            node = nodes.get(first, nodes.get(second))
            place = graph["nodes"][node]
            vessels = [graph["readings"][first[0]].get("vessel", -1), graph["readings"][second[0]].get("vessel", -1)]
            if vessels[0] != vessels[1]:
                kind = "undone"
            elif junction:
                kind = "junction"
            elif near_added is not None and near_added[int(np.clip(round(place["y"]), 0, near_added.shape[0] - 1)),
                                                       int(np.clip(round(place["x"]), 0, near_added.shape[1] - 1))]:
                kind = "extension"
            else:
                kind = "skeleton"
            joins.append({"ends": (first, second), "node": node, "x": place["x"], "y": place["y"], "provenance": kind})
    return joins


def edge_flux(graph):
    """Per edge of a whole-vessel graph what the network takes from it: the measured flux (line mass x velocity,
    signed along the edge's points; NaN where no clear, trackable speed was read), its sigma (line mass x the
    widened sigma of the speed), and the measurement it belongs to (the whole vessel's number)."""
    count = len(graph["edges"])
    q_hat, sigma, vessel = np.full(count, np.nan), np.full(count, np.nan), np.full(count, -1)
    for number, reading in enumerate(graph["readings"]):
        if reading is None:
            continue
        vessel[number] = int(reading.get("vessel", -1))
        if reading["direction"] != 0 and np.isfinite(reading["velocity"]) and reading["mass"] > 0:
            q_hat[number] = reading["mass"] * reading["velocity"]
            sigma[number] = reading["mass"] * max(float(reading.get("velocity_sigma", np.nan)), 1e-6)
    return q_hat, sigma, vessel


def flux_lumens(graph, junctions, provenance):
    """The balances of a whole-vessel graph, as section 40 takes them: per junction its reading's groups (a crossing:
    two through-lumens set aside together), and per two pieces joined end to end one through-lumen (a join section
    32 undid has none). Returns the lumens ((edge, sign), ...), their kind (LUMEN_KINDS), their group (the lumens set
    aside together), the overlap z of the crossings, and per lumen its node (for a join: the join itself, as a dict)."""
    by_node = {junction["node"]: junction for junction in junctions}
    lumens, kinds, groups, overlaps, places = [], [], [], [], []
    for number, node in enumerate(graph["nodes"]):
        if node.get("gone") or len(node.get("arms", [])) < 3:
            continue
        partition = node.get("partition", [])
        crossing = len(node["arms"]) == 4 and sorted(len(group) for group in partition) == [2, 2]
        # a bifurcation of three arms whose rival "through pair + free end" has evidence: section 27's baseline saw
        # a thin stem again beyond the through vessel's far wall (a T whose stem dives)
        dives = len(node["arms"]) == 3 and by_node.get(number, {}).get("baseline") == "crossing"
        together = len(set(groups))  # a crossing's two lumens are one reading, set aside together
        for group in partition:
            if len(group) < 2:
                continue
            lumens.append(tuple((node["arms"][arm]["edge"], 1 if node["arms"][arm]["end"] == 0 else -1) for arm in group))
            kinds.append("crossing" if crossing else "bifurcation, T" if dives and len(group) == 3
                         else {2: "through", 3: "bifurcation"}.get(len(group), "unresolved"))
            groups.append(together if crossing else len(set(groups)))
            overlaps.append(float(by_node.get(number, {}).get("tests", {}).get("overlap", np.nan)) if crossing else np.nan)
            places.append(number)
    for join in provenance:
        if join["provenance"] in ("skeleton", "extension"):
            lumens.append(tuple((edge, 1 if end == 0 else -1) for edge, end in join["ends"]))
            kinds.append(f"join, {join['provenance']}")
            groups.append(len(set(groups)))
            overlaps.append(np.nan)
            places.append(join)
    return lumens, kinds, np.array(groups, dtype=int), np.array(overlaps), places


def edge_truth(graph, segments):
    """Per edge of a graph, the true direction of the flow along its points (+1, -1; 0 where most of its points lie
    on no single true segment within TRUE_WITHIN), and the true segment."""
    points = np.concatenate([segment["points"] for segment in segments])
    owner = np.concatenate([np.full(len(segment["points"]), number) for number, segment in enumerate(segments)])
    place = np.concatenate([np.arange(len(segment["points"])) for segment in segments])
    tree = scipy.spatial.cKDTree(points)
    way, which = np.zeros(len(graph["edges"]), dtype=int), np.full(len(graph["edges"]), -1)
    for number, edge in enumerate(graph["edges"]):
        distance, nearest = tree.query(np.column_stack([edge["x"], edge["y"]]))
        on = distance <= TRUE_WITHIN
        if on.mean() <= 0.5:
            continue
        segment = np.bincount(owner[nearest[on]]).argmax()
        mine = nearest[on][owner[nearest[on]] == segment]
        if len(mine) < 2 or place[mine[-1]] == place[mine[0]]:
            continue
        way[number] = int(np.sign(place[mine[-1]] - place[mine[0]]) * np.sign(segments[segment]["speed"]))
        which[number] = segment
    return way, which


def join_right(graph, ends, truth_segment, segments):
    """Whether two edge ends joined into one vessel lie on one true segment, or on two that meet in a node of one
    lumen. None where either edge lies on no segment."""
    a, b = truth_segment[ends[0][0]], truth_segment[ends[1][0]]
    if a < 0 or b < 0:
        return None
    nodes = lambda number: {segments[number]["start"], segments[number]["end"]}
    return bool(a == b or (segments[a]["tree"] == segments[b]["tree"] and nodes(a) & nodes(b)))


def two_vessels_truth(burst, place, radius):
    """At a junction read as a crossing: whether the nearest true junction within junction_metrics' gate is two
    vessels (a projected crossing) or one lumen (an anastomosis, a split, a loop). None where there is none."""
    best, kind = np.inf, None
    for junction in burst["junctions"]:
        distance = np.hypot(junction["x"] - place[0], junction["y"] - place[1])
        if distance <= max(6.0, 2 * junction["radius"] + 2) and distance < best:
            best, kind = distance, junction["lumens"] == 2
    return kind


def lumen_confidences(kinds, overlaps, places, graph, density, calibration):
    """The structural confidence of every lumen (section 37), as section 40 takes it, from the bursts for
    developing (calibration): a two-arm join P(join right | its turn, width ratio, density ratio and provenance), each
    a calibrated log-likelihood ratio (calibrate_terms over JOIN_TERMS); a bifurcation whose rival "through pair +
    free end" has evidence ("bifurcation, T") the share of such readings that are right; a crossing P(two vessels,
    not one lumen) from its overlap z, binned; any other bifurcation, a through group inside a larger reading and an
    unresolved junction SURE. Without a calibration every lumen is SURE."""
    out = []
    for kind, overlap, place in zip(kinds, overlaps, places):
        if kind.startswith("join") and "join model" in calibration:
            row = join_features(graph, place, density)
            out.append(edge_confidence([row], calibration["join model"], terms=JOIN_TERMS)[0][0])
        elif kind in calibration.get("joins", {}):
            out.append(calibration["joins"][kind])
        elif kind == "crossing" and "crossing" in calibration:
            cuts, shares = calibration["crossing"]
            out.append(shares[bin_of(overlap, cuts)])
        else:
            out.append(SURE)
    return np.clip(np.array(out, dtype=float), 1e-3, SURE)


def lumen_imbalances(lumens, kinds, q_hat, all_arms_needed=True):
    """Section 27's flux test, per lumen whose arms all carry a measured flux and whose directions are possible (not
    all in, not all out): what flows in minus what flows out over the flux through it. Returns {kind: [ratios]}."""
    out = {}
    for lumen, kind in zip(lumens, kinds):
        flows = np.array([sign * q_hat[edge] for edge, sign in lumen])
        if not np.isfinite(flows).all():
            continue
        if abs(np.sign(flows).sum()) == len(flows):
            continue  # all in or all out: impossible, a direction is wrong; not a measure of the softness
        out.setdefault(kind, []).append(float(flows.sum() / max(np.abs(flows).sum(), 1e-12)))
    return out


def sigma_flux_by_kind(imbalances, fallback=None, least=FLUX_LEAST):
    """SIGMA_FLUX per lumen kind (section 40): the robust sd of section 27's flux test (lumen_imbalances) over the
    lumens of that kind, with their counts. Where a kind has fewer than `least`, `fallback` (the harness's value for
    that kind, or section 27's one SIGMA_FLUX) stands in. Through-lumens of one vessel and joins share one value."""
    pooled = {"through": [], "bifurcation": [], "crossing": []}
    for kind, values in imbalances.items():
        target = "bifurcation" if kind.startswith("bifurcation") else "crossing" if kind == "crossing" else "through"
        pooled[target] += values
    out = {}
    for kind, values in pooled.items():
        given = (fallback or {}).get(kind, (SIGMA_FLUX, 0, "section 27"))[0]
        out[kind] = (float(robust_sd(np.array(values))), len(values), "measured") if len(values) >= least else (given, len(values), "stands in")
    return out


def sigma_of_kind(kind, by_kind):
    """The SIGMA_FLUX a lumen of a kind is solved with."""
    return by_kind["bifurcation" if kind.startswith("bifurcation") else "crossing" if kind == "crossing" else "through"][0]


def run_flux_network(graph, junctions, provenance, by_kind, p, calibration, density, watch=None):
    """Section 40 on a whole-vessel graph: its measured fluxes and measurements (edge_flux), its balances
    (flux_lumens) with their structural confidences (lumen_confidences) and SIGMA_FLUX per kind, solved once with
    the sign step (solve_flux_network). Returns the solution, the lumens with their kinds, groups and confidences,
    and what section 40 reports: blame, the flux at the free ends, the directed cycles, the lumens that constrain a
    sign (their imbalance has a value: the rest of the network says what their arms carry)."""
    q_hat, sigma_hat, vessel = edge_flux(graph)
    lumens, kinds, groups, overlaps, places = flux_lumens(graph, junctions, provenance)
    if not lumens or not np.isfinite(q_hat).any():
        return None
    softness = np.array([sigma_of_kind(kind, by_kind) for kind in kinds])
    confidence = lumen_confidences(kinds, overlaps, places, graph, density, calibration)
    solution = fluxnet.solve_flux_network(q_hat, sigma_hat, lumens, sigma_flux=softness, p=p, measurement=vessel,
                                          lumen_confidence=confidence, lumen_group=groups, watch=watch)
    free = [(edge, end) for node in graph["nodes"] if not node.get("gone") and len(node["ends"]) == 1 for edge, end in node["ends"]]
    edges = [edge["nodes"] for edge in graph["edges"]]
    return {"solution": solution, "lumens": lumens, "kinds": kinds, "groups": groups, "places": places, "confidence": confidence,
            "q_hat": q_hat, "measured": np.isfinite(q_hat), "blame": fluxnet.blame(solution),
            "free ends": fluxnet.flux_at_free_ends(solution, free),
            "cycles": fluxnet.directed_cycles(edges, solution.direction, solution.direction_confidence, FLUX_CYCLE),
            "constraining": int(np.isfinite(solution.imbalance).sum())}


def propagated_directions(graph):
    """Per edge the direction along its points that propagate_directions (section 27) gave it from the junctions,
    where it had no measured one (0 elsewhere)."""
    way = np.zeros(len(graph["edges"]), dtype=int)
    for node in graph["nodes"]:
        if node.get("gone"):
            continue
        for arm in node.get("arms", []):
            if arm.get("inferred") and arm.get("direction_out", 0) != 0:
                way[arm["edge"]] = int(arm["direction_out"] * (1 if arm["end"] == 0 else -1))
    return way


def direction_scores(run, graph, truth_way):
    """What the network does to the directions, against the truth (0 in truth_way: unknown, left out): among the
    edges without a measured direction, those the network infers (|q| / sigma >= FLUX_INFERRED) and those
    propagate_directions does, and how many of each are right; among the measured ones, the flips right and wrong."""
    solution, measured = run["solution"], run["measured"]
    known = truth_way != 0
    inferred = ~measured & known & (solution.direction_confidence >= FLUX_INFERRED) & (solution.direction != 0)
    propagated = propagated_directions(graph)
    by_junctions = ~measured & known & (propagated != 0)
    as_read = np.sign(np.nan_to_num(run["q_hat"])).astype(int)
    return {"unmeasured": int((~measured & known).sum()),
            "network inferred": int(inferred.sum()), "network right": int((solution.direction[inferred] == truth_way[inferred]).sum()),
            "junctions inferred": int(by_junctions.sum()), "junctions right": int((propagated[by_junctions] == truth_way[by_junctions]).sum()),
            "measured": int((measured & known).sum()), "measured right": int((as_read[measured & known] == truth_way[measured & known]).sum()),
            "flips right": int((solution.flipped & measured & known & (as_read != truth_way)).sum()),
            "flips wrong": int((solution.flipped & measured & known & (as_read == truth_way)).sum()),
            "curve": [(float(solution.direction_confidence[e]), bool(solution.direction[e] == truth_way[e]))
                      for e in np.flatnonzero(~measured & known)]}

# %%
# section 40 on one synthetic tree burst first: its whole-vessel graph (section 32 on the consolidated lines), the
# joins' provenance, the balances, one solve with the sign step, filmed
burst = synthetic_burst_35(TREE_SEEDS[0], **SYNTH_THINGS)
front = ridge_front(burst, workers)
read = read_burst_ridges(burst, front, consolidate=True, speeds="whole vessels", threads=workers)
graph = read["graph"]
added, _ = consolidated_extensions(front["strength"], front["orientation"], front["mask"])
provenance = join_provenance(graph, added)
p = chosen_p(read["counts"])
by_kind = {kind: (SIGMA_FLUX, 0, "section 27") for kind in ("through", "bifurcation", "crossing")}
flux_recorder = vascnet.watch.Recorder(every=3)
run = run_flux_network(graph, read["junctions"], provenance, by_kind, p, {}, burst["mean"], watch=flux_recorder)
truth_way, _ = edge_truth(graph, burst["segments"])
scores = direction_scores(run, graph, truth_way)
solution = run["solution"]
print(f"Tree burst {TREE_SEEDS[0]}: {len(graph['edges'])} edges, {run['measured'].sum()} with a measured flux, in "
      f"{len(set(edge_flux(graph)[2][run['measured']]))} whole vessels | joins: "
      + ", ".join(f"{kind} {sum(j['provenance'] == kind for j in provenance)}" for kind in ("skeleton", "extension", "junction", "undone"))
      + f" | p = {p:.2f} from the burst's own counts")
print("   balances: " + ", ".join(f"{run['kinds'].count(kind)} {kind}" for kind in LUMEN_KINDS if kind in run["kinds"])
      + f" | {run['constraining']} constrain a sign | flips {solution.flipped.sum()}, set aside {solution.disabled.sum()}, "
      f"impossible {solution.impossible.sum()}, directed cycles {len(run['cycles'])}, blamed lumens {len(run['blame']['lumens'])}")
print(f"   against the truth: of {scores['unmeasured']} edges without a measured direction the network infers {scores['network inferred']} "
      f"({scores['network right']} right), propagate_directions {scores['junctions inferred']} ({scores['junctions right']} right) | "
      f"measured: {scores['measured right']} of {scores['measured']} right as read, flips right {scores['flips right']}, wrong {scores['flips wrong']}")
os.makedirs("vessel_pictures/fit", exist_ok=True)
positions = {number: (node["x"], -node["y"]) for number, node in enumerate(graph["nodes"])}
true_flux = truth_way * np.nan_to_num(np.abs(run["q_hat"]), nan=1.0)
vascnet.watch.animate(flux_recorder.states, lambda state: vascnet.watch.flux_frame(
    state, positions, [edge["nodes"] for edge in graph["edges"]], true_flux, size=(7.0, 5.5)), "vessel_pictures/fit/flux.gif", fps=2.0)
print(f"   the sign step handed {flux_recorder.count} candidates to the recorder; {len(flux_recorder.states)} are filmed in "
      "vessel_pictures/fit/flux.gif (green right against the truth, red wrong, orange the move, purple a reading set aside)")
del burst, front, read, graph

# %%
# acceptance step 3 on the harness. Per set, on its bursts for developing: without truth, p (section 33's counts) and
# SIGMA_FLUX per lumen kind; with truth, the structural confidences (section 41: P(join right) from the pair's turn,
# width and density ratios and provenance; the share right of T-shaped bifurcations; P(two vessels) at crossings by
# overlap z). On the held-out bursts: the network against propagate_directions. The gate is read on section 35's trees.
FLUX_SETS = (("section 35's trees", "trees", TREE_SEEDS, TREE_HELD_OUT, SYNTH_THINGS),
             ("section 30's bursts", "trees", HARNESS_SEEDS, HELD_OUT_SEEDS, {}),
             ("crossing plexus", "crossing plexus", PLEXUS_SEEDS, PLEXUS_HELD_OUT, SYNTH_THINGS),
             ("anastomotic plexus", "anastomotic plexus", PLEXUS_SEEDS, PLEXUS_HELD_OUT, SYNTH_THINGS))


def flux_reading(burst):
    """Section 32's whole-vessel graph of a burst's consolidated lines, read with flow as tests, and its joins'
    provenance."""
    front = ridge_front(burst, workers)
    read = read_burst_ridges(burst, front, consolidate=True, speeds="whole vessels", threads=workers)
    added, _ = consolidated_extensions(front["strength"], front["orientation"], front["mask"])
    return read, join_provenance(read["graph"], added)


flux_setup = {}  # per set: p, SIGMA_FLUX per kind, the structural calibration, and what they came from
for name, kind, developing, _, things in FLUX_SETS:
    counts, imbalances, join_rows, join_truth, tees, crossings = [], {}, [], [], [], []
    for seed in developing:
        burst = synthetic_burst_35(seed, kind, **things)
        read, provenance = flux_reading(burst)
        graph = read["graph"]
        counts.append(read["counts"])
        lumens, kinds, _, overlaps, places = flux_lumens(graph, read["junctions"], provenance)
        for lumen_kind, values in lumen_imbalances(lumens, kinds, edge_flux(graph)[0]).items():
            imbalances.setdefault(lumen_kind, []).extend(values)
        _, on_segment = edge_truth(graph, burst["segments"])
        for join in provenance:
            if join["provenance"] in ("skeleton", "extension"):
                right = join_right(graph, join["ends"], on_segment, burst["segments"])
                if right is not None:
                    join_rows.append(join_features(graph, join, burst["mean"]))
                    join_truth.append(right)
        for lumen_kind, overlap, place in zip(kinds, overlaps, places):
            if lumen_kind in ("crossing", "bifurcation, T"):
                node = graph["nodes"][place]
                two = two_vessels_truth(burst, (node["x"], node["y"]), node.get("thick", 0) / 2)
                if two is not None and lumen_kind == "crossing" and np.isfinite(overlap):
                    crossings.append((overlap, two))
                elif two is not None and lumen_kind == "bifurcation, T":
                    tees.append(not two)  # right: the junction there is one lumen
        del burst, read
    total = add_counts(counts)
    calibration = {"joins": {"bifurcation, T": (sum(tees) + 0.5) / (len(tees) + 1)}}
    if join_truth:
        calibration["join model"] = calibrate_terms(join_rows, join_truth, terms=JOIN_TERMS, bins=JOIN_BINS)
    values = np.array([value for value, _ in crossings])
    two = np.array([right for _, right in crossings], dtype=bool)
    cuts = np.unique(np.quantile(values, np.linspace(0, 1, OVERLAP_BINS + 1)[1:-1])) if len(values) else np.array([])
    spots = np.array([bin_of(value, cuts) for value in values], dtype=int)
    calibration["crossing"] = (cuts, np.array([(two[spots == spot].sum() + 0.5) / ((spots == spot).sum() + 1)
                                               for spot in range(len(cuts) + 2)]))
    join_confidence = edge_confidence(join_rows, calibration["join model"], terms=JOIN_TERMS)[0] if join_truth else np.array([])
    flux_setup[name] = {"p": chosen_p(total), "p from": reliability(total), "counts": total, "sigma": sigma_flux_by_kind(imbalances),
                        "calibration": calibration, "joins": (len(join_truth), int(np.sum(join_truth))),
                        "joins told apart": told_apart(join_confidence[np.array(join_truth, dtype=bool)],
                                                       join_confidence[~np.array(join_truth, dtype=bool)]) if join_truth else np.nan,
                        "tees": (len(tees), int(sum(tees))), "crossings": (len(two), int(two.sum()))}

print("On the bursts for developing: p and SIGMA_FLUX per lumen kind without truth (robust sd of section 27's flux test over the "
      "lumens with every arm measured and possible, and their count); the structural confidences with truth")
for name, setup in flux_setup.items():
    joins, tees, crossings = setup["joins"], setup["tees"], setup["crossings"]
    print(f"   {name:20s} p = {setup['p']:.2f} (three-arm junctions {setup['p from'][0]:.2f} of {setup['counts']['three arms']}, sides "
          f"{setup['p from'][1]:.2f} of {setup['counts']['sides']}) | " + ", ".join(
              f"{kind} {value:.2f} ({count}, {source})" for kind, (value, count, source) in setup["sigma"].items()))
    print(f"   {'':20s} two-arm joins: {joins[1]} of {joins[0]} right, told apart by P(join right) {setup['joins told apart']:.2f} | "
          f"T-shaped bifurcations: {tees[1]} of {tees[0]} right | crossings: {crossings[1]} of {crossings[0]} are two vessels, by "
          f"overlap z " + ", ".join(f"{share:.2f}" for share in setup["calibration"]["crossing"][1]))

flux_results = {}  # per set: per held-out burst the network's numbers and its direction scores
for name, kind, _, held_out, things in FLUX_SETS:
    setup, rows = flux_setup[name], []
    for seed in held_out:
        burst = synthetic_burst_35(seed, kind, **things)
        read, provenance = flux_reading(burst)
        graph = read["graph"]
        run = run_flux_network(graph, read["junctions"], provenance, setup["sigma"], setup["p"], setup["calibration"], burst["mean"])
        if run is None:
            continue
        truth_way, _ = edge_truth(graph, burst["segments"])
        solution = run["solution"]
        rows.append({"seed": seed, **direction_scores(run, graph, truth_way), "lumens": len(run["lumens"]),
                     "constraining": run["constraining"], "set aside": int(solution.disabled.sum()),
                     "impossible": int(solution.impossible.sum()), "cycles": len(run["cycles"]),
                     "blamed": len(run["blame"]["lumens"]), "speeds in doubt": len(run["blame"]["speeds"]),
                     "free ends": len(run["free ends"]), "edges": len(graph["edges"])})
        del burst, read
    flux_results[name] = rows
    memory_note(f"section 40, {name}")

print("\nHeld out: directions among the edges without a measured one (the truth known), the network against "
      "propagate_directions; shares with their 95% interval over bursts")
for name, rows in flux_results.items():
    network = with_interval([row["network right"] for row in rows], [row["network inferred"] for row in rows])
    junctions = with_interval([row["junctions right"] for row in rows], [row["junctions inferred"] for row in rows])
    unmeasured = sum(row["unmeasured"] for row in rows)
    print(f"   {name:20s} {len(rows)} bursts, {unmeasured} edges without a measured direction | network: {sum(r['network inferred'] for r in rows)} inferred, "
          f"{sum(r['network right'] for r in rows)} right ({network[0] * 100:.0f}%, {network[1] * 100:.0f} to {network[2] * 100:.0f}) | junctions: "
          f"{sum(r['junctions inferred'] for r in rows)} inferred, {sum(r['junctions right'] for r in rows)} right ({junctions[0] * 100:.0f}%, "
          f"{junctions[1] * 100:.0f} to {junctions[2] * 100:.0f})")
    print(f"   {'':20s} measured: {sum(r['measured right'] for r in rows)} of {sum(r['measured'] for r in rows)} right as read | flips "
          f"right {sum(r['flips right'] for r in rows)}, wrong {sum(r['flips wrong'] for r in rows)} | lumens {sum(r['lumens'] for r in rows)}, "
          f"constraining a sign {sum(r['constraining'] for r in rows)}, set aside {sum(r['set aside'] for r in rows)}, impossible "
          f"{sum(r['impossible'] for r in rows)}, directed cycles {sum(r['cycles'] for r in rows)}, blamed {sum(r['blamed'] for r in rows)}, "
          f"speeds in doubt {sum(r['speeds in doubt'] for r in rows)}")
print("\nAs description, the network's directions of the edges without a measured one at other cutoffs of |q| / sigma (inferred, right):")
for name, rows in flux_results.items():
    curve = np.array([point for row in rows for point in row["curve"]]).reshape(-1, 2)
    print(f"   {name:20s} " + " | ".join(f">= {cut}: {(curve[:, 0] >= cut).sum()}, {int(curve[curve[:, 0] >= cut, 1].sum())}"
                                         for cut in (0.5, 1.0, 1.5, 2.0, 3.0))
          + f" | propagate_directions: {sum(r['junctions inferred'] for r in rows)}, {sum(r['junctions right'] for r in rows)}")
rows = flux_results["section 35's trees"]
right = (sum(r["network right"] for r in rows), sum(r["junctions right"] for r in rows))
shares = (right[0] / max(sum(r["network inferred"] for r in rows), 1), right[1] / max(sum(r["junctions inferred"] for r in rows), 1))
verdict = right[0] > right[1] and shares[0] >= shares[1]
print(f"\nThe gate, on section 35's held-out trees: the network gives the right direction to {right[0]} edges without a measured one, "
      f"propagate_directions to {right[1]}; shares right {shares[0] * 100:.0f}% and {shares[1] * 100:.0f}% -> "
      f"{'IMPROVES: passed' if verdict else 'does not improve: FAILED'} | the plexus is reported above (expected: no change)")
os.makedirs("vessel_pictures/flux", exist_ok=True)
save_table([{"set": name, **{key: value for key, value in row.items() if key != "curve"}} for name, rows in flux_results.items()
            for row in rows], "vessel_pictures/flux/held_out.csv")
memory_note("section 40, harness")

# %%
# section 40 on the real burst: the whole-vessel graphs of section 32 (whole_graphs), their junctions read from the
# geometry, p from section 33's counts pooled over the fixations (chosen_p), SIGMA_FLUX per kind measured here where
# there are FLUX_LEAST lumens of a kind, else the harness's, and the structural confidences calibrated on section 35's
# trees (the real burst has no truth to calibrate them on). Also the consolidated graph's shares of
# impossible junctions and directed cycles: the reference of the rule by which lambda is chosen.
real_runs = []
counts = []
real_junctions = []
for number, graph in enumerate(whole_graphs):
    found = read_junctions(graph, saved["density"][number], saved["static_scores"][number], flow=False)[0]
    real_junctions.append(found)
    counts.append(direction_counts(graph, found, whole_readings[number]))
p_three, p_sides = reliability(add_counts(counts))
real_p = chosen_p(add_counts(counts))
imbalances = {}
for number, graph in enumerate(whole_graphs):
    lumens, kinds, _, _, _ = flux_lumens(graph, real_junctions[number], join_provenance(graph))
    for kind, values in lumen_imbalances(lumens, kinds, edge_flux(graph)[0]).items():
        imbalances.setdefault(kind, []).extend(values)
real_sigma = sigma_flux_by_kind(imbalances, fallback=flux_setup["section 35's trees"]["sigma"])
print(f"Real burst: p = {real_p:.2f} (chosen_p) from section 33's counts over both fixations (three-arm junctions {p_three:.2f}: "
      f"{add_counts(counts)['all in or all out']} of {add_counts(counts)['three arms']} all in or all out; sides {p_sides:.2f})")
print("   SIGMA_FLUX per lumen kind: " + ", ".join(f"{kind} {value:.2f} ({count} lumens, {source})" for kind, (value, count, source) in real_sigma.items()))
for number, graph in enumerate(whole_graphs):
    front = saved_front(saved, number)
    added, _ = consolidated_extensions(front["strength"], front["orientation"], front["mask"])
    provenance = join_provenance(graph, added)
    run = run_flux_network(graph, real_junctions[number], provenance, real_sigma, real_p,
                           flux_setup["section 35's trees"]["calibration"], saved["density"][number])
    real_runs.append(run)
    solution = run["solution"]
    lost = np.array([flux for flux, sigma in run["free ends"] if np.isfinite(sigma) and sigma > 0 and abs(flux) >= FLUX_INFERRED * sigma])
    unmeasured = ~run["measured"] & np.array([reading is not None for reading in graph["readings"]])
    inferred = unmeasured & (solution.direction_confidence >= FLUX_INFERRED)
    agree = (solution.direction[inferred] == propagated_directions(graph)[inferred]) & (propagated_directions(graph)[inferred] != 0)
    print(f"Fixation {number}: {len(graph['edges'])} edges, {run['measured'].sum()} with a measured flux | joins: "
          + ", ".join(f"{kind} {sum(j['provenance'] == kind for j in provenance)}" for kind in ("skeleton", "extension", "junction", "undone")))
    print("   balances: " + ", ".join(f"{run['kinds'].count(kind)} {kind}" for kind in LUMEN_KINDS if kind in run["kinds"])
          + f" | constraining a sign {run['constraining']}, set aside {solution.disabled.sum()}, impossible {solution.impossible.sum()}, "
          f"directed cycles {len(run['cycles'])}, flips {solution.flipped.sum()}, blamed lumens {len(run['blame']['lumens'])}, speeds in doubt "
          f"{len(run['blame']['speeds'])}")
    print(f"   free ends: {len(run['free ends'])}, of which {len(lost)} lose or gain a flux of {FLUX_INFERRED:.0f} sigma or more | edges "
          f"without a measured direction: {unmeasured.sum()}, the network infers {inferred.sum()}; propagate_directions agrees on "
          f"{agree.sum()} of them (agreement between two inferences, not a score)")
    print(f"   lambda's reference, the consolidated graph: impossible lumens {solution.impossible.sum()} of {len(run['lumens'])} "
          f"({solution.impossible.mean() * 100:.1f}%), directed cycles {len(run['cycles'])} per {len(graph['edges'])} edges")
memory_note("section 40, real burst")

# %% [markdown]
# ## 36. Junction cells (acceptance step 4)
#
# A junction cell at a pixel counts the arms of a junction there and fires in proportion to how clearly it sees them (`junction_cell`).
#
# - **The ring** (`ring_profile`). On a ring of radius ρ around the pixel, at `RING_STEPS` angles φ, the cell reads the pre-interaction score of the neuron whose orientation is the radial direction φ, interpolated between the two nearest of the 16.
# - **The arms** (`ring_arms`). The arms are the profile's local maxima above `T_LOW`, in runs that reach above `T_HIGH`; two within `SAME_ARM` degrees are one arm.
# - **The count and the rate.** `n` is the number of arms. `rate` is their scores' geometric mean times a core term: the core's density against the arms', since a junction is at least as dark as its arms.
# - **The rings.** Rings of `JUNCTION_RINGS` = 4 to 24 px are read, and the ring with the largest rate is kept.
# - **What `n` says** (`cell_kind`): an end, a line (two arms within `STRAIGHT` of straight, suppressed), a bend, or a candidate for three or four arms.
# - **Plaid cells** (`plaid_cells`): two orientations at least `PLAID_APART` = 45 degrees apart, both above `T_HIGH`.
#
# **The three checks**, before any cell map is built for a whole frame. Each is run on pictures rendered as section 35's generator renders: lumens united, transmittances blurred and multiplied, the tissue's texture and a mean frame's noise added. Each picture is scored by sections 19 and 20 (`render_check`), at blurs of 0.8, 1.5 and 3 px with eight noise realisations.
#
# 1. **A thin vessel (R = 1.5) crossing a thick one (R = 6).** Does the thin arm appear as a peak on the ring that matches the thick vessel, at what rate, and on its own ring (4 px)? If not, the second reading of the ring at the thin width is added (`width_channels`: the 16 neurons of one width before the widths compete). That reading is made beside the first, so both can be compared.
# 2. **A thin daughter (R = 1.5) leaving a thick trunk (R = 6) at 60 degrees, one lumen.** On which rings is the daughter an arm, and what does the cell keep? The ring with the most arms among those above `T_HIGH` (`most_arms_ring`) is reported beside section 36's rule as description.
# 3. **A crossing the skeleton breaks into two forks 5–10 px apart.** Two R = 3 vessels crossing at 45 degrees give two forks 8 px apart; does the cell at their midpoint say `n` = 4?
#
# **The gate** (Part III): the thin arm appears on its own ring at a rate above `T_LOW`, or the second reading at the thin width is added and the plan says so.
#
# **Result:** `check_rows` (`vessel_pictures/junction_cells/checks.csv`) and a picture of the three configurations (`checks.png`).

# %%
JUNCTION_RINGS = (4, 6, 10, 16, 24)  # pixels; the radii of the rings a junction cell reads its arms on
RING_STEPS = 72                       # angles read on a ring (5 degrees apart)
PLAID_APART = 45.0                    # degrees; two orientations this far apart, both above T_HIGH, make a plaid cell
CHECK_SHAPE = (161, 161)              # pixels; the pictures the section's checks are rendered in
CHECK_NOISE = 0.0025                  # density; the white noise of a mean of the generator's kept frames (shot noise of
                                      # SYNTH_LIGHT over 62 frames), on top of the tissue's texture
CHECK_SEEDS = tuple(range(1, 9))      # noise realisations per configuration


def ring_profile(scores, x, y, radius, steps=RING_STEPS):
    """A junction cell's view of a ring of `radius` around (x, y): at every angle phi the score of the neuron whose
    orientation is the radial direction phi, interpolated between the two nearest of the N_ORIENT orientations, read
    at the ring's point. Returns the angles (radians) and the profile."""
    phi = np.arange(steps) * 2 * np.pi / steps
    px, py = x + radius * np.cos(phi), y + radius * np.sin(phi)
    k = (phi % np.pi) / (np.pi / N_ORIENT)
    low, weight = np.floor(k).astype(int) % N_ORIENT, k - np.floor(k)
    high = (low + 1) % N_ORIENT
    profile = np.zeros(steps)
    for layer in np.unique(np.r_[low, high]):
        values = scipy.ndimage.map_coordinates(np.asarray(scores[layer], dtype=np.float32), [py, px], order=1, mode="nearest")
        profile += np.where(low == layer, 1 - weight, 0) * values + np.where(high == layer, weight, 0) * values
    return phi, profile


def ring_arms(phi, profile):
    """The arms on a ring's profile: runs of angles above T_LOW that hold a value above T_HIGH (hysteresis on the
    circle), each read at its peak; two peaks within SAME_ARM degrees are one arm. Returns [(angle, score)]."""
    above = profile > T_LOW
    if above.all() or not above.any():
        return [] if not above.any() else [(float(phi[np.argmax(profile)]), float(profile.max()))]
    start = int(np.argmin(above))  # a place below T_LOW, so that no run is cut by the end of the array
    order = np.r_[np.arange(start, len(profile)), np.arange(0, start)]
    peak = (profile >= np.roll(profile, 1)) & (profile >= np.roll(profile, -1)) & above  # local maxima on the circle
    arms, run = [], []
    for place in list(order) + [order[0]]:
        if above[place]:
            run.append(place)
        elif run:
            if profile[run].max() > T_HIGH:  # a run joined to a peak above T_HIGH: every peak in it is an arm
                arms += [(float(phi[spot]), float(profile[spot])) for spot in run if peak[spot]]
            run = []
    merged = []
    for angle, score in sorted(arms, key=lambda arm: -arm[1]):
        if all(np.degrees(abs(np.angle(np.exp(1j * (angle - other))))) >= SAME_ARM for other, _ in merged):
            merged.append((angle, score))
    return sorted(merged)


def junction_cell(scores, density, x, y, rings=JUNCTION_RINGS):
    """A junction cell at (x, y) (section 36): on each ring its arms (ring_arms), n their number, and its rate, the
    geometric mean of the arms' scores times a core term, the core's density against the arms' (a junction is at
    least as dark as its arms; 1 without a density). The ring with the largest rate is kept. Returns that ring's
    reading with every ring's reading under `rings`."""
    readings = []
    for radius in rings:
        phi, profile = ring_profile(scores, x, y, radius)
        arms = ring_arms(phi, profile)
        core = 1.0
        if density is not None and arms:
            rows, columns = density.shape
            row, column = int(np.clip(round(y), 1, rows - 2)), int(np.clip(round(x), 1, columns - 2))
            centre = float(np.nanmax(density[row - 1:row + 2, column - 1:column + 2]))
            on_arms = [float(density[int(np.clip(round(y + radius * np.sin(a)), 0, rows - 1)),
                                     int(np.clip(round(x + radius * np.cos(a)), 0, columns - 1))]) for a, _ in arms]
            core = float(np.clip(centre / max(np.mean(on_arms), 1e-9), 0, 1)) if np.mean(on_arms) > 0 else 1.0
        rate = float(np.exp(np.mean(np.log([score for _, score in arms]))) * core) if arms else 0.0
        readings.append({"ring": radius, "n": len(arms), "rate": rate, "arms": arms, "profile": profile})
    best = max(readings, key=lambda reading: reading["rate"])
    return {**best, "rings": readings}


def cell_kind(cell):
    """What a cell's n says (section 36): an end, a line (two arms within STRAIGHT of a straight line: suppressed),
    a bend, or a candidate bifurcation or crossing."""
    if cell["n"] == 2:
        (a, _), (b, _) = cell["arms"]
        turn = 180 - np.degrees(abs(np.angle(np.exp(1j * (a - b)))))
        return "line" if turn <= STRAIGHT else "bend"
    return {0: "nothing", 1: "end", 3: "three arms", 4: "four arms"}.get(cell["n"], "more arms")


def plaid_cells(scores):
    """Pixels where two orientations at least PLAID_APART degrees apart both answer above T_HIGH in the
    pre-interaction score: the weaker line of a crossing is there, before the interactions could tune it away."""
    strong = np.asarray(scores) > T_HIGH
    apart = int(np.ceil(PLAID_APART / (180 / N_ORIENT)))
    plaid = np.zeros(strong.shape[1:], dtype=bool)
    for k in range(N_ORIENT):
        for other in range(k + apart, k + N_ORIENT - apart + 1):
            plaid |= strong[k] & strong[other % N_ORIENT]
    return plaid


def width_channels(contrast, valid, quiet, width):
    """The N_ORIENT neurons of one width, before the widths compete (section 20's fields and complex unit, the
    width's own noise level from its even field on the quiet pixels, no floor): what the "second reading at the
    thin width" of section 36 reads its ring on. Returns (N_ORIENT, rows, columns) in noise units."""
    rows, columns = contrast.shape
    factor = reduction(width)
    spectrum, shape, inner = padded_spectrum(contrast, factor)
    share = blur(valid.astype(np.float32), ELONGATION * width)
    small = (columns // factor, rows // factor)
    seen = cv2.resize(share, small, interpolation=cv2.INTER_AREA) >= VALID_SHARE
    places = np.flatnonzero(cv2.resize((quiet & (share >= VALID_SHARE)).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST))
    sample = places[::max(1, len(places) // NOISE_SAMPLE)]
    out = np.zeros((N_ORIENT, rows, columns), dtype=np.float32)
    levels = []
    for k in range(N_ORIENT):
        even, odd, halves = field_answers(spectrum, shape, inner, width / factor, k)
        levels.append(robust_sd(even.ravel()[sample]) * factor ** -0.5)
        answer = np.where(seen, complex_unit(even, odd, halves), 0).astype(np.float32) * np.float32(factor ** -0.5)
        out[k] = cv2.resize(answer, (columns, rows), interpolation=cv2.INTER_LINEAR) if factor > 1 else answer
    return out / np.float32(np.median(levels))


def line_points(start, end):
    """A straight polyline from start to end, points a pixel apart."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = int(np.ceil(np.hypot(*(end - start)))) + 1
    return start + np.linspace(0, 1, count)[:, None] * (end - start)


def render_check(lumens, blur, seed, shape=CHECK_SHAPE):
    """A picture for the section's checks: `lumens` is a list of lumens, each a list of (points, radius); the
    segments of one lumen are united (max), the lumens' transmittances are blurred by `blur` and multiplied, as
    section 35's generator does, and the tissue's texture and a mean frame's white noise are added. Returns the
    density, the score (section 20 on section 19's contrast, as read_burst does), the half-width and the clean
    density."""
    rng = np.random.default_rng(seed)
    rows, columns = shape
    y, x = np.mgrid[:rows, :columns]
    pixels = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)
    transmitted = np.ones(shape)
    lines = np.zeros(shape, dtype=np.uint8)
    for lumen in lumens:
        layer = np.zeros(rows * columns)
        for points, radius in lumen:
            segment = {"points": points, "radius": radius}
            footprint(segment, pixels, True)
            layer[segment["pixels"]] = np.maximum(layer[segment["pixels"]], segment["chord"])
            cv2.polylines(lines, np.rint(points).astype(np.int32)[None], False, 1)
        transmitted *= cv2.GaussianBlur(np.exp(-layer.reshape(shape)), (0, 0), blur)
    clean = -np.log(transmitted)
    texture = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), SYNTH_GRAIN)
    density = (clean + texture * SYNTH_TEXTURE / texture.std() + rng.normal(0, CHECK_NOISE, shape)).astype(np.float32)
    valid = np.ones(shape, dtype=bool)
    wall = cv2.distanceTransform((lines == 0).astype(np.uint8), cv2.DIST_L2, 5)
    quiet = wall > 3 * max(radius for lumen in lumens for _, radius in lumen)
    contrast = contrast_layer(density, robust_sd(density[quiet]), valid)
    contrast = contrast / robust_sd(contrast[quiet])
    scores, radius, _, _ = orientation_hypercolumns(contrast, valid, quiet)
    strength, orientation, half_width = winning_neurons(scores, radius)
    return {"density": density, "scores": scores, "strength": strength, "orientation": orientation, "half_width": half_width,
            "contrast": contrast, "valid": valid, "quiet": quiet, "clean": clean}


def check_configurations(apart=45.0):
    """The three configurations of section 36's checks, centred in a CHECK_SHAPE picture: a thick vessel (R = 6)
    crossed at right angles by a thin one (R = 1.5), two lumens; a thin daughter (R = 1.5) leaving a thick trunk
    (R = 6) at 60 degrees, one lumen; and two vessels (R = 3) crossing at `apart` degrees, at which the skeleton breaks
    the crossing into two forks 5 to 10 px apart (measured: 8 px at 45 degrees, 5.5 at 60, 21 at 30, one fork at 75). Returns {name: (lumens, (x, y) of the junction, the thin arm's
    angles in degrees, or None)}."""
    rows, columns = CHECK_SHAPE
    c = np.array([(columns - 1) / 2, (rows - 1) / 2])
    away = lambda degrees, length: c + length * np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
    return {
        "thin crossing a thick vessel": ([[(line_points(away(180, 75), away(0, 75)), 6.0)],
                                          [(line_points(away(270, 75), away(90, 75)), 1.5)]], tuple(c), (90, 270)),
        "thin daughter of a thick trunk": ([[(line_points(away(180, 75), away(0, 75)), 6.0),
                                             (line_points(c, away(-60, 75)), 1.5)]], tuple(c), (300,)),
        "crossing split into two forks": ([[(line_points(away(180, 75), away(0, 75)), 3.0)],
                                           [(line_points(away(180 + apart, 75), away(apart, 75)), 3.0)]], tuple(c), None),
    }


def most_arms_ring(cell):
    """A rule beside section 36's (the ring with the largest rate): the ring with the most arms among those whose rate
    is above T_HIGH. A thin daughter leaving a thick trunk shows only on the rings outside the trunk, and there the
    geometric mean with its weaker arm stays below the two thick arms' rate on a smaller ring (measured)."""
    strong = [reading for reading in cell["rings"] if reading["rate"] > T_HIGH]
    return max(strong, key=lambda reading: (reading["n"], reading["rate"])) if strong else cell


def fork_places(strength, orientation, near, within=30):
    """The forks of the joined ridges (section 37) within `within` px of a place: the centres of the fork pixels'
    connected groups."""
    joined = joined_ridges(strength, orientation, strength > 0)
    forks = skeleton_parts(joined["skeleton"])[1]
    count, labels = cv2.connectedComponents(forks.astype(np.uint8), connectivity=8)
    places = [np.argwhere(labels == label).mean(axis=0)[::-1] for label in range(1, count)]
    return [place for place in places if np.hypot(*(place - np.asarray(near))) < within]

# %%
# section 36's three checks, before any cell map is built for a whole frame (acceptance step 4): each configuration
# rendered at three blurs, CHECK_SEEDS noise realisations each, and the junction cell read at the junction (for the
# crossing the skeleton splits, at the midpoint of its two forks)
CHECK_BLURS = (0.8, 1.5, 3.0)  # pixels; the generator's sharpest blur, the real burst's (section 31), a defocused tree


def arm_near(arms, target, within=15):
    """The score of the arm within `within` degrees of a direction (0 when there is none)."""
    return max([score for angle, score in arms if abs((np.degrees(angle) - target + 180) % 360 - 180) <= within], default=0.0)


check_rows = []
configurations = check_configurations()
for name, (lumens, place, thin) in configurations.items():
    for defocus in CHECK_BLURS:
        for seed in CHECK_SEEDS:
            picture = render_check(lumens, defocus, seed)
            x, y = place
            forks = []
            if thin is None:
                forks = fork_places(picture["strength"], picture["orientation"], place)
                if len(forks) >= 2:
                    x, y = np.mean(forks, axis=0)
            cell = junction_cell(picture["scores"], picture["density"], x, y)
            other = most_arms_ring(cell)
            row = {"configuration": name, "blur": defocus, "seed": seed, "kept ring": cell["ring"], "n": cell["n"], "kind": cell_kind(cell),
                   "rate": cell["rate"], "most arms ring": other["ring"], "most arms n": other["n"], "forks": len(forks),
                   "forks apart": max((float(np.hypot(*(a - b))) for a in forks for b in forks), default=0.0)}
            for reading in cell["rings"]:
                row[f"n, ring {reading['ring']}"] = reading["n"]
                if thin is not None:
                    row[f"thin arm, ring {reading['ring']}"] = min(arm_near(reading["arms"], target) for target in thin)
            if name.startswith("thin crossing"):
                channels = width_channels(picture["contrast"], picture["valid"], picture["quiet"], HYPER_WIDTHS[0])
                for radius in JUNCTION_RINGS:
                    arms = ring_arms(*ring_profile(channels, x, y, radius))
                    row[f"second reading, ring {radius}"] = min(arm_near(arms, target) for target in thin)
            check_rows.append(row)


def share_above(rows, key):
    """The share of rows whose value under `key` is above T_LOW, and the values' median."""
    return f"{np.mean([row[key] > T_LOW for row in rows]) * 100:.0f}% ({np.median([row[key] for row in rows]):.1f})"


print(f"Section 36's checks: {len(CHECK_SEEDS)} noise realisations per configuration and blur. Per ring, the share of realisations in which "
      f"the thin arm is an arm above T_LOW = {T_LOW} (in brackets the median score of that arm), and the n the cell keeps")
for name, (lumens, place, thin) in configurations.items():
    for defocus in CHECK_BLURS:
        rows = [row for row in check_rows if row["configuration"] == name and row["blur"] == defocus]
        kept = {kind: sum(row["kind"] == kind for row in rows) for kind in sorted({row["kind"] for row in rows})}
        text = f"   {name:30s} blur {defocus:3.1f}: kept {kept}, on ring {np.median([row['kept ring'] for row in rows]):.0f} (median)"
        if thin is not None:
            text += " | thin arm " + ", ".join(f"ring {radius}: {share_above(rows, f'thin arm, ring {radius}')}" for radius in JUNCTION_RINGS)
            text += f" | the most-arms rule keeps n = {sorted({row['most arms n'] for row in rows})}"
        else:
            text += (f" | {np.mean([row['forks'] for row in rows]):.1f} forks, {np.median([row['forks apart'] for row in rows]):.1f} px apart "
                     "(median) | four arms on ring " + ", ".join(
                         f"{radius}: {np.mean([row[f'n, ring {radius}'] == 4 for row in rows]) * 100:.0f}%" for radius in JUNCTION_RINGS))
        print(text)
        if name.startswith("thin crossing"):
            print(f"   {'':30s} the second reading, at the thin width: " + ", ".join(
                f"ring {radius}: {share_above(rows, f'second reading, ring {radius}')}" for radius in JUNCTION_RINGS))
thin_rows = [row for row in check_rows if row["configuration"].startswith("thin crossing")]
own = np.mean([row[f"thin arm, ring {JUNCTION_RINGS[0]}"] > T_LOW for row in thin_rows])
print(f"\nThe gate: the thin arm of a thin vessel crossing a thick one is an arm above T_LOW on its own ring ({JUNCTION_RINGS[0]} px) in "
      f"{own * 100:.0f}% of the {len(thin_rows)} pictures, every blur -> {'PASSED' if own >= 0.9 else 'not on its own ring: the second reading decides'}")
os.makedirs("vessel_pictures/junction_cells", exist_ok=True)
save_table(check_rows, "vessel_pictures/junction_cells/checks.csv")

# the three configurations at the real burst's blur: the density, and the arms the cell finds on every ring
figure, axes = plt.subplots(1, len(configurations), figsize=(15, 5))
for axis, (name, (lumens, place, thin)) in zip(axes, configurations.items()):
    picture = render_check(lumens, 1.5, CHECK_SEEDS[0])
    x, y = place
    if thin is None:
        forks = fork_places(picture["strength"], picture["orientation"], place)
        for fx, fy in forks:
            axis.plot(fx, fy, "x", color="orange", ms=8, mew=2)
        if len(forks) >= 2:
            x, y = np.mean(forks, axis=0)
    cell = junction_cell(picture["scores"], picture["density"], x, y)
    axis.imshow(picture["density"], cmap="gray_r", vmin=-0.02, vmax=np.percentile(picture["density"], 99.9))
    for reading in cell["rings"]:
        kept = reading["ring"] == cell["ring"]
        axis.add_patch(plt.Circle((x, y), reading["ring"], fill=False, lw=0.8, color="tab:red" if kept else "tab:blue"))
        for angle, score in reading["arms"]:
            axis.plot(x + reading["ring"] * np.cos(angle), y + reading["ring"] * np.sin(angle), "o", ms=3,
                      color="tab:red" if kept else "tab:blue")
    axis.set_xlim(x - 40, x + 40)
    axis.set_ylim(y + 40, y - 40)
    axis.set_title(f"{name}\nkept ring {cell['ring']} px: n = {cell['n']} ({cell_kind(cell)}), rate {cell['rate']:.0f}", fontsize=9)
    axis.axis("off")
figure.suptitle("section 36's checks at a blur of 1.5 px: the rings (red: the one the cell keeps) and the arms found on each; orange: the "
                "skeleton's forks", fontsize=10)
figure.tight_layout()
figure.savefig("vessel_pictures/junction_cells/checks.png", dpi=90)
plt.show()

# %% [markdown]
# ## 38(a). The model as splines, rendered and scored (acceptance step 5)
#
# **The graph as a view of the splines** (`spline_to_graph`, section 37's rules):
#
# - k ends within `RIDGE_JOIN` are one vertex.
# - An end on another spline's body splits the body there (a T).
# - Two bodies passing within a pixel at `SAME_ARM` degrees or more cross; at less they touch at a glance and make no vertex.
# - A twin (`read_arms`' rule) is marked and makes no vertex.
# - Vertices within one core are one.
# - A spline that crosses itself makes a loop; multi-edges and self-loops are allowed.
# - The cycles are counted as edges − vertices + components.
#
# **The initial model** (`initial_model`):
#
# - section 37's pieces, their graph with the junctions read from the geometry;
# - the lumens: every edge united through its junction's groups of two or three and through two pieces meeting end to end; a crossing's two pairs stay two lumens;
# - the darkness: R and κ per window outside the cores (`window_kappa`), fitted on the density with the other lumens' render taken off, the thickest first, three passes (`kappa_passes`).
#
# **The blur is fitted on the profiles.** Section 38 fits R and κ per window; with R fixed from the ridge's half-width, κ came out 2–3 times too large on synthetic bursts. The blur is fitted per picture as the median over windows of the blur that fits each profile best (`fitted_blur`). Section 29's `picture_blur` cannot be less than the blur plus the thinnest vessel's own width, and came out at 2.2–2.8 px against true blurs of 1.3–2.1. The fitted blur came out at 1.75 px.
#
# **The renderer** (`render_graph`, `draw_lumen`): each lumen drawn on its own canvas as chord density, the max within the lumen; its transmittance blurred; densities added across lumens. A twin is not drawn.
#
# **The scoring** (`scoring_frame`, `channel_answers`, `ridges_of`): section 19's contrast with the data's own noise map and gain, then section 20's fields per channel, 16 orientations × 8 widths, before the widths compete, in noise units of the data's eight constants. The render is never measured. On the real burst the noise map is rebuilt from the saved density with section 12's `local_rms`, on quiet pixels taken as those with no neuron above `T_LOW` within 8 px, since section 29 saves neither; the data and the render go through the same rebuilt scoring. The ridges are `ridge_crests` on the winner, with no interactions.
#
# **The self-consistency number** (`self_consistency`): the render's ridges found again in the data's within `REPEAT_WITHIN`, and the data's in the render's, with what chance finds taken off (`repeated`). It is measured over the whole frame and within the tubes. **The zero point** of the topology residual per channel (`zero_points`) is the median of data minus render on the model's own centerline points. The topology residual is `topology_residual`. **Windowed re-scoring** (`rescore_window`) works on windows aligned to 20 px with a 110 px margin.
#
# **The gate** (Part III, step 5): within the tubes the agreement reaches the data's own repeatability on the real burst; the windowed re-score equals the full one inside the window within 0.005; a move's evaluation (one windowed re-score) takes under a few seconds (5 s is the line drawn here).
#
# **Result:** `real_model`, the numbers printed, `vessel_pictures/render/`, and the darkness fit's film `vessel_pictures/fit/kappa.gif`.

# %%
WINDOW_ALIGN = 20        # pixels; a re-scored window starts and ends on multiples of this (the widths' reduced grids:
                         # factors 2, 2, 4 and 5 for widths 6 and up)
WINDOW_MARGIN = 110      # pixels; around a re-scored window, so that the widest field sees what it sees in the frame
KAPPA_PASSES = 3         # passes of the darkness fit over the lumens (Gauss-Seidel, the thickest first)
PROFILE_STEP = 0.5       # pixels between the samples of a density profile across a vessel
RIDGE_FINE = 3           # sub-pixel points a side when a vessel thinner than 2 px is drawn
NOISE_WINDOW = 48        # pixels; section 12's window for the local noise of the density (fit_background's default)


# ---------------------------------------------------------------- spline_to_graph (section 37)

def spline_to_graph(splines, join=RIDGE_JOIN):
    """The graph as a view of a set of splines (section 37's rules). Each spline is a dict with x, y and radius
    (half-width) arrays, points about a pixel apart.

      ends    k ends within `join` of each other are one vertex of k arms
      T       an end within `join` of another spline's body (away from its ends) splits the body there
      cross   two bodies that pass within a pixel of each other at SAME_ARM degrees or more cross: both are split at
              the closest points, one vertex of four arms; at less they touch at a glance and make no vertex
      twin    a spline that runs beside a longer one for most of its length (read_arms' rule: within SAME_ARM
              degrees, closer than the larger width plus RING_MARGIN) is marked and makes no vertex
      core    two vertices joined by a piece no longer than one core (half the widest incident width plus
              RING_MARGIN) are one, as contract_cores does, while the merged vertex stays within two cores
      loop    a spline that crosses itself is split there; multi-edges and self-loops are allowed
    Returns {"vertices": [{"x", "y", "ends": [(edge, end)]}], "edges": [{"x", "y", "radius", "spline", "span",
    "vertices"}], "twins": the twin splines, "cycles": edges - vertices + components}."""
    cuts = [{0, len(spline["x"]) - 1} for spline in splines]
    links = []  # (spline, place), (spline, place): one vertex
    ends = [(number, place) for number, spline in enumerate(splines) for place in (0, len(spline["x"]) - 1)]
    end_xy = np.array([[splines[n]["x"][p], splines[n]["y"][p]] for n, p in ends]).reshape(-1, 2)
    if len(ends):
        for a, b in scipy.spatial.cKDTree(end_xy).query_pairs(join):
            links.append((ends[a], ends[b]))
    owner = np.concatenate([np.full(len(spline["x"]), number) for number, spline in enumerate(splines)])
    place = np.concatenate([np.arange(len(spline["x"])) for spline in splines])
    xy = np.column_stack([np.concatenate([spline["x"] for spline in splines]), np.concatenate([spline["y"] for spline in splines])])
    tree = scipy.spatial.cKDTree(xy)
    size = np.array([len(spline["x"]) for spline in splines])
    body = lambda spot: join < place[spot] < size[owner[spot]] - 1 - join
    # T: an end that lands on another spline's body
    for (number, spot), (x, y) in zip(ends, end_xy):
        for other in tree.query_ball_point((x, y), join):
            if owner[other] != number and body(other):
                cuts[owner[other]].add(int(place[other]))
                links.append(((number, spot), (int(owner[other]), int(place[other]))))
    # crossings, glancing contacts and loops: points of two bodies (or two far parts of one) within a pixel
    def tangent(spot):
        number, at = owner[spot], place[spot]
        lo, hi = max(at - 2, 0), min(at + 2, size[number] - 1)
        start = np.flatnonzero(owner == number)[0]
        d = xy[start + hi] - xy[start + lo]
        return d / max(np.hypot(*d), 1e-9)
    pairs = np.array(sorted(tree.query_pairs(1.0)), dtype=int).reshape(-1, 2)
    pairs = pairs[[body(a) and body(b) and (owner[a] != owner[b] or abs(place[a] - place[b]) > 4 * join + 2) for a, b in pairs]] \
        if len(pairs) else pairs
    taken = set()
    for a, b in sorted(pairs.tolist(), key=lambda pair: np.hypot(*(xy[pair[0]] - xy[pair[1]]))):
        key = (owner[a], place[a] // 6, owner[b], place[b] // 6)
        if key in taken:
            continue  # one contact is one crossing, at its closest pair
        turn = np.degrees(np.arccos(np.clip(abs(np.dot(tangent(a), tangent(b))), 0, 1)))
        taken.update({(owner[a], place[a] // 6 + d1, owner[b], place[b] // 6 + d2) for d1 in (-1, 0, 1) for d2 in (-1, 0, 1)})
        if turn < SAME_ARM:
            continue  # a glance: no vertex
        cuts[owner[a]].add(int(place[a]))
        cuts[owner[b]].add(int(place[b]))
        links.append(((int(owner[a]), int(place[a])), (int(owner[b]), int(place[b]))))
    # twins
    twins = set()
    lengths = size.astype(float)
    if len(xy):
        widths = 2 * np.concatenate([spline["radius"] for spline in splines]).astype(float) / CYLINDER
        theta = np.concatenate([np.arctan2(np.gradient(spline["y"]), np.gradient(spline["x"])) for spline in splines])
        i, j, distance = near_pairs(xy[:, 0], xy[:, 1], widths + RING_MARGIN)
        turned = np.abs(theta[i] - theta[j]) % np.pi
        beside = ((owner[i] != owner[j]) & (lengths[owner[j]] > lengths[owner[i]])
                  & (np.degrees(np.minimum(turned, np.pi - turned)) < SAME_ARM) & (distance <= np.maximum(widths[i], widths[j]) + RING_MARGIN))
        alongside = np.zeros(len(xy), dtype=bool)
        alongside[i[beside]] = True
        for number in range(len(splines)):
            if alongside[owner == number].mean() > 0.5:
                twins.add(number)
    # the vertices: every cut of every spline, joined by the links
    keys = [(number, spot) for number in range(len(splines)) for spot in sorted(cuts[number])]
    index = {key: number for number, key in enumerate(keys)}
    parent = list(range(len(keys)))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in links:
        if a in index and b in index:
            parent[root(index[a])] = root(index[b])
    spot_xy = np.array([[splines[n]["x"][p], splines[n]["y"][p]] for n, p in keys]).reshape(-1, 2)
    # the pieces between cuts, and the cores contracted as contract_cores (section 26) does: two vertices joined by a
    # piece no longer than the larger core (half the widest incident width plus RING_MARGIN) are one, shortest piece
    # first, only while the merged vertex stays within two cores; the piece is then the vertex's core, not an edge
    pieces = []
    for number, spline in enumerate(splines):
        spots = sorted(cuts[number])
        for first, last in zip(spots[:-1], spots[1:]):
            pieces.append((number, first, last))
    width_at = lambda number, spot: 2 * float(splines[number]["radius"][spot]) / CYLINDER
    core = {}
    for number, first, last in pieces:
        for spot in (first, last):
            node = root(index[(number, spot)])
            core[node] = max(core.get(node, 0.0), width_at(number, spot) / 2 + RING_MARGIN)
    box = {node: [*spot_xy[node], *spot_xy[node]] for node in range(len(keys))}
    for node in range(len(keys)):
        top = root(node)
        box[top] = [min(box[top][0], spot_xy[node][0]), min(box[top][1], spot_xy[node][1]),
                    max(box[top][2], spot_xy[node][0]), max(box[top][3], spot_xy[node][1])]
    length = lambda piece: float(np.sum(np.hypot(np.diff(splines[piece[0]]["x"][piece[1]:piece[2] + 1]),
                                                 np.diff(splines[piece[0]]["y"][piece[1]:piece[2] + 1]))))
    inside = set()
    for piece in sorted(pieces, key=length):
        a, b = root(index[(piece[0], piece[1])]), root(index[(piece[0], piece[2])])
        reach = max(core.get(a, RING_MARGIN), core.get(b, RING_MARGIN))
        if length(piece) > reach:
            break
        if a == b:
            if length(piece) <= 2 * join:
                inside.add(piece)  # a piece inside one vertex's core disappears into it
            continue
        merged = [min(box[a][0], box[b][0]), min(box[a][1], box[b][1]), max(box[a][2], box[b][2]), max(box[a][3], box[b][3])]
        if max(merged[2] - merged[0], merged[3] - merged[1]) > 2 * reach:
            continue
        parent[b], box[a] = a, merged
        core[a] = max(core.get(a, 0.0), core.get(b, 0.0))
        inside.add(piece)
    roots = sorted({root(node) for node in range(len(keys))})
    vertex_of = {node: roots.index(root(node)) for node in range(len(keys))}
    vertices = [{"x": 0.0, "y": 0.0, "ends": [], "members": 0} for _ in roots]
    for node, (x, y) in enumerate(spot_xy):
        vertex = vertices[vertex_of[node]]
        vertex["x"] += x
        vertex["y"] += y
        vertex["members"] += 1
    for vertex in vertices:
        vertex["x"] /= max(vertex["members"], 1)
        vertex["y"] /= max(vertex["members"], 1)
    edges = []
    for number, first, last in pieces:
        if (number, first, last) in inside:
            continue
        a, b = vertex_of[index[(number, first)]], vertex_of[index[(number, last)]]
        spline = splines[number]
        edge = {name: np.asarray(spline[name][first:last + 1]) for name in ("x", "y", "radius")}
        edge.update({"spline": number, "span": (first, last), "vertices": (a, b), "twin": number in twins})
        vertices[a]["ends"].append((len(edges), 0))
        vertices[b]["ends"].append((len(edges), 1))
        edges.append(edge)
    used = sorted({v for edge in edges for v in edge["vertices"]})
    parent_of = {v: v for v in used}

    def top(v):
        while parent_of[v] != v:
            v = parent_of[v]
        return v

    for edge in edges:
        parent_of[top(edge["vertices"][0])] = top(edge["vertices"][1])
    components = len({top(v) for v in used})
    return {"vertices": vertices, "edges": edges, "twins": sorted(twins), "cycles": len(edges) - len(used) + components}


# ---------------------------------------------------------------- the scoring of a picture (section 38a)

def scoring_frame(density, valid, quiet, noise=None):
    """What a picture and its render are scored with (section 19 then 20): the data's noise (a map, section 12's
    local noise of the density on the quiet pixels, rebuilt from the saved density since section 29 does not save
    it; or one number, as read_burst takes it on synthetic bursts), the gain that brings the quiet pixels' contrast
    to a spread of 1, and the per-width noise constants of the data's even fields (channel_answers). The render is
    never measured: it has no noise."""
    if noise is None:
        smooth = blur(np.where(valid, density, 0).astype(np.float32), 1)
        noise = local_rms(smooth - np.median(smooth[quiet]), quiet, NOISE_WINDOW)
    contrast = contrast_layer(density, noise, valid)
    gain = float(robust_sd(contrast[quiet]))
    frame = {"noise": noise, "gain": gain, "valid": valid, "quiet": quiet}
    frame["levels"] = channel_levels(contrast / gain, valid, quiet)
    return frame


def to_contrast(picture, frame):
    """A picture (the data or a render) as section 19's contrast, with the data's noise and gain."""
    return contrast_layer(picture, frame["noise"], frame["valid"]) / np.float32(frame["gain"])


def channel_levels(contrast, valid, quiet, widths=HYPER_WIDTHS):
    """The per-width noise constants of section 20: the robust spread of the even field over the quiet pixels (the
    median over orientations), as orientation_hypercolumns measures them."""
    rows, columns = contrast.shape
    levels = []
    for width in widths:
        factor = reduction(width)
        spectrum, shape, inner = padded_spectrum(contrast, factor)
        share = blur(valid.astype(np.float32), ELONGATION * width)
        small = (columns // factor, rows // factor)
        places = np.flatnonzero(cv2.resize((quiet & (share >= VALID_SHARE)).astype(np.uint8), small, interpolation=cv2.INTER_NEAREST))
        sample = places[::max(1, len(places) // NOISE_SAMPLE)]
        levels.append(float(np.median([robust_sd(field_answers(spectrum, shape, inner, width / factor, k)[0].ravel()[sample])
                                       for k in range(0, N_ORIENT, 4)])) * factor ** -0.5)
    return np.array(levels)


def channel_answers(contrast, valid, levels, width_index, threads=None):
    """The N_ORIENT neurons of one width before the widths compete, in noise units of the data's constant for that
    width (no floor). Returns (N_ORIENT, rows, columns) float32."""
    rows, columns = contrast.shape
    width = HYPER_WIDTHS[width_index]
    factor = reduction(width)
    spectrum, shape, inner = padded_spectrum(contrast, factor)
    small = (columns // factor, rows // factor)
    seen = cv2.resize(blur(valid.astype(np.float32), ELONGATION * width), small, interpolation=cv2.INTER_AREA) >= VALID_SHARE
    out = np.zeros((N_ORIENT, rows, columns), dtype=np.float32)

    def one(k):
        even, odd, halves = field_answers(spectrum, shape, inner, width / factor, k)
        answer = np.where(seen, complex_unit(even, odd, halves), 0).astype(np.float32) * np.float32(factor ** -0.5)
        out[k] = cv2.resize(answer, (columns, rows), interpolation=cv2.INTER_LINEAR) if factor > 1 else answer

    with ThreadPoolExecutor(max_workers=threads or 1) as pool:
        list(pool.map(one, range(N_ORIENT)))
    return out / np.float32(levels[width_index])


def channel_winner(contrast, valid, levels, threads=None):
    """The competition of section 20 on the channels (the largest over widths, then over orientations), without the
    floor: strength, orientation and half-width per pixel, as winning_neurons gives them."""
    best = np.zeros((N_ORIENT, *contrast.shape), dtype=np.float32)
    width = np.zeros((N_ORIENT, *contrast.shape), dtype=np.uint8)
    for place in range(len(HYPER_WIDTHS)):
        answers = channel_answers(contrast, valid, levels, place, threads)
        better = answers > best
        best[better], width[better] = answers[better], place
    orientation = np.argmax(best, axis=0).astype(np.uint8)
    strength = np.take_along_axis(best, orientation[None], axis=0)[0]
    half_width = np.asarray(HYPER_WIDTHS, dtype=np.float32)[np.take_along_axis(width, orientation[None], axis=0)[0]]
    return strength, orientation, half_width


def ridges_of(contrast, valid, levels, threads=None):
    """The ridges of a picture's channels after the competition (ridge_crests on the winner, no interactions): what
    the self-consistency number compares."""
    strength, orientation, _ = channel_winner(contrast, valid, levels, threads)
    region = apply_hysteresis_threshold(strength, T_LOW, T_HIGH) & valid
    return ridge_crests(strength, orientation, region), strength


def aligned_window(window, shape, margin=WINDOW_MARGIN, align=WINDOW_ALIGN):
    """A window (top, left, bottom, right) grown to multiples of `align` and its crop with `margin` around it
    (clipped to the frame, still aligned)."""
    rows, columns = shape
    top, left = (window[0] // align) * align, (window[1] // align) * align
    bottom, right = min(-(-window[2] // align) * align, rows), min(-(-window[3] // align) * align, columns)
    # the crop's far edges on multiples of `align` too, unless it reaches the frame's edge: the wide channels are
    # computed on grids reduced 2, 4 and 5 times, which match the frame's only on such a crop (and only at a frame
    # edge whose size is a multiple of `align`, as the real burst's 1200 x 1920 are)
    crop = (max(top - margin, 0) // align * align, max(left - margin, 0) // align * align,
            min(-(-(bottom + margin) // align) * align, rows), min(-(-(right + margin) // align) * align, columns))
    return (top, left, bottom, right), crop


def rescore_window(contrast, valid, levels, width_index, window, threads=None):
    """One width's channels re-scored in a window only: computed on the window with WINDOW_MARGIN around it, aligned
    to WINDOW_ALIGN, and cut back to the window. Returns the window and its (N_ORIENT, rows, columns) channels."""
    inner, crop = aligned_window(window, contrast.shape)
    t, l, b, r = crop
    answers = channel_answers(contrast[t:b, l:r], valid[t:b, l:r], levels, width_index, threads)
    return inner, answers[:, inner[0] - t:inner[2] - t, inner[1] - l:inner[3] - l]


# ---------------------------------------------------------------- darkness and the renderer (section 38)

def blurred_chord(radius, blur_sigma, offsets):
    """The density across a cylinder of `radius` and unit absorption, blurred by `blur_sigma`, at `offsets` from its
    axis (a fine grid convolved with the Gaussian, then read off)."""
    reach = radius + 4 * blur_sigma + 2
    grid = np.arange(-reach, reach + 1e-9, 0.1)
    chord = 2 * np.sqrt(np.maximum(radius ** 2 - grid ** 2, 0))
    kernel = np.exp(-0.5 * (np.arange(-4 * blur_sigma, 4 * blur_sigma + 1e-9, 0.1) / max(blur_sigma, 1e-3)) ** 2)
    smooth = np.convolve(chord, kernel / kernel.sum(), mode="same")
    return np.interp(offsets, grid, smooth, left=0.0, right=0.0)


def sharp_radius(half_width, blur_sigma):
    """A vessel's radius from the half-width its line shows, with the blur taken out (as tube_reach does); at least
    a third of a pixel, where the line is no wider than the blur."""
    return np.maximum(np.sqrt(np.maximum(np.asarray(half_width, dtype=float) ** 2 - blur_sigma ** 2, 0)) / CYLINDER, 1 / 3)


BLUR_TRIES = (0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)  # pixels; the blurs a picture's fit tries
BLUR_WINDOWS = 200       # windows on the picture's thicker vessels its blur is fitted on


def fitted_blur(density, edges, cores, tries=BLUR_TRIES, count=BLUR_WINDOWS):
    """The picture's blur, fitted on the density profiles across its vessels: at each of up to `count` windows on
    vessels whose half-width is at least 2.5 px (where R and the blur are separable), the blur that fits the
    profile best (an offset plus kappa times the blurred chord, R free), and the median over the windows. A sum of
    the windows' residuals is ruled by the few that catch a neighbouring vessel (it chose the largest blur tried on
    synthetic burst 1); the median over windows came out at 1.75 px on bursts 1, 2 and 1001, whose trees' true mean
    blurs are 1.3 to 2.1 px. Section 29's picture_blur (the 5% quantile of the ridges' half-widths) cannot be less
    than the blur plus the thinnest vessel's own width: 2.2 to 2.8 px on the same bursts. Returns the blur and every
    window's best blur."""
    spots = [(e, int(place)) for e, edge in enumerate(edges) if np.median(edge["radius"]) >= 2.5 for place in edge_windows(edge, cores[e])]
    if not spots:
        return float(tries[len(tries) // 2]), np.array([])
    picks = np.random.default_rng(0).choice(len(spots), size=min(count, len(spots)), replace=False)
    image = np.nan_to_num(density).astype(np.float32)
    best = []
    for pick in picks:
        e, i = spots[pick]
        edge = edges[e]
        lo, hi = max(i - 2, 0), min(i + 2, len(edge["x"]) - 1)
        d = np.array([edge["x"][hi] - edge["x"][lo], edge["y"][hi] - edge["y"][lo]], dtype=float)
        d /= max(np.hypot(*d), 1e-9)
        normal = np.array([-d[1], d[0]])
        reach = PROFILE_REACH * float(edge["radius"][i]) + 8
        offsets = np.arange(-reach, reach + 1e-9, PROFILE_STEP)
        values = scipy.ndimage.map_coordinates(image, [edge["y"][i] + offsets * normal[1], edge["x"][i] + offsets * normal[0]],
                                               order=1, mode="nearest")
        errors = []
        for sigma in tries:
            error = np.inf
            for radius in np.geomspace(0.5, max(12.0, 3 * float(edge["radius"][i])), 30):
                design = np.column_stack([np.ones_like(offsets), blurred_chord(radius, sigma, offsets)])
                solution, *_ = np.linalg.lstsq(design, values, rcond=None)
                error = min(error, float(np.sum((values - design @ solution) ** 2)))
            errors.append(error)
        best.append(tries[int(np.argmin(errors))])
    best = np.array(best)
    return float(np.median(best)), best


def edge_windows(edge, cores):
    """The places (point numbers) along an edge where its darkness is read: one every KNOT_SPACING points, none whose
    centre lies within the core radius of either of its end vertices (`cores`: the two radii)."""
    count = len(edge["x"])
    places = np.arange(int(KNOT_SPACING // 2), count, int(KNOT_SPACING))
    from_start = np.hypot(edge["x"][places] - edge["x"][0], edge["y"][places] - edge["y"][0])
    from_end = np.hypot(edge["x"][places] - edge["x"][-1], edge["y"][places] - edge["y"][-1])
    return places[(from_start > cores[0]) & (from_end > cores[1])]


RADIUS_TRIES = 24       # radii tried per window when R is fitted (log-spaced)


def window_kappa(density, edge, place, blur_sigma):
    """R and kappa (density per pixel of chord) at one window (section 38): the density profile across the edge
    fitted with an offset c plus kappa times the cylinder's chord of radius R blurred by the picture's blur, least
    squares in c and kappa for each of RADIUS_TRIES radii, the best kept. Where the line is no wider than the blur
    (the radius from its half-width, R_ridge, is at most the blur) R and kappa are not separable: R stays R_ridge and
    only the line mass pi kappa R^2 means something. Returns kappa, its sigma and R."""
    i = int(place)
    lo, hi = max(i - 2, 0), min(i + 2, len(edge["x"]) - 1)
    d = np.array([edge["x"][hi] - edge["x"][lo], edge["y"][hi] - edge["y"][lo]], dtype=float)
    d /= max(np.hypot(*d), 1e-9)
    normal = np.array([-d[1], d[0]])
    half_width = float(edge["radius"][i])
    ridge = float(sharp_radius(half_width, blur_sigma))
    tries = [ridge] if ridge <= blur_sigma else np.geomspace(max(0.5, 0.4 * ridge), 2.5 * ridge + 2, RADIUS_TRIES)
    reach = max(PROFILE_REACH * half_width, max(tries) + 2) + 2 * blur_sigma
    offsets = np.arange(-reach, reach + 1e-9, PROFILE_STEP)
    px, py = edge["x"][i] + offsets * normal[0], edge["y"][i] + offsets * normal[1]
    values = scipy.ndimage.map_coordinates(np.nan_to_num(density).astype(np.float32), [py, px], order=1, mode="nearest")
    best = None
    for radius in tries:
        design = np.column_stack([np.ones_like(offsets), blurred_chord(radius, blur_sigma, offsets)])
        solution, *_ = np.linalg.lstsq(design, values, rcond=None)
        error = float(np.sum((values - design @ solution) ** 2))
        if best is None or error < best[0]:
            best = (error, solution, design, float(radius))
    error, solution, design, radius = best
    spread = np.sqrt(error / max(len(values) - 3, 1))
    cov = spread ** 2 * np.linalg.pinv(design.T @ design)
    return float(solution[1]), float(np.sqrt(max(cov[1, 1], 0))), radius


def lumen_groups(graph):
    """The lumens of a vessel_graph whose junctions were read (read_junctions): its edges united through every group
    of two or three in a junction's reading and through every two pieces that meet end to end; a crossing's two pairs
    stay two lumens. Returns per edge its lumen number."""
    parent = list(range(len(graph["edges"])))

    def root(edge):
        while parent[edge] != edge:
            parent[edge] = parent[parent[edge]]
            edge = parent[edge]
        return edge

    for node in graph["nodes"]:
        if node.get("gone"):
            continue
        if "arms" in node:
            for group in node.get("partition", []):
                if 2 <= len(group) <= 3:
                    members = [node["arms"][arm]["edge"] for arm in group]
                    for other in members[1:]:
                        parent[root(other)] = root(members[0])
        elif len(node["ends"]) == 2:
            parent[root(node["ends"][0][0])] = root(node["ends"][1][0])
    roots = [root(edge) for edge in range(len(graph["edges"]))]
    names = {value: number for number, value in enumerate(sorted(set(roots)))}
    return np.array([names[value] for value in roots])


def draw_lumen(edges, kappa, blur_sigma, shape, radii=None):
    """One lumen drawn as section 38's renderer draws it: every edge's chord density (2 kappa sqrt(R^2 - d^2), R the
    radius per point, `radii`, as kappa_passes fits it, else the sharp radius from its half-width; kappa per point),
    the edges united by the max, on a canvas around the lumen; the canvas' transmittance blurred by the picture's
    blur, returned as density. Returns (box, density) with box = (top, left, bottom, right)."""
    radii = [sharp_radius(edge["radius"], blur_sigma) for edge in edges] if radii is None else [np.asarray(r, dtype=float) for r in radii]
    rows, columns = shape
    xs = np.concatenate([edge["x"] for edge in edges])
    ys = np.concatenate([edge["y"] for edge in edges])
    reach = max(float(np.max(np.concatenate(radii))), 1) + 4 * blur_sigma + 2
    top, left = int(max(np.floor(ys.min() - reach), 0)), int(max(np.floor(xs.min() - reach), 0))
    bottom, right = int(min(np.ceil(ys.max() + reach) + 1, rows)), int(min(np.ceil(xs.max() + reach) + 1, columns))
    if bottom <= top or right <= left:
        return (0, 0, 0, 0), np.zeros((0, 0), dtype=np.float32)
    canvas = np.zeros((bottom - top, right - left))
    for edge, value, radius_of in zip(edges, kappa, radii):
        points = np.column_stack([edge["x"], edge["y"]]).astype(float)
        radii_here = np.broadcast_to(radius_of, (len(points),))
        value = np.broadcast_to(np.asarray(value, dtype=float), (len(points),))
        bound = float(np.max(radii_here)) + 1.5
        # only the pixels of the edge's own box can lie within `bound` of it: a lumen's box spans its whole tree
        et, el = int(max(np.floor(points[:, 1].min() - bound), top)), int(max(np.floor(points[:, 0].min() - bound), left))
        eb, er = int(min(np.ceil(points[:, 1].max() + bound) + 1, bottom)), int(min(np.ceil(points[:, 0].max() + bound) + 1, right))
        if eb <= et or er <= el:
            continue
        y, x = np.mgrid[et:eb, el:er]
        pixels = np.column_stack([x.ravel(), y.ravel()]).astype(float)
        distance, near = scipy.spatial.cKDTree(points).query(pixels, distance_upper_bound=bound)
        inside = np.flatnonzero(np.isfinite(distance))
        if not len(inside):
            continue
        fine = RIDGE_FINE if np.min(radii_here) < 2 else 1
        offsets = (np.arange(fine) + 0.5) / fine - 0.5
        chord = np.zeros(len(inside))
        for dy, dx in itertools.product(offsets, offsets):
            spot = pixels[inside] + (dx, dy)
            best = np.full(len(inside), np.inf)
            for step in (-1, 0):
                first = np.clip(near[inside] + step, 0, max(len(points) - 2, 0))
                a, b = points[first], points[np.minimum(first + 1, len(points) - 1)]
                ab = b - a
                t = np.clip(((spot - a) * ab).sum(axis=1) / np.maximum((ab ** 2).sum(axis=1), 1e-12), 0, 1)
                best = np.minimum(best, np.hypot(*(spot - (a + t[:, None] * ab)).T))
            r = radii_here[near[inside]]
            chord += 2 * np.sqrt(np.maximum(r ** 2 - best ** 2, 0)) * value[near[inside]]
        flat = canvas.ravel()
        spots = (y.ravel()[inside] - top) * (right - left) + (x.ravel()[inside] - left)
        flat[spots] = np.maximum(flat[spots], chord / fine ** 2)
    density = -np.log(np.maximum(cv2.GaussianBlur(np.exp(-canvas), (0, 0), blur_sigma), 1e-9))
    return (top, left, bottom, right), density.astype(np.float32)


def render_graph(edges, lumens, kappa, blur_sigma, shape, drawn=None, radii=None):
    """The renderer of section 38: every lumen drawn on its own canvas (draw_lumen), densities added across lumens
    (their transmittances multiplied). `drawn` per edge says whether it is drawn. Returns the picture and the
    canvases {lumen: (box, density)}."""
    picture = np.zeros(shape, dtype=np.float32)
    canvases = {}
    drawn = np.ones(len(edges), dtype=bool) if drawn is None else np.asarray(drawn)
    for lumen in sorted(set(lumens.tolist())):
        members = [e for e in np.flatnonzero(lumens == lumen) if drawn[e]]
        if not members:
            continue
        box, density = draw_lumen([edges[e] for e in members], [kappa[e] for e in members], blur_sigma, shape,
                                  None if radii is None else [radii[e] for e in members])
        if density.size:
            picture[box[0]:box[2], box[1]:box[3]] += density
            canvases[lumen] = (box, density)
    return picture, canvases


def kappa_passes(edges, lumens, density, blur_sigma, cores, drawn=None, passes=KAPPA_PASSES, watch=None):
    """The darkness of every edge (section 37's initialisation, step 4): kappa per window outside the cores
    (window_kappa), read on the density with the other lumens' render taken off (exact under the renderer, since
    densities add across lumens), the thickest lumens first, repeated `passes` times (Gauss-Seidel). An edge
    without a window takes its lumen's median kappa (section 38's windowless edges), else one window at its middle.
    Returns kappa and R per edge (one per point, smoothed along the edge), the windows' kappa, sigma and R, and the
    last pass' largest change of kappa in sigmas. `watch` (a vascnet.watch.Recorder) is handed the render before
    the first pass and after every pass."""
    drawn = np.ones(len(edges), dtype=bool) if drawn is None else np.asarray(drawn)
    kappa = [np.full(len(edge["x"]), SYNTH_ABSORB) for edge in edges]
    order = sorted(set(lumens.tolist()), key=lambda lumen: -max(float(np.max(edges[e]["radius"])) for e in np.flatnonzero(lumens == lumen)))
    windows = {}
    change = np.inf
    radii = [sharp_radius(edge["radius"], blur_sigma) for edge in edges]
    picture, canvases = render_graph(edges, lumens, kappa, blur_sigma, density.shape, drawn, radii)
    rest = np.nan_to_num(density).astype(np.float32) - picture  # the density the render does not explain
    if watch is not None:
        watch({"kind": "kappa", "taken": True, "pass": 0, "change": np.nan, "render": picture.copy()})
    for number in range(passes):
        change = 0.0
        for lumen in order:
            members = np.flatnonzero(lumens == lumen)
            if lumen in canvases:  # the lumen's own render back into what it is fitted on
                box, values = canvases[lumen]
                rest[box[0]:box[2], box[1]:box[3]] += values
            for e in members:
                places = edge_windows(edges[e], cores[e])
                found = [window_kappa(rest, edges[e], place, blur_sigma) for place in places]
                if found:
                    values = np.array([value for value, _, _ in found])
                    sigmas = np.array([sigma for _, sigma, _ in found])
                    fitted = np.array([radius for _, _, radius in found])
                    old = windows.get(e)
                    if old is not None:
                        change = max(change, float(np.max(np.abs(values - old[1]) / np.maximum(sigmas, 1e-9))))
                    windows[e] = (places, values, sigmas, fitted)
            for e in members:
                if e in windows:
                    places, values, _, fitted = windows[e]
                    along = np.arange(len(edges[e]["x"]))
                    kappa[e] = np.interp(along, places, scipy.ndimage.median_filter(values, size=3, mode="nearest"))
                    radii[e] = np.interp(along, places, scipy.ndimage.median_filter(fitted, size=3, mode="nearest"))
            known = [np.median(windows[e][1]) for e in members if e in windows]
            for e in members:
                if e not in windows:
                    if known:
                        kappa[e] = np.full(len(edges[e]["x"]), float(np.median(known)))
                    else:
                        value, _, radius = window_kappa(rest, edges[e], len(edges[e]["x"]) // 2, blur_sigma)
                        kappa[e] = np.full(len(edges[e]["x"]), max(value, 0.0))
                        radii[e] = np.full(len(edges[e]["x"]), radius)
            for e in members:
                kappa[e] = np.maximum(kappa[e], 0.0)
            shown = [e for e in members if drawn[e]]
            canvases.pop(lumen, None)
            if shown:  # the lumen drawn again with its new darkness, and taken out of what the next ones are fitted on
                box, values = draw_lumen([edges[e] for e in shown], [kappa[e] for e in shown], blur_sigma, density.shape,
                                         [radii[e] for e in shown])
                if values.size:
                    rest[box[0]:box[2], box[1]:box[3]] -= values
                    canvases[lumen] = (box, values)
        if watch is not None:
            watch({"kind": "kappa", "taken": True, "pass": number + 1, "change": change,
                   "render": (np.nan_to_num(density).astype(np.float32) - rest).copy(), "last": number == passes - 1})
    return kappa, radii, windows, change


def tube_mask(edges, blur_sigma, shape, drawn=None):
    """The tubes of the drawn edges: within the radius plus TUBE_BLURS sigmas of the blur of their centerlines."""
    mask = np.zeros(shape, dtype=np.uint8)
    drawn = np.ones(len(edges), dtype=bool) if drawn is None else drawn
    for edge, shown in zip(edges, drawn):
        if not shown:
            continue
        reach = tube_reach(np.asarray(edge["radius"], dtype=np.float32), blur_sigma)
        for x, y, r in zip(edge["x"][::2], edge["y"][::2], reach[::2]):
            cv2.circle(mask, (int(round(x)), int(round(y))), int(np.ceil(r)), 1, -1)
    return mask > 0


def self_consistency(render_ridges, data_ridges, valid, region=None):
    """section 38's self-consistency number: the render's ridges found again in the data's within REPEAT_WITHIN, and
    the data's in the render's (found_again; `repeated` for the share above chance), over `region` (the whole frame,
    or the tubes). Returns {"render in data": (found, chance, above), "data in render": (...)}."""
    region = valid if region is None else region & valid
    same = np.eye(3)
    return {"render in data": repeated(render_ridges & region, valid, data_ridges, valid, same),
            "data in render": repeated(data_ridges & region, valid, render_ridges, valid, same)}


def zero_points(data_contrast, render_contrast, valid, levels, edges, drawn=None, threads=None):
    """The zero point of the topology residual per channel (section 38a): on the model's own centerline points, the
    median of data minus render in the channel of the point's orientation and width (the nearest of the 16 and the
    8). On a right line the data's rectified answer carries noise with a positive expectation and the render's carries
    none. Returns (N_ORIENT, widths) with NaN where a channel has fewer than 20 points, and the counts."""
    drawn = np.ones(len(edges), dtype=bool) if drawn is None else drawn
    xs = np.concatenate([edge["x"] for edge, shown in zip(edges, drawn) if shown])
    ys = np.concatenate([edge["y"] for edge, shown in zip(edges, drawn) if shown])
    theta = np.concatenate([np.arctan2(np.gradient(edge["y"]), np.gradient(edge["x"])) for edge, shown in zip(edges, drawn) if shown])
    radius = np.concatenate([edge["radius"] for edge, shown in zip(edges, drawn) if shown])
    k = np.rint((theta % np.pi) / (np.pi / N_ORIENT)).astype(int) % N_ORIENT
    w = np.argmin(np.abs(np.log(np.asarray(radius, dtype=float)[:, None] / np.asarray(HYPER_WIDTHS)[None])), axis=1)
    rows, columns = valid.shape
    row, column = np.clip(np.rint(ys).astype(int), 0, rows - 1), np.clip(np.rint(xs).astype(int), 0, columns - 1)
    zero = np.full((N_ORIENT, len(HYPER_WIDTHS)), np.nan)
    counts = np.zeros((N_ORIENT, len(HYPER_WIDTHS)), dtype=int)
    for place in range(len(HYPER_WIDTHS)):
        mine = w == place
        if not mine.any():
            continue
        data = channel_answers(data_contrast, valid, levels, place, threads)
        render = channel_answers(render_contrast, valid, levels, place, threads)
        for orientation in range(N_ORIENT):
            spots = mine & (k == orientation)
            counts[orientation, place] = spots.sum()
            if spots.sum() >= 20:
                zero[orientation, place] = float(np.median(data[orientation, row[spots], column[spots]]
                                                           - render[orientation, row[spots], column[spots]]))
    return zero, counts


def topology_residual(data_contrast, render_contrast, valid, levels, zero, threads=None):
    """The topology residual (section 38a): per channel data minus render minus the channel's zero point (0 where
    none was measured), and its strongest positive and negative channel at every pixel with their orientations:
    a positive ridge is a line the data has and the render lacks, a negative one a line the render has and the data
    lacks. Returns (positive, positive orientation, negative, negative orientation)."""
    shape = data_contrast.shape
    positive, negative = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    pos_k, neg_k = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    for place in range(len(HYPER_WIDTHS)):
        difference = (channel_answers(data_contrast, valid, levels, place, threads)
                      - channel_answers(render_contrast, valid, levels, place, threads))
        difference -= np.nan_to_num(zero[:, place], nan=0.0)[:, None, None].astype(np.float32)
        top, bottom = difference.max(axis=0), (-difference).max(axis=0)
        better = top > positive
        positive[better], pos_k[better] = top[better], np.argmax(difference, axis=0)[better]
        worse = bottom > negative
        negative[worse], neg_k[worse] = bottom[worse], np.argmax(-difference, axis=0)[worse]
    return positive, pos_k, negative, neg_k


def initial_model(front, density, valid, threads=None, watch=None):
    """Section 37's initialisation, up to the first network: the pieces (cyclic_pieces), their graph with the
    junctions read from the geometry (vessel_graph, read_junctions), the lumens (lumen_groups), the picture's blur
    (fitted_blur), and the darkness of every edge (kappa_passes), then the render. A twin is not drawn: it and the
    piece it runs beside are one line mass (twins_and_stubs)."""
    pieces, joined = cyclic_pieces(front, valid)
    graph = vessel_graph(pieces, valid, density)
    read_junctions(graph, density, front["scores"], flow=False)
    edges, nodes = graph["edges"], graph["nodes"]
    for edge in edges:
        edge["length"] = chain_length(edge)
        edge["loose"] = tuple(len(nodes[node]["ends"]) == 1 for node in edge["nodes"])
    core = lambda node: nodes[node]["ring"][0] if "ring" in nodes[node] else 0.0
    cores = [(core(edge["nodes"][0]), core(edge["nodes"][1])) for edge in edges]
    blur_sigma, window_blurs = fitted_blur(density, edges, cores)
    kinds = twins_and_stubs(edges, blur_sigma, valid)
    drawn = np.array([kind != "twin" for kind in kinds])
    lumens = lumen_groups(graph)
    kappa, radii, windows, change = kappa_passes(edges, lumens, density, blur_sigma, cores, drawn, watch=watch)
    render, canvases = render_graph(edges, lumens, kappa, blur_sigma, density.shape, drawn, radii)
    return {"pieces": pieces, "joined": joined, "graph": graph, "edges": edges, "cores": cores, "blur": blur_sigma,
            "window blurs": window_blurs, "kinds": kinds, "drawn": drawn, "lumens": lumens, "kappa": kappa, "radii": radii,
            "windows": windows, "change": change, "render": render, "canvases": canvases}


def kappa_frame(state, density, box, size=(8.0, 4.2), dpi=80):
    """A frame of the darkness fit's film: the density and the render over one box, side by side, with the pass and
    the largest change of kappa in sigmas."""
    import matplotlib
    matplotlib.use("Agg")
    top, left, bottom, right = box
    high = float(np.nanpercentile(density[top:bottom, left:right], 99.5))
    figure, axes = plt.subplots(1, 2, figsize=size, dpi=dpi)
    for axis, picture, title in ((axes[0], density, "density"), (axes[1], state["render"], "render")):
        axis.imshow(np.nan_to_num(picture[top:bottom, left:right]), cmap="gray_r", vmin=0, vmax=high)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    change = state.get("change", np.nan)
    figure.suptitle(f"darkness fit: pass {state['pass']}" + (f", largest change of kappa {change:.2f} sigma" if np.isfinite(change) else
                                                               " (kappa at its starting value)"), fontsize=9)
    frame = vascnet.watch.frame_of(figure)
    plt.close(figure)
    return frame

# %%
# acceptance step 5 on one synthetic burst first: the initial model rendered and scored, its self-consistency against
# the data's ridges (whole frame and within the tubes), the data's own repeatability (the second fixation), the
# generator's own truth through the same scoring, the zero points, and spline_to_graph against vessel_graph
burst = synthetic_burst_35(HARNESS_SEEDS[0], second=True)
front = ridge_front(burst, workers)
model = initial_model(front, burst["mean"], burst["valid"], workers)
frame = scoring_frame(burst["mean"], burst["valid"], burst["quiet"], noise=robust_sd(burst["mean"][burst["quiet"]]))
data_contrast, render_contrast = to_contrast(burst["mean"], frame), to_contrast(model["render"], frame)
data_ridges, _ = ridges_of(data_contrast, burst["valid"], frame["levels"], workers)
render_ridges, _ = ridges_of(render_contrast, burst["valid"], frame["levels"], workers)
tubes = tube_mask(model["edges"], model["blur"], burst["valid"].shape, model["drawn"])
whole = self_consistency(render_ridges, data_ridges, burst["valid"])
inside = self_consistency(render_ridges, data_ridges, burst["valid"], tubes)
second = burst["second"]
second_frame = scoring_frame(second["mean"], second["valid"], second["quiet"], noise=robust_sd(second["mean"][second["quiet"]]))
second_ridges, _ = ridges_of(to_contrast(second["mean"], second_frame), second["valid"], second_frame["levels"], workers)
own = repeated(data_ridges, burst["valid"], second_ridges, second["valid"], second["to second"])
truth_ridges, _ = ridges_of(to_contrast(burst["layers"].sum(axis=0), frame), burst["valid"], frame["levels"], workers)
truth = self_consistency(truth_ridges, data_ridges, burst["valid"])
zero, zero_counts = zero_points(data_contrast, render_contrast, burst["valid"], frame["levels"], model["edges"], model["drawn"], workers)
focus = np.linspace(SYNTH_FOCUS[0], SYNTH_FOCUS[1], SYNTH_FRAMES)
true_blurs = sorted({round(float(np.mean(SYNTH_SHARP + SYNTH_DEFOCUS * np.abs(s["depth"] - focus))), 2) for s in burst["segments"]})
shares = lambda reading: " | ".join(f"{name}: found {found * 100:.1f}%, by chance {chance * 100:.1f}%, above chance {above * 100:.1f}%"
                                    for name, (found, chance, above) in reading.items())
print(f"Synthetic burst {HARNESS_SEEDS[0]}: {len(model['edges'])} edges ({model['drawn'].sum()} drawn, the rest twins) in "
      f"{model['lumens'].max() + 1} lumens | blur fitted on the profiles {model['blur']:.2f} px (section 29's picture_blur "
      f"{picture_blur(front['half_width'], model['joined']['ridges']):.2f}; the trees' true mean blurs {true_blurs})")
print(f"   darkness: kappa median {np.median([np.median(k) for k in model['kappa']]):.4f} (the generator's {SYNTH_ABSORB}), "
      f"{sum(len(w[0]) for w in model['windows'].values())} windows; the last pass changed kappa by {model['change']:.2f} sigma at the most")
print(f"   self-consistency, whole frame: {shares(whole)}")
print(f"   self-consistency, within the tubes: {shares(inside)}")
print(f"   the data's own repeatability (against the second fixation): found {own[0] * 100:.1f}%, by chance {own[1] * 100:.1f}%, "
      f"above chance {own[2] * 100:.1f}%")
print(f"   the generator's own noiseless truth, scored the same way: {shares(truth)}")
print("   zero point per width (median over the orientations measured; points): " + ", ".join(
    f"{HYPER_WIDTHS[w]}: {np.nanmedian(zero[:, w]):+.2f} ({zero_counts[:, w].sum()})" for w in range(len(HYPER_WIDTHS)) if np.isfinite(zero[:, w]).any()))
# spline_to_graph against vessel_graph: the junctions of vessel_graph found again among spline_to_graph's vertices
view = spline_to_graph(model["pieces"])
junctions = [(node["x"], node["y"]) for node in model["graph"]["nodes"] if not node.get("gone") and len(node.get("arms", [])) >= 3]
vertices = np.array([(v["x"], v["y"]) for v in view["vertices"] if len(v["ends"]) >= 3]).reshape(-1, 2)
found = [bool(len(vertices)) and np.min(np.hypot(*(vertices - place).T)) <= 5 for place in junctions]
print(f"   spline_to_graph: {len(view['vertices'])} vertices, {len(view['edges'])} edges, {view['cycles']} cycles, {len(view['twins'])} twins | of "
      f"vessel_graph's {len(junctions)} junctions, {sum(found)} have a vertex of three or more arms within 5 px")
os.makedirs("vessel_pictures/render", exist_ok=True)
figure, axes = plt.subplots(1, 3, figsize=(18, 4.6))
high = np.percentile(burst["mean"], 99.8)
axes[0].imshow(burst["mean"], cmap="gray_r", vmin=0, vmax=high)
axes[0].set_title("the mean density", fontsize=10)
axes[1].imshow(model["render"], cmap="gray_r", vmin=0, vmax=high)
axes[1].set_title(f"the initial model rendered (blur {model['blur']:.2f} px)", fontsize=10)
overlay = np.dstack([data_ridges & ~render_ridges, render_ridges & data_ridges, render_ridges & ~data_ridges]).astype(np.uint8) * 255
axes[2].imshow(overlay)
axes[2].set_title("ridges: green both, red the data's only, blue the render's only", fontsize=10)
for axis in axes:
    axis.axis("off")
figure.tight_layout()
figure.savefig("vessel_pictures/render/one_burst.png", dpi=80)
plt.show()
del burst, front, second, data_contrast, render_contrast

# %%
# acceptance step 5 on the real burst: the initial model of the anchor's fixation rendered and scored with the data's
# own constants; the self-consistency number, whole frame and within the tubes, against the data's own repeatability
# (the anchor's ridges against the other fixation's); the zero point per channel; the cost of a full re-score and of a
# windowed one, and whether the windowed one equals the full one. The darkness fit is filmed.
import time


def saved_quiet(front, valid, away=8):
    """The quiet pixels of a fixation from section 29's saved maps: where there is data and no neuron of the static
    score answers above T_LOW within `away` px. (Section 12's quiet pixels, judged on the frames, are not saved.)"""
    lines = (front["strength"] > T_LOW).astype(np.uint8)
    return valid & (cv2.distanceTransform(1 - lines, cv2.DIST_L2, 5) > away)


real_fronts = {number: saved_front(saved, number) for number in range(len(saved["times"]))}
real_frames, real_ridges = {}, {}
for number, front in real_fronts.items():
    valid = saved["valid"][number]
    real_frames[number] = scoring_frame(saved["density"][number], valid, saved_quiet(front, valid))
    real_ridges[number], _ = ridges_of(to_contrast(saved["density"][number], real_frames[number]), valid, real_frames[number]["levels"], workers)
others = [number for number in real_fronts if number != anchor]
own = repeated(real_ridges[anchor], saved["valid"][anchor], real_ridges[others[0]], saved["valid"][others[0]], saved["to anchor"][others[0]])
density, valid, frame = saved["density"][anchor], saved["valid"][anchor], real_frames[anchor]
kappa_recorder = vascnet.watch.Recorder(every=1)
started = time.time()
real_model = initial_model(real_fronts[anchor], density, valid, workers, watch=kappa_recorder)
model_time = time.time() - started
render_contrast = to_contrast(real_model["render"], frame)
started = time.time()
render_ridges, _ = ridges_of(render_contrast, valid, frame["levels"], workers)
full_time = time.time() - started
tubes = tube_mask(real_model["edges"], real_model["blur"], valid.shape, real_model["drawn"])
whole = self_consistency(render_ridges, real_ridges[anchor], valid)
inside = self_consistency(render_ridges, real_ridges[anchor], valid, tubes)
data_contrast = to_contrast(density, frame)
zero, zero_counts = zero_points(data_contrast, render_contrast, valid, frame["levels"], real_model["edges"], real_model["drawn"], workers)
# the windowed re-score: a 320 px window in the middle of the frame, all eight widths, against the full frame's channels
rows, columns = valid.shape
middle = (rows // 2 - 160, columns // 2 - 160, rows // 2 + 160, columns // 2 + 160)
largest, window_time = 0.0, 0.0
for place in range(len(HYPER_WIDTHS)):
    full = channel_answers(render_contrast, valid, frame["levels"], place, workers)
    started = time.time()
    inner, part = rescore_window(render_contrast, valid, frame["levels"], place, middle, workers)
    window_time += time.time() - started
    largest = max(largest, float(np.abs(full[:, inner[0]:inner[2], inner[1]:inner[3]] - part).max()))
    del full, part
view = spline_to_graph(real_model["pieces"])
degree = np.array([len(vertex["ends"]) for vertex in view["vertices"]])
places = np.array([(vertex["x"], vertex["y"]) for vertex in view["vertices"]]).reshape(-1, 2)
close = sum(1 for a, b in scipy.spatial.cKDTree(places).query_pairs(2 * RING_MARGIN)) if len(places) else 0
short = sum(chain_length(edge) < 2 for edge in view["edges"])

shares = lambda reading: " | ".join(f"{name}: found {found * 100:.1f}%, by chance {chance * 100:.1f}%, above chance {above * 100:.1f}%"
                                    for name, (found, chance, above) in reading.items())
print(f"Real burst, fixation {anchor}: the initial model in {model_time / 60:.1f} min: {len(real_model['edges'])} edges "
      f"({real_model['drawn'].sum()} drawn, {(~real_model['drawn']).sum()} twins not drawn) in {real_model['lumens'].max() + 1} lumens | "
      f"blur fitted on the profiles {real_model['blur']:.2f} px (section 29's picture_blur 1.50) | kappa median "
      f"{np.median([np.median(k) for k in real_model['kappa']]):.4f}; the last pass changed kappa by {real_model['change']:.2f} sigma at the most")
print(f"   spline_to_graph of the pieces: {len(view['vertices'])} vertices ({(degree == 1).sum()} ends, {(degree == 2).sum()} of two arms, "
      f"{(degree == 3).sum()} of three, {(degree >= 4).sum()} of four or more), {len(view['edges'])} edges, {view['cycles']} cycles, "
      f"{len(view['twins'])} twins | clean: {close} pairs of vertices closer than {2 * RING_MARGIN:.0f} px, {short} edges under 2 px")
print(f"   the self-consistency number, whole frame: {shares(whole)}")
print(f"   within the tubes: {shares(inside)}")
print(f"   the data's own repeatability (fixation {anchor}'s ridges against fixation {others[0]}'s): found {own[0] * 100:.1f}%, by chance "
      f"{own[1] * 100:.1f}%, above chance {own[2] * 100:.1f}%")
print("   zero point per width (median over the orientations measured; points): " + ", ".join(
    f"{HYPER_WIDTHS[w]}: {np.nanmedian(zero[:, w]):+.2f} ({zero_counts[:, w].sum()})" for w in range(len(HYPER_WIDTHS)) if np.isfinite(zero[:, w]).any()))
print(f"   cost: a full re-score (all 16 x 8 channels and the ridges, {columns} x {rows}) {full_time:.1f} s; a windowed one (320 px, "
      f"{WINDOW_MARGIN} px margin, all widths) {window_time:.1f} s; the windowed channels differ from the full ones inside the window by "
      f"{largest:.4f} at the most")
reached = all(inside[name][2] >= own[2] for name in inside)
print(f"\nThe gate: within the tubes the agreement reaches the data's own repeatability ({'yes' if reached else 'no'}: "
      + ", ".join(f"{name} {inside[name][2] * 100:.1f}%" for name in inside) + f" against {own[2] * 100:.1f}% above chance); the windowed "
      f"re-score equals the full one within 0.005 ({'yes' if largest <= 0.005 else 'no'}: {largest:.4f}); a move's evaluation (one windowed "
      f"re-score) under a few seconds ({'yes' if window_time <= 5 else 'no'}: {window_time:.1f} s) -> "
      f"{'PASSED' if reached and largest <= 0.005 and window_time <= 5 else 'FAILED'}")
os.makedirs("vessel_pictures/fit", exist_ok=True)
box = (max(rows // 2 - 200, 0), max(columns // 2 - 300, 0), min(rows // 2 + 200, rows), min(columns // 2 + 300, columns))
vascnet.watch.animate(kappa_recorder.states, lambda state: kappa_frame(state, density, box), "vessel_pictures/fit/kappa.gif", fps=1.0)
os.makedirs("vessel_pictures/render", exist_ok=True)
Image.fromarray(np.uint8(np.clip(real_model["render"] / max(DENSITY_RANGE[1], 1e-6), 0, 1) * 255)).save("vessel_pictures/render/real_render.png")
print(f"   the darkness fit is filmed in vessel_pictures/fit/kappa.gif ({len(kappa_recorder.states)} frames); the render is in vessel_pictures/render")
memory_note("section 38, real burst")

# %% [markdown]
# ## 38(b). The core's matched filter (acceptance step 6)
#
# At a four-arm vertex, geometry reads a planar four-way as a crossing, and the flux test passes both readings (section 40). The core's matched filter is the one test that can tell two vessels crossing from one lumen of four arms (`core_z`):
#
# - **The template** (`core_template`) is the difference between the two compositions at the core: the arms drawn as two lumens (densities add) minus the arms drawn as one (the max). The arms' κ and R come from section 38's window fit outside the core (`fitted_arms`).
# - **The residual** is the density minus the one-lumen composition.
# - **z** is their inner product, whitened by the noise's power spectrum, over the template's whitened norm. Under one lumen z has mean 0 and spread 1; under two vessels its mean is the template's whitened norm.
# - **The spectrum** (`noise_spectrum`) is measured on noise fields made as the generator makes a mean frame's noise: the tissue's texture and the kept frames' white noise. The texture is not white, and the core z rides on both.
#
# **The table** covers radii of 1 to 5 px (`CORE_PAIRS`) and blurs of 0.8, 1.5, 2 and 3.5 px (`CORE_BLURS`). It uses two vessels at right angles, rendered as two lumens and as one, each with 12 noise realisations. Each cell gives the median z under each truth, the z the template expects, and `told_apart` between the two.
#
# **The gate** (Part III, step 6): decisive (z ≥ 3) for R ≥ 3 at the real burst's blur (1.5 px). Capillaries are reported as they come out.
#
# **Result:** `core_rows` (`vessel_pictures/core/z_table.csv`).

# %%
CORE_PAIRS = ((1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (3.0, 3.0), (5.0, 3.0), (5.0, 5.0))  # radii of the two vessels at a core
CORE_BLURS = (0.8, 1.5, 2.0, 3.5)   # pixels; the generator's range, with the real burst's blur among them
CORE_SEEDS = tuple(range(1, 13))    # noise realisations per radius, blur and composition
CORE_ANGLE = 90.0                    # degrees between the two vessels
NOISE_SAMPLES = 24                   # noise fields the spectrum is measured on


def noise_spectrum(shape, samples=NOISE_SAMPLES, seed=1000):
    """The noise's power spectrum (section 38b), measured on noise fields made as the generator makes a mean frame's
    noise: the tissue's texture (SYNTH_TEXTURE, grain SYNTH_GRAIN) and the white noise of the kept frames
    (CHECK_NOISE). Measured, not modelled: the texture is not white, and the core z rides on both. Returns the mean
    of |FFT|^2 over the samples (the variance of every frequency of a field of this shape)."""
    rng = np.random.default_rng(seed)
    total = np.zeros(shape)
    for _ in range(samples):
        texture = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), SYNTH_GRAIN)
        field = texture * SYNTH_TEXTURE / texture.std() + rng.normal(0, CHECK_NOISE, shape)
        total += np.abs(np.fft.fft2(field - field.mean())) ** 2
    return total / samples


def core_template(arms, blur_sigma, shape):
    """The two compositions of a core rendered from its arms' kappa and R (section 38b): the arms as two lumens
    (two vessels crossing: densities add) and as one (one lumen: the max). `arms` is a list of two edges (x, y,
    radius as half-width) with their kappa and R. Returns (two vessels, one lumen) as densities of `shape`."""
    two = np.zeros(shape, dtype=np.float32)
    for edge, kappa, radius in arms:
        box, density = draw_lumen([edge], [kappa], blur_sigma, shape, radii=[np.full(len(edge["x"]), radius)])
        two[box[0]:box[2], box[1]:box[3]] += density
    one = np.zeros(shape, dtype=np.float32)
    box, density = draw_lumen([edge for edge, _, _ in arms], [kappa for _, kappa, _ in arms], blur_sigma, shape,
                              radii=[np.full(len(edge["x"]), radius) for edge, _, radius in arms])
    one[box[0]:box[2], box[1]:box[3]] += density
    return two, one


def core_z(density, arms, blur_sigma, spectrum, region=None):
    """The core's matched filter (section 38b): the template is the difference between the two compositions
    (core_template), the residual the density minus the one-lumen composition (less its median); z is their inner
    product whitened by the noise's spectrum, over the template's whitened norm. Under one lumen z has mean 0 and
    spread 1; under two vessels its mean is the template's whitened norm. `region` (a mask) limits both to the core.
    Returns z and the template's whitened norm (the z two vessels are expected to give)."""
    two, one = core_template(arms, blur_sigma, density.shape)
    template = (two - one).astype(float)
    residual = np.nan_to_num(density).astype(float) - one
    residual -= np.median(residual)
    if region is not None:
        template, residual = template * region, residual * region
    t, r = np.fft.fft2(template - template.mean()), np.fft.fft2(residual - residual.mean())
    # no weight where the noise has no measured power: the mean (taken off every noise field) and anything below a
    # thousandth of the median power
    floor = 1e-3 * np.median(spectrum)
    weight = np.where(spectrum > floor, 1 / np.maximum(spectrum, floor), 0.0)
    weight[0, 0] = 0.0
    norm = float(np.sqrt(np.sum(np.abs(t) ** 2 * weight)))
    return float(np.real(np.sum(r * np.conj(t) * weight)) / max(norm, 1e-12)), norm


def fitted_arms(density, lines, blur_sigma, core):
    """The arms' kappa and R as section 38 reads them: the median of window_kappa over windows more than `core` px
    from the junction. `lines` are (points, half-width) per vessel through the junction."""
    arms = []
    for points, half_width in lines:
        edge = {"x": points[:, 0], "y": points[:, 1], "radius": np.full(len(points), half_width)}
        centre = points[len(points) // 2]
        places = [p for p in range(6, len(points) - 6, 6) if np.hypot(*(points[p] - centre)) > core]
        found = [window_kappa(density, edge, p, blur_sigma) for p in places]
        arms.append((edge, float(np.median([k for k, _, _ in found])), float(np.median([r for _, _, r in found]))))
    return arms


def core_configuration(radii, angle=CORE_ANGLE):
    """Two straight vessels of the given radii through the centre of a CHECK_SHAPE picture, `angle` degrees apart:
    as lines (points, radius), each 150 px long."""
    rows, columns = CHECK_SHAPE
    c = np.array([(columns - 1) / 2, (rows - 1) / 2])
    away = lambda degrees, length: c + length * np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
    return [(line_points(away(180, 75), away(0, 75)), radii[0]), (line_points(away(180 + angle, 75), away(angle, 75)), radii[1])]

# %%
# acceptance step 6: the core z by radius and blur (section 38b). Two vessels crossing at right angles, rendered as two
# lumens (densities add) and as one lumen (the max: an anastomosis), CORE_SEEDS noise realisations each; the arms'
# kappa and R fitted outside the core (fitted_arms), the template and z (core_z) with the noise's measured spectrum.
spectrum = noise_spectrum(CHECK_SHAPE)
core_rows = []
for pair in CORE_PAIRS:
    lines = core_configuration(pair)
    for defocus in CORE_BLURS:
        for composition in ("two vessels", "one lumen"):
            lumens = [[line] for line in lines] if composition == "two vessels" else [lines]
            for seed in CORE_SEEDS:
                picture = render_check(lumens, defocus, seed)
                core = max(pair) + 2 * defocus + RING_MARGIN
                measured = []
                for points, radius in lines:
                    rows_, columns_ = np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)
                    far = np.hypot(points[:, 0] - points[len(points) // 2, 0], points[:, 1] - points[len(points) // 2, 1]) > core + 4
                    measured.append((points, float(np.median(picture["half_width"][rows_[far], columns_[far]]))))
                arms = fitted_arms(picture["density"], measured, defocus, core)
                z, expected = core_z(picture["density"], arms, defocus, spectrum)
                core_rows.append({"radii": f"{pair[0]:g} and {pair[1]:g}", "smaller": min(pair), "blur": defocus,
                                  "composition": composition, "seed": seed, "z": z, "expected": expected,
                                  "kappa": np.mean([kappa for _, kappa, _ in arms]), "R": np.mean([radius for _, _, radius in arms])})

print(f"The core's matched filter: z for two vessels against one lumen, at a right-angle core, {len(CORE_SEEDS)} realisations each "
      f"(median z under each truth, the z the template expects for two vessels, and told_apart between the two)")
print(f"   {'radii':>10s} " + "".join(f"{f'blur {blur} px':>34s}" for blur in CORE_BLURS))
for pair in CORE_PAIRS:
    name = f"{pair[0]:g} and {pair[1]:g}"
    cells = []
    for defocus in CORE_BLURS:
        rows = [row for row in core_rows if row["radii"] == name and row["blur"] == defocus]
        two = [row["z"] for row in rows if row["composition"] == "two vessels"]
        one = [row["z"] for row in rows if row["composition"] == "one lumen"]
        cells.append(f"{np.median(two):6.1f} / {np.median(one):5.1f} ({np.median([r['expected'] for r in rows]):5.1f}) {told_apart(two, one):4.2f}")
    print(f"   {name:>10s} " + "".join(f"{cell:>34s}" for cell in cells))
print("   (each cell: median z with two vessels / with one lumen (the expected z) told_apart)")
gate_blur = min(CORE_BLURS, key=lambda blur: abs(blur - 1.5))
thick = [row for row in core_rows if row["smaller"] >= 3 and row["blur"] == gate_blur]
decisive = {name: float(np.median([row["z"] for row in thick if row["radii"] == name and row["composition"] == "two vessels"]))
            for name in sorted({row["radii"] for row in thick})}
print(f"\nThe gate: decisive (z >= 3) for R >= 3 at the real burst's blur ({gate_blur} px; section 29's picture_blur 1.50): "
      + ", ".join(f"R {name}: median z {value:.1f}" for name, value in decisive.items())
      + f" -> {'PASSED' if all(value >= 3 for value in decisive.values()) else 'FAILED'} | capillaries as they come out, in the table")
os.makedirs("vessel_pictures/core", exist_ok=True)
save_table(core_rows, "vessel_pictures/core/z_table.csv")

# %% [markdown]
# ## 39. The recursion (acceptance step 7)
#
# The model is section 38's initial model as a set of splines: section 37's pieces, each with its darkness and radius per point, its lumen, and whether it is drawn (`piece_state`). Every structural change is a move, taken only when J falls:
#
# **J = χ²_topo + λ Σ k ln n** (`objective`):
#
# - **χ²_topo** (`chi_topo`) runs over every channel (16 orientations × 8 widths, before the competition). In each it takes the data's answer minus the render's, less the channel's zero point where the render answers above `T_LOW`. The square is weighted per width by 1 / (`CHANNEL_AREA` × width²), the share of an independent sample a channel-pixel is.
# - **The price** charges each drawn spline its knots: x and y at the knot spacing, the half-width `RADIUS_KNOTS` times farther apart. n is the frame's independent samples.
# - **A spline that is not drawn pays nothing.**
#
# **The moves of a round** come from the topology residual of the model as it stands. There are at most `CANDIDATES_MOST`, the strongest evidence first:
#
# | move | evidence | plan |
# |---|---|---|
# | remove a spline from the rendering (it stays in the model, not drawn) | its negative residual above `T_HIGH` on half its points; or the render shows no line along it (its darkness read as about 0), where only the price speaks | step 3b |
# | add a spline | a positive ridge of the residual off the drawn splines (outside their tubes), cut as section 37 cuts the data's ridges; an end within its tube's reach of a spline's body is joined to it (a branch, a lumen of its own) | steps 3, 5 |
# | extend an end | the same, where the new line's end lies at a drawn spline's end: it joins that lumen | steps 1, 3 |
# | one lumen ↔ two vessels | the core filter (`core_z`) at a vertex of four arms contradicts the composition by more than `Z_PASS` | step 4 |
# | bridge two free ends | reported with its line-mass similarity and whether a positive ridge continues across the gap; **never drawn** | step 1 |
#
# **How a round runs.** Every candidate is applied to a copy of the model, with the touched lumens drawn again. J is then evaluated over the whole frame, which on frames of this size is what a window with its margin covers (step 5). The candidates are taken in order of their gain. Once a move has been taken, each later candidate is evaluated again on the model as it then stands, and is taken only if J still falls. The rounds end when a round takes nothing, or after `ROUNDS_MOST` = 5.
#
# **Not built:**
#
# - step 2 (re-reading the partitions to merge arms through a junction);
# - step 3's centerline shifts;
# - step 3b's vertex merges and splits;
# - step 5's linking of a crossing vessel's two tails through a spline;
# - step 6, the flux network's closing round.
#
# Their places in the plan stay as written.
#
# **The numbers** (Part III, step 7) come from three runs:
#
# - one synthetic burst made to need each move;
# - the held-out trees: found-and-typed-right per round, and the edge confidence's `told_apart` after the rounds;
# - a central crop of the real burst, where λ is swept.
#
# **The gate:**
#
# - J falls at every taken move;
# - found-and-typed-right does not fall from one round to the next;
# - the first principle's number (`told_apart` ≥ 0.75) holds after the rounds.
#
# **Films:** `vessel_pictures/fit/round_<r>.gif`.

# %%
ROUNDS_MOST = 5            # rounds of the recursion at the most (section 39)
CHANNEL_AREA = 2.5         # width^2; a rectified channel's correlation area (section 38a, measured within 10 to 40%): a
                           # channel-pixel counts as 1 / (CHANNEL_AREA width^2) of an independent sample in chi^2_topo
EVIDENCE_LEAST = 10        # pixels; the shortest positive ridge of the topology residual proposed as a new spline
REMOVE_SHARE = 0.5         # a drawn spline is proposed for removal when its negative residual is above T_HIGH along this share
CANDIDATES_MOST = 30       # candidates evaluated per round at the most: a third removals, a sixth compositions, the rest additions
BRIDGE_WIDTHS = 10         # BRIDGE_REACH: two free ends are a bridge's candidates within this many times the larger width
CORE_REGION = 3            # the core filter of a move reads the density within this many core radii of the vertex
LAMBDA_TRIES = (0.25, 1.0, 4.0, 16.0)  # the multipliers of the price swept on the real burst
WATCH_EVERY = 5            # the recorder keeps one candidate in this many, every taken one and the last


# ---------------------------------------------------------------- the objective

def recursion_scene(density, valid, frame, threads=None):
    """What every evaluation of J compares a render with (section 39): the data's contrast and its channels at every
    width before the competition (channel_answers with the data's constants), each width's weight (the share of an
    independent sample a channel-pixel is: 1 / (CHANNEL_AREA width^2)), ln n with n the frame's independent samples,
    and the data's ridges (the competition and ridge_crests on its channels, as ridges_of)."""
    contrast = to_contrast(density, frame)
    channels = [channel_answers(contrast, valid, frame["levels"], place, threads) for place in range(len(HYPER_WIDTHS))]
    weights = 1 / (CHANNEL_AREA * np.asarray(HYPER_WIDTHS, dtype=float) ** 2)
    best = np.zeros_like(channels[0])
    for answers in channels:
        np.maximum(best, answers, out=best)
    orientation = np.argmax(best, axis=0).astype(np.uint8)
    strength = np.take_along_axis(best, orientation[None], axis=0)[0]
    ridges = ridge_crests(strength, orientation, apply_hysteresis_threshold(strength, T_LOW, T_HIGH) & valid)
    return {"density": np.nan_to_num(density).astype(np.float32), "valid": valid, "frame": frame, "contrast": contrast,
            "channels": channels, "weights": weights, "ln n": float(np.log(N_ORIENT * valid.sum() * weights.sum())),
            "ridges": ridges, "threads": threads}


def own_channels(state):
    """Every point of the drawn splines with the channel of its own line: the orientation of its tangent (the nearest
    of N_ORIENT) and the width nearest its half-width, as zero_points reads them. Returns (piece, row, column,
    orientation, width) arrays."""
    shown = np.flatnonzero(state["drawn"])
    if not len(shown):
        return tuple(np.zeros(0, int) for _ in range(5))
    rows, columns = state["render"].shape
    pieces = [state["pieces"][p] for p in shown]
    owner = np.concatenate([np.full(len(piece["x"]), p) for p, piece in zip(shown, pieces)])
    x, y = np.concatenate([piece["x"] for piece in pieces]), np.concatenate([piece["y"] for piece in pieces])
    theta = np.concatenate([np.arctan2(np.gradient(piece["y"]), np.gradient(piece["x"])) if len(piece["x"]) > 1 else np.zeros(1) for piece in pieces])
    radius = np.concatenate([piece["radius"] for piece in pieces]).astype(float)
    k = np.rint((theta % np.pi) / (np.pi / N_ORIENT)).astype(int) % N_ORIENT
    w = np.argmin(np.abs(np.log(np.maximum(radius, 1e-3)[:, None] / np.asarray(HYPER_WIDTHS)[None])), axis=1)
    return owner, np.clip(np.rint(y).astype(int), 0, rows - 1), np.clip(np.rint(x).astype(int), 0, columns - 1), k, w


def chi_topo(scene, render, zero, maps=False, points=None):
    """chi^2_topo of J (section 39): per channel (16 orientations x 8 widths, before the competition) the data's answer
    minus the render's, less the channel's zero point where the render answers above T_LOW (the zero point is measured
    on right lines, section 38a), squared, each width's channel-pixels weighted by the scene's weights. With `maps`
    also the residual's strongest positive and negative channel per pixel (value, orientation, width) and the render's
    ridges (the competition and ridge_crests on its channels, as ridges_of); with `points` (own_channels) also the
    residual at every point in its own channel. Returns chi^2 and the maps (or None)."""
    contrast = to_contrast(render, scene["frame"])
    valid, levels = scene["valid"], scene["frame"]["levels"]
    shape = render.shape
    total = 0.0
    out = None
    if maps:
        out = {name: np.zeros(shape, np.float32) for name in ("positive", "negative")}
        out.update({name: np.zeros(shape, np.uint8) for name in ("positive k", "positive w", "negative k", "negative w")})
        best = np.zeros((N_ORIENT, *shape), np.float32)
        if points is not None:
            out["own"], out["points"] = np.zeros(len(points[0]), np.float32), points
    for place, weight in enumerate(scene["weights"]):
        answers = channel_answers(contrast, valid, levels, place, scene["threads"])
        residual = scene["channels"][place] - answers
        shift = np.nan_to_num(np.asarray(zero, dtype=float)[:, place], nan=0.0).astype(np.float32)[:, None, None]
        residual -= np.where(answers >= T_LOW, shift, np.float32(0))
        total += weight * float(np.sum(np.square(residual), dtype=np.float64))
        if maps:
            for sign, name in ((1, "positive"), (-1, "negative")):
                signed = residual if sign > 0 else -residual
                top = signed.max(axis=0)
                better = top > out[name]
                out[name][better] = top[better]
                out[name + " k"][better] = np.argmax(signed, axis=0)[better]
                out[name + " w"][better] = place
            np.maximum(best, answers, out=best)
            if points is not None:
                mine = points[4] == place
                out["own"][mine] = residual[points[3][mine], points[1][mine], points[2][mine]]
    if maps:
        orientation = np.argmax(best, axis=0).astype(np.uint8)
        strength = np.take_along_axis(best, orientation[None], axis=0)[0]
        out["ridges"] = ridge_crests(strength, orientation, apply_hysteresis_threshold(strength, T_LOW, T_HIGH) & valid)
        out["render strength"] = strength
    return total, out


def spline_parameters(piece):
    """k of a spline in J's price: the knots of its place (x and y each) and of its half-width, as spline_vessel fits
    them (KNOT_SPACING, or the spacing section 37's refinement settled on; RADIUS_KNOTS times farther for the width)."""
    length = float(piece.get("length", chain_length(piece)))
    spacing = float(piece.get("knot spacing", KNOT_SPACING))
    return 2 * (int(np.ceil(length / spacing)) + 1) + int(np.ceil(length / (RADIUS_KNOTS * KNOT_SPACING))) + 1


def objective(scene, state, zero, lam, chi=None):
    """J = chi^2_topo + lambda sum over the drawn splines of k ln n (section 39's principle). A spline that is not
    drawn has no say in the render and pays nothing. Returns J, chi^2_topo and the price."""
    chi = chi_topo(scene, state["render"], zero)[0] if chi is None else chi
    price = lam * scene["ln n"] * sum(spline_parameters(piece) for piece, shown in zip(state["pieces"], state["drawn"]) if shown)
    return chi + price, chi, price


def round_zero(scene, state):
    """The zero point per channel at round 0 (zero_points on the model's drawn splines)."""
    render_contrast = to_contrast(state["render"], scene["frame"])
    return zero_points(scene["contrast"], render_contrast, scene["valid"], scene["frame"]["levels"], state["pieces"],
                       state["drawn"], scene["threads"])[0]


def quiet_of(front, valid, away=8):
    """The quiet pixels of a fixation from section 29's saved maps, as step 5's cell takes them: where there is data
    and no neuron of the static score answers above T_LOW within `away` px."""
    lines = (front["strength"] > T_LOW).astype(np.uint8)
    return valid & (cv2.distanceTransform(1 - lines, cv2.DIST_L2, 5) > away)


# ---------------------------------------------------------------- the model as the recursion edits it

def fresh_render(scene, state):
    """The state with its render drawn from nothing (render_graph): every drawn lumen's canvas."""
    render, canvases = render_graph(state["pieces"], state["lumens"], state["kappa"], state["blur"], scene["density"].shape,
                                    state["drawn"], state["radii"])
    return dict(state, render=render, canvases=canvases)


def fresh_state(pieces, scene, blur_sigma, lumens=None, drawn=None):
    """A model of splines alone (for the checks): every piece its own lumen unless `lumens` says otherwise, its
    darkness from kappa_passes (no cores), rendered."""
    lumens = np.arange(len(pieces)) if lumens is None else np.asarray(lumens)
    drawn = np.ones(len(pieces), dtype=bool) if drawn is None else np.asarray(drawn, dtype=bool)
    pieces = [dict(piece, key=("initial", number)) for number, piece in enumerate(pieces)]
    kappa, radii, _, _ = kappa_passes(pieces, lumens, scene["density"], blur_sigma, [(0.0, 0.0)] * len(pieces), drawn)
    state = {"pieces": pieces, "kappa": list(kappa), "radii": list(radii), "drawn": drawn.copy(), "lumens": lumens.copy(),
             "origin": ["initial"] * len(pieces), "blur": float(blur_sigma)}
    return fresh_render(scene, state)


def piece_state(model, scene):
    """Section 38's initial model (initial_model) as the recursion edits it: per piece of section 37 (the splines), its
    darkness and radius per point from the edges cut from it (each edge's span on its piece), its lumen (the union of
    its edges' lumens) and whether it is drawn (most of its points on drawn edges: a twin is not). A piece no edge
    was cut from lies inside a core and is not drawn. Rendered (fresh_render)."""
    pieces = [dict(piece, key=("initial", number)) for number, piece in enumerate(model["pieces"])]
    kappa = [np.full(len(piece["x"]), np.nan) for piece in pieces]
    radii = [np.full(len(piece["x"]), np.nan) for piece in pieces]
    shown = [np.zeros(len(piece["x"]), dtype=bool) for piece in pieces]
    parent = list(range(len(pieces)))

    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    first = {}
    for e, edge in enumerate(model["edges"]):
        p, (a, b) = edge["chain"], edge["span"]
        if b - a + 1 != len(edge["x"]):
            continue
        kappa[p][a:b + 1], radii[p][a:b + 1] = model["kappa"][e], model["radii"][e]
        shown[p][a:b + 1] |= bool(model["drawn"][e])
        lumen = int(model["lumens"][e])
        if lumen in first:
            parent[root(p)] = root(first[lumen])
        else:
            first[lumen] = p
    known = [float(np.nanmedian(k)) for k in kappa if np.isfinite(k).any()]
    for p, piece in enumerate(pieces):
        good = np.isfinite(kappa[p])
        along = np.arange(len(good))
        if good.any():
            kappa[p] = np.interp(along, along[good], kappa[p][good])
            radii[p] = np.interp(along, along[good], radii[p][good])
        else:
            kappa[p] = np.full(len(good), float(np.median(known)) if known else SYNTH_ABSORB)
            radii[p] = sharp_radius(piece["radius"], model["blur"])
    roots = [root(p) for p in range(len(pieces))]
    names = {value: number for number, value in enumerate(sorted(set(roots)))}
    state = {"pieces": pieces, "kappa": kappa, "radii": radii, "drawn": np.array([s.mean() > 0.5 for s in shown], dtype=bool),
             "lumens": np.array([names[value] for value in roots]), "origin": ["initial"] * len(pieces), "blur": float(model["blur"])}
    return fresh_render(scene, state)


def without_pieces(scene, state, gone):
    """The state with some pieces taken out of the model altogether (not merely undrawn), rendered again."""
    keep = [p for p in range(len(state["pieces"])) if p not in set(gone)]
    lumens = state["lumens"][keep]
    names = {value: number for number, value in enumerate(sorted(set(lumens.tolist())))}
    return fresh_render(scene, dict(state, pieces=[state["pieces"][p] for p in keep], kappa=[state["kappa"][p] for p in keep],
                                    radii=[state["radii"][p] for p in keep], drawn=state["drawn"][keep].copy(),
                                    lumens=np.array([names[value] for value in lumens.tolist()]),
                                    origin=[state["origin"][p] for p in keep]))


def redraw_lumens(state, lumens):
    """The state's render (changed in place) with the canvases of `lumens` drawn again (draw_lumen; densities add
    across lumens, so a lumen's canvas is taken off and its new one put on)."""
    render, canvases = state["render"], state["canvases"]
    for lumen in sorted(lumens):
        if lumen in canvases:
            box, values = canvases.pop(lumen)
            render[box[0]:box[2], box[1]:box[3]] -= values
        members = [p for p in np.flatnonzero(state["lumens"] == lumen) if state["drawn"][p]]
        if members:
            box, values = draw_lumen([state["pieces"][p] for p in members], [state["kappa"][p] for p in members], state["blur"],
                                     render.shape, [state["radii"][p] for p in members])
            if values.size:
                render[box[0]:box[2], box[1]:box[3]] += values
                canvases[lumen] = (box, values)


def piece_darkness(rest, piece, blur_sigma):
    """A new spline's darkness and radius per point: window_kappa at its windows (no cores) on the density the render
    does not explain, as kappa_passes reads an edge; a spline too short for two windows has one, at its middle."""
    places = edge_windows(piece, (0.0, 0.0))
    if len(places) < 2:
        places = np.array([len(piece["x"]) // 2])
    found = [window_kappa(rest, piece, place, blur_sigma) for place in places]
    values = np.maximum([value for value, _, _ in found], 0.0)
    fitted = np.array([radius for _, _, radius in found])
    along = np.arange(len(piece["x"]))
    if len(places) > 1:
        return (np.interp(along, places, scipy.ndimage.median_filter(values, size=3, mode="nearest")),
                np.interp(along, places, scipy.ndimage.median_filter(fitted, size=3, mode="nearest")))
    return np.full(len(along), float(values[0])), np.full(len(along), float(fitted[0]))


def applied_move(scene, state, move):
    """A move applied to a copy of the state, the touched lumens drawn again:

      remove       a spline taken out of the rendering (it stays in the model, not drawn: section 39, step 3b)
      add, extend  a new spline (its darkness read on the density the render does not explain); an extension joins
                   the lumen of the spline whose end it continues, an addition is a lumen of its own
      one lumen    two vessels at a core made one lumen (their lumens united: section 39, step 4)
      two vessels  one lumen at a core made two: one vessel's pieces given a lumen of their own
    """
    new = dict(state, pieces=list(state["pieces"]), kappa=list(state["kappa"]), radii=list(state["radii"]),
               drawn=state["drawn"].copy(), lumens=state["lumens"].copy(), origin=list(state["origin"]),
               render=state["render"].copy(), canvases=dict(state["canvases"]))
    kind = move["kind"]
    if kind == "remove":
        new["drawn"][move["piece"]] = False
        touched = {int(new["lumens"][move["piece"]])}
    elif kind in ("add", "extend"):
        piece = move["new"]
        piece.setdefault("key", ("added", len(piece["x"]), float(piece["x"][0]), float(piece["y"][0]), float(piece["x"][-1]), float(piece["y"][-1])))
        kappa, radii = piece_darkness(scene["density"] - state["render"], piece, state["blur"])
        lumen = int(new["lumens"].max()) + 1 if move.get("lumen") is None else int(move["lumen"])
        new["pieces"].append(piece)
        new["kappa"].append(kappa)
        new["radii"].append(radii)
        new["drawn"] = np.r_[new["drawn"], True]
        new["lumens"] = np.r_[new["lumens"], lumen]
        new["origin"].append(kind)
        touched = {lumen}
    elif kind == "one lumen":
        a, b = int(new["lumens"][move["first"][0]]), int(new["lumens"][move["second"][0]])
        new["lumens"][new["lumens"] == b] = a
        touched = {a, b}
    elif kind == "two vessels":
        fresh = int(new["lumens"].max()) + 1
        touched = {int(new["lumens"][p]) for p in move["second"]} | {fresh}
        new["lumens"][list(move["second"])] = fresh
    else:
        raise ValueError(kind)
    redraw_lumens(new, touched)
    return new


# ---------------------------------------------------------------- the candidates of a round

def removal_candidates(state, residual):
    """Section 39, step 3b: every drawn spline along which the topology residual in its own channel (own_channels:
    the orientation and width of its line) lies below -T_HIGH on REMOVE_SHARE of its points (a line the render has
    and the data lacks), with its evidence (the sum of the squares there); and every drawn spline the render itself
    shows no line along (its strongest channel below T_LOW on REMOVE_SHARE of its points: its darkness was read as
    about zero), where only the price speaks, evidence 0. `residual` is chi_topo's maps with `points` given."""
    owner, rows, columns, _, _ = residual["points"]
    own, shown = residual["own"], residual["render strength"]
    out = []
    for p in np.flatnonzero(state["drawn"]):
        piece = state["pieces"][p]
        mine = owner == p
        if not mine.any():
            continue
        values = -own[mine]
        if np.mean(values >= T_HIGH) >= REMOVE_SHARE:
            out.append({"kind": "remove", "piece": int(p), "evidence": float(np.sum(values[values > 0] ** 2)), "x": piece["x"], "y": piece["y"]})
        elif np.mean(shown[rows[mine], columns[mine]] < T_LOW) >= REMOVE_SHARE:
            out.append({"kind": "remove", "piece": int(p), "evidence": 0.0, "x": piece["x"], "y": piece["y"]})
    return out


def addition_candidates(scene, state, residual, maps=None):
    """Section 39, steps 1, 3 and 5: the positive ridges of the topology residual off the drawn splines (outside their
    tubes), joined and cut as section 37 cuts the data's ridges (joined_ridges, ridge_lines to the forks), each piece
    of EVIDENCE_LEAST px or more a new spline (spline_vessel). An end that lies within its tube's reach (plus
    RIDGE_JOIN) of a drawn spline is joined to it: to its end (an extension, in its lumen) or to its body (a branch,
    a lumen of its own). `maps` may hold the fixation's `has_flow` and `flow_strength` and the data's `strength`
    (the static score the initial pieces carry), else the residual's own is kept. Returns the candidates with their
    evidence (the sum of the squared residual along them)."""
    positive = residual["positive"]
    shape = positive.shape
    maps = maps or {}
    has_flow = maps.get("has_flow", np.zeros(shape, dtype=bool))
    flow_strength = maps.get("flow_strength", np.zeros(shape, dtype=np.float32))
    tubes = tube_mask(state["pieces"], state["blur"], shape, state["drawn"])
    region = apply_hysteresis_threshold(positive, T_LOW, T_HIGH) & scene["valid"] & ~tubes
    if not region.any():
        return []
    joined = joined_ridges(positive, residual["positive k"], region)
    if skeleton_parts(joined["skeleton"])[3] <= 1:
        return []  # no piece between forks: ridge_lines needs one
    half_width = np.asarray(HYPER_WIDTHS, dtype=np.float32)[residual["positive w"]]
    lines = ridge_lines(joined["skeleton"], positive, residual["positive k"], half_width, least=0, to_forks=True)
    shown = np.flatnonzero(state["drawn"])
    owner = np.concatenate([np.full(len(state["pieces"][p]["x"]), p) for p in shown]) if len(shown) else np.zeros(0, int)
    spot = np.concatenate([np.arange(len(state["pieces"][p]["x"])) for p in shown]) if len(shown) else np.zeros(0, int)
    xy = np.column_stack([np.concatenate([state["pieces"][p]["x"] for p in shown]),
                          np.concatenate([state["pieces"][p]["y"] for p in shown])]) if len(shown) else np.zeros((0, 2))
    reach = (np.concatenate([tube_reach(np.asarray(state["pieces"][p]["radius"], dtype=np.float32), state["blur"]) for p in shown])
             + RIDGE_JOIN + 1) if len(shown) else np.zeros(0)
    tree = scipy.spatial.cKDTree(xy) if len(xy) else None
    rows, columns = shape
    out = []
    for line in lines:
        if chain_length(line) < EVIDENCE_LEAST:
            continue
        line, kind, lumen = dict(line), "add", None
        loose = list(line.get("loose", (True, True)))
        for side in (0, 1):
            size = len(line["x"])
            here = 0 if side == 0 else size - 1
            if tree is None:
                break
            distance, nearest = tree.query((line["x"][here], line["y"][here]))
            if not np.isfinite(distance) or distance > reach[nearest]:
                continue
            p, place = int(owner[nearest]), int(spot[nearest])
            last = len(state["pieces"][p]["x"]) - 1
            at_end = place <= 2 or place >= last - 2
            target = (state["pieces"][p]["x"][0 if place <= 2 else last], state["pieces"][p]["y"][0 if place <= 2 else last]) \
                if at_end else (state["pieces"][p]["x"][place], state["pieces"][p]["y"][place])
            for name, value in list(line.items()):  # the joining point put before the line's first or after its last
                if isinstance(value, np.ndarray) and len(value) == size:
                    extra = np.asarray([target[0] if name == "x" else target[1] if name == "y" else value[here]], dtype=value.dtype)
                    line[name] = np.concatenate([extra, value]) if side == 0 else np.concatenate([value, extra])
            loose[side] = False
            if at_end and lumen is None:
                kind, lumen = "extend", int(state["lumens"][p])
        line["loose"] = tuple(loose)
        if "strength" in maps:
            line["strength"] = maps["strength"][np.clip(np.rint(line["y"]).astype(int), 0, rows - 1),
                                                np.clip(np.rint(line["x"]).astype(int), 0, columns - 1)].astype(np.float32)
        line["flow share"] = flow_share(line, has_flow)
        vessel = spline_vessel(line, flow_strength)
        vessel.update({"loose": line["loose"], "knot spacing": KNOT_SPACING, "split": False})
        values = positive[np.clip(np.rint(vessel["y"]).astype(int), 0, rows - 1), np.clip(np.rint(vessel["x"]).astype(int), 0, columns - 1)]
        out.append({"kind": kind, "new": vessel, "lumen": lumen, "evidence": float(np.sum(values ** 2)),
                    "x": vessel["x"], "y": vessel["y"]})
    return out


def residual_spectrum(scene, state, quiet):
    """The noise's power spectrum for the core filter (section 38b) where no generator made the noise: the density less
    the render on the quiet pixels (the rest set to 0, the power divided by their share), smoothed over neighbouring
    frequencies (a single picture's power at one frequency scatters by its own size)."""
    quiet = quiet & scene["valid"]
    rest = (scene["density"] - state["render"]).astype(float)
    field = np.where(quiet, rest - np.median(rest[quiet]), 0.0)
    power = np.abs(np.fft.fft2(field)) ** 2 / max(float(quiet.mean()), 1e-6)
    return np.fft.ifftshift(scipy.ndimage.gaussian_filter(np.fft.fftshift(power), 2.0))


def composition_candidates(scene, state, quiet, spectrum=None):
    """Section 39, step 4: at every vertex of four arms of the drawn splines' graph (spline_to_graph), the arms paired
    into the two straightest vessels, and the core filter (core_z, section 38b) read on the density less the other
    lumens' render, within CORE_REGION core radii, with the arms' darkness and radius. Where it contradicts the
    composition by more than Z_PASS the other composition is proposed: two vessels whose z falls Z_PASS below what two
    vessels give, and below Z_PASS, become one lumen; one lumen with z above Z_PASS becomes two vessels. Returns a
    reading per vertex with its `kind` (the proposed move, or None) and z."""
    shown = np.flatnonzero(state["drawn"])
    if len(shown) < 2:
        return []
    view = spline_to_graph([state["pieces"][p] for p in shown])
    spectrum = residual_spectrum(scene, state, quiet) if spectrum is None else spectrum
    rows, columns = scene["density"].shape
    blur_sigma = state["blur"]
    out = []
    for vertex in view["vertices"]:
        if len(vertex["ends"]) != 4:
            continue
        centre = np.array([vertex["x"], vertex["y"]])
        arms = []
        for e, end in vertex["ends"]:
            edge = view["edges"][e]
            points = np.column_stack([edge["x"], edge["y"]]).astype(float)
            first, last = edge["span"]
            places = np.arange(first, last + 1)
            if end == 1:
                points, places = points[::-1], places[::-1]
            p = int(shown[edge["spline"]])
            arms.append({"points": points, "places": places, "piece": p,
                         "direction": (points[min(10, len(points) - 1)] - points[0]) / max(np.hypot(*(points[min(10, len(points) - 1)] - points[0])), 1e-9)})
        if any(len(arm["points"]) < 4 for arm in arms):
            continue
        pairing = min((((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))),
                      key=lambda pairs: sum(float(np.dot(arms[a]["direction"], arms[b]["direction"])) for a, b in pairs))
        vessels = []
        for a, b in pairing:
            kappa = float(np.median(np.concatenate([state["kappa"][arms[i]["piece"]][arms[i]["places"]] for i in (a, b)])))
            radius = float(np.median(np.concatenate([state["radii"][arms[i]["piece"]][arms[i]["places"]] for i in (a, b)])))
            vessels.append({"arms": (a, b), "pieces": sorted({arms[a]["piece"], arms[b]["piece"]}), "kappa": kappa, "radius": radius})
        core = max(vessel["radius"] for vessel in vessels) + 2 * blur_sigma + RING_MARGIN
        reach = CORE_REGION * core + 10
        for vessel in vessels:
            a, b = vessel["arms"]
            near = lambda arm: arm["points"][np.hypot(*(arm["points"] - centre).T) <= reach]
            line = np.vstack([near(arms[a])[::-1], near(arms[b])[1:]])
            half = float(np.sqrt((CYLINDER * vessel["radius"]) ** 2 + blur_sigma ** 2))
            vessel["edge"] = {"x": line[:, 0], "y": line[:, 1], "radius": np.full(len(line), half)}
        lumen_sets = [{int(state["lumens"][p]) for p in vessel["pieces"]} for vessel in vessels]
        if any(len(lumens) != 1 for lumens in lumen_sets):
            continue
        composition = "two vessels" if lumen_sets[0] != lumen_sets[1] else "one lumen"
        others = state["render"].copy()
        for lumen in set().union(*lumen_sets):
            if lumen in state["canvases"]:
                box, values = state["canvases"][lumen]
                others[box[0]:box[2], box[1]:box[3]] -= values
        y, x = np.mgrid[:rows, :columns]
        region = (np.hypot(x - centre[0], y - centre[1]) <= CORE_REGION * core).astype(float)
        z, expected = core_z(scene["density"] - others, [(vessel["edge"], vessel["kappa"], vessel["radius"]) for vessel in vessels],
                             blur_sigma, spectrum, region)
        kind = None
        if composition == "two vessels" and z < Z_PASS and expected - z > Z_PASS:
            kind = "one lumen"
        elif composition == "one lumen" and z > Z_PASS:
            kind = "two vessels"
        out.append({"kind": kind, "composition": composition, "z": z, "expected": expected, "x": float(centre[0]), "y": float(centre[1]),
                    "first": vessels[0]["pieces"], "second": vessels[1]["pieces"], "evidence": abs(z - (expected if composition == "two vessels" else 0.0))})
    return out


def bridge_candidates(scene, state, positive):
    """Section 39, step 1, reported and never drawn: pairs of free ends of drawn splines (no other spline within
    RIDGE_JOIN + 1 px, farther than FIELD_EDGE from where the data ends) within BRIDGE_REACH (BRIDGE_WIDTHS times the
    larger width, at least RIDGE_REACH), each end's outward direction within SIGMA_BRANCH of the line to the other.
    Per bridge: the similarity of the line masses (pi kappa R^2 over the last 12 points of each end, exp(-t^2 / 2)
    with t their log ratio over SIGMA_DENSITY) and whether a positive ridge of the topology residual continues
    across the gap (its median along the chord at least T_LOW)."""
    shown = np.flatnonzero(state["drawn"])
    if len(shown) < 2:
        return []
    owner = np.concatenate([np.full(len(state["pieces"][p]["x"]), p) for p in shown])
    xy = np.column_stack([np.concatenate([state["pieces"][p]["x"] for p in shown]), np.concatenate([state["pieces"][p]["y"] for p in shown])])
    tree = scipy.spatial.cKDTree(xy)
    rows, columns = scene["valid"].shape
    border = cv2.distanceTransform(scene["valid"].astype(np.uint8), cv2.DIST_L2, 5)
    ends = []
    for p in shown:
        piece = state["pieces"][p]
        size = len(piece["x"])
        for end, step in ((0, 1), (size - 1, -1)):
            spot = np.array([piece["x"][end], piece["y"][end]], dtype=float)
            if any(owner[i] != p for i in tree.query_ball_point(spot, RIDGE_JOIN + 1)):
                continue
            if border[int(np.clip(round(spot[1]), 0, rows - 1)), int(np.clip(round(spot[0]), 0, columns - 1))] <= FIELD_EDGE:
                continue
            back = int(np.clip(end + 5 * step, 0, size - 1))
            direction = spot - np.array([piece["x"][back], piece["y"][back]], dtype=float)
            if np.hypot(*direction) < 1e-6:
                continue
            last = slice(0, 12) if end == 0 else slice(max(size - 12, 0), size)
            mass = float(np.median(np.pi * state["kappa"][p][last] * state["radii"][p][last] ** 2))
            ends.append({"piece": int(p), "end": int(end), "xy": spot, "direction": direction / np.hypot(*direction),
                         "width": 2 * float(piece["radius"][end]) / CYLINDER, "mass": max(mass, 1e-9)})
    bridges = []
    for a, b in itertools.combinations(ends, 2):
        if a["piece"] == b["piece"]:
            continue
        gap = b["xy"] - a["xy"]
        distance = float(np.hypot(*gap))
        if distance < 1e-6 or distance > max(BRIDGE_WIDTHS * max(a["width"], b["width"]), RIDGE_REACH):
            continue
        turn = lambda direction, towards: np.degrees(np.arccos(np.clip(float(np.dot(direction, towards / distance)), -1, 1)))
        if max(turn(a["direction"], gap), turn(b["direction"], -gap)) > SIGMA_BRANCH:
            continue
        chord = line_points(a["xy"], b["xy"])
        along = positive[np.clip(np.rint(chord[:, 1]).astype(int), 0, rows - 1), np.clip(np.rint(chord[:, 0]).astype(int), 0, columns - 1)]
        t = np.log(a["mass"] / b["mass"]) / SIGMA_DENSITY
        bridges.append({"ends": ((a["piece"], a["end"]), (b["piece"], b["end"])), "x": float((a["xy"][0] + b["xy"][0]) / 2),
                        "y": float((a["xy"][1] + b["xy"][1]) / 2), "distance": distance, "similarity": float(np.exp(-t ** 2 / 2)),
                        "continued": bool(np.median(along) >= T_LOW), "chord": chord})
    return bridges


# ---------------------------------------------------------------- the rounds

def round_candidates(scene, state, residual, quiet=None, maps=None, most=CANDIDATES_MOST):
    """A round's candidates, the strongest evidence first within each kind: at most a third removals and a sixth
    compositions (the core filter needs the quiet pixels), the rest additions and extensions."""
    removals = sorted(removal_candidates(state, residual), key=lambda c: -c["evidence"])[:most // 3]
    compositions = []
    if quiet is not None:
        compositions = sorted([c for c in composition_candidates(scene, state, quiet) if c["kind"]], key=lambda c: -c["evidence"])[:most // 6]
    additions = sorted(addition_candidates(scene, state, residual, maps), key=lambda c: -c["evidence"])
    return removals + compositions + additions[:most - len(removals) - len(compositions)]


def still_applies(state, move):
    """Whether a move proposed at the start of a round still means something after other moves were taken."""
    if move["kind"] == "remove":
        return bool(state["drawn"][move["piece"]])
    if move["kind"] == "one lumen":
        return state["lumens"][move["first"][0]] != state["lumens"][move["second"][0]]
    if move["kind"] == "two vessels":
        return state["lumens"][move["first"][0]] == state["lumens"][move["second"][0]]
    return True


def recursion_rounds(scene, state, zero, lam, rounds=ROUNDS_MOST, quiet=None, maps=None, score=None, watch=None,
                     most=CANDIDATES_MOST):
    """Section 39's rounds, at most `rounds`: per round the topology residual of the current model (chi_topo with its
    maps) gives the candidates (round_candidates); each is applied to a copy and J evaluated on the whole frame; then
    the candidates are taken in the order of their gain, each re-evaluated on the model as it is by then once a move
    has been taken, and taken only when J falls. The rounds end when a round takes nothing. Bridges are reported at
    the end and never drawn. `score` (state -> anything) is called on the model of every round; `watch` (a
    vascnet.watch.Recorder) is handed every candidate. Returns the final state and the history: J, chi^2, price,
    self-consistency (render in data, data in render, above chance) and score per round (round 0 first), every move
    (proposed and taken, with its gain and the J after it), and the bridges."""
    total, chi, price = objective(scene, state, zero, lam)
    _, residual = chi_topo(scene, state["render"], zero, maps=True, points=own_channels(state))
    agreement = lambda maps: tuple(self_consistency(maps["ridges"], scene["ridges"], scene["valid"])[name][2]
                                   for name in ("render in data", "data in render"))
    history = {"lambda": lam, "J": [total], "chi": [chi], "price": [price], "self": [agreement(residual)],
               "scores": [score(state) if score else None], "moves": [], "proposed": [0], "taken": [0]}
    shown = lambda s: [(s["pieces"][p]["x"], s["pieces"][p]["y"]) for p in np.flatnonzero(s["drawn"])]
    for number in range(1, rounds + 1):
        candidates = round_candidates(scene, state, residual, quiet, maps, most)
        evaluated = []
        for move in candidates:
            changed = applied_move(scene, state, move)
            value = objective(scene, changed, zero, lam)[0]
            evaluated.append((value - total, move, changed))
        taken = 0
        for gain, move, changed in sorted(evaluated, key=lambda item: item[0]):
            record = {"round": number, "kind": move["kind"], "proposed gain": gain, "taken": False, "x": move.get("x"), "y": move.get("y")}
            if gain < 0 and still_applies(state, move):
                if taken:
                    changed = applied_move(scene, state, move)
                    gain = objective(scene, changed, zero, lam)[0] - total
                if gain < 0:
                    state, total = changed, total + gain
                    taken += 1
                    record["taken"] = True
            record.update({"gain": gain, "J": total})
            history["moves"].append(record)
            if watch is not None:
                watch({"round": number, "kind": move["kind"], "taken": record["taken"], "gain": gain, "J": total,
                       "shown": shown(state), "x": move.get("x"), "y": move.get("y")})
        total, chi, price = objective(scene, state, zero, lam)
        _, residual = chi_topo(scene, state["render"], zero, maps=True, points=own_channels(state))
        for name, value in (("J", total), ("chi", chi), ("price", price), ("self", agreement(residual)),
                            ("scores", score(state) if score else None), ("proposed", len(candidates)), ("taken", taken)):
            history[name].append(value)
        if not taken:
            break
    history["bridges"] = bridge_candidates(scene, state, residual["positive"])
    history["residual"] = residual
    return state, history


def model_score(burst, front, state, cache, threads=None):
    """found-and-typed-right of a model on a synthetic burst: its drawn pieces read as section 37's pieces are read
    (confidence_curve: a kymograph per piece, cached by the piece's key; the graph; the junctions with flow as tests)
    and scored against the truth (score_reading)."""
    shown = [int(p) for p in np.flatnonzero(state["drawn"])]
    new = [p for p in shown if state["pieces"][p]["key"] not in cache]
    if new:
        readings, _ = segment_readings([state["pieces"][p] for p in new], lambda position: burst["frames"][position],
                                       burst["times"], burst["pairs"], burst["fps"], threads=threads)
        for p, reading in zip(new, readings):
            cache[state["pieces"][p]["key"]] = reading
    if len(shown) < 2:
        return score_reading(burst, {"chains": [], "graph": {"nodes": []}, "junctions": []})
    readings = [dict(cache[state["pieces"][p]["key"]]) for p in shown]
    vessels = [downstream(dict(state["pieces"][p]), reading) for p, reading in zip(shown, readings)]
    graph = vessel_graph(vessels, burst["valid"], burst["mean"], readings, burst["pairs"], burst["fps"])
    count = len(burst["times"])
    halves = (np.arange(0, count, 2), np.arange(1, count, 2)) if count >= SPLIT_FRAMES else None
    found = read_junctions(graph, burst["mean"], front["scores"], halves)[0]
    return score_reading(burst, {"chains": vessels, "readings": readings, "graph": graph, "junctions": found})


def bridge_right(bridge, lines):
    """A bridge is right when its chord lies on a true centerline (`lines`, a picture): most of its points within
    TRUE_WITHIN, as true_pieces judges a piece."""
    distance = distance_to(lines)
    rows, columns = lines.shape
    chord = bridge["chord"]
    return bool(np.mean(distance[np.clip(np.rint(chord[:, 1]).astype(int), 0, rows - 1),
                                 np.clip(np.rint(chord[:, 0]).astype(int), 0, columns - 1)] <= TRUE_WITHIN) > 0.5)


def round_frame(state, density, box=None, size=(7.0, 5.4), dpi=80):
    """A frame of a round's film: the density, the drawn splines (blue), and the candidate: green an addition or
    extension, red a removal, orange a composition; full when taken, faded and dashed when refused; J in the caption."""
    import matplotlib
    matplotlib.use("Agg")
    top, left, bottom, right = box if box is not None else (0, 0, *density.shape)
    figure, axis = plt.subplots(figsize=size, dpi=dpi)
    axis.imshow(np.nan_to_num(density[top:bottom, left:right]), cmap="gray_r", vmin=0,
                vmax=float(np.nanpercentile(density[top:bottom, left:right], 99.5)))
    for x, y in state["shown"]:
        axis.plot(np.asarray(x) - left, np.asarray(y) - top, color="tab:blue", lw=0.7, alpha=0.8)
    color = {"remove": "red", "add": "lime", "extend": "lime", "one lumen": "orange", "two vessels": "orange"}[state["kind"]]
    style = {"alpha": 1.0 if state["taken"] else 0.4, "ls": "-" if state["taken"] else "--"}
    if np.ndim(state["x"]) == 0:
        axis.plot([state["x"] - left], [state["y"] - top], "o", color=color, ms=12, mfc="none", mew=2, alpha=style["alpha"])
    else:
        axis.plot(np.asarray(state["x"]) - left, np.asarray(state["y"]) - top, color=color, lw=2.2, **style)
    axis.set_xlim(0, right - left)
    axis.set_ylim(bottom - top, 0)
    axis.axis("off")
    axis.set_title(f"round {state['round']}: {state['kind']} {'taken' if state['taken'] else 'refused'}, gain {state['gain']:+.0f} | "
                   f"J {state['J']:.0f}", fontsize=9)
    frame = vascnet.watch.frame_of(figure)
    plt.close(figure)
    return frame

# %%
# acceptance step 7 on one synthetic burst made to need each move: section 35's first tree burst for developing, its
# initial model (section 38) with the longest drawn true piece taken out of the model (the additions have to find that
# vessel again) and a false spline drawn across empty background as dark as the model's median (the removals have to
# take it out); its gaps, crossings and loops are the generator's. The rounds at lambda = 1, filmed; per round J, the
# self-consistency number and found-and-typed-right; the bridges reported.
import time

started = time.time()
recursion_burst = synthetic_burst_35(TREE_SEEDS[0], **SYNTH_THINGS)
recursion_front = ridge_front(recursion_burst, workers)
recursion_model = initial_model(recursion_front, recursion_burst["mean"], recursion_burst["valid"], workers)
recursion_frame = scoring_frame(recursion_burst["mean"], recursion_burst["valid"], recursion_burst["quiet"],
                                noise=robust_sd(recursion_burst["mean"][recursion_burst["quiet"]]))
recursion_space = recursion_scene(recursion_burst["mean"], recursion_burst["valid"], recursion_frame, workers)
start_state = piece_state(recursion_model, recursion_space)
truth = true_pieces(start_state["pieces"], recursion_burst["lines"])
missing = max((p for p in np.flatnonzero(start_state["drawn"]) if truth[p]), key=lambda p: start_state["pieces"][p]["length"])
missing_line = np.column_stack([start_state["pieces"][missing]["x"], start_state["pieces"][missing]["y"]])
missing_length = float(start_state["pieces"][missing]["length"])
start_state = without_pieces(recursion_space, start_state, [missing])
# the false spline: a straight 60 px line where every point lies 20 px or more from every true centerline and piece
occupied = recursion_burst["lines"].copy()
for piece in start_state["pieces"]:
    occupied[np.clip(np.rint(piece["y"]).astype(int), 0, occupied.shape[0] - 1), np.clip(np.rint(piece["x"]).astype(int), 0, occupied.shape[1] - 1)] = True
away = distance_to(occupied)
away[~cv2.erode(recursion_burst["valid"].astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)] = 0
rows, columns = away.shape
places = [(x, y, angle) for y in range(30, rows - 30, 10) for x in range(30, columns - 30, 10) for angle in (0, 45, 90, 135)]
reach = lambda x, y, angle: line_points((x - 30 * np.cos(np.radians(angle)), y - 30 * np.sin(np.radians(angle))),
                                        (x + 30 * np.cos(np.radians(angle)), y + 30 * np.sin(np.radians(angle))))
clearance = lambda place: float(away[np.rint(reach(*place)[:, 1]).astype(int), np.rint(reach(*place)[:, 0]).astype(int)].min())
best = max(places, key=clearance)
points = reach(*best)
drawn_half = float(np.median(np.concatenate([p["radius"] for p, shown in zip(start_state["pieces"], start_state["drawn"]) if shown])))
false_chain = {"x": points[:, 0].astype(np.float32), "y": points[:, 1].astype(np.float32),
               "theta": np.full(len(points), np.radians(best[2]) % np.pi, np.float32), "radius": np.full(len(points), drawn_half, np.float32),
               "strength": np.full(len(points), T_HIGH, np.float32), "bridge": np.zeros(len(points), bool), "closed": False,
               "loose": (True, True)}
false_chain["flow share"] = flow_share(false_chain, recursion_front["has_flow"])
false_piece = spline_vessel(false_chain, recursion_front["flow_strength"])
false_piece.update({"loose": (True, True), "knot spacing": KNOT_SPACING, "split": False, "key": ("false", 0)})
median_kappa = float(np.median([np.median(k) for k, shown in zip(start_state["kappa"], start_state["drawn"]) if shown]))
median_radius = float(np.median([np.median(r) for r, shown in zip(start_state["radii"], start_state["drawn"]) if shown]))
start_state = fresh_render(recursion_space, dict(
    start_state, pieces=start_state["pieces"] + [false_piece], kappa=start_state["kappa"] + [np.full(len(false_piece["x"]), median_kappa)],
    radii=start_state["radii"] + [np.full(len(false_piece["x"]), median_radius)], drawn=np.r_[start_state["drawn"], True],
    lumens=np.r_[start_state["lumens"], start_state["lumens"].max() + 1], origin=start_state["origin"] + ["false"]))
false_number = len(start_state["pieces"]) - 1
recursion_zero = round_zero(recursion_space, start_state)
round_recorder = vascnet.watch.Recorder(every=WATCH_EVERY)
reading_cache = {}
recursion_maps = {"has_flow": recursion_front["has_flow"], "flow_strength": recursion_front["flow_strength"], "strength": recursion_front["strength"]}
final_state, recursion_history = recursion_rounds(
    recursion_space, start_state, recursion_zero, 1.0, quiet=recursion_burst["quiet"], maps=recursion_maps,
    score=lambda state: model_score(recursion_burst, recursion_front, state, reading_cache, workers), watch=round_recorder)
spent = time.time() - started

print(f"Tree burst {TREE_SEEDS[0]}: {len(start_state['pieces'])} pieces in the model ({start_state['drawn'].sum()} drawn), blur "
      f"{start_state['blur']:.2f} px | taken out: the longest drawn true piece ({missing_length:.0f} px "
      f"from ({missing_line[0, 0]:.0f}, {missing_line[0, 1]:.0f}) to ({missing_line[-1, 0]:.0f}, {missing_line[-1, 1]:.0f})) | put in: a false spline "
      f"of 60 px at ({best[0]}, {best[1]}), {best[2]} degrees, {clearance(best):.0f} px from anything, kappa {median_kappa:.4f} | {spent / 60:.1f} min")
print(f"   {'round':>5s} {'proposed':>9s} {'taken':>6s} {'J':>12s} {'chi2_topo':>12s} {'price':>9s} {'render in data':>15s} {'data in render':>15s} "
      f"{'centerline P/R':>16s} {'junction P/R':>14s} {'found and typed right':>22s}")
for number, value in enumerate(recursion_history["J"]):
    row = ablation_row([recursion_history["scores"][number]])
    print(f"   {number:5d} {recursion_history['proposed'][number]:9d} {recursion_history['taken'][number]:6d} {value:12.0f} "
          f"{recursion_history['chi'][number]:12.0f} {recursion_history['price'][number]:9.0f} {recursion_history['self'][number][0] * 100:14.1f}% "
          f"{recursion_history['self'][number][1] * 100:14.1f}% {row['centerline precision'][0] * 100:7.1f}/{row['centerline recall'][0] * 100:5.1f}% "
          f"{row['junction precision'][0] * 100:6.1f}/{row['junction recall'][0] * 100:5.1f}% {row['found and typed right'][0] * 100:21.1f}%")
moves = recursion_history["moves"]
kinds = sorted({move["kind"] for move in moves})
print("   moves proposed / taken by kind: " + ", ".join(f"{kind} {sum(m['kind'] == kind for m in moves)} / "
                                                     f"{sum(m['kind'] == kind and m['taken'] for m in moves)}" for kind in kinds))
taken_values = [recursion_history["J"][0]] + [move["J"] for move in moves if move["taken"]]
falls = all(b < a for a, b in zip(taken_values[:-1], taken_values[1:]))
again = objective(recursion_space, fresh_render(recursion_space, final_state), recursion_zero, 1.0)[0]
print(f"   J at every taken move falls: {'yes' if falls else 'NO'} ({len(taken_values) - 1} moves, from {taken_values[0]:.0f} to {taken_values[-1]:.0f}); "
      f"the final model drawn afresh gives J {again:.0f} (the moves' sum {taken_values[-1]:.0f}, off by {abs(again - taken_values[-1]) / max(abs(again), 1) * 100:.3f}%)")
added = [p for p, origin in enumerate(final_state["origin"]) if origin in ("add", "extend") and final_state["drawn"][p]]
taken_out = [p for p in range(len(start_state["pieces"])) if start_state["drawn"][p] and not final_state["drawn"][p]]
start_truth = true_pieces(start_state["pieces"], recursion_burst["lines"])
tree_of_missing = scipy.spatial.cKDTree(missing_line)
covered = np.zeros(len(missing_line), bool)
for p in added:
    distance, nearest = tree_of_missing.query(np.column_stack([final_state["pieces"][p]["x"], final_state["pieces"][p]["y"]]))
    covered[nearest[distance <= TRUE_WITHIN]] = True
added_truth = true_pieces([final_state["pieces"][p] for p in added], recursion_burst["lines"]) if added else np.array([], bool)
print(f"   the vessel taken out: {covered.mean() * 100:.0f}% of its points have an added spline within {TRUE_WITHIN:.0f} px | the false spline: "
      f"{'taken out of the rendering' if not final_state['drawn'][false_number] else 'STILL DRAWN'} | added splines drawn at the end: {len(added)}, "
      f"{added_truth.sum()} of them true | splines taken out of the rendering: {len(taken_out)} ("
      + ", ".join("the false one" if p == false_number else "a true piece" if start_truth[p] else "a false piece" for p in taken_out) + ")")
junction_places = np.array([(junction["x"], junction["y"]) for junction in recursion_burst["junctions"]]).reshape(-1, 2)
for move in moves:
    if move["kind"] in ("one lumen", "two vessels"):
        distance = np.hypot(*(junction_places - (move["x"], move["y"])).T)
        nearest = int(np.argmin(distance)) if len(distance) else None
        print(f"   composition move {move['kind']} at ({move['x']:.0f}, {move['y']:.0f}), {'taken' if move['taken'] else 'refused'} (gain "
              f"{move['gain']:+.0f}): the nearest true junction is a {recursion_burst['junctions'][nearest]['type']} {distance[nearest]:.0f} px away"
              if nearest is not None else "")
bridges = recursion_history["bridges"]
print(f"   bridges (reported, never drawn): {len(bridges)}, of which {sum(bridge_right(b, recursion_burst['lines']) for b in bridges)} right; "
      f"{sum(b['continued'] for b in bridges)} with a positive ridge across the gap | the generator's gaps: "
      f"{sum(1 for s in recursion_burst['segments'] if s.get('gap'))}")
os.makedirs("vessel_pictures/fit", exist_ok=True)
for number in sorted({state["round"] for state in round_recorder.states}):
    states = [state for state in round_recorder.states if state["round"] == number]
    vascnet.watch.animate(states, lambda state: round_frame(state, recursion_burst["mean"]), f"vessel_pictures/fit/round_{number}.gif", fps=2.0)
print(f"   the rounds are filmed in vessel_pictures/fit/round_<r>.gif ({len(round_recorder.states)} frames of {round_recorder.count} candidates)")
memory_note("section 39, one burst")

# %%
# acceptance step 7 on the harness: section 35's held-out trees (TREE_HELD_OUT), each with its initial model (section 38)
# and the rounds at lambda = 1 (the real burst's sweep is the next cell; the plan's rule for lambda needs the flux
# network on every model, which is not run); found-and-typed-right per round over the bursts, and the first principle's
# number after the rounds: the edge confidence (section 37's five terms, calibrated on the trees for developing as
# acceptance step 2 calibrated it) told apart between true and false pieces at round 0 and after the rounds. After the
# rounds the terms are read on the model's pieces less the ones the rounds took out of the rendering (an added piece
# that takes the place of one taken out would otherwise make both twins), and those are counted once left out and once
# with their confidence down to 0, as the plan has them.
import time

started = time.time()
calibration_rows, calibration_truth = [], []
for seed in TREE_SEEDS:
    table = edge_table(synthetic_burst_35(seed, **{**SYNTH_THINGS, "second": True}), workers)
    calibration_rows += table["terms"]
    calibration_truth += list(table["truth"])
    del table
tree_edge_model = calibrate_terms(calibration_rows, calibration_truth)
calibration_time = time.time() - started
recursion_runs = []
for seed in TREE_HELD_OUT:
    burst = synthetic_burst_35(seed, **{**SYNTH_THINGS, "second": True})
    front = ridge_front(burst, workers)
    model = initial_model(front, burst["mean"], burst["valid"], workers)
    frame = scoring_frame(burst["mean"], burst["valid"], burst["quiet"], noise=robust_sd(burst["mean"][burst["quiet"]]))
    space = recursion_scene(burst["mean"], burst["valid"], frame, workers)
    state = piece_state(model, space)
    zero = round_zero(space, state)
    cache = {}
    maps = {"has_flow": front["has_flow"], "flow_strength": front["flow_strength"], "strength": front["strength"]}
    final, history = recursion_rounds(space, state, zero, 1.0, quiet=burst["quiet"], maps=maps,
                                      score=lambda candidate: model_score(burst, front, candidate, cache, workers))
    history.pop("residual")
    second = burst["second"]
    other = ridge_front(second, workers)
    other_lines = joined_ridges(other["strength"], other["orientation"], other["mask"])["skeleton"]
    ridge_blur = picture_blur(front["half_width"], model["joined"]["ridges"])
    confidence_of = lambda pieces: edge_confidence(edge_terms(pieces, burst["valid"], ridge_blur, edge_repeatability(
        pieces, burst["valid"], other_lines, second["valid"], second["to second"])[2]), tree_edge_model)[0]
    removed = np.array([p < len(state["pieces"]) and bool(state["drawn"][p]) and not final["drawn"][p] for p in range(len(final["pieces"]))])
    kept = [p for p in range(len(final["pieces"])) if not removed[p]]
    after = confidence_of([final["pieces"][p] for p in kept])
    truth_after = true_pieces(final["pieces"], burst["lines"])
    taken = [history["J"][0]] + [move["J"] for move in history["moves"] if move["taken"]]
    recursion_runs.append({
        "seed": seed, "history": history, "J falls": all(b < a for a, b in zip(taken[:-1], taken[1:])),
        "J again": objective(space, fresh_render(space, final), zero, 1.0)[0], "J tracked": taken[-1],
        "before": confidence_of(state["pieces"]), "truth before": true_pieces(state["pieces"], burst["lines"]),
        "after": after, "truth after": truth_after[kept], "after down": np.r_[after, np.zeros(removed.sum())],
        "truth after down": np.r_[truth_after[kept], truth_after[removed]], "truth final": truth_after,
        "origin after": list(final["origin"]), "drawn after": final["drawn"].copy(), "drawn before": state["drawn"].copy(),
        "bridges": [(b["similarity"], b["continued"], bridge_right(b, burst["lines"])) for b in history["bridges"]],
        "compositions": [(m["kind"], m["taken"], *min(((j["type"], float(np.hypot(j["x"] - m["x"], j["y"] - m["y"]))) for j in burst["junctions"]),
                                                      key=lambda pair: pair[1], default=("none", np.inf)))
                         for m in history["moves"] if m["kind"] in ("one lumen", "two vessels")]})
    del burst, front, model, frame, space, state, final, cache, second, other, other_lines
    print(f"   tree burst {seed}: {len(recursion_runs[-1]['history']['J']) - 1} rounds, {sum(m['taken'] for m in recursion_runs[-1]['history']['moves'])} "
          f"moves taken | {(time.time() - started) / 60:.1f} min since the start", flush=True)

rounds_run = max(len(run["history"]["scores"]) for run in recursion_runs)
at_round = lambda run, number: run["history"]["scores"][min(number, len(run["history"]["scores"]) - 1)]
per_round = [ablation_row([at_round(run, number) for run in recursion_runs]) for number in range(rounds_run)]
print(f"\nThe rounds on section 35's {len(recursion_runs)} held-out trees at lambda = 1 (the calibration on {len(TREE_SEEDS)} trees for developing "
      f"took {calibration_time / 60:.1f} min; all {(time.time() - started) / 60:.1f} min). A burst whose rounds ended keeps its last model.")
shown = ("centerline precision", "centerline recall", "junction precision", "junction recall", "found and typed right")
print(f"   {'round':>5s} {'bursts moving':>14s} " + "".join(f"{column:>24s}" for column in shown))
for number, row in enumerate(per_round):
    moving = sum(len(run["history"]["J"]) > number for run in recursion_runs)
    print(f"   {number:5d} {moving:14d} " + "".join(f"{row[c][0] * 100:7.1f} ({row[c][1] * 100:4.1f}-{row[c][2] * 100:5.1f})".rjust(24) for c in shown))
moves = [move for run in recursion_runs for move in run["history"]["moves"]]
print("   moves proposed / taken by kind: " + ", ".join(f"{kind} {sum(m['kind'] == kind for m in moves)} / {sum(m['kind'] == kind and m['taken'] for m in moves)}"
                                                     for kind in sorted({m["kind"] for m in moves})))
off = max(abs(run["J again"] - run["J tracked"]) / max(abs(run["J again"]), 1) for run in recursion_runs)
print(f"   J falls at every taken move in {sum(run['J falls'] for run in recursion_runs)} of {len(recursion_runs)} bursts; the final models drawn afresh "
      f"give the J the moves summed to within {off * 100:.3f}%")


def told_with_interval(confidences, truths):
    told = told_apart(np.concatenate([c[t] for c, t in zip(confidences, truths)]), np.concatenate([c[~t] for c, t in zip(confidences, truths)]))
    picks = np.random.default_rng(0).integers(len(confidences), size=(500, len(confidences)))
    resampled = [told_apart(np.concatenate([confidences[b][truths[b]] for b in pick]), np.concatenate([confidences[b][~truths[b]] for b in pick]))
                 for pick in picks]
    return (told, *np.percentile(resampled, [2.5, 97.5]))


principle = {name: told_with_interval([run[key] for run in recursion_runs], [run[truth] for run in recursion_runs])
             for name, key, truth in (("round 0", "before", "truth before"), ("after the rounds, pieces taken out left out", "after", "truth after"),
                                      ("after the rounds, pieces taken out at 0", "after down", "truth after down"))}
for name, (told, low, high) in principle.items():
    print(f"   the edge confidence told apart, {name}: {told:.3f} ({low:.3f} to {high:.3f}, 95% over bursts)")
added = [(run["truth final"][p], run["drawn after"][p]) for run in recursion_runs for p, origin in enumerate(run["origin after"]) if origin in ("add", "extend")]
taken_out = [(run["truth before"][p]) for run in recursion_runs for p in range(len(run["drawn before"])) if run["drawn before"][p] and not run["drawn after"][p]]
print(f"   pieces added and drawn at the end: {sum(d for _, d in added)}, {sum(t and d for t, d in added)} of them true | pieces taken out of the "
      f"rendering: {len(taken_out)}, {sum(taken_out)} of them true")
bridges = [bridge for run in recursion_runs for bridge in run["bridges"]]
right = np.array([r for _, _, r in bridges], bool)
similarity = np.array([s for s, _, _ in bridges])
print(f"   bridges (reported, never drawn): {len(bridges)}, {right.sum()} right | the line-mass similarity told apart between right and wrong "
      f"{told_apart(similarity[right], similarity[~right]):.2f} | a positive ridge across the gap: {sum(c for _, c, r in bridges if r)} of the right, "
      f"{sum(c for _, c, r in bridges if not r)} of the wrong")
compositions = [c for run in recursion_runs for c in run["compositions"]]
for kind in ("one lumen", "two vessels"):
    mine = [c for c in compositions if c[0] == kind]
    near = lambda c: c[2] if c[3] <= 10 else "no true junction within 10 px"
    print(f"   composition moves to {kind}: {len(mine)} proposed, {sum(c[1] for c in mine)} taken | the taken ones at: "
          + (", ".join(f"{name} {sum(near(c) == name for c in mine if c[1])}" for name in sorted({near(c) for c in mine if c[1]})) or "none"))
shares = [row["found and typed right"][0] for row in per_round]
holds = all(b >= a for a, b in zip(shares[:-1], shares[1:]))
falls = all(run["J falls"] for run in recursion_runs)
kept = min(told for told, _, _ in list(principle.values())[1:]) >= 0.75
print(f"\nThe gate: J falls at every taken move ({'yes' if falls else 'no'}); found-and-typed-right does not fall from one round to the next "
      f"({'yes' if holds else 'no'}: " + " -> ".join(f"{share * 100:.1f}" for share in shares) + f"); the first principle's number after the rounds "
      f"at least 0.75 ({'yes' if kept else 'no'}: " + ", ".join(f"{told:.3f}" for told, _, _ in list(principle.values())[1:]) + ") -> "
      + ("PASSED" if falls and holds and kept else "FAILED"))
os.makedirs("vessel_pictures/recursion", exist_ok=True)
save_table([{"seed": run["seed"], "round": number, "J": run["history"]["J"][number], "chi2": run["history"]["chi"][number],
             "price": run["history"]["price"][number], "render in data": run["history"]["self"][number][0],
             "data in render": run["history"]["self"][number][1], "proposed": run["history"]["proposed"][number],
             "taken": run["history"]["taken"][number], "true junctions": run["history"]["scores"][number]["junctions"]["n_gt"],
             "found and typed right": sum(run["history"]["scores"][number]["junctions"]["per_type"][kind]["type_correct"] for kind in TRUE_TYPES),
             "centerline precision": run["history"]["scores"][number]["centerline"]["precision"],
             "centerline recall": run["history"]["scores"][number]["centerline"]["recall"]}
            for run in recursion_runs for number in range(len(run["history"]["J"]))], "vessel_pictures/recursion/held_out_rounds.csv")
memory_note("section 39, harness")

# %%
# acceptance step 7 on the real burst: the rounds on a central 640 x 640 crop of the anchor's fixation (an evaluation of
# J over the whole frame takes about a minute here, step 5), lambda swept over LAMBDA_TRIES; per lambda and round J,
# the moves taken by kind and the self-consistency number. The plan chooses lambda as the largest at which the flux
# network's shares of impossible junctions and directed cycles on the model stay within the consolidated graph's
# (step 3); the network is not run on these models (section 39's closing round is not built), so lambda is not chosen.
import time

rows, columns = saved["valid"][anchor].shape
crop_top, crop_left = (rows // 2 - 320) // WINDOW_ALIGN * WINDOW_ALIGN, (columns // 2 - 320) // WINDOW_ALIGN * WINDOW_ALIGN
crop = (slice(crop_top, crop_top + 640), slice(crop_left, crop_left + 640))
whole_front = saved_front(saved, anchor)
crop_front = {name: np.ascontiguousarray(value[:, crop[0], crop[1]] if name == "scores" else value[crop]) for name, value in whole_front.items()}
crop_density = np.ascontiguousarray(saved["density"][anchor][crop])
crop_valid = np.ascontiguousarray(saved["valid"][anchor][crop])
crop_quiet = quiet_of(crop_front, crop_valid)
started = time.time()
crop_model = initial_model(crop_front, crop_density, crop_valid, workers)
crop_frame = scoring_frame(crop_density, crop_valid, crop_quiet)
crop_space = recursion_scene(crop_density, crop_valid, crop_frame, workers)
crop_state = piece_state(crop_model, crop_space)
crop_zero = round_zero(crop_space, crop_state)
crop_maps = {"has_flow": crop_front["has_flow"], "flow_strength": crop_front["flow_strength"], "strength": crop_front["strength"]}
model_time = time.time() - started
lambda_runs = {}
crop_recorder = vascnet.watch.Recorder(every=WATCH_EVERY)
for lam in LAMBDA_TRIES:
    started = time.time()
    final, history = recursion_rounds(crop_space, crop_state, crop_zero, lam, quiet=crop_quiet, maps=crop_maps,
                                      watch=crop_recorder if lam == 1.0 else None)
    history.pop("residual")
    taken = [history["J"][0]] + [move["J"] for move in history["moves"] if move["taken"]]
    lambda_runs[lam] = {"final": final, "history": history, "time": time.time() - started,
                        "J falls": all(b < a for a, b in zip(taken[:-1], taken[1:])), "J tracked": taken[-1],
                        "J again": objective(crop_space, fresh_render(crop_space, final), crop_zero, lam)[0]}

print(f"Real burst, fixation {anchor}, the crop rows {crop_top}-{crop_top + 640}, columns {crop_left}-{crop_left + 640}: the initial model in "
      f"{model_time / 60:.1f} min, {len(crop_state['pieces'])} pieces ({crop_state['drawn'].sum()} drawn) in {crop_state['lumens'].max() + 1} lumens, "
      f"blur fitted {crop_state['blur']:.2f} px | ln n {crop_space['ln n']:.1f}")
print("   zero point per width (median over the orientations measured): " + ", ".join(
    f"{HYPER_WIDTHS[w]}: {np.nanmedian(crop_zero[:, w]):+.2f}" for w in range(len(HYPER_WIDTHS)) if np.isfinite(crop_zero[:, w]).any()))
print(f"   {'lambda':>6s} {'round':>5s} {'proposed':>9s} {'taken':>6s} {'J':>12s} {'chi2_topo':>12s} {'price':>10s} {'render in data':>15s} {'data in render':>15s}")
for lam, run in lambda_runs.items():
    history = run["history"]
    for number, value in enumerate(history["J"]):
        print(f"   {lam:6.2f} {number:5d} {history['proposed'][number]:9d} {history['taken'][number]:6d} {value:12.0f} {history['chi'][number]:12.0f} "
              f"{history['price'][number]:10.0f} {history['self'][number][0] * 100:14.1f}% {history['self'][number][1] * 100:14.1f}%")
for lam, run in lambda_runs.items():
    moves = run["history"]["moves"]
    final = run["final"]
    print(f"   lambda {lam:g}: moves proposed / taken: " + ", ".join(
        f"{kind} {sum(m['kind'] == kind for m in moves)} / {sum(m['kind'] == kind and m['taken'] for m in moves)}" for kind in sorted({m['kind'] for m in moves}))
          + f" | drawn pieces {crop_state['drawn'].sum()} -> {final['drawn'].sum()} | bridges reported {len(run['history']['bridges'])}, "
          f"{sum(b['continued'] for b in run['history']['bridges'])} with a positive ridge across the gap | J falls at every taken move: "
          f"{'yes' if run['J falls'] else 'NO'}; drawn afresh {run['J again']:.0f} against {run['J tracked']:.0f} | {run['time'] / 60:.1f} min")
print(f"   the data's own repeatability (whole frame, step 5): 74.8% above chance | lambda is not chosen: the flux network is not run on these models")
os.makedirs("vessel_pictures/fit", exist_ok=True)
for number in sorted({state["round"] for state in crop_recorder.states}):
    states = [state for state in crop_recorder.states if state["round"] == number]
    vascnet.watch.animate(states, lambda state: round_frame(state, crop_density), f"vessel_pictures/fit/real_round_{number}.gif", fps=2.0)
print(f"   the rounds at lambda 1 are filmed in vessel_pictures/fit/real_round_<r>.gif ({len(crop_recorder.states)} frames)")
memory_note("section 39, real burst")
