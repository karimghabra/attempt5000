"""Why acceptance step 5's windowed re-score is off on the real burst.

The real render (the 8-bit copy the step 5 cell saves, `notebooks/vessel_pictures/render/real_render.png`) is scored
with the real frame's own constants (`scoring_frame`, the quiet pixels as the step 5 cell makes them). For each width
it prints the largest difference between the full frame's channels and a 320 px window's, at margins of 110 to 280 px,
the largest answer in the window, and the window's cost per margin.

    python scripts/window_margin_check.py

Needs section 29's saved inputs (`notebooks/section_29_inputs`) and a run of the step 5 cell before it.
"""
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from notebook import definitions  # noqa: E402

MARGINS = (110, 160, 220, 280)
THREADS = 6


def main():
    nb = definitions()
    folder = ROOT / "notebooks" / "section_29_inputs"
    valid = np.load(folder / "valid_0.npy")
    density = np.load(folder / "density_0.npy")
    strength = np.load(folder / "static_strength_0.npy")
    with open(folder / "section_29.pkl", "rb") as file:
        small = pickle.load(file)
    picture = ROOT / "notebooks" / "vessel_pictures" / "render" / "real_render.png"
    render = np.asarray(Image.open(picture), dtype=np.float32) / 255 * small["density range"][1]
    lines = (strength > nb["T_LOW"]).astype(np.uint8)
    quiet = valid & (cv2.distanceTransform(1 - lines, cv2.DIST_L2, 5) > 8)
    frame = nb["scoring_frame"](density, valid, quiet)
    contrast = nb["to_contrast"](render, frame)
    rows, columns = valid.shape
    middle = (rows // 2 - 160, columns // 2 - 160, rows // 2 + 160, columns // 2 + 160)
    spent = {margin: 0.0 for margin in MARGINS}
    for place, width in enumerate(nb["HYPER_WIDTHS"]):
        full = nb["channel_answers"](contrast, valid, frame["levels"], place, THREADS)
        differences = []
        for margin in MARGINS:
            inner, crop = nb["aligned_window"](middle, contrast.shape, margin)
            t, l, b, r = crop
            started = time.time()
            part = nb["channel_answers"](contrast[t:b, l:r], valid[t:b, l:r], frame["levels"], place, THREADS)
            spent[margin] += time.time() - started
            part = part[:, inner[0] - t:inner[2] - t, inner[1] - l:inner[3] - l]
            differences.append(float(np.abs(full[:, inner[0]:inner[2], inner[1]:inner[3]] - part).max()))
        largest = float(full[:, inner[0]:inner[2], inner[1]:inner[3]].max())
        print(f"width {width}: largest answer in the window {largest:.1f} | difference at margins "
              + " / ".join(str(margin) for margin in MARGINS) + ": " + " / ".join(f"{value:.4f}" for value in differences), flush=True)
    print("window time, all widths, per margin: " + ", ".join(f"{margin} px {spent[margin]:.1f} s" for margin in MARGINS))


if __name__ == "__main__":
    main()
