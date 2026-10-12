"""FFT convolution with a fixed kernel bank.

Every stage of the network is a convolution with hand-built kernels, so the
cheapest exact implementation is to transform each kernel once, cache it, and
reuse it for every frame of a video.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft


def _embed(kernel: np.ndarray, fshape: tuple[int, int]) -> np.ndarray:
    """Place a centred, odd-sized kernel at the origin of a periodic grid."""
    out = np.zeros(fshape, dtype=np.float32)
    kh, kw = kernel.shape
    out[:kh, :kw] = kernel
    return np.roll(out, (-(kh // 2), -(kw // 2)), axis=(0, 1))


class FourierBank:
    """A bank of kernels pre-transformed for images of one shape.

    Images are reflect-padded by the largest kernel radius, so outputs carry no
    wrap-around artefacts and behave like ``scipy.ndimage.convolve(mode='reflect')``.
    """

    def __init__(self, kernels: list[np.ndarray], image_shape: tuple[int, int]):
        self.pad = max(max(k.shape) for k in kernels) // 2
        h, w = image_shape
        self.shape = (h, w)
        self.fshape = (
            sfft.next_fast_len(h + 2 * self.pad, real=True),
            sfft.next_fast_len(w + 2 * self.pad, real=True),
        )
        self.kf = np.stack(
            [sfft.rfft2(_embed(k, self.fshape), workers=-1) for k in kernels]
        ).astype(np.complex64)

    def __len__(self) -> int:
        return len(self.kf)

    def spectrum(self, image: np.ndarray) -> np.ndarray:
        """Spectrum of one image (H, W) or a stack (C, H, W)."""
        p = self.pad
        pad = [(0, 0)] * (image.ndim - 2) + [(p, p), (p, p)]
        padded = np.pad(image.astype(np.float32, copy=False), pad, mode="reflect")
        return sfft.rfft2(padded, s=self.fshape, workers=-1).astype(np.complex64)

    def _crop(self, full: np.ndarray) -> np.ndarray:
        p = self.pad
        h, w = self.shape
        return full[..., p : p + h, p : p + w]

    def apply(self, spec: np.ndarray, idx=slice(None)) -> np.ndarray:
        """Convolve the image whose spectrum is ``spec`` with kernels ``idx``."""
        out = sfft.irfft2(spec * self.kf[idx], s=self.fshape, workers=-1)
        return np.ascontiguousarray(self._crop(out), dtype=np.float32)

    def apply_mixed(self, specs: np.ndarray, routing: list[list[tuple[int, int]]]) -> np.ndarray:
        """Multi-channel convolution: output o = sum_{(c, k) in routing[o]} spec_c * K_k."""
        outs = []
        for terms in routing:
            acc = None
            for c, k in terms:
                t = specs[c] * self.kf[k]
                acc = t if acc is None else acc + t
            outs.append(acc)
        full = sfft.irfft2(np.stack(outs), s=self.fshape, workers=-1)
        return np.ascontiguousarray(self._crop(full), dtype=np.float32)
