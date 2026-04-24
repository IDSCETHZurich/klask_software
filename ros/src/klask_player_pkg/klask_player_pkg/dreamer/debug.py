"""Debug utilities for Dreamer inference — visualize image tensors with OpenCV."""

import cv2
import numpy as np
import torch


def show_image_tensor(tensor: torch.Tensor, window_name: str = "dreamer_debug") -> None:
    """Show an image tensor in an OpenCV window.

    Accepts shape (1, H, W, 3) or (H, W, 3), RGB, float in [0, 255] or [0, 1],
    or uint8. Batch dim is dropped; floats are rescaled and clipped; RGB is
    converted to BGR. Uses a non-blocking waitKey(1) so this can be called
    from the inference loop.
    """
    img = tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img = img.numpy()

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img_bgr)
    cv2.waitKey(1)
