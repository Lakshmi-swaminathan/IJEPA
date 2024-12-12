# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from multiprocessing import Value
from logging import getLogger
import torch

logger = getLogger(__name__)

class MaskCollator:
    """
    A collator that samples encoder and predictor masks for a batch of images.

    This class divides input images into patch grids and samples masks for both 
    'encoder' and 'predictor' tasks. It supports configuration of mask size, 
    aspect ratio, and shape (rectangular or elliptical).

    Attributes:
        input_size (tuple): Spatial size of the input images (height, width).
        patch_size (int): The size of each patch along one dimension.
        enc_mask_scale (tuple): (min_scale, max_scale) for encoder masks.
        pred_mask_scale (tuple): (min_scale, max_scale) for predictor masks.
        aspect_ratio (tuple): (min_ar, max_ar) aspect ratio for masks.
        nenc (int): Number of encoder masks sampled per image.
        npred (int): Number of predictor masks sampled per image.
        min_keep (int): Minimum number of patches that must remain unmasked.
        allow_overlap (bool): Whether encoder and predictor masks may overlap.
        mask_shape (str): Shape of the mask to be sampled: 'rectangle' or 'ellipse'.
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        mask_shape='ellipse',  # default to ellipse
        max_retries=50
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        
        # Parameter validation
        if patch_size <= 0 or patch_size > input_size[0] or patch_size > input_size[1]:
            raise ValueError("patch_size must be positive and no larger than input dimensions.")
        if not (0 < enc_mask_scale[0] <= enc_mask_scale[1] <= 1):
            raise ValueError("enc_mask_scale values must be between 0 and 1 and min <= max.")
        if not (0 < pred_mask_scale[0] <= pred_mask_scale[1] <= 1):
            raise ValueError("pred_mask_scale values must be between 0 and 1 and min <= max.")
        if not (0 < aspect_ratio[0] <= aspect_ratio[1]):
            raise ValueError("aspect_ratio must have positive values with min <= max.")
        if mask_shape not in ['rectangle', 'ellipse']:
            raise ValueError("mask_shape must be either 'rectangle' or 'ellipse'.")

        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self.mask_shape = mask_shape
        self.max_retries = max_retries

        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        
        # Sample block aspect ratio
        min_ar, max_ar = aspect_ratio_scale
        ar = min_ar + _rand * (max_ar - min_ar)
        
        # Compute block height and width given scale and aspect ratio
        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))

        # Ensure h and w fit within the grid
        h = min(h, self.height)
        w = min(w, self.width)

        # If h or w is too large, decrement
        while h >= self.height and h > 0:
            h -= 1
        while w >= self.width and w > 0:
            w -= 1

        # Ensure at least 1x1
        h = max(h, 1)
        w = max(w, 1)

        return (h, w)

    def _create_rectangular_mask(self, h, w, top, left):
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask[top:top+h, left:left+w] = 1
        return mask

    def _create_elliptical_mask(self, h, w, top, left):
        """
        Creates an elliptical mask within the bounding box defined by
        (top, left) and (h, w).

        The ellipse is centered in the middle of the box and its axes are aligned 
        with the patch grid. The ellipse equation:
        
        ((x - cx)^2 / (rx^2) + (y - cy)^2 / (ry^2)) <= 1

        where (cx, cy) is the center, rx and ry are the ellipse radii.
        """
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)

        y_coords = torch.arange(self.height).unsqueeze(1).float()
        x_coords = torch.arange(self.width).float()

        # Ellipse center
        cy = top + h/2.0
        cx = left + w/2.0

        # Radii
        ry = h / 2.0
        rx = w / 2.0

        # Compute ellipse mask
        # Condition: ((x - cx)^2 / rx^2 + (y - cy)^2 / ry^2) <= 1
        ellipse_area = (((x_coords - cx)**2) / (rx**2)) + (((y_coords - cy)**2) / (ry**2))
        mask[ellipse_area <= 1] = 1
        return mask

    def _create_mask(self, shape, h, w, top, left):
        if shape == 'rectangle':
            return self._create_rectangular_mask(h, w, top, left)
        elif shape == 'ellipse':
            return self._create_elliptical_mask(h, w, top, left)
        else:
            raise NotImplementedError(f"Unknown shape: {shape}")

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        tries = 0

        while tries < self.max_retries:
            top = torch.randint(0, self.height - h + 1, (1,)).item()
            left = torch.randint(0, self.width - w + 1, (1,)).item()

            # Create mask based on specified shape
            mask = self._create_mask(self.mask_shape, h, w, top, left)

            # Constrain mask if acceptable regions are provided
            if acceptable_regions is not None and len(acceptable_regions) > 0:
                for region_mask in acceptable_regions:
                    mask = mask * region_mask

            nonzero_elements = torch.nonzero(mask.flatten())
            if len(nonzero_elements) > self.min_keep:
                # Valid mask found
                mask_complement = 1 - mask
                return nonzero_elements.squeeze(), mask_complement

            tries += 1
            if tries % 10 == 0:
                logger.warning(f"Mask generation still failing after {tries} tries.")

        # Fall back if no valid mask after max_retries
        logger.error("Failed to generate a valid mask after maximum retries. Returning fallback mask.")
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask[0, 0] = 1
        mask_complement = 1 - mask
        return torch.nonzero(mask.flatten()).squeeze(), mask_complement

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch:
        
        Steps:
        1. Use a global step-based seed to sample block sizes for encoder & predictor masks.
        2. For each image in the batch:
           - Sample pred masks using pre-determined block size and shape.
           - Sample enc masks similarly, possibly constrained by acceptable regions 
             from the pred masks if allow_overlap=False.
        3. Return collated batch and masks.
        """
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Sample block sizes (deterministic per step)
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(B):
            masks_p, masks_C = [], []
            # Predictor masks
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = None if self.allow_overlap else masks_C

            # Encoder masks
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Truncate masks to smallest found keep size for consistent batch shape
        collated_masks_pred = [[m[:min_keep_pred] for m in mp] for mp in collated_masks_pred]
        collated_masks_enc = [[m[:min_keep_enc] for m in me] for me in collated_masks_enc]

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
