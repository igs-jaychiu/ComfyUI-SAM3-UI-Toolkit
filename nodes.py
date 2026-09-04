import json
import os

import torch
import torch.nn.functional as F

from . import auto_filter


def _masks_to_bool_list(masks, size=None):
    """MASK tensor (B,H,W) -> list of bool numpy arrays, resized to `size` (H,W) if given."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if size is not None and masks.shape[-2:] != tuple(size):
        masks = F.interpolate(masks.unsqueeze(1), size=tuple(size), mode="nearest").squeeze(1)
    return [m >= 0.5 for m in masks.detach().cpu().numpy()]


def _bool_list_to_masks(bool_masks, like, size):
    import numpy as np

    if not bool_masks:
        return torch.zeros((1, size[0], size[1]), dtype=like.dtype, device=like.device)
    stacked = np.stack(bool_masks).astype("float32")
    return torch.from_numpy(stacked).to(dtype=like.dtype, device=like.device)


def _image_to_uint8(image):
    """First image of an IMAGE tensor -> uint8 RGB numpy."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    rgb = image[0, ..., :3].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
    return rgb.round().astype("uint8")


def _pad_to_square(image, mask):
    """Pad BCHW tensors on the right/bottom so MAT sees a square crop."""
    _, _, height, width = image.shape
    side = max(height, width)
    pad_right = side - width
    pad_bottom = side - height
    if pad_right or pad_bottom:
        # Replication is stable for UI gradients and also works for tiny crops.
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")
        mask = F.pad(mask, (0, pad_right, 0, pad_bottom), mode="constant", value=0)
    return image, mask, (height, width, side)


def _resize_for_mat(image, mask, size=512):
    image, mask, original = _pad_to_square(image, mask)
    if image.shape[-1] != size:
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=(size, size), mode="nearest")
    return image, mask, original


def _restore_from_mat(image, original):
    height, width, side = original
    if image.shape[-2:] != (side, side):
        image = F.interpolate(image, size=(side, side), mode="bilinear", align_corners=False)
    return image[:, :, :height, :width]


class SAM3BatchCropToObjects:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "padding": (
                    "INT",
                    {"default": 2, "min": 0, "max": 256, "step": 1},
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.031, "min": 0.001, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGES", "MASKS", "X", "Y", "WIDTH", "HEIGHT")
    OUTPUT_IS_LIST = (True, True, True, True, True, True)
    FUNCTION = "crop_batch"
    CATEGORY = "image/crop"

    def crop_batch(self, images, masks, padding=2, alpha_threshold=0.031):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        image_count = images.shape[0]
        mask_count = masks.shape[0]
        if image_count != mask_count:
            if image_count == 1:
                images = images.repeat(mask_count, 1, 1, 1)
            elif mask_count == 1:
                masks = masks.repeat(image_count, 1, 1)
            else:
                raise ValueError(
                    f"Image/mask batch mismatch: {image_count} images, {mask_count} masks"
                )

        cropped_images = []
        cropped_masks = []
        xs, ys, widths, heights = [], [], [], []

        for image, mask in zip(images, masks):
            foreground = mask >= alpha_threshold
            coordinates = torch.nonzero(foreground, as_tuple=False)
            if coordinates.numel() == 0:
                continue

            y1 = max(0, int(coordinates[:, 0].min().item()) - padding)
            y2 = min(mask.shape[0], int(coordinates[:, 0].max().item()) + 1 + padding)
            x1 = max(0, int(coordinates[:, 1].min().item()) - padding)
            x2 = min(mask.shape[1], int(coordinates[:, 1].max().item()) + 1 + padding)

            cropped_images.append(image[y1:y2, x1:x2, :].unsqueeze(0))
            cropped_masks.append(mask[y1:y2, x1:x2].unsqueeze(0))
            xs.append(x1)
            ys.append(y1)
            widths.append(x2 - x1)
            heights.append(y2 - y1)

        if not cropped_images:
            raise ValueError("No non-empty masks were available to crop")

        return cropped_images, cropped_masks, xs, ys, widths, heights


class SAM3MergeMaskBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MERGED_MASK",)
    FUNCTION = "merge_masks"
    CATEGORY = "image/mask"

    def merge_masks(self, masks):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        return (torch.amax(masks, dim=0, keepdim=True),)


class SAM3SelectionOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "opacity": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "color": (
                    "STRING",
                    {"default": "#ffd400", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("PREVIEW",)
    FUNCTION = "overlay"
    CATEGORY = "image/detection"

    def overlay(self, image, mask, opacity=0.35, color="#ffd400"):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.shape[-2:] != image.shape[1:3]:
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        batch_size = max(image.shape[0], mask.shape[0])
        if image.shape[0] == 1 and batch_size > 1:
            image = image.repeat(batch_size, 1, 1, 1)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.repeat(batch_size, 1, 1)

        value = color.strip().lstrip("#")
        if len(value) != 6:
            raise ValueError("Overlay color must use #RRGGBB format")
        rgb = torch.tensor(
            [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
            dtype=image.dtype,
            device=image.device,
        ).view(1, 1, 1, 3)

        alpha = mask.clamp(0.0, 1.0).unsqueeze(-1) * float(opacity)
        preview = image[..., :3] * (1.0 - alpha) + rgb * alpha
        return (preview,)


class SAM3MATInpaintSequence:
    """Inpaint every SAM3 object separately, then composite it back in order.

    Processing small crops gives MAT enough resolution for thin UI text and avoids
    letting a generative model redesign every label on the full screen at once.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpaint_model": ("INPAINT_MODEL",),
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "context_padding": (
                    "INT",
                    {"default": 64, "min": 8, "max": 512, "step": 8},
                ),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("INPAINTED_IMAGE",)
    FUNCTION = "inpaint_sequence"
    CATEGORY = "image/inpaint"

    def inpaint_sequence(
        self,
        inpaint_model,
        image,
        masks,
        context_padding=64,
        mask_threshold=0.5,
        seed=0,
    ):
        from comfy import model_management

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        # The staged workflow passes one canvas forward at a time.
        result = image[:1, ..., :3].clone()
        canvas_height, canvas_width = result.shape[1:3]
        if masks.shape[-2:] != (canvas_height, canvas_width):
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(canvas_height, canvas_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        device = model_management.get_torch_device()
        inpaint_model.to(device)
        try:
            for index, source_mask in enumerate(masks):
                binary = source_mask >= float(mask_threshold)
                coordinates = torch.nonzero(binary, as_tuple=False)
                if coordinates.numel() == 0:
                    continue

                y1 = max(0, int(coordinates[:, 0].min().item()) - int(context_padding))
                y2 = min(canvas_height, int(coordinates[:, 0].max().item()) + 1 + int(context_padding))
                x1 = max(0, int(coordinates[:, 1].min().item()) - int(context_padding))
                x2 = min(canvas_width, int(coordinates[:, 1].max().item()) + 1 + int(context_padding))

                crop = result[:, y1:y2, x1:x2, :].permute(0, 3, 1, 2)
                crop_mask = binary[y1:y2, x1:x2].to(crop.dtype).unsqueeze(0).unsqueeze(0)
                work_image, work_mask, original = _resize_for_mat(crop, crop_mask, 512)

                torch.manual_seed(int(seed) + index)
                generated = inpaint_model(work_image.to(device), work_mask.to(device))
                generated = _restore_from_mat(generated, original).to(result.device)

                # Preserve every pixel outside the requested mask exactly.
                alpha = crop_mask.to(result.device)
                original_crop = crop.to(result.device)
                composited = original_crop + (generated - original_crop) * alpha
                result[:, y1:y2, x1:x2, :] = composited.permute(0, 2, 3, 1)
        finally:
            inpaint_model.cpu()

        return (result.clamp(0.0, 1.0),)


def _parse_mask_indices(value, count):
    indices = set()
    for token in str(value).replace(" ", "").split(","):
        if not token:
            continue
        try:
            if "-" in token:
                start, end = token.split("-", 1)
                start, end = int(start), int(end)
                if start > end:
                    start, end = end, start
                indices.update(range(start, end + 1))
            else:
                indices.add(int(token))
        except ValueError as error:
            raise ValueError(
                f"Invalid mask index '{token}'. Use values such as 2,5,8-10."
            ) from error
    return {index for index in indices if 1 <= index <= count}


def _grow_mask_batch(masks, pixels):
    if pixels <= 0:
        return masks
    kernel = int(pixels) * 2 + 1
    return F.max_pool2d(
        masks.unsqueeze(1), kernel_size=kernel, stride=1, padding=int(pixels)
    ).squeeze(1)


class SAM3MaskReview:
    """Number SAM3 masks, exclude unwanted instances, and preview the real fill mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "exclude_indices": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "grow_pixels": (
                    "INT",
                    {"default": 4, "min": 0, "max": 64, "step": 1},
                ),
                "opacity": (
                    "FLOAT",
                    {"default": 0.32, "min": 0.0, "max": 0.85, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("NUMBERED_PREVIEW", "ASSET_MASKS", "FILL_MASKS", "SUMMARY")
    FUNCTION = "review"
    CATEGORY = "image/detection"

    def review(self, image, masks, exclude_indices="", grow_pixels=4, opacity=0.32):
        from PIL import Image, ImageDraw, ImageFont

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[-2:] != image.shape[1:3]:
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=image.shape[1:3],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        excluded = _parse_mask_indices(exclude_indices, masks.shape[0])
        kept_indices = [i for i in range(masks.shape[0]) if i + 1 not in excluded]
        if not kept_indices:
            raise ValueError("All SAM3 masks were excluded; keep at least one mask.")

        asset_masks = masks[kept_indices].clamp(0.0, 1.0)
        fill_masks = _grow_mask_batch(asset_masks, int(grow_pixels)).clamp(0.0, 1.0)
        union = torch.amax(fill_masks, dim=0).clamp(0.0, 1.0)

        base = image[0, ..., :3].detach().cpu().clamp(0.0, 1.0)
        preview = base * (1.0 - union.unsqueeze(-1) * float(opacity))
        yellow = torch.tensor([1.0, 0.82, 0.0], dtype=base.dtype)
        preview += yellow * union.unsqueeze(-1) * float(opacity)
        array = (preview.numpy() * 255.0).round().astype("uint8")
        canvas = Image.fromarray(array, mode="RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for batch_index, mask in enumerate(masks):
            coords = torch.nonzero(mask >= 0.5, as_tuple=False)
            if coords.numel() == 0:
                continue
            y1 = int(coords[:, 0].min().item())
            y2 = int(coords[:, 0].max().item())
            x1 = int(coords[:, 1].min().item())
            x2 = int(coords[:, 1].max().item())
            number = batch_index + 1
            is_excluded = number in excluded
            color = (235, 55, 55) if is_excluded else (25, 220, 100)
            label = f"X{number}" if is_excluded else f"#{number}"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            box = draw.textbbox((x1, max(0, y1 - 13)), label, font=font)
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((x1, max(0, y1 - 13)), label, fill=color, font=font)

        output = torch.from_numpy(
            __import__("numpy").asarray(canvas).copy()
        ).to(dtype=image.dtype, device=image.device).unsqueeze(0) / 255.0
        summary = (
            f"Detected {masks.shape[0]} | kept {len(kept_indices)} | "
            f"excluded {','.join(map(str, sorted(excluded))) or 'none'} | "
            f"fill grow {int(grow_pixels)}px"
        )
        return (output, asset_masks, fill_masks, summary)


class SAM3ApprovalGate:
    """Stop a global queue after the preview until the user explicitly approves it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preview_image": ("IMAGE",),
                "image": ("IMAGE",),
                "asset_masks": ("MASK",),
                "fill_masks": ("MASK",),
                "approved": ("BOOLEAN", {"default": False}),
                "stage_name": (
                    "STRING",
                    {"default": "stage", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "ASSET_MASKS", "FILL_MASKS")
    FUNCTION = "approve"
    CATEGORY = "image/workflow"

    def approve(self, preview_image, image, asset_masks, fill_masks, approved=False, stage_name="stage"):
        if not approved:
            raise RuntimeError(
                f"{stage_name}: preview was saved. Review mask numbers, set exclusions, "
                "then enable approved and run again."
            )
        return (image, asset_masks, fill_masks)


class SAM3MaskBatchConcat:
    """Concatenate up to four MASK batches (e.g. several SAM3 prompts) into one batch."""

    SLOTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(2, cls.SLOTS + 1):
            optional[f"masks_{i}"] = ("MASK",)
        for i in range(1, cls.SLOTS + 1):
            optional[f"labels_in_{i}"] = ("STRING", {"forceInput": True})
        return {
            "required": {
                "masks_1": ("MASK",),
                "label_names": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("MASKS", "LABELS_JSON")
    FUNCTION = "concat"
    CATEGORY = "image/mask"

    def concat(self, masks_1, label_names="", **kwargs):
        """Concatenate mask batches and carry a per-mask source label alongside.

        `label_names` is a comma separated name per input slot; `labels_in_N` accepts a
        LABELS_JSON from an upstream concat so labels survive chaining.
        """
        names = [s.strip() for s in label_names.split(",")] if label_names.strip() else []
        incoming = [masks_1] + [kwargs.get(f"masks_{i}") for i in range(2, self.SLOTS + 1)]
        batches, labels = [], []
        for slot, masks in enumerate(incoming):
            if masks is None:
                continue
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if batches and masks.shape[-2:] != batches[0].shape[-2:]:
                masks = F.interpolate(
                    masks.unsqueeze(1), size=batches[0].shape[-2:], mode="nearest"
                ).squeeze(1)
            masks = masks.to(batches[0].device) if batches else masks
            batches.append(masks)
            upstream = kwargs.get(f"labels_in_{slot + 1}")
            if upstream:
                try:
                    parsed = json.loads(upstream)
                except (TypeError, ValueError):
                    parsed = []
                if len(parsed) == masks.shape[0]:
                    labels.extend(str(x) for x in parsed)
                    continue
            name = names[slot] if slot < len(names) else f"src{slot + 1}"
            labels.extend([name] * masks.shape[0])
        return (torch.cat(batches, dim=0), json.dumps(labels))


class SAM3AutoFilterMasks:
    """Turn a raw SAM3 individual-mask batch into one clean mask per UI element, no manual review.

    Removes duplicates, masks that are really an excluded reference element (a text prompt that
    returned a whole button), fragments inside a larger kept mask (single glyphs), and optionally
    re-assembles text rows and closes holes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "dedupe_iou": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
                "drop_contained": ("BOOLEAN", {"default": True}),
                "contain_ratio": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
                "min_area": ("INT", {"default": 30, "min": 1, "max": 100000, "step": 1}),
                "max_area_frac": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_fill": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exclude_overlap": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01}),
                "row_merge": ("BOOLEAN", {"default": False}),
                "merge_gap_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "split_gap_ratio": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "close_holes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "exclude_masks_1": ("MASK",),
                "exclude_masks_2": ("MASK",),
                "exclude_masks_3": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "STRING")
    RETURN_NAMES = ("FILTERED_MASKS", "COUNT", "SUMMARY")
    FUNCTION = "filter"
    CATEGORY = "image/detection"

    def filter(self, masks, dedupe_iou=0.85, drop_contained=True, contain_ratio=0.85, min_area=30,
               max_area_frac=0.5, min_fill=0.0, exclude_overlap=0.5, row_merge=False,
               merge_gap_ratio=1.0, split_gap_ratio=1.5, close_holes=False,
               exclude_masks_1=None, exclude_masks_2=None, exclude_masks_3=None):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        size = masks.shape[-2:]
        raw = _masks_to_bool_list(masks)
        exclude = []
        for ref in (exclude_masks_1, exclude_masks_2, exclude_masks_3):
            if ref is not None:
                exclude.extend(_masks_to_bool_list(ref, size))
        kept, summary = auto_filter.filter_masks(
            raw, exclude=exclude, dedupe_iou=float(dedupe_iou), drop_contained=bool(drop_contained),
            contain_ratio=float(contain_ratio), min_area=int(min_area), max_area_frac=float(max_area_frac),
            min_fill=float(min_fill), exclude_overlap=float(exclude_overlap), row_merge=bool(row_merge),
            merge_gap_ratio=float(merge_gap_ratio), split_gap_ratio=float(split_gap_ratio),
            close_holes=bool(close_holes),
        )
        return (_bool_list_to_masks(kept, masks, size), len(kept), summary)


DEFAULT_PROMPT_BANK = """# name | prompt | threshold      (lines starting with # are ignored)
text_a   | text:30                                              | 0.25
text_b   | text:60                                              | 0.24
label_a  | text label:20, number:10                             | 0.25
label_b  | text label:30, number:20                             | 0.24
icon_a   | icon:20                                              | 0.28
icon_b   | icon:30                                              | 0.26
round_a  | circular icon button:10                              | 0.30
round_b  | circular button:15, round badge:12                   | 0.28
button_a | button:12                                            | 0.30
button_b | button:15                                            | 0.28
banner_a | ribbon banner:8                                      | 0.30
banner_b | ribbon banner:10, title plate:6                      | 0.28
card_a   | card:10                                              | 0.30
card_b   | card:12, table:6                                     | 0.28
panel_a  | panel:8                                              | 0.30
panel_b  | panel:10, window frame:4                             | 0.28
char_a   | cartoon character:4, mascot:3                        | 0.30
char_b   | cartoon character:5, mascot:4, animal:4              | 0.28
object_a | decorative object:15                                 | 0.30
object_b | decorative object:20                                 | 0.28
coin_a   | coin:10, gem:6, star:8                               | 0.30
coin_b   | coin:12, gem:8, star:10, badge:10                    | 0.28
bar_a    | slider:6, progress bar:6, toggle switch:4            | 0.30
bar_b    | slider:8, progress bar:8, toggle switch:6, scroll bar:4 | 0.28
avatar_a | avatar portrait:8                                    | 0.30
avatar_b | avatar portrait:10                                   | 0.28
row_a    | table row:10, list row:10                            | 0.30
nature_b | food:8, plant:6, lantern:6, moon:2                   | 0.28
"""


def _parse_prompt_bank(text):
    """Parse 'name | prompt | threshold' lines into probe tuples."""
    probes = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) == 1:
            name, prompt, thr = f"p{len(probes) + 1}", parts[0], 0.28
        elif len(parts) == 2:
            name, prompt, thr = parts[0], parts[1], 0.28
        else:
            try:
                thr = float(parts[2])
            except ValueError:
                thr = 0.28
            name, prompt = parts[0], parts[1]
        if prompt:
            probes.append((name, prompt, thr))
    return probes


class SAM3PromptBank:
    """Run a whole bank of SAM3 text prompts in one node and pool the masks.

    Wiring one CLIPTextEncode plus one SAM3_Detect per prompt made the prompt list a rewiring
    job and over half the graph. Here the list is a text field: edit a line, rerun. The pooled
    output feeds SAM3AutoLayerMasks directly, and every mask keeps the probe name that found it
    so the consensus vote and the asset labels stay meaningful.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "prompts": ("STRING", {"default": DEFAULT_PROMPT_BANK, "multiline": True}),
                "refine_iterations": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "threshold_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "STRING")
    RETURN_NAMES = ("MASKS", "LABELS_JSON", "SUMMARY")
    FUNCTION = "run"
    CATEGORY = "image/detection"

    @staticmethod
    def _unwrap(result):
        """Return a plain tuple from whatever a node hands back.

        Core nodes on the V3 schema return a NodeOutput wrapper rather than a tuple, so reach
        through it instead of assuming either shape.
        """
        if isinstance(result, (tuple, list)):
            return tuple(result)
        for attr in ("result", "results", "values", "outputs"):
            value = getattr(result, attr, None)
            if isinstance(value, (tuple, list)):
                return tuple(value)
        if hasattr(result, "__getitem__"):
            try:
                return (result[0],)
            except Exception:  # noqa: BLE001 - fall through to the single-value case
                pass
        return (result,)

    @staticmethod
    def _detector():
        try:
            from comfy_extras.nodes_sam3 import SAM3_Detect
        except ImportError as error:
            raise RuntimeError(
                "SAM3PromptBank needs the built-in SAM3 nodes (comfy_extras.nodes_sam3). "
                "Update ComfyUI to a build that ships SAM3."
            ) from error
        instance = SAM3_Detect()
        entry = getattr(SAM3_Detect, "FUNCTION", None)
        if not entry or not hasattr(instance, entry):
            raise RuntimeError("SAM3_Detect does not expose its FUNCTION entry point")
        return getattr(instance, entry)

    def run(self, model, clip, image, prompts, refine_iterations=0, threshold_scale=1.0):
        probes = _parse_prompt_bank(prompts)
        if not probes:
            raise ValueError("SAM3PromptBank: no usable prompt lines")
        detect = self._detector()
        if image.ndim == 3:
            image = image.unsqueeze(0)

        batches, labels, notes = [], [], []
        for name, prompt, thr in probes:
            tokens = clip.tokenize(prompt)
            if hasattr(clip, "encode_from_tokens_scheduled"):
                conditioning = clip.encode_from_tokens_scheduled(tokens)
            else:  # older cores
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                conditioning = [[cond, {"pooled_output": pooled}]]
            result = detect(model=model, image=image, conditioning=conditioning,
                            threshold=float(thr) * float(threshold_scale),
                            refine_iterations=int(refine_iterations), individual_masks=True)
            unwrapped = self._unwrap(result)
            masks = unwrapped[0] if unwrapped else None
            if masks is not None and not hasattr(masks, "ndim"):
                masks = None
            if masks is None:
                notes.append(f"{name}:0")
                continue
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if batches and masks.shape[-2:] != batches[0].shape[-2:]:
                masks = F.interpolate(masks.unsqueeze(1), size=batches[0].shape[-2:],
                                      mode="nearest").squeeze(1)
            masks = masks.to(batches[0].device) if batches else masks
            batches.append(masks)
            labels.extend([name.split("_")[0]] * masks.shape[0])
            notes.append(f"{name}:{masks.shape[0]}")

        if not batches:
            raise ValueError("SAM3PromptBank: every prompt returned zero masks")
        pooled = torch.cat(batches, dim=0)
        summary = f"probes={len(probes)} masks={pooled.shape[0]} | " + " ".join(notes)
        return (pooled, json.dumps(labels), summary)


class SAM3AutoLayerMasks:
    """Pool SAM3 masks from many prompts and split them into z-order layers automatically.

    Layer 1 holds the leaf elements (text, icons, props), layer 2 the plates and buttons that
    contain them, layer 3 the cards, and so on. The split comes from mask containment, not from
    which prompt found what, so the same node works on any layout without per-image tuning.
    """

    MAX_LAYERS = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "dedupe_iou": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 1.0, "step": 0.01}),
                "contain_ratio": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01}),
                "min_area": ("INT", {"default": 40, "min": 1, "max": 1000000, "step": 1}),
                "max_area_frac": ("FLOAT", {"default": 0.98, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_fill": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_dim": ("INT", {"default": 6, "min": 1, "max": 512, "step": 1}),
                "close_holes_from": ("INT", {"default": 3, "min": 0, "max": 8, "step": 1}),
                "min_votes": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "despeckle_frac": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "labels_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MASK",) * 8 + ("STRING", "STRING")
    RETURN_NAMES = tuple(f"LAYER_{i}" for i in range(1, 9)) + ("META_JSON", "SUMMARY")
    FUNCTION = "split"
    CATEGORY = "image/detection"

    def split(self, masks, dedupe_iou=0.8, contain_ratio=0.85, min_area=40, max_area_frac=0.98,
              min_fill=0.0, min_dim=6, close_holes_from=3, min_votes=2, despeckle_frac=0.06,
              labels_json=""):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        size = masks.shape[-2:]
        raw = _masks_to_bool_list(masks)
        name_list = None
        if labels_json:
            try:
                parsed = json.loads(labels_json)
            except (TypeError, ValueError):
                parsed = []
            if len(parsed) == len(raw):
                name_list = [str(x) for x in parsed]
        layers, layer_labels, summary, layer_meta = auto_filter.auto_layers(
            raw, labels=name_list, dedupe_iou=float(dedupe_iou), contain_ratio=float(contain_ratio),
            min_area=int(min_area), max_area_frac=float(max_area_frac), min_fill=float(min_fill),
            min_dim=int(min_dim), max_layers=self.MAX_LAYERS, close_holes_from=int(close_holes_from),
            min_votes=int(min_votes), despeckle_frac=float(despeckle_frac),
        )
        while len(layers) < self.MAX_LAYERS:
            layers.append([])
            layer_labels.append([])
            layer_meta.append([])
        outs = [_bool_list_to_masks(layer, masks, size) for layer in layers[:self.MAX_LAYERS]]
        payload = json.dumps({f"layer_{i + 1}": layer_meta[i] for i in range(self.MAX_LAYERS)},
                             ensure_ascii=False)
        return (*outs, payload, summary)


class SAM3NineSlice:
    """Work out 9-slice borders for each extracted sprite and record them.

    A button exported as a flat PNG can only ever be drawn at the size it was cut. With borders
    the engine redraws it at any width while the rounded corners and bevel stay pixel-exact.
    Sprites that cannot be stretched - round icons, text, characters, anything that came out in
    pieces - are reported with a reason instead of a wrong guess.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "min_center": ("INT", {"default": 6, "min": 2, "max": 256, "step": 1}),
                "flat_frac": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_confidence": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_opaque": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "silhouette_tol": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.005}),
                "max_circularity": ("FLOAT", {"default": 0.82, "min": 0.1, "max": 1.0, "step": 0.01}),
                "write_9png": ("BOOLEAN", {"default": False}),
                "prefix": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "coords_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("NINESLICE_JSON", "SUMMARY")
    OUTPUT_IS_LIST = (False, False)
    # its product is a file on disk, so it must run even when nothing consumes its string outputs
    OUTPUT_NODE = True
    FUNCTION = "detect"
    CATEGORY = "image/detection"

    @staticmethod
    def _first(value, default=None):
        """INPUT_IS_LIST wraps every widget in a list; take the single value back out."""
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    @staticmethod
    def _write_nine_patch(path, rgba, borders):
        """Android .9.png: a 1px frame where black marks the stretchable spans."""
        import numpy as np

        height, width = rgba.shape[:2]
        out = np.zeros((height + 2, width + 2, 4), np.uint8)
        out[1:-1, 1:-1] = rgba
        black = np.array([0, 0, 0, 255], np.uint8)
        left, right = borders["left"], borders["right"]
        top, bottom = borders["top"], borders["bottom"]
        if borders["stretch_x"]:
            out[0, 1 + left:1 + width - right] = black          # stretch span
            out[-1, 1 + left:1 + width - right] = black         # content span
        if borders["stretch_y"]:
            out[1 + top:1 + height - bottom, 0] = black
            out[1 + top:1 + height - bottom, -1] = black
        import cv2
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))

    def detect(self, images, min_center=6, flat_frac=0.15, min_confidence=0.55, min_opaque=0.35,
               silhouette_tol=0.04, max_circularity=0.82, write_9png=False, prefix="",
               coords_json=None):
        import numpy as np

        min_center = int(self._first(min_center, 6))
        flat_frac = float(self._first(flat_frac, 0.15))
        min_confidence = float(self._first(min_confidence, 0.55))
        min_opaque = float(self._first(min_opaque, 0.35))
        silhouette_tol = float(self._first(silhouette_tol, 0.04))
        max_circularity = float(self._first(max_circularity, 0.82))
        write_9png = bool(self._first(write_9png, False))
        prefix = str(self._first(prefix, "") or "").strip()

        rows = []
        raw_coords = self._first(coords_json, None)
        if raw_coords:
            try:
                rows = json.loads(raw_coords)
            except (TypeError, ValueError):
                rows = []

        if not isinstance(images, list):
            images = [images]

        out_dir = None
        if write_9png and prefix:
            try:
                import folder_paths

                out_dir = folder_paths.get_output_directory()
            except Exception as error:  # noqa: BLE001 - never fail detection over bookkeeping
                print(f"[SAM3NineSlice] cannot resolve output directory: {error}")

        results = []
        stretchable = 0
        for index, image in enumerate(images, 1):
            tensor = image[0] if image.ndim == 4 else image
            array = (tensor.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
            if array.shape[2] == 3:
                array = np.dstack([array, np.full(array.shape[:2], 255, np.uint8)])
            if array.shape[0] < 2 or array.shape[1] < 2:
                continue
            borders = auto_filter.nine_slice(
                array, min_center=min_center, flat_frac=flat_frac,
                min_confidence=min_confidence, min_opaque=min_opaque,
                silhouette_tol=silhouette_tol, max_circularity=max_circularity)
            record = {"index": index, "w": int(array.shape[1]), "h": int(array.shape[0])}
            if index - 1 < len(rows) and isinstance(rows[index - 1], dict):
                src = rows[index - 1]
                for key in ("uid", "layer", "label", "x", "y"):
                    if key in src:
                        record[key] = src[key]
            record.update(borders)
            results.append(record)
            if borders["nine_slice"]:
                stretchable += 1
                if out_dir:
                    name = record.get("uid") or f"sprite_{index:03d}"
                    target = os.path.join(out_dir, prefix + f"_{name}.9.png")
                    try:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        self._write_nine_patch(target, array, borders)
                    except Exception as error:  # noqa: BLE001
                        print(f"[SAM3NineSlice] could not write {target}: {error}")

        payload = json.dumps(results, indent=1, ensure_ascii=False)
        if prefix and out_dir:
            try:
                target = os.path.join(out_dir, prefix + "_nineslice.json")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(payload)
            except Exception as error:  # noqa: BLE001
                print(f"[SAM3NineSlice] could not write json: {error}")
        summary = f"{stretchable}/{len(results)} sprites are 9-sliceable"
        return (payload, summary)


class SAM3CropToRGBA:
    """Cut every mask out of the image as a transparent RGBA sprite and record its coordinates."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "padding": ("INT", {"default": 2, "min": 0, "max": 256, "step": 1}),
                "feather": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "coords_prefix": ("STRING", {"default": "", "multiline": False}),
                "layer": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "matte": (["difference", "off"],),
                "matte_low": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "matte_high": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01}),
                "align_siblings": ("BOOLEAN", {"default": True}),
                "align_tolerance": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
            # kept optional so a graph saved before these existed still validates
            "optional": {
                "matte_min_coverage": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01}),
                "matte_tight_edge": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 5.0, "step": 0.05}),
                "meta_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("RGBA_IMAGES", "COORDS_JSON")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "crop"
    CATEGORY = "image/crop"

    @classmethod
    def _sibling_sizes_global(cls, meta_json, tolerance=0.12):
        """Group repeated elements across every layer, keyed by uid.

        The three purchase buttons of a shop panel do not all land on the same layer - one holds
        a play icon as well as its text, so it sits a level up. Grouping per layer would miss
        them, while the crop window is free to agree across layers because it has no bearing on
        the peel order.
        """
        try:
            data = json.loads(meta_json)
        except (TypeError, ValueError, AttributeError):
            return {}
        rows = [r for layer in data.values() for r in layer if isinstance(r, dict)]
        if len(rows) < 2:
            return {}
        boxes = [(r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]) for r in rows]
        groups = cls._sibling_sizes(boxes, tolerance)
        return {rows[i].get("uid"): size for i, size in groups.items() if rows[i].get("uid")}

    @staticmethod
    def _sibling_sizes(boxes, tolerance=0.12):
        """Find repeated layout elements and agree one crop size for each group.

        Three columns of the same card come back 214, 200 and 217 px wide because the masks are
        traced independently. Anything that is the same element repeated across a row or column
        should export at one size, or the sprites will not line up in the game.
        """
        groups = {}
        idx = [i for i, b in enumerate(boxes) if b is not None]
        used = set()
        for i in idx:
            if i in used:
                continue
            bi = boxes[i]
            wi, hi = bi[2] - bi[0], bi[3] - bi[1]
            members = [i]
            for j in idx:
                if j == i or j in used:
                    continue
                bj = boxes[j]
                wj, hj = bj[2] - bj[0], bj[3] - bj[1]
                if abs(wj - wi) > tolerance * max(wi, wj):
                    continue
                if abs(hj - hi) > tolerance * max(hi, hj):
                    continue
                # a repeated element sits on a shared baseline (a row) or a shared edge (a column)
                same_row = abs(bj[1] - bi[1]) <= 0.25 * max(hi, hj)
                same_col = abs(bj[0] - bi[0]) <= 0.25 * max(wi, wj)
                if same_row or same_col:
                    members.append(j)
            if len(members) < 2:
                continue
            tw = max(boxes[m][2] - boxes[m][0] for m in members)
            th = max(boxes[m][3] - boxes[m][1] for m in members)
            for m in members:
                groups[m] = (tw, th)
                used.add(m)
        return groups

    def crop(self, image, masks, padding=2, feather=1, coords_prefix="", layer=0,
             matte="difference", matte_low=0.10, matte_high=0.35,
             matte_min_coverage=0.40, matte_tight_edge=0.90,
             align_siblings=True, align_tolerance=0.12, meta_json=""):
        import cv2
        import numpy as np

        if image.ndim == 3:
            image = image.unsqueeze(0)
        height, width = image.shape[1:3]
        rgb = _image_to_uint8(image)
        bool_masks = _masks_to_bool_list(masks, (height, width))
        # the layer metadata from SAM3AutoLayerMasks lines up 1:1 with this layer's mask batch,
        # so uid / label / votes / parent can travel with the sprite coordinates
        meta_rows = []
        if meta_json and layer:
            try:
                meta_rows = json.loads(meta_json).get(f"layer_{int(layer)}", [])
            except (TypeError, ValueError, AttributeError):
                meta_rows = []
        boxes = [auto_filter.bbox(m) for m in bool_masks]
        target = {}
        if align_siblings:
            by_uid = self._sibling_sizes_global(meta_json, float(align_tolerance)) if meta_json else {}
            if by_uid and meta_rows:
                for i, row in enumerate(meta_rows):
                    size = by_uid.get(row.get("uid"))
                    if size:
                        target[i] = size
            else:
                target = self._sibling_sizes(boxes, float(align_tolerance))

        images = []
        coords = []
        for index, mask in enumerate(bool_masks, 1):
            box = boxes[index - 1]
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            if (index - 1) in target:
                tw, th = target[index - 1]
                # grow the crop window symmetrically to the group size, so three cards that are
                # the same card in the layout come out as three same-sized sprites
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                nx1, nx2 = int(round(cx - tw / 2.0)), int(round(cx + tw / 2.0))
                ny1, ny2 = int(round(cy - th / 2.0)), int(round(cy + th / 2.0))
                if nx1 >= 0 and ny1 >= 0 and nx2 <= width and ny2 <= height:
                    x1, y1, x2, y2 = nx1, ny1, nx2, ny2
            if matte == "difference":
                # SAM3 hands back text as a filled plate; recover the real glyph shape by
                # measuring how far each pixel departs from an estimate of what is behind it
                soft = auto_filter.difference_matte(
                    rgb, mask, low=float(matte_low), high=float(matte_high),
                    min_coverage=float(matte_min_coverage), tight_edge=float(matte_tight_edge))
                alpha = (soft[y1:y2, x1:x2] * 255).round().astype(np.uint8)
            else:
                alpha = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
            if feather > 0:
                alpha = cv2.GaussianBlur(alpha, (2 * feather + 1, 2 * feather + 1), 0)
            rgba = np.dstack([rgb[y1:y2, x1:x2], alpha]).astype(np.float32) / 255.0
            images.append(torch.from_numpy(rgba).to(dtype=image.dtype, device=image.device).unsqueeze(0))
            record = {"index": index, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
            if index - 1 < len(meta_rows):
                m = meta_rows[index - 1]
                record.update({"uid": m.get("uid"), "layer": m.get("layer"),
                               "label": m.get("label"), "votes": m.get("votes"),
                               "parent": m.get("parent"), "area": m.get("area")})
            coords.append(record)
        if not images:
            # An empty layer is normal in a generic pipeline; emit a 1x1 transparent stub so the
            # graph keeps running instead of aborting the whole extraction.
            images = [torch.zeros((1, 1, 1, 4), dtype=image.dtype, device=image.device)]
        payload = json.dumps(coords, indent=1, ensure_ascii=False)
        if coords_prefix.strip():
            try:
                import folder_paths

                out_dir = folder_paths.get_output_directory()
                target = os.path.join(out_dir, coords_prefix.strip() + "_coords.json")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(payload)
            except Exception as error:  # noqa: BLE001 - never fail the crop because of bookkeeping
                print(f"[SAM3CropToRGBA] could not write coords file: {error}")
        return (images, payload)


class SAM3DeterministicInpaint:
    """Remove masked UI content without diffusion or generative hallucinations.

    `interp` (recommended) fills each hole by edge-aware linear interpolation from its
    surroundings and smooths the interior; `grow` / `shadow_reach` enlarge the fill mask so that
    anti-aliased edges and soft drop shadows disappear together with the element.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
                "method": (["interp", "telea", "navier_stokes", "gradient"],),
                "radius": (
                    "INT",
                    {"default": 5, "min": 1, "max": 64, "step": 1},
                ),
                "gradient_ring": (
                    "INT",
                    {"default": 20, "min": 2, "max": 128, "step": 1},
                ),
            },
            "optional": {
                "auto_scale": ("BOOLEAN", {"default": False}),
                "grow": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "shadow_reach": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "shadow_thresh": ("FLOAT", {"default": 14.0, "min": 1.0, "max": 128.0, "step": 0.5}),
                "bg_std_max": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 128.0, "step": 0.5}),
                "max_expand": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.05}),
                "sim_scale": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 128.0, "step": 0.5}),
                "blur_scale": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_max": ("INT", {"default": 41, "min": 3, "max": 255, "step": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("INPAINTED_IMAGE",)
    FUNCTION = "inpaint"
    CATEGORY = "image/inpaint"

    @staticmethod
    def _gradient_fill(rgb, binary, ring_size):
        import cv2
        import numpy as np

        result = rgb.astype(np.float32).copy()
        count, labels = cv2.connectedComponents(binary, connectivity=8)
        height, width = binary.shape
        yy, xx = np.mgrid[0:height, 0:width]
        for label in range(1, count):
            component = labels == label
            area = int(component.sum())
            if area == 0:
                continue
            kernel_size = int(ring_size) * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            expanded = cv2.dilate(component.astype(np.uint8), kernel, iterations=1) > 0
            ring = expanded & ~component & (binary == 0)
            ry, rx = np.nonzero(ring)
            if len(rx) < 12:
                continue
            # Keep the least-squares system bounded on large regions.
            if len(rx) > 12000:
                step = max(1, len(rx) // 12000)
                rx, ry = rx[::step], ry[::step]
            x_norm = rx / max(1, width - 1)
            y_norm = ry / max(1, height - 1)
            design = np.column_stack((np.ones_like(x_norm), x_norm, y_norm))
            cy, cx = np.nonzero(component)
            component_design = np.column_stack((
                np.ones_like(cx, dtype=np.float32),
                cx / max(1, width - 1),
                cy / max(1, height - 1),
            ))
            for channel in range(3):
                coefficients, *_ = np.linalg.lstsq(
                    design, result[ry, rx, channel], rcond=None
                )
                result[cy, cx, channel] = component_design @ coefficients
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _auto_scale(bool_masks):
        """Pick grow / shadow_reach from how big this layer's elements actually are.

        A 25px tall label and a 480px card need the same couple of pixels to swallow their
        anti-aliased edge, but wildly different reach to swallow their drop shadow, so the edge
        term stays nearly constant while the shadow term tracks element size.
        """
        import numpy as np

        dims = []
        for m in bool_masks:
            box = auto_filter.bbox(m)
            if box:
                dims.append(min(box[2] - box[0], box[3] - box[1]))
        if not dims:
            return 4, 12
        typical = float(np.median(dims))
        grow = int(round(min(8.0, max(3.0, 3.0 + typical / 150.0))))
        reach = int(round(min(36.0, max(8.0, typical * 0.12))))
        return grow, reach

    def inpaint(self, image, masks, method="interp", radius=5, gradient_ring=20, grow=0,
                shadow_reach=0, shadow_thresh=14.0, bg_std_max=30.0, max_expand=0.6,
                sim_scale=12.0, blur_scale=0.25, blur_max=41, auto_scale=False):
        import cv2
        import numpy as np

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        size = tuple(image.shape[1:3])
        bool_masks = _masks_to_bool_list(masks, size)
        if auto_scale:
            grow, shadow_reach = self._auto_scale(bool_masks)

        outputs = []
        for source in image[..., :3]:
            rgb = (
                source.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
            ).round().astype(np.uint8)
            fill = np.zeros(size, dtype=bool)
            for mask in bool_masks:
                if not mask.any():
                    continue
                if int(grow) > 0 or int(shadow_reach) > 0:
                    fill |= auto_filter.shadow_grow(
                        rgb, mask, reach=int(shadow_reach), thresh=float(shadow_thresh),
                        base=int(grow), bg_std_max=float(bg_std_max), max_expand=float(max_expand),
                    )
                else:
                    fill |= mask
            if not fill.any():
                outputs.append(source)
                continue
            if method == "gradient":
                filled = self._gradient_fill(rgb, (fill.astype(np.uint8) * 255), int(gradient_ring))
            elif method == "interp":
                filled = auto_filter.inpaint_interp(
                    rgb, fill, True, float(blur_scale), float(sim_scale), int(blur_max)
                )
            else:
                flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
                filled = cv2.inpaint(rgb, (fill.astype(np.uint8) * 255), float(radius), flag)
            generated = torch.from_numpy(filled.astype(np.float32) / 255.0).to(
                dtype=image.dtype, device=image.device
            )
            union = torch.from_numpy(fill).to(device=image.device)
            # Composite in torch so non-mask pixels retain the original tensor
            # values exactly instead of passing through uint8 quantization.
            outputs.append(torch.where(union.unsqueeze(-1), generated, source))
        return (torch.stack(outputs),)


NODE_CLASS_MAPPINGS = {
    "SAM3BatchCropToObjects": SAM3BatchCropToObjects,
    "SAM3MergeMaskBatch": SAM3MergeMaskBatch,
    "SAM3SelectionOverlay": SAM3SelectionOverlay,
    "SAM3MATInpaintSequence": SAM3MATInpaintSequence,
    "SAM3MaskReview": SAM3MaskReview,
    "SAM3ApprovalGate": SAM3ApprovalGate,
    "SAM3DeterministicInpaint": SAM3DeterministicInpaint,
    "SAM3MaskBatchConcat": SAM3MaskBatchConcat,
    "SAM3AutoFilterMasks": SAM3AutoFilterMasks,
    "SAM3CropToRGBA": SAM3CropToRGBA,
    "SAM3AutoLayerMasks": SAM3AutoLayerMasks,
    "SAM3PromptBank": SAM3PromptBank,
    "SAM3NineSlice": SAM3NineSlice,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3MaskBatchConcat": "Concat SAM3 Mask Batches",
    "SAM3AutoFilterMasks": "Auto Filter SAM3 Masks",
    "SAM3CropToRGBA": "Crop SAM3 Masks To RGBA Sprites",
    "SAM3AutoLayerMasks": "Auto Layer SAM3 Masks (z-order)",
    "SAM3PromptBank": "SAM3 Prompt Bank (run many prompts)",
    "SAM3NineSlice": "Detect 9-Slice Borders",
    "SAM3BatchCropToObjects": "Crop SAM3 Batch To Objects",
    "SAM3MergeMaskBatch": "Merge SAM3 Mask Batch",
    "SAM3SelectionOverlay": "Overlay SAM3 Selection",
    "SAM3MATInpaintSequence": "MAT Inpaint SAM3 Objects Sequentially",
    "SAM3MaskReview": "Review and Filter SAM3 Masks",
    "SAM3ApprovalGate": "Require SAM3 Preview Approval",
    "SAM3DeterministicInpaint": "Deterministic UI Inpaint",
}
