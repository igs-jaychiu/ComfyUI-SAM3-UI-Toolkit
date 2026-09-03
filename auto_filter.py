"""Deterministic mask filtering and inpainting helpers shared by the SAM3 UI Toolkit nodes.

Everything here works on numpy arrays so it can be unit-tested without ComfyUI. The node
classes in nodes.py convert torch tensors to/from these helpers.
"""
import cv2
import numpy as np


# --------------------------------------------------------------------------- mask utilities

def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def dedupe(masks, threshold):
    kept = []
    for mask in sorted(masks, key=lambda m: -m.sum()):
        if any(iou(mask, k) > threshold for k in kept):
            continue
        kept.append(mask)
    return kept


def fill_holes(mask):
    """Close interior holes of a binary mask (flood fill from the border)."""
    height, width = mask.shape
    padded = np.pad(mask.astype(np.uint8), 1)
    scratch = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(padded, scratch, (0, 0), 2)
    holes = padded[1:-1, 1:-1] == 0
    return mask | holes


def grow(mask, pixels):
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def filter_masks(masks, exclude=(), dedupe_iou=0.85, drop_contained=True, contain_ratio=0.85,
                 min_area=30, max_area_frac=0.5, min_fill=0.0, exclude_overlap=0.5, row_merge=False,
                 merge_gap_ratio=1.0, split_gap_ratio=1.5, close_holes=False):
    """Reduce a raw SAM3 individual-mask batch to one clean mask per UI element.

    Steps: size / fill-ratio gate -> IoU dedupe -> drop masks that *are* an excluded reference
    (e.g. a text prompt that returned a whole button) -> drop fragments contained in a larger kept
    mask (single glyphs inside a text line) -> optional text row split/merge -> optional hole fill.
    Returns (kept_masks, summary_string).
    """
    if not masks:
        return [], "no masks"
    height, width = masks[0].shape
    sized = []
    for mask in masks:
        area = int(mask.sum())
        if not (min_area <= area <= max_area_frac * height * width):
            continue
        if min_fill > 0:
            x1, y1, x2, y2 = bbox(mask)
            if area / ((x2 - x1) * (y2 - y1)) < min_fill:
                continue
        sized.append(mask)
    kept = dedupe(sized, dedupe_iou)
    n_dedupe = len(kept)

    if exclude:
        remaining = []
        for mask in kept:
            is_reference = False
            for ref in exclude:
                inter = np.logical_and(mask, ref).sum()
                if inter / max(1, ref.sum()) > exclude_overlap:
                    is_reference = True
                    break
            if not is_reference:
                remaining.append(mask)
        kept = remaining
    n_excluded = len(kept)

    if drop_contained:
        remaining = []
        for i, mask in enumerate(kept):
            area = mask.sum()
            contained = False
            for j, other in enumerate(kept):
                if j == i or other.sum() <= area:
                    continue
                if np.logical_and(mask, other).sum() / area > contain_ratio:
                    contained = True
                    break
            if not contained:
                remaining.append(mask)
        kept = remaining
    n_contained = len(kept)

    if row_merge:
        split = []
        for mask in kept:
            x1, y1, x2, y2 = bbox(mask)
            line_height = y2 - y1
            columns = np.nonzero(mask.any(axis=0))[0]
            gaps = np.nonzero(np.diff(columns) > split_gap_ratio * line_height)[0]
            if len(gaps) == 0:
                split.append(mask)
                continue
            starts = [columns[0]] + [columns[g + 1] for g in gaps]
            ends = [columns[g] + 1 for g in gaps] + [columns[-1] + 1]
            for start, end in zip(starts, ends):
                piece = np.zeros_like(mask)
                piece[:, start:end] = mask[:, start:end]
                if piece.sum() >= min_area:
                    split.append(piece)
        kept = split
        changed = True
        while changed:
            changed = False
            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    a = bbox(kept[i])
                    b = bbox(kept[j])
                    ha = a[3] - a[1]
                    hb = b[3] - b[1]
                    vertical_overlap = min(a[3], b[3]) - max(a[1], b[1])
                    if vertical_overlap < 0.6 * min(ha, hb):
                        continue
                    if abs(ha - hb) > 0.6 * max(ha, hb):
                        continue
                    gap = max(a[0], b[0]) - min(a[2], b[2])
                    if gap < merge_gap_ratio * max(ha, hb):
                        kept[i] = np.logical_or(kept[i], kept[j])
                        del kept[j]
                        changed = True
                        break
                if changed:
                    break
        kept = dedupe(kept, dedupe_iou)

    if close_holes:
        kept = [fill_holes(mask) for mask in kept]

    kept.sort(key=lambda m: (bbox(m)[1] // 20, bbox(m)[0]))
    summary = (
        f"in={len(masks)} sized={len(sized)} dedupe={n_dedupe} exclude={n_excluded} "
        f"contained={n_contained} final={len(kept)}"
    )
    return kept, summary


# --------------------------------------------------------------------------- fill-mask growth

def shadow_grow(image, mask, reach=24, thresh=14.0, base=3, bg_std_max=30.0, max_expand=0.6):
    """Extend a mask over the soft drop shadow / halo of a UI element.

    Pixels within `reach` whose colour differs from the local background (median of a far ring)
    by more than `thresh` and that touch the object are added. Two safety gates keep the growth
    from swallowing neighbouring structure: the ring must be reasonably uniform (`bg_std_max`)
    and the added area may not exceed `max_expand` of the object area.
    """
    near = grow(mask, base)
    if reach <= 0:
        return near
    far = grow(mask, reach)
    ring = far & ~grow(mask, reach // 2)
    if ring.sum() < 20:
        return near
    ring_pixels = image[ring].astype(np.float32)
    background = np.median(ring_pixels, axis=0)
    if float(ring_pixels.std(axis=0).mean()) > bg_std_max:
        return near
    diff = np.abs(image.astype(np.float32) - background).max(axis=2)
    candidate = (far & (diff > thresh)) | near
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[near])
    keep = keep[keep != 0]
    out = np.isin(labels, keep)
    out = cv2.morphologyEx(
        out.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ) > 0
    out = out | near
    if (out.sum() - near.sum()) > max_expand * max(1, near.sum()):
        return near
    return out


# --------------------------------------------------------------------------- inpainting

def _interp_axis(sub, comp, known, axis, sim_scale):
    """Linear interpolation along one axis. Returns (fill, weight).

    weight = similarity(endpoint colours) / distance-to-nearest-known. A line whose two known
    endpoints differ a lot crosses an edge (e.g. a panel frame) and is distrusted.
    """
    height, width = comp.shape
    fill = sub.copy()
    weight = np.zeros((height, width), np.float32)
    n_lines = height if axis == 0 else width
    for i in range(n_lines):
        line = comp[i] if axis == 0 else comp[:, i]
        if not line.any():
            continue
        kline = known[i] if axis == 0 else known[:, i]
        k = np.nonzero(kline)[0]
        if len(k) < 2:
            continue
        t = np.nonzero(line)[0]
        inside = (t > k[0]) & (t < k[-1])
        if not inside.any():
            continue
        t = t[inside]
        values = sub[i, k, :] if axis == 0 else sub[k, i, :]
        idx = np.searchsorted(k, t)
        left = values[idx - 1]
        right = values[idx]
        for c in range(3):
            v = np.interp(t, k, values[:, c])
            if axis == 0:
                fill[i, t, c] = v
            else:
                fill[t, i, c] = v
        dl = t - k[idx - 1]
        dr = k[idx] - t
        colour_gap = np.abs(left - right).max(axis=1)
        similarity = 1.0 / (1.0 + (colour_gap / sim_scale) ** 2)
        w = similarity / np.maximum(1, np.minimum(dl, dr))
        if axis == 0:
            weight[i, t] = w
        else:
            weight[t, i] = w
    return fill, weight


def inpaint_interp(image, mask, blur=True, blur_scale=0.25, sim_scale=12.0, blur_max=41):
    """Edge-aware bidirectional linear interpolation per connected component, then interior smoothing.

    Preserves horizontal / vertical gradients of flat UI surfaces and never invents content.
    """
    out = image.astype(np.float32).copy()
    height, width = mask.shape
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        component = labels == label
        ys, xs = np.nonzero(component)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        pad = 4
        Y1, Y2 = max(0, y1 - pad), min(height, y2 + pad + 1)
        X1, X2 = max(0, x1 - pad), min(width, x2 + pad + 1)
        sub = out[Y1:Y2, X1:X2]
        comp = component[Y1:Y2, X1:X2]
        known = ~mask[Y1:Y2, X1:X2]
        fh, wh = _interp_axis(sub, comp, known, 0, sim_scale)
        fv, wv = _interp_axis(sub, comp, known, 1, sim_scale)
        total = wh + wv
        blend = (fh * wh[..., None] + fv * wv[..., None]) / np.maximum(total, 1e-6)[..., None]
        fill = np.where((total > 0)[..., None], blend, sub)
        missing = comp & (total == 0)
        if missing.any():
            fallback = cv2.inpaint(
                np.clip(fill, 0, 255).astype(np.uint8), (missing * 255).astype(np.uint8), 5, cv2.INPAINT_TELEA
            ).astype(np.float32)
            fill = np.where(missing[..., None], fallback, fill)
        if blur:
            size = int(max(3, min(blur_max, int(min(y2 - y1, x2 - x1) * blur_scale) * 2 + 1)))
            if size % 2 == 0:
                size += 1
            blurred = cv2.GaussianBlur(fill, (size, size), 0)
            interior = cv2.erode(
                comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            ) > 0
            fill = np.where(interior[..., None], blurred, fill)
        sub[comp] = fill[comp]
    return np.clip(out, 0, 255).astype(np.uint8)


def inpaint_gradient(image, mask, ring=20):
    """Per component least-squares plane fit from the surrounding ring."""
    result = image.astype(np.float32).copy()
    binary = mask.astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    height, width = binary.shape
    for label in range(1, count):
        component = labels == label
        kernel = np.ones((ring * 2 + 1, ring * 2 + 1), np.uint8)
        expanded = cv2.dilate(component.astype(np.uint8), kernel, iterations=1) > 0
        ring_mask = expanded & ~component & (binary == 0)
        ry, rx = np.nonzero(ring_mask)
        if len(rx) < 12:
            continue
        if len(rx) > 12000:
            step = max(1, len(rx) // 12000)
            rx, ry = rx[::step], ry[::step]
        design = np.column_stack((np.ones_like(rx, dtype=np.float32), rx / max(1, width - 1), ry / max(1, height - 1)))
        cy, cx = np.nonzero(component)
        component_design = np.column_stack((
            np.ones_like(cx, dtype=np.float32), cx / max(1, width - 1), cy / max(1, height - 1)))
        for channel in range(3):
            coefficients, *_ = np.linalg.lstsq(design, result[ry, rx, channel], rcond=None)
            result[cy, cx, channel] = component_design @ coefficients
    return np.clip(result, 0, 255).astype(np.uint8)


def inpaint(image, fill_mask, method="interp", radius=5, gradient_ring=20,
            sim_scale=12.0, blur_scale=0.25, blur_max=41):
    if method == "gradient":
        return inpaint_gradient(image, fill_mask, int(gradient_ring))
    if method == "interp":
        return inpaint_interp(image, fill_mask, True, blur_scale, sim_scale, int(blur_max))
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, (fill_mask * 255).astype(np.uint8), float(radius), flag)
