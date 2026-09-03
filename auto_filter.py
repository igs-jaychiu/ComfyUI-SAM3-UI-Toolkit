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



def despeckle(mask, keep_frac=0.06, max_components=0):
    """Drop tiny disconnected specks so the mask's bounding box reflects the real object.

    SAM3 sometimes returns a clean object plus a handful of stray pixels far away; the stray
    pixels blow up the bbox and make an element look like a much larger region.
    """
    u = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    order = np.argsort(-areas)
    keep = []
    for rank, ci in enumerate(order):
        if areas[ci] < keep_frac * biggest:
            break
        if max_components and rank >= max_components:
            break
        keep.append(ci + 1)
    if not keep:
        return mask
    return np.isin(labels, keep)

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


# --------------------------------------------------------------------------- automatic layering

def _mask_from_rle(entry):
    return entry


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _crop_iou(ma, ba, aa, mb, bb, ab):
    """Pixel IoU computed only over the overlapping bbox window."""
    ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
    ix2, iy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = int(np.logical_and(ma[iy1:iy2, ix1:ix2], mb[iy1:iy2, ix1:ix2]).sum())
    union = aa + ab - inter
    return inter / union if union else 0.0


def _crop_inter(ma, ba, mb, bb):
    ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
    ix2, iy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int(np.logical_and(ma[iy1:iy2, ix1:ix2], mb[iy1:iy2, ix1:ix2]).sum())


def dedupe_indexed(masks, threshold):
    """IoU dedupe that keeps track of which inputs were merged into each survivor.

    Bounding boxes pre-filter the pixel comparison, and the pixel AND runs only on the
    overlapping window, so this stays fast on large canvases with hundreds of masks.

    Returns (kept_masks, groups) where groups[i] is the list of input indices folded into kept[i].
    """
    areas = [int(m.sum()) for m in masks]
    boxes = [bbox(m) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: -areas[i])
    kept, groups, kboxes, kareas = [], [], [], []
    for i in order:
        ba, aa = boxes[i], areas[i]
        if ba is None:
            continue
        hit = -1
        for k in range(len(kept)):
            # box IoU is an upper bound on pixel IoU, so it can reject cheaply
            if box_iou(ba, kboxes[k]) <= threshold:
                continue
            if _crop_iou(masks[i], ba, aa, kept[k], kboxes[k], kareas[k]) > threshold:
                hit = k
                break
        if hit >= 0:
            groups[hit].append(i)
        else:
            kept.append(masks[i])
            groups.append([i])
            kboxes.append(ba)
            kareas.append(aa)
    return kept, groups


def contains(small, big, ratio=0.85):
    """True when `small` sits inside `big` (most of small's area overlaps big, and big is larger)."""
    a = small.sum()
    if a == 0:
        return False
    if big.sum() <= a * 1.02:
        return False
    return np.logical_and(small, big).sum() / a > ratio


def layer_heights(masks, contain_ratio=0.85):
    """Assign each mask a height and its direct parent.

    Height: leaves = 1, a mask containing height-h children = h+1. Parent: the smallest mask
    that contains it, which is what a caller needs to rebuild the UI tree.

    This is a z-order that generalises across layouts: text/icons/props come out at height 1,
    the plates and buttons that hold them at 2, the cards at 3, the window at 4 - without any
    per-image prompt bookkeeping.
    """
    n = len(masks)
    if n == 0:
        return [], []
    areas = [int(m.sum()) for m in masks]
    boxes = [bbox(m) for m in masks]
    inside = [[] for _ in range(n)]          # inside[p] = children of p
    order = sorted(range(n), key=lambda i: areas[i])
    for ci in range(n):
        bc, ac = boxes[ci], areas[ci]
        if bc is None or ac == 0:
            continue
        for pi in range(n):
            if ci == pi or areas[pi] <= ac * 1.02:
                continue
            bp = boxes[pi]
            if bp is None:
                continue
            # cheap reject: how much of ci's box can possibly fall inside pi's box
            ix1, iy1 = max(bc[0], bp[0]), max(bc[1], bp[1])
            ix2, iy2 = min(bc[2], bp[2]), min(bc[3], bp[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            if (ix2 - ix1) * (iy2 - iy1) < contain_ratio * ac:
                continue
            if _crop_inter(masks[ci], bc, masks[pi], bp) / ac > contain_ratio:
                inside[pi].append(ci)
    height = [0] * n
    for i in order:  # ascending area guarantees children resolve first
        kids = inside[i]
        height[i] = 1 + max((height[c] for c in kids), default=0)
    # direct parent = the smallest mask that contains this one
    parent = [None] * n
    for pi in range(n):
        for ci in inside[pi]:
            if parent[ci] is None or areas[pi] < areas[parent[ci]]:
                parent[ci] = pi
    return height, parent


def auto_layers(masks, labels=None, dedupe_iou=0.85, contain_ratio=0.85, min_area=40,
                max_area_frac=0.98, min_fill=0.0, min_dim=6, max_layers=6, close_holes_from=3,
                label_priority=None, despeckle_frac=0.06, min_votes=1, straddle_lo=0.0,
                straddle_hi=0.0, drop_same_label_children=False):
    """Pool masks from many prompts, clean them, and split into z-order layers (leaves first).

    Returns (layers, labels_per_layer, summary, meta_per_layer). layers[k] is a list of bool
    masks; meta[k][i] carries uid / label / votes / parent uid / box for that element.
    """
    if not masks:
        return [], [], 'no masks', []
    height_px, width_px = masks[0].shape
    total = height_px * width_px
    labels = list(labels) if labels is not None else [''] * len(masks)

    sized, sized_labels = [], []
    for m, lb in zip(masks, labels):
        if despeckle_frac > 0:
            m = despeckle(m, despeckle_frac)
        area = int(m.sum())
        if area < min_area or area > max_area_frac * total:
            continue
        box = bbox(m)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if (x2 - x1) < min_dim or (y2 - y1) < min_dim:
            continue
        if min_fill > 0 and area / ((x2 - x1) * (y2 - y1)) < min_fill:
            continue
        sized.append(m)
        sized_labels.append(lb)
    if not sized:
        return [], [], f'in={len(masks)} sized=0', []

    kept, groups = dedupe_indexed(sized, dedupe_iou)
    prio = label_priority or []

    def pick_label(idxs):
        names = [sized_labels[i] for i in idxs if sized_labels[i]]
        if not names:
            return ''
        for p in prio:
            if p in names:
                return p
        return max(set(names), key=names.count)

    kept_labels = [pick_label(g) for g in groups]
    votes = [len(g) for g in groups]
    n_dedupe = len(kept)

    # --- consensus: an element that several independent prompts agree on is real; something a
    # single prompt hallucinated usually is not.
    if min_votes > 1:
        sel = [i for i, v in enumerate(votes) if v >= min_votes]
        kept = [kept[i] for i in sel]
        kept_labels = [kept_labels[i] for i in sel]
        votes = [votes[i] for i in sel]
    n_votes = len(kept)

    # --- straddle suppression: a mask that half-overlaps another (neither disjoint nor cleanly
    # contained) is a bad cut across two elements; keep whichever has more prompt agreement.
    if straddle_hi > straddle_lo > 0:
        boxes_k = [bbox(m) for m in kept]
        areas_k = [int(m.sum()) for m in kept]
        drop = set()
        for i in range(len(kept)):
            if i in drop:
                continue
            for j in range(i + 1, len(kept)):
                if j in drop:
                    continue
                bi, bj = boxes_k[i], boxes_k[j]
                if bi is None or bj is None:
                    continue
                inter = _crop_inter(kept[i], bi, kept[j], bj)
                if inter == 0:
                    continue
                fi = inter / max(1, areas_k[i])
                fj = inter / max(1, areas_k[j])
                small = max(fi, fj)
                if straddle_lo < small < straddle_hi:
                    loser = i if votes[i] < votes[j] else j
                    if votes[i] == votes[j]:
                        loser = i if areas_k[i] > areas_k[j] else j
                    drop.add(loser)
        if drop:
            sel = [i for i in range(len(kept)) if i not in drop]
            kept = [kept[i] for i in sel]
            kept_labels = [kept_labels[i] for i in sel]
            votes = [votes[i] for i in sel]
    n_straddle = len(kept)

    # --- granularity collapse: a piece contained in a bigger piece that the *same* kind of prompt
    # found is over-segmentation (a glyph inside its word, a knob inside its slider). Text inside a
    # button survives because the labels differ.
    if drop_same_label_children:
        boxes_k = [bbox(m) for m in kept]
        areas_k = [int(m.sum()) for m in kept]
        drop = set()
        for ci in range(len(kept)):
            bc, ac = boxes_k[ci], areas_k[ci]
            if bc is None or ac == 0:
                continue
            for pi in range(len(kept)):
                if ci == pi or areas_k[pi] <= ac * 1.02:
                    continue
                if kept_labels[ci] != kept_labels[pi] or not kept_labels[ci]:
                    continue
                bp = boxes_k[pi]
                if bp is None:
                    continue
                ix1, iy1 = max(bc[0], bp[0]), max(bc[1], bp[1])
                ix2, iy2 = min(bc[2], bp[2]), min(bc[3], bp[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) < contain_ratio * ac:
                    continue
                if _crop_inter(kept[ci], bc, kept[pi], bp) / ac > contain_ratio:
                    drop.add(ci)
                    break
        if drop:
            sel = [i for i in range(len(kept)) if i not in drop]
            kept = [kept[i] for i in sel]
            kept_labels = [kept_labels[i] for i in sel]
            votes = [votes[i] for i in sel]
    n_granular = len(kept)

    heights, parents = layer_heights(kept, contain_ratio)

    # place every kept mask on a layer, then sort each layer in reading order
    placement = []   # (layer_no, kept_index)
    for h in range(1, max_layers + 1):
        if h == max_layers:
            sel = [i for i, v in enumerate(heights) if v >= max_layers]
        else:
            sel = [i for i, v in enumerate(heights) if v == h]
        sel.sort(key=lambda i: (bbox(kept[i])[1] // 24, bbox(kept[i])[0]))
        placement.append(sel)

    # stable uid per element so a parent can be referenced from another layer
    uid_of = {}
    for li, sel in enumerate(placement, 1):
        for pos, ki in enumerate(sel, 1):
            uid_of[ki] = f'L{li}_{pos}'

    layers, layer_labels, layer_meta = [], [], []
    for li, sel in enumerate(placement, 1):
        group = [kept[i] for i in sel]
        if close_holes_from and li >= close_holes_from:
            group = [fill_holes(m) for m in group]
        layers.append(group)
        layer_labels.append([kept_labels[i] for i in sel])
        meta = []
        for pos, ki in enumerate(sel, 1):
            x1, y1, x2, y2 = bbox(kept[ki])
            meta.append({
                'uid': uid_of[ki],
                'layer': li,
                'index': pos,
                'label': kept_labels[ki],
                'votes': votes[ki],
                'parent': uid_of.get(parents[ki]) if parents[ki] is not None else None,
                'area': int(kept[ki].sum()),
                'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
            })
        layer_meta.append(meta)

    summary = (f'in={len(masks)} sized={len(sized)} dedupe={n_dedupe} votes={n_votes} '
               f'straddle={n_straddle} granular={n_granular} layers=' + '/'.join(str(len(l)) for l in layers))
    return layers, layer_labels, summary, layer_meta


# --------------------------------------------------------------------------- alpha refinement

def difference_matte(image, mask, low=0.10, high=0.35, min_coverage=0.40, pad=6,
                     smooth=1, keep_largest=True, max_interior_hole=0.25, tight_edge=0.9):
    """Turn a blob-shaped mask into a shape-accurate alpha using a difference matte.

    SAM3 returns text as a filled rectangle, so a "text" sprite comes out with its plate baked
    in. The fix does not need another model: estimate what sits *behind* the mask by inpainting
    it away, then set alpha from how far each pixel departs from that estimate. Glyph strokes
    depart a lot, the plate behind them does not, and the transition band keeps the original
    anti-aliasing.

    `low` / `high` are colour distances in 0..1 units (max channel difference). Anything below
    `low` becomes transparent, above `high` opaque, in between it ramps.

    Guard: if the element barely differs from its surroundings (a pale panel on a pale panel)
    the matte would erase it, so anything under `min_coverage` falls back to the original mask.
    """
    box = bbox(mask)
    if box is None:
        return mask.astype(np.float32)

    # If SAM3 already traced this shape onto a real image edge, the mask is not a loose plate and
    # re-cutting it can only hurt: a stud whose middle matches the brick it sits on would be
    # hollowed into a ring. Only masks whose border runs over flat colour get re-matted.
    if tight_edge > 0:
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        edges = cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
                              cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
        u = mask.astype(np.uint8)
        rim = (cv2.dilate(u, np.ones((3, 3), np.uint8))
               - cv2.erode(u, np.ones((3, 3), np.uint8))) > 0
        if rim.sum() >= 8:
            if float(np.percentile(edges[rim], 60)) / 255.0 > tight_edge:
                return mask.astype(np.float32)

    h_img, w_img = mask.shape
    x1 = max(0, box[0] - pad)
    y1 = max(0, box[1] - pad)
    x2 = min(w_img, box[2] + pad)
    y2 = min(h_img, box[3] + pad)
    sub = image[y1:y2, x1:x2].astype(np.float32)
    sm = mask[y1:y2, x1:x2]
    if sm.sum() < 12:
        return mask.astype(np.float32)

    background = inpaint_interp(sub.astype(np.uint8), sm, blur=True, blur_scale=0.35,
                                sim_scale=14.0, blur_max=61).astype(np.float32)
    diff = np.abs(sub - background).max(axis=2) / 255.0
    alpha = np.clip((diff - low) / max(1e-6, high - low), 0.0, 1.0)
    alpha[~sm] = 0.0

    if keep_largest:
        solid = (alpha > 0.5).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            biggest = areas.max()
            # glyphs of one word are separate components, so keep every component that is not
            # a speck rather than only the single largest one
            keep = [i + 1 for i in range(len(areas)) if areas[i] >= max(6, 0.02 * biggest)]
            if keep:
                alpha = alpha * np.isin(lab, keep)

    coverage = float((alpha > 0.5).sum()) / max(1, int(sm.sum()))
    if coverage < min_coverage:
        return mask.astype(np.float32)

    # A difference matte hollows out anything whose middle matches what surrounds it - a brick
    # stud sitting on the brick becomes a ring. Tell that apart from a glyph counter by whether
    # the removed area is sealed inside the shape: a stud's middle is, the gaps between strokes
    # of a word are not, because they run out to the sprite edge.
    if max_interior_hole > 0:
        opaque = alpha > 0.5
        free = (~opaque).astype(np.uint8)          # everything the matte would let through
        n_f, lab_f = cv2.connectedComponents(np.pad(free, 1, constant_values=1), connectivity=4)
        outside_label = lab_f[0, 0]                # the padding ring is one connected region
        inner = lab_f[1:-1, 1:-1]
        # a transparent pixel that cannot reach the outside is walled in by the element itself
        sealed = free.astype(bool) & (inner != outside_label) & sm
        if sealed.sum() > max_interior_hole * max(1, int(sm.sum())):
            return mask.astype(np.float32)

    if smooth > 0:
        k = 2 * int(smooth) + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    out = np.zeros(mask.shape, np.float32)
    out[y1:y2, x1:x2] = alpha
    return out
