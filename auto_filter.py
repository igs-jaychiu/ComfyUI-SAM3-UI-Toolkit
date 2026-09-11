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


def claim_changed(masks, changed, cap=64, relative=0.0):
    """Give every pixel a peel touched to exactly one of the elements it removed.

    Two neighbouring buttons share the gap between them, and both of their shadows land in it.
    Handing that gap to both means the second sprite paints over the first and neither lands
    where it should, while capping the reach at a fixed radius leaves the outer part of a large
    element's shadow belonging to nobody at all. Nearest element wins instead, so the claims
    tile the changed area exactly once.

    One distance transform decides all of it. Doing one per mask costs a full-image pass each
    time, which is minutes once a screen has a few hundred elements on it.
    """
    if not masks:
        return []
    height, width = changed.shape
    union = np.zeros((height, width), np.uint8)
    for mask in masks:
        union |= mask.astype(np.uint8)
    if not union.any():
        return list(masks)
    count, components = cv2.connectedComponents(union, connectivity=8)
    distance, nearest = cv2.distanceTransformWithLabels(
        (union == 0).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
    # cv2 numbers the zero-set components its own way; read the mapping off the seed pixels
    owner_of = np.zeros(int(nearest.max()) + 1, np.int32) - 1
    side_of = np.zeros(int(nearest.max()) + 1, np.float32)
    for index, mask in enumerate(masks):
        tags = nearest[mask]
        if tags.size == 0:
            continue
        box = bbox(mask)
        side = float(min(box[2] - box[0], box[3] - box[1])) if box else 0.0
        for tag in np.unique(tags):
            if owner_of[tag] < 0:
                owner_of[tag] = index
                side_of[tag] = side
    limit = (np.minimum(float(cap), np.maximum(4.0, side_of * float(relative)))
             if relative > 0 else np.full_like(side_of, float(cap)))
    reachable = changed & (distance <= limit[nearest])
    owner = owner_of[nearest]
    return [mask | (reachable & (owner == index)) for index, mask in enumerate(masks)]


def fade_margin(alpha, own, fade=3):
    """Ramp the claimed margin out instead of cutting it off in a straight line.

    Everything outside the element's own mask is there because the peel changed it - its shadow,
    its anti-aliased rim. That claim ends abruptly at whatever radius the change stopped, and an
    abrupt end through flat paint is exactly what reads as a torn file. Weighting the margin down
    with distance keeps the shadow and loses the line.
    """
    if fade <= 0 or not own.any():
        return alpha
    outside = ~own
    distance = cv2.distanceTransform(outside.astype(np.uint8), cv2.DIST_L2, 3)
    ramp = np.clip(1.0 - (distance - 1.0) / float(fade), 0.0, 1.0)
    faded = alpha.astype(np.float32)
    faded[outside] *= ramp[outside]
    return faded.round().clip(0, 255).astype(alpha.dtype)


def tidy_alpha(alpha, drop_island=0.06, fill_hole=0.15):
    """Make a sprite's alpha one whole shape instead of a torn one.

    Two things make an exported file unusable on its own even when the rebuilt screen looks
    perfect. A stray island - a few pixels of a neighbour the mask caught - reads as debris and
    puts the sprite's origin in the wrong place. And a hole punched through the middle, which is
    what the difference matte leaves when part of an element happens to match what is behind it,
    reads as a bite taken out of the artwork. Islands far smaller than the body go, holes far
    smaller than the body are filled back to opaque - the colour underneath them is the element's
    own, so filling is restoring, not inventing. A real ring or window frame keeps its opening,
    because that opening is not small.
    """
    solid = (alpha > 100).astype(np.uint8)
    if not solid.any():
        return alpha
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count > 2:
        areas = {i: int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)}
        biggest = max(areas.values())
        keep = {i for i, a in areas.items() if a >= drop_island * biggest}
        if len(keep) < len(areas):
            alpha = np.where(np.isin(labels, list(keep)), alpha, 0).astype(alpha.dtype)
            solid = (alpha > 100).astype(np.uint8)
    body = int(solid.sum())
    if body == 0:
        return alpha
    padded = np.pad(solid, 1)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 2)
    enclosed = (flood[1:-1, 1:-1] == 0)
    if enclosed.any():
        holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
            enclosed.astype(np.uint8), connectivity=8)
        small = [i for i in range(1, holes)
                 if int(hole_stats[i, cv2.CC_STAT_AREA]) < fill_hole * body]
        if small:
            alpha = np.where(np.isin(hole_labels, small), 255, alpha).astype(alpha.dtype)
    return alpha


def close_pockets(mask, image, tol=20.0, min_frac=0.02, max_frac=1.50, walled=0.60):
    """Give an element back the middle of itself that no prompt described.

    A rim prompt wins the vote often enough that a round button arrives as an annulus or, with
    one nick in the rim, as a C. Neither hole filling nor the alpha tidy can help: a flood fill
    from the border reaches straight into a C, and the opening of a ring is far too large to be
    treated as a bite.

    So work from the element's own outline instead. A pocket is what the convex outline walls in
    but the mask does not cover, and it belongs to the element when it looks nothing like what
    shows around the element - a real ring's opening shows the same surface as beside it, an
    unpainted interior does not. A pocket much larger than the shape around it is left alone:
    that is a thin frame with a scene behind it, not an object missing its middle.

    Measured on the roulette screen against the game's own textures: four hollowed buttons went
    from IoU 0.27-0.38 to 0.69-0.78, files reproducing a texture 25 -> 29, nothing regressed.
    """
    if image is None or not mask.any():
        return mask
    body = max(1, int(mask.sum()))
    pieces, piece_label, piece_stats, _pc = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    hull = np.zeros(mask.shape, np.uint8)
    for i in range(1, pieces):
        if int(piece_stats[i, cv2.CC_STAT_AREA]) < 0.05 * body:
            continue
        part = (piece_label == i).astype(np.uint8)
        contours, _h = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(hull, [cv2.convexHull(np.vstack(contours))], -1, 1,
                         thickness=cv2.FILLED)
    pocket = (hull > 0) & ~mask
    if not pocket.any():
        return mask
    band = cv2.dilate((hull > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    band &= ~(hull > 0)
    if int(band.sum()) < 50:
        return mask
    rgb = image[..., :3].astype(np.float32)
    if rgb.max() <= 1.001:
        rgb = rgb * 255.0
    outside = np.median(rgb[band], axis=0)
    count, label, stats, _c = cv2.connectedComponentsWithStats(pocket.astype(np.uint8),
                                                               connectivity=8)
    add = np.zeros(mask.shape, bool)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_frac * body or area > max_frac * body:
            continue
        pick = label == i
        rim = cv2.dilate(pick.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        rim &= ~pick
        if int(rim.sum()) and float((rim & mask).sum()) / int(rim.sum()) < walled:
            continue
        if float(np.abs(np.median(rgb[pick], axis=0) - outside).max()) <= tol:
            continue
        add |= pick
    return mask | add


def repaint_filled(colour, before, after):
    """Repaint a hole the alpha tidy closed, using the sprite's own paint.

    Filling a hole back to opaque restores the shape but not the picture: the colour sitting
    there came from the layer inpaint, which was guessing what is *behind* the element, and for
    a hole that is inside the element the answer is wrong - a peeled label leaves a smear of
    plate colour across the middle of a trophy. The pixels around an enclosed hole are the
    element's own surface, so propagating them inward is restoring rather than inventing.
    """
    holes = (after > 100) & ~(before > 100)
    if not holes.any():
        return colour
    return cv2.inpaint(np.ascontiguousarray(colour.astype(np.uint8)),
                       (holes * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)


def feather_edge(mask, image, band=2, min_span=20.0):
    """Give a hard mask back the half-covered pixels along its edge.

    Every guard inside `difference_matte` ends by handing back the mask it started from, and that
    is the right call - the guards fire when re-cutting would destroy the sprite - but what comes
    back is SAM3's binary mask. Binary is what a staircase edge is: every delivered file measured
    0.0% partially transparent, and the boundary of a straight wedge wandered by two to four
    pixels.

    A pixel on that boundary is genuinely part sprite and part background, and how much of each
    is written in the picture: take the sprite's colour just inside, the background's just
    outside, and see where the pixel falls between them. On a synthetic sprite whose true
    coverage is known this reproduces the edge exactly (mean error 0/255 against 54/255 for the
    binary mask), and it also straightens a mask deliberately made to wobble by 84 pixels,
    because the coverage is read from the picture rather than from the mask.

    Where the two ends of the blend are the same colour there is nothing to read, so those
    pixels keep the mask's own verdict.
    """
    solid = mask.astype(np.uint8)
    if not solid.any():
        return mask.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(solid, kernel, iterations=int(band)) > 0
    outer = cv2.dilate(solid, kernel, iterations=int(band)) > 0
    edge = outer & ~inner
    if not edge.any() or not inner.any():
        return mask.astype(np.float32)
    rgb = image.astype(np.float32)
    reach = max(9, 6 * int(band) + 3)
    inside_colour = cv2.blur(np.where(inner[..., None], rgb, 0.0), (reach, reach))
    inside_weight = cv2.blur(inner.astype(np.float32), (reach, reach))
    outside_colour = cv2.blur(np.where(~outer[..., None], rgb, 0.0), (reach, reach))
    outside_weight = cv2.blur((~outer).astype(np.float32), (reach, reach))
    with np.errstate(divide="ignore", invalid="ignore"):
        fore = inside_colour / np.maximum(inside_weight, 1e-3)[..., None]
        back = outside_colour / np.maximum(outside_weight, 1e-3)[..., None]
    span = fore - back
    length = np.linalg.norm(span, axis=2)
    projection = ((rgb - back) * span).sum(axis=2) / np.maximum((span * span).sum(axis=2), 1e-3)
    alpha = mask.astype(np.float32)
    readable = edge & (length >= float(min_span))
    alpha[readable] = np.clip(projection[readable], 0.0, 1.0)
    alpha[inner] = 1.0
    alpha[~outer] = 0.0
    return alpha


def solve_layer_sprite(source, under, support, alpha, floor=0.50, exact_tol=2.0):
    """Express what a peel removed as RGBA that composites back over the peel's own result.

    Estimating "what is behind this element" was always a guess, and the guess had to agree
    with a second, differently-parameterised guess made later by the layer inpaint or the rim
    came out wrong. But the inpaint result is the ground truth for what will sit underneath the
    sprite when the screen is rebuilt, so given it there is nothing to estimate: solve
    C = a*F + (1-a)*U for the colour, then take alpha as the point on the segment U -> F closest
    to C. Where the two agree there is nothing to recover and the matte's own alpha stands.
    """
    colour = source.astype(np.float32)
    below = under.astype(np.float32)
    a = alpha.astype(np.float32)
    a3 = a[..., None]
    reliable = a >= floor
    fore = colour.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        solved = (colour - (1.0 - a3) * below) / np.maximum(a3, 1e-6)
    # A solve that runs off the end of the range has not recovered a colour, it has saturated:
    # clipping it leaves a white or black rim on the sprite, which composites correctly and
    # looks wrong the moment anyone opens the asset on its own.
    saturated = ((solved < -1.0) | (solved > 256.0)).any(axis=2)
    trustworthy = reliable & ~saturated
    fore[trustworthy] = np.clip(solved, 0.0, 255.0)[trustworthy]

    # a faint rim or a soft shadow has no usable ratio of its own; it takes the colour of the
    # nearest pixel that did
    unknown = support & ~trustworthy
    if unknown.any() and trustworthy.any():
        fore = cv2.inpaint(fore.astype(np.uint8), unknown.astype(np.uint8) * 255, 3,
                           cv2.INPAINT_TELEA).astype(np.float32)
    span = fore - below
    denom = (span * span).sum(axis=2)
    numer = ((colour - below) * span).sum(axis=2)
    usable = support & (denom > 4.0)
    projected = np.divide(numer, denom, out=a.copy(), where=usable)
    out = np.where(usable, np.clip(projected, 0.0, 1.0), a)
    out[~support] = 0.0

    # In this art style every element is drawn with a near-black ink outline, and an outline is
    # not a blend of anything: no mix of the fill colour and the plate behind it lands on it, so
    # the segment cannot reach and the stroke came back washed out. Wherever the decomposition
    # cannot explain the pixel it was, keep that pixel as it was and call it opaque - which is
    # what an ink outline is.
    if exact_tol > 0:
        blended = out[..., None] * fore + (1.0 - out[..., None]) * below
        residual = np.abs(blended - colour).max(axis=2)
        force = support & (residual > float(exact_tol))
        if force.any():
            fore = np.where(force[..., None], colour, fore)
            out = np.where(force, 1.0, out)
    return np.clip(fore, 0, 255).astype(np.uint8), out


def halo_alpha(image, extra, background):
    """Turn the drop shadow / glow around an element into alpha the sprite can carry.

    The inpaint that peels a layer erases the element together with its shadow, but the sprite
    was only ever cut to the element itself, so the shadow used to vanish from both the asset
    and the background it was standing on. A shadow is the background seen through something
    dark: C = (1-a)*B, so a = 1 - C/B recovers it exactly, and the mirror form recovers a glow.
    Exported that way the shadow travels with its button and darkens whatever it is dropped on.

    `background` has to be the per-pixel plate, not one averaged colour: a shadow lying across
    wood grain or a paw-print pattern would otherwise come back as the mean of the texture.
    """
    if extra is None or not extra.any():
        return None, None
    colour = image.astype(np.float32)
    plate = background.astype(np.float32)
    darker = np.clip((plate - colour) / np.maximum(plate, 1.0), 0.0, 1.0).max(axis=2)
    lighter = np.clip((colour - plate) / np.maximum(255.0 - plate, 1.0), 0.0, 1.0).max(axis=2)
    towards_dark = colour.mean(axis=2) <= plate.mean(axis=2)
    alpha = np.where(towards_dark, darker, lighter).astype(np.float32)
    alpha[~extra] = 0.0
    fore = np.where(towards_dark[..., None], 0.0, 255.0).astype(np.float32)
    return alpha, fore


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


def _repeat_offsets(known, sub, limit, count=12, min_strength=0.25):
    """Offsets at which this patch of texture repeats, best first.

    Autocorrelation of the known pixels finds the lattice a UI background is drawn on - a grid
    of board squares, a wallpaper of paw prints, a run of planks - without knowing anything
    about what is drawn.
    """
    grey = sub.mean(axis=2).astype(np.float32)
    grey = np.where(known, grey - grey[known].mean(), 0.0)
    spectrum = np.fft.rfft2(grey)
    power = np.fft.irfft2(spectrum * np.conj(spectrum), s=grey.shape).real
    power = np.fft.fftshift(power)
    height, width = power.shape
    cy, cx = height // 2, width // 2
    reach = int(min(limit, min(cy, cx)))
    if reach < 4:
        return []
    y0, x0 = cy - reach, cx - reach
    window = power[y0:cy + reach + 1, x0:cx + reach + 1].copy()
    # an even-sized axis leaves the centre off by one, so the offsets come from the real slice
    yy, xx = np.mgrid[y0 - cy:y0 - cy + window.shape[0], x0 - cx:x0 - cx + window.shape[1]]
    zero = float(power[cy, cx])
    window[(np.abs(yy) < 3) & (np.abs(xx) < 3)] = -np.inf     # the origin is not a period
    order = np.argsort(window.ravel())[::-1][:count * 4]
    seen, offsets = set(), []
    for flat in order:
        # A patch with no real lattice still has a highest correlation somewhere, and taking it
        # copies a thin slice over and over - the stripes that used to appear across small
        # non-repeating elements. Insist the peak is a real fraction of the zero-lag energy.
        if zero > 0 and float(window.ravel()[flat]) < min_strength * zero:
            break
        dy, dx = int(yy.ravel()[flat]), int(xx.ravel()[flat])
        key = (abs(dy) // 2, abs(dx) // 2)
        if key in seen:
            continue
        seen.add(key)
        offsets.append((dy, dx))
        if len(offsets) >= count:
            break
    return offsets


def _fill_pieces(hole, block=96):
    """Work list for periodic_fill: connected components, big ones cut into blocks.

    One offset for a whole large hole has to be right everywhere in it, and it never is - the
    band around it matches while the middle of the source sits on some neighbouring sprite, so
    a cat lands in the middle of the board. Cut into blocks, each piece picks its own offset and
    is judged on its own short boundary, which is a test that means something.
    """
    pieces = []
    count, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        component = labels == label
        box = bbox(component)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if max(x2 - x1, y2 - y1) <= block * 1.5:
            pieces.append(component)
            continue
        for by in range(y1, y2, block):
            for bx in range(x1, x2, block):
                piece = np.zeros_like(component)
                piece[by:by + block, bx:bx + block] = component[by:by + block, bx:bx + block]
                if piece.any():
                    pieces.append(piece)
    return pieces


def periodic_fill(image, hole, tol=20.0, min_overlap=200, block=96, max_span=200):
    """Fill a hole by copying the texture that repeats around it.

    Linear interpolation across a hole is right for a flat panel and wrong for anything with a
    pattern: a board of squares comes back as a smear. Nothing has to be invented here - the
    same texture is almost always present a period away, so it is copied rather than guessed,
    which keeps the peel deterministic. Returns (filled, remaining_hole).
    """
    out = image.astype(np.float32).copy()
    remaining = hole.copy()
    height, width = hole.shape
    for component in _fill_pieces(hole, block):
        box = bbox(component)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        span = max(x2 - x1, y2 - y1)
        if max_span and span > max_span:
            # Measured on punched-out background: beyond this the copy stops beating plain
            # interpolation, because an interior block has no trusted boundary left to judge it.
            continue
        pad = int(max(32, span * 2.0))
        X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
        X2, Y2 = min(width, x2 + pad), min(height, y2 + pad)
        sub = out[Y1:Y2, X1:X2]
        comp = component[Y1:Y2, X1:X2]
        # blocks filled earlier are fair game as sources, so the copy can march across a hole
        known = ~remaining[Y1:Y2, X1:X2]
        if known.sum() < min_overlap or comp.sum() == 0:
            continue
        todo = comp.copy()
        # Sourcing may use pixels this loop has already filled, but judging an offset may not:
        # scoring against its own output makes every offset look perfect. Judge it on the band
        # hugging the hole, too - a whole-window score passes an offset that happens to land
        # its source on a neighbouring sprite, and then a cat gets copied into the board.
        trusted = ~hole[Y1:Y2, X1:X2]
        band = grow(comp, 6) & ~comp & trusted
        offsets = _repeat_offsets(known, sub, limit=pad + span)
        # A wide hole cannot be reached in one step: the texture a period away is itself still
        # missing near the middle. Sweeping the offsets again lets the filled edge become the
        # source for the next ring inwards, so the copy marches in from all sides.
        for dy, dx in [o for _ in range(3) for o in offsets]:
            if not todo.any():
                break
            shifted = np.roll(np.roll(sub, dy, axis=0), dx, axis=1)
            shifted_known = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
            # rolling wraps, so only trust the part that did not come round the edge
            inside = np.ones_like(known)
            if dy > 0:
                inside[:dy] = False
            elif dy < 0:
                inside[dy:] = False
            if dx > 0:
                inside[:, :dx] = False
            elif dx < 0:
                inside[:, dx:] = False
            shifted_trusted = np.roll(np.roll(trusted, dy, axis=0), dx, axis=1)
            agree = trusted & shifted_trusted & inside
            close = band & shifted_trusted & inside
            if agree.sum() < min_overlap or close.sum() < 60:
                continue
            error = float(np.abs(sub[agree] - shifted[agree]).max(axis=1).mean())
            edge = float(np.abs(sub[close] - shifted[close]).max(axis=1).mean())
            if error > tol or edge > tol:
                continue
            # A genuine period repeats at twice the offset as well. A spurious one - the best
            # correlation a non-repeating patch happens to have - does not, and copying it lays
            # a thin slice down again and again.
            twice = np.roll(np.roll(sub, dy * 2, axis=0), dx * 2, axis=1)
            twice_trusted = np.roll(np.roll(trusted, dy * 2, axis=0), dx * 2, axis=1)
            far = np.ones_like(known)
            if dy > 0:
                far[:dy * 2] = False
            elif dy < 0:
                far[dy * 2:] = False
            if dx > 0:
                far[:, :dx * 2] = False
            elif dx < 0:
                far[:, dx * 2:] = False
            repeats = trusted & twice_trusted & far
            if repeats.sum() >= 60:
                again = float(np.abs(sub[repeats] - twice[repeats]).max(axis=1).mean())
                if again > tol * 1.5:
                    continue
            usable = todo & shifted_known & inside
            if not usable.any():
                continue
            # The hole is meant to come out empty. An offset can satisfy the boundary band and
            # still be sourcing content - the other line of text on the same plate - so refuse
            # a source that is markedly busier than the surround the hole sits in.
            if band.any():
                busy = float(shifted[usable].std(axis=0).mean())
                calm = float(sub[band].std(axis=0).mean())
                if busy > calm * 1.8 + 4.0:
                    continue
            sub[usable] = shifted[usable]
            known = known | usable          # a filled pixel can source the next offset
            todo &= ~usable
        filled = comp & ~todo
        if filled.any():
            out[Y1:Y2, X1:X2] = sub
            remaining[Y1:Y2, X1:X2] &= ~filled
    return out, remaining


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


def fill_auditioned(image, fill, margin=0.70, band=5, tol=20.0, block=96, max_span=200,
                    blur_scale=0.25, sim_scale=12.0, blur_max=41):
    """Choose, per hole, between copying the surrounding texture and interpolating it.

    Copying a repeating surface is the better fill when there is one - measured on holes punched
    into repeating background, MAE 0.80 against 1.14. Across a colour boundary it is the worse
    fill and by more: 6.41 against 4.87, with p90 20.2 against 11.5 and seven times as many
    pixels off by over 64/255, because the lattice gets laid over an edge that has none and the
    interpolation fallback never sees those pixels again. This is what left a smear across the
    base of a trophy whose label had been peeled.

    Nothing at fill time knows which kind of hole it is looking at, so audition both: widen the
    hole by a ring of pixels that *are* known, fill the wider hole both ways, and score each on
    that ring. The copy has to win by a margin, since a lattice over an edge scores close on the
    ring and badly in the middle. Per family that gives MAE 0.85 / 4.92 against the copy-first
    pipeline's 0.80 / 6.41 - it keeps almost all of the win where copying belongs and gives back
    the loss where it does not.
    """
    if not fill.any():
        return image, "nothing to fill"
    kernel = np.ones((band * 2 + 1, band * 2 + 1), np.uint8)
    ring = (cv2.dilate(fill.astype(np.uint8), kernel) > 0) & ~fill
    wide = fill | ring

    def copy_then_interp(mask):
        stage, left = periodic_fill(image, mask, tol=tol, block=block, max_span=max_span)
        base = np.clip(stage, 0, 255).astype(np.uint8)
        if not left.any():
            return base
        return inpaint_interp(base, left, True, blur_scale, sim_scale, int(blur_max))

    def interp_only(mask):
        return inpaint_interp(image, mask, True, blur_scale, sim_scale, int(blur_max))

    truth = image.astype(np.int16)
    trials = {}
    if ring.any():
        for name, fn in (("copy", copy_then_interp), ("plain", interp_only)):
            trial = fn(wide).astype(np.int16)
            trials[name] = np.abs(trial - truth).max(axis=2)

    copy_all = copy_then_interp(fill)
    plain_all = interp_only(fill)
    count, labels = cv2.connectedComponents(fill.astype(np.uint8), connectivity=8)
    take_copy = np.zeros(fill.shape, bool)
    picks = {"copy": 0, "plain": 0}
    reach = np.ones((band * 2 + 3, band * 2 + 3), np.uint8)
    for index in range(1, count):
        component = labels == index
        near = (cv2.dilate(component.astype(np.uint8), reach) > 0) & ring
        pick = "plain"
        if trials and near.any():
            if float(trials["copy"][near].mean()) <= margin * float(
                    trials["plain"][near].mean()):
                pick = "copy"
        picks[pick] += 1
        if pick == "copy":
            take_copy |= component
    out = np.where(take_copy[..., None], copy_all, plain_all)
    out = np.where(fill[..., None], out, image)
    return (np.clip(out, 0, 255).astype(np.uint8),
            f"{picks['copy']} holes copied, {picks['plain']} interpolated")


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


def colour_parts(image, mask, min_frac=0.05, max_frac=0.80, clusters=7, min_dim=6,
                 min_area=500):
    """Find the pieces a UI element was drawn from, by colour, inside the element itself.

    A prompt finds "the button". The art it was built from is a plate, a 9-slice frame, a strip
    of tape and an icon, and none of those is a separate object to look at - so no amount of
    prompting returns them. They are separate *colours* though, laid out in flat regions the way
    UI art always is, so clustering the colours inside an element and taking the connected pieces
    recovers them without inventing anything.
    """
    box = bbox(mask)
    if box is None:
        return []
    x1, y1, x2, y2 = box
    sub = image[y1:y2, x1:x2]
    inside = mask[y1:y2, x1:x2]
    area = int(inside.sum())
    if area < min_area or min(x2 - x1, y2 - y1) < 2 * min_dim:
        return []
    lab = cv2.cvtColor(sub, cv2.COLOR_RGB2LAB).astype(np.float32)
    samples = lab[inside]
    count = int(min(clusters, max(2, len(samples) // 200)))
    if count < 2:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _err, assignment, _centres = cv2.kmeans(samples, count, None, criteria, 3,
                                            cv2.KMEANS_PP_CENTERS)
    index = np.full(inside.shape, -1, np.int32)
    index[inside] = assignment.ravel()
    # Label every pixel of the element, then keep the pieces worth keeping and hand the
    # leftovers to whichever kept piece is nearest. Dropping the small clusters outright left
    # gaps, and a piece set with gaps can never add back up to the shape it came from - which
    # is what a coin on a plate needs, since no single colour covers it.
    tag = np.full(inside.shape, 0, np.int32)
    next_tag = 1
    sizes = {}
    for cluster in range(count):
        band = (index == cluster).astype(np.uint8)
        if not band.any():
            continue
        found, labelled, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        for comp in range(1, found):
            size = int(stats[comp, cv2.CC_STAT_AREA])
            tag[labelled == comp] = next_tag
            sizes[next_tag] = size
            next_tag += 1
    keep = {t for t, size in sizes.items()
            if size >= min_frac * area and size <= max_frac * area}
    keep = {t for t in keep
            if min(*(lambda b: (b[2] - b[0], b[3] - b[1]))(bbox(tag == t))) >= min_dim}
    if not keep:
        return []
    orphan = inside & ~np.isin(tag, list(keep))
    if orphan.any():
        # cv2's label output numbers the zero pixels its own way, so it cannot be read as an
        # index into the tag map. One distance transform per kept piece and take the nearest.
        best = np.full(tag.shape, np.inf, np.float32)
        home = np.zeros(tag.shape, np.int32)
        for t in keep:
            distance = cv2.distanceTransform((tag != t).astype(np.uint8), cv2.DIST_L2, 3)
            closer = distance < best
            best = np.where(closer, distance, best)
            home = np.where(closer, np.int32(t), home)
        tag = np.where(orphan, home, tag)
    pieces = []
    for t in keep:
        piece = np.zeros(mask.shape, bool)
        piece[y1:y2, x1:x2] = tag == t
        if piece.any():
            pieces.append((int(piece.sum()), piece))
    pieces.sort(key=lambda t: -t[0])
    return [p for _size, p in pieces]


def auto_layers(masks, labels=None, dedupe_iou=0.85, contain_ratio=0.85, min_area=40,
                max_area_frac=0.98, min_fill=0.0, min_dim=6, max_layers=6, close_holes_from=3,
                label_priority=None, despeckle_frac=0.06, min_votes=1, straddle_lo=0.0,
                straddle_hi=0.0, drop_same_label_children=False, image=None,
                split_parts=False, split_min_frac=0.06, split_max_parts=4,
                split_depth=1, absorb_min_share=0.0, absorb_max_share=0.95,
                absorb_max_children=2):
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

    # An element missing its own middle would be layered as a container of whatever shows
    # through, and peeled hollow on top of that, so repair the shapes before containment.
    if image is not None:
        kept = [close_pockets(m, image) for m in kept]

    heights, parents = layer_heights(kept, contain_ratio)

    # --- absorb: a child that is a substantial share of a leaf-like parent is not a separate
    # asset, it is the same object cut in two by the prompts (a coin and the star on its face,
    # a button and its icon). Containment alone cannot say that - everything is contained in the
    # screen-wide card - so the share has to be bounded at *both* ends, and a parent that holds
    # several children is a real container and is left alone. Measured on the wheel screen at a
    # 0.20 floor: 30 fragments absorbed, 129 files down to 99, and not one file that reproduces
    # a project texture was lost. At 0.12 two are lost, at 0.06 five.
    if absorb_min_share > 0:
        areas_a = [int(m.sum()) for m in kept]
        child_count = {}
        for index, parent in enumerate(parents):
            if parent is not None:
                child_count[parent] = child_count.get(parent, 0) + 1
        merge_into = {}
        for index, parent in enumerate(parents):
            if parent is None:
                continue
            share = areas_a[index] / max(1, areas_a[parent])
            if not absorb_min_share <= share <= absorb_max_share:
                continue
            if child_count.get(parent, 0) > int(absorb_max_children):
                continue
            merge_into.setdefault(parent, []).append(index)
        if merge_into:
            drop = set()
            for parent, children in merge_into.items():
                for child in children:
                    kept[parent] = kept[parent] | kept[child]
                    votes[parent] = max(votes[parent], votes[child])
                    drop.add(child)
            sel = [i for i in range(len(kept)) if i not in drop]
            kept = [kept[i] for i in sel]
            kept_labels = [kept_labels[i] for i in sel]
            votes = [votes[i] for i in sel]
            heights, parents = layer_heights(kept, contain_ratio)
            print(f"[auto_layers] absorbed {len(drop)} fragments into "
                  f"{len(merge_into)} parents")
    n_absorb = len(kept)

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

    # --- the art an element was built from is not a set of separate objects, so prompting
    # never returns it: a button is a plate, a 9-slice frame, a strip of tape and an icon. Those
    # are separate colours though, so clustering inside each finished element recovers them.
    # They are derived *after* the layers are settled and handed back on their own - putting
    # them in the pool made every one a child of the element it came from, and the peel then
    # hollowed out its own parent.
    parts, parts_meta = [], []
    if split_parts and image is not None:
        boxes_k = [bbox(m) for m in kept]
        areas_k = [int(m.sum()) for m in kept]
        frontier = list(kept)
        for _round in range(max(1, int(split_depth))):
            produced = []
            for origin in frontier:
                for piece in colour_parts(image, origin, min_frac=float(split_min_frac))[
                        :int(split_max_parts)]:
                    pb, pa = bbox(piece), int(piece.sum())
                    if pb is None or pa < min_area:
                        continue
                    twin = False
                    for other, ob, oa in zip(kept + parts,
                                             boxes_k + [bbox(p) for p in parts],
                                             areas_k + [int(p.sum()) for p in parts]):
                        if ob is None:
                            continue
                        inter = _crop_inter(piece, pb, other, ob)
                        if inter / max(1, pa + oa - inter) > dedupe_iou:
                            twin = True
                            break
                    if not twin:
                        parts.append(piece)
                        produced.append(piece)
            frontier = produced
            if not frontier:
                break
        for index, piece in enumerate(parts, 1):
            x1, y1, x2, y2 = bbox(piece)
            parts_meta.append({'uid': f'P_{index}', 'layer': 0, 'index': index,
                               'label': 'part', 'votes': 0, 'parent': None,
                               'area': int(piece.sum()),
                               'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1})
        print(f"[auto_layers] {len(kept)} elements, {len(parts)} extra pieces derived after "
              f"layering")

    summary = (f'in={len(masks)} sized={len(sized)} dedupe={n_dedupe} votes={n_votes} '
               f'straddle={n_straddle} granular={n_granular} absorb={n_absorb} '
               f'parts={len(parts)} layers='
               + '/'.join(str(len(l)) for l in layers))
    return layers, layer_labels, summary, layer_meta, parts, parts_meta


# --------------------------------------------------------------------------- alpha refinement

def difference_matte(image, mask, low=0.10, high=0.35, min_coverage=0.40, pad=6,
                     smooth=1, keep_largest=True, max_interior_hole=0.25, tight_edge=0.9,
                     max_pieces=12):
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

    # Splitting a word into its glyphs is the point; shattering a scene prop into dozens of
    # specks is not. Past a sane piece count the matte is destroying the element, so keep the
    # mask it started from.
    if max_pieces > 0:
        solid_u = (alpha > 0.5).astype(np.uint8)
        n_p, _, stats_p, _ = cv2.connectedComponentsWithStats(solid_u, connectivity=8)
        if n_p > 1:
            sizes = stats_p[1:, cv2.CC_STAT_AREA]
            if int((sizes >= 30).sum()) > max_pieces:
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


# --------------------------------------------------------------------------- 9-slice borders

def _axis_profile(rgba, axis):
    """Mean absolute difference between neighbouring lines along `axis`.

    axis=1 walks columns (for horizontal stretching), axis=0 walks rows. Alpha is included so a
    change in silhouette counts as a change, which is what makes a rounded corner show up.
    """
    plane = rgba.astype(np.float32)
    if axis == 0:
        a, b = plane[:-1, :, :], plane[1:, :, :]
        return np.abs(a - b).mean(axis=(1, 2)) / 255.0
    a, b = plane[:, :-1, :], plane[:, 1:, :]
    return np.abs(a - b).mean(axis=(0, 2)) / 255.0


def _longest_flat_run(profile, threshold):
    best_start = best_len = 0
    start = None
    for i, v in enumerate(profile):
        if v <= threshold:
            if start is None:
                start = i
        else:
            if start is not None and i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    if start is not None and len(profile) - start > best_len:
        best_start, best_len = start, len(profile) - start
    return best_start, best_len


def _axis_slice(profile, length, min_center, flat_frac):
    """Return (low_inset, high_inset, confidence) or None when the axis cannot be stretched."""
    if profile.size < 3:
        return None
    lo = float(np.percentile(profile, 30))
    hi = float(np.percentile(profile, 95))
    threshold = max(1.5 / 255.0, lo + flat_frac * (hi - lo))
    start, run = _longest_flat_run(profile, threshold)
    if run < min_center:
        return None
    low = int(start)
    high = int(length - (start + run + 1))
    if low < 0 or high < 0:
        return None
    centre = float(profile[start:start + run].mean())
    border = np.concatenate([profile[:start], profile[start + run:]])
    outer = float(border.mean()) if border.size else 0.0
    confidence = 0.0 if outer <= 1e-6 else max(0.0, min(1.0, 1.0 - centre / outer))
    if border.size == 0:
        confidence = 1.0          # perfectly uniform along this axis
    return low, high, confidence



def _runs_per_line(alpha, axis):
    """Median number of separate opaque spans along each line. A solid plate gives 1."""
    lines = alpha if axis == 0 else alpha.T
    counts = []
    for line in lines:
        edges = np.diff(np.concatenate(([0], line.astype(np.uint8), [0])))
        n = int((edges == 1).sum())
        if n:
            counts.append(n)
    return float(np.median(counts)) if counts else 0.0


def nine_slice(rgba, min_center=6, flat_frac=0.15, min_confidence=0.55, min_inset=2,
               min_opaque=0.35, min_side=48, max_parts=1, silhouette_tol=0.04,
               max_circularity=0.82, max_runs=1, min_band_frac=0.25):
    """Work out 9-slice borders for a UI sprite.

    A 9-slice sprite has a middle band that repeats along the stretch axis, so scanning the
    difference between neighbouring rows and columns finds it directly: the corners and the
    bevel change fast, the stretchable middle does not.

    Returns a dict with left/right/top/bottom insets, whether each axis may stretch at all, and
    a confidence. An element with a diagonal gloss or a centred ornament has no flat band and is
    reported as not stretchable rather than given a wrong guess.
    """
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("nine_slice expects an HxWx3 or HxWx4 array")
    if rgba.shape[2] == 3:
        rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
    height, width = rgba.shape[:2]

    reject = {"left": 0, "right": 0, "top": 0, "bottom": 0,
              "stretch_x": False, "stretch_y": False,
              "confidence_x": 0.0, "confidence_y": 0.0, "nine_slice": False}

    # 9-slice only means anything for a solid plate. A sprite that is mostly holes - a card whose
    # contents were peeled out, a ribbon traced around its own text - has no coherent border to
    # keep, and measuring one produces confident nonsense.
    if height < min_side or width < min_side:
        return dict(reject, reason="too small")
    alpha = rgba[:, :, 3] > 127
    opaque = float(alpha.mean())
    if opaque < min_opaque:
        return dict(reject, reason=f"only {opaque:.0%} opaque")

    # One plate, not a handful of leftovers. A card whose contents were peeled out, or a ribbon
    # traced around its own text, comes back in pieces and has no border worth keeping.
    count, _, stats, _ = cv2.connectedComponentsWithStats(alpha.astype(np.uint8), connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        parts = int((areas >= 0.05 * areas.max()).sum())
        if parts > max_parts:
            return dict(reject, reason=f"{parts} disconnected parts")

    # A disc has no straight side to stretch along, and its interior looks flat enough to fool
    # the profile, so measure the outline directly: circularity near 1 means a round icon.
    contours, _ = cv2.findContours(alpha.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(biggest, True)
        if perimeter > 0:
            circularity = 4.0 * np.pi * cv2.contourArea(biggest) / (perimeter * perimeter)
            if circularity > max_circularity:
                return dict(reject, reason=f"round shape (circularity {circularity:.2f})")

    horizontal = _axis_slice(_axis_profile(rgba, 1), width, min_center, flat_frac)
    vertical = _axis_slice(_axis_profile(rgba, 0), height, min_center, flat_frac)

    # The band a 9-slice stretches must have straight sides and be solid across. A round icon
    # has a flat-looking interior but an outline that curves the whole way, and a word is a row
    # of strokes with gaps - stretching either one is wrong.
    if horizontal is not None:
        low, high, _ = horizontal
        band = alpha[:, low:width - high]
        run = band.sum(axis=0).astype(np.float32)
        if run.size < 2 or run.std() > silhouette_tol * height:
            horizontal = None
        elif max_runs > 0 and _runs_per_line(band, 1) > max_runs:
            horizontal = None
    if vertical is not None:
        low, high, _ = vertical
        band = alpha[low:height - high, :]
        run = band.sum(axis=1).astype(np.float32)
        if run.size < 2 or run.std() > silhouette_tol * width:
            vertical = None
        elif max_runs > 0 and _runs_per_line(band, 0) > max_runs:
            vertical = None

    left = right = top = bottom = 0
    stretch_x = stretch_y = False
    conf_x = conf_y = 0.0
    if horizontal is not None:
        left, right, conf_x = horizontal
        # A sliver of a middle is not worth stretching, and a detection that finds one is
        # usually reading a glyph's own bowl rather than a repeating band.
        band_ok = (width - left - right) >= min_band_frac * width
        stretch_x = band_ok and conf_x >= min_confidence and (left >= min_inset
                                                              or right >= min_inset
                                                              or conf_x >= 0.95)
    if vertical is not None:
        top, bottom, conf_y = vertical
        band_ok = (height - top - bottom) >= min_band_frac * height
        stretch_y = band_ok and conf_y >= min_confidence and (top >= min_inset
                                                              or bottom >= min_inset
                                                              or conf_y >= 0.95)
    if not stretch_x:
        left = right = 0
    if not stretch_y:
        top = bottom = 0
    result = {
        "left": int(left), "right": int(right), "top": int(top), "bottom": int(bottom),
        "stretch_x": bool(stretch_x), "stretch_y": bool(stretch_y),
        "confidence_x": round(float(conf_x), 3), "confidence_y": round(float(conf_y), 3),
        "nine_slice": bool(stretch_x or stretch_y),
    }
    if not result["nine_slice"]:
        result["reason"] = "no repeating band on either axis"
    return result


def nine_slice_resize(rgba, width, height, borders):
    """Scale a sprite the way an engine would, so a detection can be checked against the source."""
    src_h, src_w = rgba.shape[:2]
    left, right = int(borders.get("left", 0)), int(borders.get("right", 0))
    top, bottom = int(borders.get("top", 0)), int(borders.get("bottom", 0))
    left = min(left, max(0, src_w - 1))
    right = min(right, max(0, src_w - 1 - left))
    top = min(top, max(0, src_h - 1))
    bottom = min(bottom, max(0, src_h - 1 - top))
    xs_src = [(0, left), (left, src_w - right), (src_w - right, src_w)]
    ys_src = [(0, top), (top, src_h - bottom), (src_h - bottom, src_h)]
    xs_dst = [(0, left), (left, width - right), (width - right, width)]
    ys_dst = [(0, top), (top, height - bottom), (height - bottom, height)]
    out = np.zeros((height, width, rgba.shape[2]), rgba.dtype)
    for (sy0, sy1), (dy0, dy1) in zip(ys_src, ys_dst):
        for (sx0, sx1), (dx0, dx1) in zip(xs_src, xs_dst):
            if sy1 <= sy0 or sx1 <= sx0 or dy1 <= dy0 or dx1 <= dx0:
                continue
            patch = rgba[sy0:sy1, sx0:sx1]
            if (dy1 - dy0, dx1 - dx0) != patch.shape[:2]:
                patch = cv2.resize(patch, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_LINEAR)
            out[dy0:dy1, dx0:dx1] = patch
    return out


def estimate_background(image, mask, pad=6):
    """Estimate what sits behind a mask by inpainting it away, on a padded crop."""
    box = bbox(mask)
    if box is None:
        return None, None
    height, width = mask.shape
    x1 = max(0, box[0] - pad)
    y1 = max(0, box[1] - pad)
    x2 = min(width, box[2] + pad)
    y2 = min(height, box[3] + pad)
    sub = image[y1:y2, x1:x2]
    sm = mask[y1:y2, x1:x2]
    if sm.sum() < 12:
        return None, None
    filled = inpaint_interp(sub.astype(np.uint8), sm, blur=True, blur_scale=0.35,
                            sim_scale=14.0, blur_max=61)
    return filled.astype(np.float32), (x1, y1, x2, y2)


def unmix_foreground(rgb, alpha, background, alpha_floor=0.80):
    """Recover the object's own colour from a composited edge, and the alpha that goes with it.

    Every partly transparent pixel is a blend: C = a*F + (1-a)*B. Cutting a sprite by simply
    pairing the source pixels with an alpha keeps B in the result, which is the halo you see
    around an extracted button - the old page colour still sitting on its rim. Solving for F
    removes it. Below `alpha_floor` the division is too ill-conditioned to trust, so those
    pixels take their colour from the nearest reliable neighbour instead.

    Borrowing a neighbour's colour leaves the pair (F, a) no longer able to reproduce C, which
    is why a rebuilt screen used to show every sprite outlined. So once the colours are settled,
    alpha is re-solved as the point on the segment B -> F that lands closest to the pixel that
    was actually there. The rim then both loses its halo and composites back to the original.
    """
    a = alpha.astype(np.float32)
    if a.max() <= 0:
        return rgb, alpha
    a3 = a[..., None]
    source = rgb.astype(np.float32)
    back = background.astype(np.float32)
    reliable = a >= alpha_floor
    fore = source.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        solved = (source - (1.0 - a3) * back) / np.maximum(a3, 1e-6)
    solved = np.clip(solved, 0.0, 255.0)
    fore[reliable] = solved[reliable]

    # carry colour outward into the faint rim, where the algebra is unstable
    unknown = (a > 0.0) & ~reliable
    if unknown.any() and reliable.any():
        fore = cv2.inpaint(fore.astype(np.uint8), unknown.astype(np.uint8) * 255, 3,
                           cv2.INPAINT_TELEA).astype(np.float32)

    edge = (a > 0.0) & (a < 1.0)
    if edge.any():
        span = fore - back
        denom = (span * span).sum(axis=2)
        numer = ((source - back) * span).sum(axis=2)
        # where the object colour and what is behind it agree there is no ratio to recover,
        # so those pixels keep the alpha the matte gave them
        usable = edge & (denom > 4.0)
        projected = np.divide(numer, denom, out=a.copy(), where=usable)
        a = np.where(usable, np.clip(projected, 0.0, 1.0), a)
    return np.clip(fore, 0, 255).astype(np.uint8), a
