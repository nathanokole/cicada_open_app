import numpy as np
from PIL import Image, ImageEnhance
import cv2
from scipy.ndimage import binary_fill_holes
import matplotlib.colors as mcolors
import random
import matplotlib.cm as cm
import pandas as pd
from skimage.measure import regionprops, label as sk_label
from utils.constants import MIN_DISPLAY_SIZE
import copy

def cm2_to_px(area_cm2: float, dpi: float) -> float:
    px_per_cm = dpi / 2.54
    return area_cm2 * (px_per_cm**2)

def merge_boxes_smart_xyxy_square(
    boxes, img_shape, iou=0.25, ioa=0.75, gap=10, dilate=2, use_ioa=False, use_gap=True
):
    """Merge if IoU>=iou OR (IoA>=ioa if use_ioa) OR (gap<=gap if use_gap); then square+clamp."""
    b = np.asarray(boxes, np.float32)
    if b.size == 0: return b.reshape(0,4).astype(np.int32)
    x1 = np.minimum(b[:,0], b[:,2]); y1 = np.minimum(b[:,1], b[:,3])
    x2 = np.maximum(b[:,0], b[:,2]); y2 = np.maximum(b[:,1], b[:,3])
    H, W = img_shape[:2]
    if dilate>0:
        x1 = np.clip(x1-dilate, 0, W); y1 = np.clip(y1-dilate, 0, H)
        x2 = np.clip(x2+dilate, 0, W); y2 = np.clip(y2+dilate, 0, H)
    w, h = x2-x1, y2-y1
    keep = (w>0)&(h>0)
    x1,y1,x2,y2,w,h = x1[keep],y1[keep],x2[keep],y2[keep],w[keep],h[keep]
    area = w*h; N = len(x1)
    if N==0: return np.empty((0,4), np.int32)

    xx1 = np.maximum(x1[:,None], x1[None,:]); yy1 = np.maximum(y1[:,None], y1[None,:])
    xx2 = np.minimum(x2[:,None], x2[None,:]); yy2 = np.minimum(y2[:,None], y2[None,:])
    iw = np.clip(xx2-xx1, 0, None); ih = np.clip(yy2-yy1, 0, None)
    inter = iw*ih
    union = area[:,None] + area[None,:] - inter
    iou_m = inter/np.maximum(union,1e-6)
    ioa_m = inter/np.maximum(np.minimum.outer(area, area), 1e-6)
    dx = np.maximum(0, np.maximum(x1[:,None], x1[None,:]) - np.minimum(x2[:,None], x2[None,:]))
    dy = np.maximum(0, np.maximum(y1[:,None], y1[None,:]) - np.minimum(y2[:,None], y2[None,:]))
    gap_m = np.maximum(dx, dy)

    # build adjacency using selected rules
    adj = (iou_m >= float(iou))
    if use_ioa:
        adj = adj | (ioa_m >= float(ioa))
    if use_gap:
        adj = adj | (gap_m <= float(gap))

    np.fill_diagonal(adj, True)
    seen = np.zeros(N, bool); merged = []
    for i in range(N):
        if seen[i]: continue
        stack=[i]; seen[i]=True; comp=[]
        while stack:
            j = stack.pop(); comp.append(j)
            nb = np.where(adj[j] & ~seen)[0]; seen[nb]=True; stack.extend(nb.tolist())
        merged.append([x1[comp].min(), y1[comp].min(), x2[comp].max(), y2[comp].max()])
    m = np.asarray(merged, np.float32)

    # square+clamp
    cx = (m[:,0]+m[:,2])/2; cy = (m[:,1]+m[:,3])/2
    side = np.clip(np.maximum(m[:,2]-m[:,0], m[:,3]-m[:,1]), 1, min(W,H))
    nx1 = np.clip((cx - side/2).astype(int), 0, W - side.astype(int))
    ny1 = np.clip((cy - side/2).astype(int), 0, H - side.astype(int))
    return np.stack([nx1, ny1, nx1+side.astype(int), ny1+side.astype(int)], 1).astype(np.int32)

def make_square_boxes(boxes, img_shape, min_size=200, pad=30):
    h_img, w_img = img_shape[:2]
    boxes = np.asarray(boxes, dtype=float)
    cx = boxes[:,0] + boxes[:,2]/2
    cy = boxes[:,1] + boxes[:,3]/2
    base = np.maximum(boxes[:,2], boxes[:,3])
    size = np.where(base < min_size, min_size, base + 2*pad).astype(int)
    size = np.minimum(size, min(h_img, w_img))
    x1 = np.clip((cx - size/2).astype(int), 0, w_img - size)
    y1 = np.clip((cy - size/2).astype(int), 0, h_img - size)
    x2 = x1 + size
    y2 = y1 + size
    return np.vstack([x1, y1, x2, y2]).T.astype(int)

def build_insect_mask(img, dpi,
                      min_cm2=0.01, max_cm2=1.0,
                      thr_main=180, thr_inside_max=160):
    """Return cleaned binary mask (uint8 0/255) for components in [min_cm2,max_cm2]."""
    ch = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ch = np.array(ImageEnhance.Brightness(ImageEnhance.Contrast(Image.fromarray(ch)).enhance(1.25)).enhance(1.5))

    min_px, max_px = cm2_to_px(min_cm2, dpi), cm2_to_px(max_cm2, dpi)

    # main threshold + small, sturdy morph
    _, b1 = cv2.threshold(ch, thr_main, 255, cv2.THRESH_BINARY_INV)
    b1 = cv2.morphologyEx(b1, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=3)
    b1 = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
    b1 = cv2.dilate(b1, np.ones((5,5), np.uint8), iterations=2)

    n1, lab1, st1, _ = cv2.connectedComponentsWithStats(b1, 8)
    areas1 = st1[:, cv2.CC_STAT_AREA]
    keep1 = np.where((areas1 > min_px) & (areas1 < max_px))[0]
    keep1 = keep1[keep1 != 0]
    m1 = np.isin(lab1, keep1).astype(np.uint8) * 255

    # split very large comps
    big = np.where(areas1 > max_px)[0]; big = big[big != 0]
    m2 = np.zeros_like(m1)
    if len(big):
        big_mask = np.isin(lab1, big)
        gray_big = np.where(big_mask, ch, 0).astype(np.uint8)
        b2 = np.where((gray_big > 0) & (gray_big < thr_inside_max), 255, 0).astype(np.uint8)
        b2 = cv2.morphologyEx(b2, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=3)
        b2 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=2)
        b2 = cv2.dilate(b2, np.ones((5,5), np.uint8), iterations=2)
        b2 = binary_fill_holes(b2.astype(bool)).astype(np.uint8)*255

        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(b2, 8)
        areas2 = st2[:, cv2.CC_STAT_AREA]
        keep2 = np.where((areas2 > min_px) & (areas2 < max_px))[0]
        keep2 = keep2[keep2 != 0]
        if len(keep2): m2 = np.isin(lab2, keep2).astype(np.uint8) * 255

    final = ((m1>0) | (m2>0)).astype(np.uint8)*255
    final = binary_fill_holes(final.astype(bool)).astype(np.uint8)*255
    return final

def detect_insect_candidates(
    img, dpi,
    min_cm2=0.008, max_cm2=1.0,
    max_square=None, max_sq_frac=None,
    merge_boxes=False
):
    mask = build_insect_mask(img, dpi, min_cm2=min_cm2, max_cm2=max_cm2)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return (np.empty((0,4), int), np.empty((0,), int), mask,
                np.empty((0,2), np.float32), [], [])

    stats, cents = stats[1:], cents[1:]
    boxes_xywh = stats[:, :4].astype(float)     # x,y,w,h
    # square to xyxy
    sq = make_square_boxes(boxes_xywh, img_shape=img.shape, min_size=160, pad=30)

    # recompute areas/centroids to match current sq set
    if sq.size:
        side = (sq[:,2]-sq[:,0]).astype(int)
        areas = (side * side).astype(int)
        cents = np.column_stack(((sq[:,0]+sq[:,2])/2.0, (sq[:,1]+sq[:,3])/2.0)).astype(np.float32)
    else:
        areas = np.empty((0,), int)
        cents = np.empty((0,2), np.float32)

    # clamp by max size (abs or fraction of min(H,W))
    if sq.size:
        H, W = img.shape[:2]
        if max_square is None and max_sq_frac is not None:
            max_square = int(max(1, round(min(H, W)*float(max_sq_frac))))
        if max_square is not None:
            keep = side <= int(max_square)
            sq, cents, areas = sq[keep], cents[keep], areas[keep]

    # optional merging (IoU-based); recompute areas/centroids afterwards
    if merge_boxes and sq.size:
        sq = merge_boxes_smart_xyxy_square(sq, img.shape, iou=0.1, use_ioa=False, use_gap=False)
        if sq.size:
            side = (sq[:,2]-sq[:,0]).astype(int)
            areas = (side * side).astype(int)
            cents = np.column_stack(((sq[:,0]+sq[:,2])/2.0, (sq[:,1]+sq[:,3])/2.0)).astype(np.float32)
        else:
            areas = np.empty((0,), int)
            cents = np.empty((0,2), np.float32)

    # ROIs
    rois, roi_masks = [], []
    for x1,y1,x2,y2 in sq.astype(int):
        x1 = max(0, min(img.shape[1]-1, x1)); x2 = max(0, min(img.shape[1], x2))
        y1 = max(0, min(img.shape[0]-1, y1)); y2 = max(0, min(img.shape[0], y2))
        rois.append(img[y1:y2, x1:x2].copy())
        roi_masks.append((mask[y1:y2, x1:x2] > 0).astype(np.uint8)*255)

    return sq.astype(int), areas.astype(int), mask, cents.astype(np.float32), rois, roi_masks

def prep_roi(roi, dpi, out_size=256, seam=5, ksize=(11,11)):
    target_dpi = 1600
    if dpi and target_dpi and dpi != target_dpi:
        scale = float(target_dpi) / float(dpi)
        new_w = int(roi.shape[1] * scale)
        new_h = int(roi.shape[0] * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    h, w = roi.shape[:2]
    # pick median border color
    brd   = np.vstack([roi[0], roi[-1], roi[:,0], roi[:,-1]]).reshape(-1,3)
    color = tuple(np.median(brd, 0).astype(int).tolist())

    if h < out_size or w < out_size:
        # compute needed padding to reach out_size
        dh = out_size - h
        dw = out_size - w
        t, b = dh//2,  dh - dh//2
        l, r = dw//2,  dw - dw//2
    else:
        # pad to square of side max(h, w)
        s  = max(h, w)
        dh = s - h
        dw = s - w
        t, b = dh//2, dh - dh//2
        l, r = dw//2, dw - dw//2

    # clamp to zero so no border is negative
    t, b, l, r = map(lambda x: max(x, 0), (t, b, l, r))

    # apply padding
    roi  = cv2.copyMakeBorder(roi,  t, b, l, r, cv2.BORDER_CONSTANT, value=color)
    
    # if we’re in the “too-small” case, blur seams and center-crop
    if h < out_size or w < out_size:
        blur = cv2.GaussianBlur(roi, ksize, 5)
        # blur only the seams
        for y1, y2 in ((t-seam, t+seam), (-b-seam, -b+seam)):
            roi[y1:y2, :] = blur[y1:y2, :]
        for x1, x2 in ((l-seam, l+seam), (-r-seam, -r+seam)):
            roi[:, x1:x2] = blur[:, x1:x2]

        # center-crop to exact out_size×out_size
        cy, cx = roi.shape[0]//2, roi.shape[1]//2
        hs = out_size // 2
        roi  = roi[ cy-hs : cy-hs+out_size, cx-hs : cx-hs+out_size ]

    # for big insects, crop to size
    if not roi.shape[0] == out_size or not roi.shape[1] == out_size:
        cy, cx = roi.shape[0]//2, roi.shape[1]//2
        hs = out_size // 2
        roi  = roi[ cy-hs : cy-hs+out_size, cx-hs : cx-hs+out_size ]
    
    return roi

def prep_roi_and_mask(roi, mask, dpi, out_size=256, seam=5, ksize=(11,11), target_dpi=1600):
    # --- scale (same factor for roi & mask) ---
    if dpi and target_dpi and dpi != target_dpi:
        s = float(target_dpi) / float(dpi)
        roi  = cv2.resize(roi,  (int(round(roi.shape[1]*s)),  int(round(roi.shape[0]*s))),  cv2.INTER_LANCZOS4)
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), cv2.INTER_NEAREST)

    # mask to single-channel uint8 {0,255}
    if mask.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8: mask = (mask > 0).astype(np.uint8)*255

    h, w = roi.shape[:2]

    # --- padding (same t,b,l,r for both) ---
    brd   = np.vstack([roi[0], roi[-1], roi[:,0], roi[:,-1]]).reshape(-1,3)
    color = tuple(np.median(brd, 0).astype(int).tolist())  # border color for ROI
    if h < out_size or w < out_size:
        dh, dw = out_size - h, out_size - w
    else:
        s  = max(h, w); dh, dw = s - h, s - w
    t, b = max(dh//2,0), max(dh - dh//2,0)
    l, r = max(dw//2,0), max(dw - dw//2,0)

    roi  = cv2.copyMakeBorder(roi,  t,b,l,r, cv2.BORDER_CONSTANT, value=color)
    mask = cv2.copyMakeBorder(mask, t,b,l,r, cv2.BORDER_CONSTANT, value=0)

    # --- resize to out_size (ROI high-quality; mask nearest) ---
    roi  = cv2.resize(roi,  (out_size, out_size), cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, (out_size, out_size), cv2.INTER_NEAREST)

    # --- seam blur + matched crop only for too-small case ---
    if h < out_size or w < out_size:
        blur = cv2.GaussianBlur(roi, ksize, 5)
        for y1, y2 in ((t-seam, t+seam), (-b-seam, -b+seam)):
            roi[max(0,y1):max(0,y2), :] = blur[max(0,y1):max(0,y2), :]
        for x1, x2 in ((l-seam, l+seam), (-r-seam, -r+seam)):
            roi[:, max(0,x1):max(0,x2)] = blur[:, max(0,x1):max(0,x2)]
        cy, cx = roi.shape[0]//2, roi.shape[1]//2
        hs = out_size//2
        sl_y = slice(cy-hs, cy-hs+out_size); sl_x = slice(cx-hs, cx-hs+out_size)
        roi  = roi[sl_y, sl_x]
        mask = mask[sl_y, sl_x]

    return roi, mask

def downsample_for_display(image, max_dim=2000):
    h,w = image.shape[:2]
    scale = min(1.0, max_dim / max(h,w))
    img8 = image
    if scale < 1.0:
        img8 = cv2.resize(img8, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img8, scale

def compute_mm_per_px(dpi):
    return 25.4 / dpi if dpi else None

def compute_morphology(pred_labels, masks, mm_per_px):
    """Compute per-ROI morphology and a label-wise summary.

    Handles mixed/None labels, 3‑channel masks, and empty regions.
    Returns (df_rows, df_summary).
    """
    files = [f"roi_{i:03d}.png" for i in range(len(masks or []))]
    rows = []
    mm = float(mm_per_px) if mm_per_px else None

    def _norm_label(x):
        # Make labels groupby-safe (hashable, consistent)
        if isinstance(x, (list, tuple, np.ndarray)):
            x = x[0] if len(x) else None
        return "Unknown" if x is None else str(x)

    for fn, lab, mask in zip(files, pred_labels or [], masks or []):
        if mask is None:
            continue
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            continue

        lbl = sk_label(m)
        if lbl.max() == 0:
            continue
        props = regionprops(lbl)
        if not props:
            continue
        p = max(props, key=lambda x: x.area)

        a = float(p.area)
        per = float(getattr(p, "perimeter", 0.0)) or 0.0
        maj = float(getattr(p, "major_axis_length", 0.0))
        minr = float(getattr(p, "minor_axis_length", 0.0))

        a_mm2 = (a * (mm ** 2)) if mm else None
        maj_mm = (maj * mm) if mm else None
        min_mm = (minr * mm) if mm else None

        rows.append(dict(
            file=fn,
            label=_norm_label(lab),
            area_mm2=a_mm2,
            major_mm=maj_mm,
            minor_mm=min_mm,
            aspect_ratio=(maj_mm / min_mm) if (maj_mm and min_mm and min_mm > 0) else None,
            circularity=(4 * np.pi * a) / (per ** 2) if per > 0 else None,
            solidity=float(getattr(p, "solidity", np.nan)) if getattr(p, "solidity", None) is not None else np.nan,
            extent=float(getattr(p, "extent", np.nan)) if getattr(p, "extent", None) is not None else np.nan,
            eccentricity=float(getattr(p, "eccentricity", np.nan)) if getattr(p, "eccentricity", None) is not None else np.nan,
            orientation=float(getattr(p, "orientation", np.nan)) if getattr(p, "orientation", None) is not None else np.nan,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    df["label"] = df["label"].astype(str)
    summary = (
        df.groupby("label", dropna=False)
          .agg(
              count=("file", "size"),
              mean_area=("area_mm2", "mean"),
              mean_major=("major_mm", "mean"),
              mean_minor=("minor_mm", "mean"),
          )
          .reset_index()
    )
    return df, summary


def ensure_min_sz(bxs, ms=MIN_DISPLAY_SIZE):
    out = copy.deepcopy(bxs)
    for i,(x,y,w,h) in enumerate(bxs):
        if w<ms or h<ms:
            dx = (ms-w)/2
            dy = (ms-h)/2
            out[i] = (x-dx, y-dy, ms, ms)
    return out

def _boxes_to_xywh(boxes, H, W):
    out = []
    for b in boxes:
        x1, y1, a, b2 = map(float, b)
        if a > x1 and b2 > y1 and a <= W and b2 <= H:
            x, y, w, h = x1, y1, a - x1, b2 - y1
        else:
            x, y, w, h = x1, y1, a, b2
        x = max(0.0, min(x, W - 1))
        y = max(0.0, min(y, H - 1))
        w = max(1.0, min(w, W - x))
        h = max(1.0, min(h, H - y))
        out.append((x, y, w, h))
    return out

