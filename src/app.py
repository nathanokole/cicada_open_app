# src/app.py
from __future__ import annotations
import hashlib
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import torch

from utils.models import ModifiedInception, get_predictions as model_predict
from utils.constants import MODEL_PATHS, MIN_DISPLAY_SIZE, M1_LABELS, M2_LABELS
from utils.io_utils import read_image_pil
from utils.viz import plot_detections, overlay_mask
from utils.image_processing import (
    detect_insect_candidates,
    prep_roi_and_mask,
    downsample_for_display,
    compute_mm_per_px,
    compute_morphology,
    ensure_min_sz,
    _boxes_to_xywh,
)

# =====================================================
# Page & minimal global state
# =====================================================
st.set_page_config(layout="wide", page_title="Planthopper Detection & Analysis")
st.title("Planthopper Detection & Analysis")
st.markdown(
    """
    <style>html,body,[class*='css']{margin:0;padding:0}.main{overflow:auto}img{max-width:100%}</style>
    """,
    unsafe_allow_html=True,
)

# bootstrap state keys (one source of truth)
for k, v in {
    "selected_idx": None,
    "ignore_next_sel": False,
    # models
    "models": {},            # name -> torch model
    "out_sizes": {},         # name -> input size used for ROI prep
    "labels_map": {},        # name -> labels list
    "primary_name": "M1",
    # overlay
    "show_overlay_mask": True,
    # annotate/edit
    "annotate_mode": False,
    "pending_disp_boxes": [],   # list of (x,y,w,h) in DISPLAY coords (dashed)
    "custom_boxes": [],         # list[(x,y,w,h)] in ORIGINAL coords (committed)
    "custom_disp_rois": [],     # list[np.ndarray] (384)
    "custom_disp_masks": [],    # list[np.ndarray] (384)
    "custom_m1_labels": [],     # list[str]
    "custom_m2_labels": [],     # list[Optional[str]]
    # for base predictions (so user edits persist across reruns for a file)
    "base_m1_labels": None,     # list[str] aligned with detections
    "base_m2_map": None,        # dict[int,str]
    # file tracking
    "current_file_hash": None,
}.items():
    st.session_state.setdefault(k, v)

# =====================================================
# Cache helpers (I/O, heavy ops)
# =====================================================
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
import json, os

def download_from_drive(file_id, dest_path):
    #d = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"].replace("\\n", "\n"))
    #st.code(d)
    print(repr(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"][1946:]), flush=True)
    e = json.loads("{" + os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"][1946:])
    
    st.code(e)
    creds = service_account.Credentials.from_service_account_file(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]))
    print("Cred: ", creds, flush=True)
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print("Download: ", done, flush=True)
            print("Download %d%%." % int(status.progress() * 100), flush=True)

@st.cache_resource(show_spinner=True)
def load_model(path: str, file_id: str, backbone_out_: int, num_classes_: int):
    print(os.path.exists(path), flush=True)
    print(file_id, flush=True)
    print(path, flush=True)
    download_from_drive(file_id, path)
    print(os.path.exists(path), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ModifiedInception(pretrained_path=path, backbone_out=backbone_out_, num_classes=num_classes_)
    return m.to(device).eval()

@st.cache_data(show_spinner=True)
def detect_stage(img: np.ndarray, dpi: int):
    return detect_insect_candidates(img, dpi, max_square=1500, merge_boxes=True)

@st.cache_data(show_spinner=True)
def prep_stage(rois: List[np.ndarray], masks: List[np.ndarray], dpi: Optional[int], size: int):
    pairs = [prep_roi_and_mask(r, m, dpi, out_size=size) for r, m in zip(rois, masks)]
    pr_rois, pr_masks = map(list, zip(*pairs)) if pairs else ([], [])
    inputs = [r.astype("float32") / 255.0 for r in pr_rois]
    return pr_rois, pr_masks, inputs

@st.cache_data(show_spinner=True)
def morph_stage(lbls: List[str], msks: List[np.ndarray], dpi_val: Optional[int]):
    mm_per_px = compute_mm_per_px(dpi_val) if dpi_val else compute_mm_per_px(1600)
    return compute_morphology(lbls, msks, mm_per_px)

# =====================================================
# Utilities
# =====================================================
def _conf(name: str) -> Dict:
    path, file_id, backbone_out, out_size, num_classes = MODEL_PATHS[name]
    return {"path": path, "file_id": file_id, "backbone_out": backbone_out, "out_size": out_size, "num_classes": num_classes}

def _norm_pred(p) -> str:
    if p is None:
        return "Unknown"
    if isinstance(p, dict):
        if "label" in p:
            return str(p["label"])
        try:
            return str(max(p.items(), key=lambda kv: float(kv[1]))[0])
        except Exception:
            return str(p)
    if isinstance(p, (list, tuple, np.ndarray)):
        return str(p[0]) if len(p) else "Unknown"
    return str(p)

def predict_with(model_name: str, imgs: List[np.ndarray]) -> List[str]:
    if not imgs:
        return []
    m = st.session_state.models.get(model_name)
    if m is None:
        raise RuntimeError(f"Model '{model_name}' not loaded")
    labels = st.session_state.labels_map.get(model_name, M2_LABELS if model_name == "M2" else M1_LABELS)
    out = model_predict(m, imgs, labels=labels)
    preds = out[0] if isinstance(out, (list, tuple)) and len(out) > 1 else out
    return [_norm_pred(p) for p in preds]

def _bar_figure(counts: pd.Series) -> go.Figure:
    cols = ["red" if v == counts.max() else "#4e79a7" for v in counts]
    bar = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=cols, width=0.5))
    bar.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=30))
    return bar

# dashed preview rectangles for pending selections
_DEF_DASH = dict(color="black", width=2, dash="dash")

def add_pending_shapes(fig: go.Figure, pending_xywh: List[Tuple[float, float, float, float]]):
    for (x, y, w, h) in pending_xywh:
        fig.add_shape(
            type="rect",
            x0=x, y0=y, x1=x + w, y1=y + h,
            line=_DEF_DASH,
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

# quick-and-robust mask from ROI crop (largest component)
def _roi_mask_from_crop(roi: np.ndarray) -> np.ndarray:
    labL = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)[..., 0]
    inv = cv2.bitwise_not(labL)
    tmp = cv2.convertScaleAbs(inv, alpha=1.25, beta=-50)
    tmp = cv2.dilate(tmp, np.ones((3, 3), np.uint8), iterations=1)
    _, bw = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return bw
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8) * 255

# convert one display box to committed custom ROI entries

def commit_one_display_box_to_custom(box_disp_xywh: Tuple[float, float, float, float],
                                     image_arr: np.ndarray,
                                     dpi: Optional[int],
                                     scale: float,
                                     m1_name: str,
                                     m2_name: str,
                                     ):  # mutates session_state
    H, W = image_arr.shape[:2]
    inv = 1.0 / max(scale, 1e-9)
    x, y, w, h = box_disp_xywh
    xo, yo, wo, ho = int(round(x * inv)), int(round(y * inv)), int(round(w * inv)), int(round(h * inv))
    xo = max(0, min(W - 1, xo)); yo = max(0, min(H - 1, yo))
    wo = max(1, min(W - xo, wo)); ho = max(1, min(H - yo, ho))

    roi = image_arr[yo:yo + ho, xo:xo + wo].copy()
    if roi.size == 0:
        return

    raw_mask = _roi_mask_from_crop(roi)

    # inference sizes
    m1_size = st.session_state.out_sizes.get(m1_name, MODEL_PATHS[m1_name][3])
    roi_m1, _ = prep_roi_and_mask(roi, raw_mask, dpi, out_size=m1_size)
    roi_inf = roi_m1.astype("float32") / 255.0

    lab_m1 = predict_with(m1_name, [roi_inf])[0] if len(roi_inf) else "Unknown"

    lab_m2 = None
    if (m2_name in st.session_state.models) and lab_m1 in {"Cixiidae", "Planthopper"}:
        m2_size = st.session_state.out_sizes.get(m2_name, MODEL_PATHS[m2_name][3])
        if m2_size != m1_size:
            roi_m2, _ = prep_roi_and_mask(roi, raw_mask, dpi, out_size=m2_size)
            roi_inf2 = roi_m2.astype("float32") / 255.0
        else:
            roi_inf2 = roi_inf
        lab_m2 = predict_with(m2_name, [roi_inf2])[0]

    roi_disp, mask_disp = prep_roi_and_mask(roi, raw_mask, dpi, out_size=384)

    st.session_state.custom_boxes.append((xo, yo, wo, ho))
    st.session_state.custom_disp_rois.append(roi_disp)
    st.session_state.custom_disp_masks.append(mask_disp)
    st.session_state.custom_m1_labels.append(lab_m1)
    st.session_state.custom_m2_labels.append(lab_m2)

# commit all pending selections

def commit_all_pending(image_arr: np.ndarray, dpi: Optional[int], scale: float, m1_name: str, m2_name: str):
    if not st.session_state.pending_disp_boxes:
        return 0
    total = len(st.session_state.pending_disp_boxes)
    with st.status(f"Committing {total} ROI(s)…", expanded=True) as s:
        for i, box in enumerate(list(st.session_state.pending_disp_boxes)):
            s.update(label=f"Processing {i+1}/{total}…")
            commit_one_display_box_to_custom(box, image_arr, dpi, scale, m1_name, m2_name)
        s.update(label="Done", state="complete")
    st.session_state.pending_disp_boxes = []
    return total

# =====================================================
# Model preload (lightweight, no fragments to avoid callback quirks)
# =====================================================
loaded, errors = [], {}
for base, labels in (("M1", M1_LABELS), ("M2", M2_LABELS)):
    if base in MODEL_PATHS and base not in st.session_state.models:
        cfg = _conf(base)
        try:
            st.session_state.models[base] = load_model(cfg["path"], cfg["file_id"], cfg["backbone_out"], cfg["num_classes"])
            st.session_state.out_sizes[base] = cfg["out_size"]
            st.session_state.labels_map[base] = labels
            loaded.append(base)
            print(cfg, flush=True)
        except Exception as e:
            errors[base] = str(e)
            print(e, flush=True)

# =====================================================
# Sidebar: models + annotate controls
# =====================================================
with st.sidebar.expander("Models", expanded=True):
    if loaded:
        st.success("Preloaded: " + ", ".join(loaded))
    if errors:
        for k, e in errors.items():
            st.error(f"Failed to preload {k}: {e}")

    base_loaded = list(st.session_state.models.keys())
    st.markdown("**Loaded:** " + ", ".join([f"`{n}`" for n in base_loaded]) if base_loaded else "No models loaded")

    if base_loaded:
        cur = st.session_state.get("primary_name", "M1")
        default_idx = base_loaded.index(cur) if cur in base_loaded else 0
        choice = st.selectbox("Primary model (UI only)", base_loaded, index=default_idx)
        st.session_state.primary_name = choice

with st.sidebar.expander("Annotate / Edit", expanded=True):
    st.toggle("Annotate mode (box-select to add ROI)", key="annotate_mode", help="When ON, existing boxes are dimmed and not clickable.")
    cols_btn = st.columns(2)
    with cols_btn[0]:
        if st.button("Clear my annotations"):
            st.session_state.custom_boxes = []
            st.session_state.custom_disp_rois = []
            st.session_state.custom_disp_masks = []
            st.session_state.custom_m1_labels = []
            st.session_state.custom_m2_labels = []
            st.toast("Annotations cleared")
    with cols_btn[1]:
        if st.button("Clear pending"):
            st.session_state.pending_disp_boxes = []
            st.toast("Pending cleared")

# overlay toggle (single source of truth)
st.sidebar.checkbox("Overlay mask", key="show_overlay_mask", value=st.session_state.get("show_overlay_mask", True))

# =====================================================
# File upload
# =====================================================

def _reset_on_new():
    # clear selection & all per-image session state
    for k in ("selected_idx", "ignore_next_sel", "pending_disp_boxes"):
        st.session_state[k] = None if k == "selected_idx" else []
    for k in ("custom_boxes", "custom_disp_rois", "custom_disp_masks", "custom_m1_labels", "custom_m2_labels"):
        st.session_state[k] = []
    st.session_state.base_m1_labels = None
    st.session_state.base_m2_map = None
    st.session_state.current_file_hash = None

uploaded = st.file_uploader(
    "Choose an image…", ["png", "jpg", "jpeg", "tif", "tiff"], on_change=_reset_on_new
)
if not uploaded:
    st.stop()

image_arr, dpi, file_hash = read_image_pil(uploaded)
st.session_state.current_file_hash = file_hash

# =====================================================
# Detection → ROI prep (base detections)
# =====================================================
boxes, areas, full_mask, cents, rois, masks = detect_stage(image_arr, dpi)
if len(boxes) == 0:
    st.error("No detections found. Adjust parameters or try another image.")
    st.stop()

st.sidebar.markdown(f"Detections: **{len(cents)}**")

# Stage‑1 classifier: always M1
m1_name = "M1"
if m1_name not in st.session_state.models:
    st.error("M1 not loaded; cannot classify.")
    st.stop()

_, _, m1_inputs = prep_stage(rois, masks, dpi, st.session_state.out_sizes.get(m1_name, MODEL_PATHS[m1_name][2]))
disp_rois, disp_masks, _ = prep_stage(rois, masks, dpi, 384)

# predictions (initialize base_* once per file to keep edits)
if st.session_state.base_m1_labels is None:
    st.session_state.base_m1_labels = predict_with(m1_name, m1_inputs) if m1_inputs else []

# Stage‑2 classifier (M2) only for M1 in {Cixiidae, Planthopper}
m2_name = "M2"
if st.session_state.base_m2_map is None:
    needs_m2 = {i for i, lab in enumerate(st.session_state.base_m1_labels) if lab in {"Cixiidae", "Planthopper"}}
    m2_map: Dict[int, str] = {}
    if needs_m2 and (m2_name in st.session_state.models):
        m1_size = st.session_state.out_sizes.get(m1_name, MODEL_PATHS[m1_name][3])
        m2_size = st.session_state.out_sizes.get(m2_name, MODEL_PATHS[m2_name][3])
        sel_idxs = sorted(needs_m2)
        if m1_size == m2_size:
            sel_imgs = [m1_inputs[i] for i in sel_idxs]
        else:
            _, _, m2_inputs_all = prep_stage(rois, masks, dpi, m2_size)
            sel_imgs = [m2_inputs_all[i] for i in sel_idxs]
        sel_preds = predict_with(m2_name, sel_imgs)
        for idx, lab in zip(sel_idxs, sel_preds):
            m2_map[idx] = lab
    st.session_state.base_m2_map = m2_map

# =====================================================
# Display image and scaled boxes
# =====================================================
H, W = image_arr.shape[:2]
boxes_xywh = _boxes_to_xywh(boxes, H, W)

disp_img, scale = downsample_for_display(image_arr, 2000)
scaled = [(x * scale, y * scale, w * scale, h * scale) for (x, y, w, h) in boxes_xywh]
min_disp = max(1.0, float(MIN_DISPLAY_SIZE) * float(scale))
disp_boxes_base = ensure_min_sz(scaled, min_disp)

# =====================================================
# Compose base + custom for the rest of the app
# =====================================================
# boxes for display
custom_disp_boxes = [
    (bx * scale, by * scale, bw * scale, bh * scale)
    for (bx, by, bw, bh) in st.session_state.custom_boxes
]
all_disp_boxes = list(disp_boxes_base) + custom_disp_boxes

# rois/masks/labels for display & stats
all_disp_rois = list(disp_rois) + list(st.session_state.custom_disp_rois)
all_disp_masks = list(disp_masks) + list(st.session_state.custom_disp_masks)
all_m1_labels  = list(st.session_state.base_m1_labels) + list(st.session_state.custom_m1_labels)

# M2 mapping over combined indices
m2_labels_by_idx_all: Dict[int, str] = dict(st.session_state.base_m2_map)
base_len = len(st.session_state.base_m1_labels)
for i, lab in enumerate(st.session_state.custom_m2_labels):
    if lab:
        m2_labels_by_idx_all[base_len + i] = lab

# =====================================================
# Controls row above plot (annotate status + commit buttons)
# =====================================================
ui_top = st.container()
with ui_top:
    c1, c2 = st.columns([3, 2])
    with c1:
        if st.session_state.annotate_mode:
            st.info("Annotate mode: existing boxes are dimmed and not clickable.")
        else:
            st.caption("Click a colored box to open ROI on the right.")
    with c2:
        pending_n = len(st.session_state.pending_disp_boxes)
        b_cols = st.columns([2,1,1])
        with b_cols[0]:
            st.markdown(f"**Pending:** {pending_n}")
        with b_cols[1]:
            if st.button("Commit pending", use_container_width=True, disabled=pending_n==0):
                n = commit_all_pending(image_arr, dpi, scale, m1_name, m2_name)
                if n:
                    st.toast(f"Committed {n} ROI(s)")
        with b_cols[2]:
            if st.button("Discard", use_container_width=True, disabled=pending_n==0):
                st.session_state.pending_disp_boxes = []

# =====================================================
# Sidebar filters
# =====================================================
unique = sorted(set(all_m1_labels))
opts = ["All"] + (["All without Artefact"] if "Artefact" in unique else []) + unique
selected_label = st.sidebar.radio("Filter detections", opts, key="label_filter")
if selected_label != st.session_state.get("label_filter_prev", "All"):
    st.session_state.selected_idx = None
    st.session_state.ignore_next_sel = True
    st.session_state.label_filter_prev = selected_label

# =====================================================
# Plot + right panel (zoom + label editing)
# =====================================================

# helper: build fig once, add dashed pending overlays, dim traces in annotate

def build_main_figure(img, boxes_xywh_disp, labels_plot, uirev: str) -> go.Figure:
    fig = plot_detections(img, boxes_xywh_disp, labels_plot, uirevision=uirev)
    if st.session_state.annotate_mode:
        # dim existing traces
        for tr in fig.data:
            if hasattr(tr, "opacity"):
                tr.opacity = 0.25
        # overlay dashed rectangles for pending selections
        add_pending_shapes(fig, st.session_state.pending_disp_boxes)
    return fig

# choose subset by filter
sel = st.session_state.get("label_filter", "All")
if sel == "All":
    keep = list(range(len(all_m1_labels)))
elif sel == "All without Artefact":
    keep = [i for i, p in enumerate(all_m1_labels) if p != "Artefact"]
else:
    keep = [i for i, p in enumerate(all_m1_labels) if p == sel]
keep = [i for i in keep if 0 <= i < len(all_disp_boxes)]  # guard

pick = lambda xs: [xs[i] for i in keep]
boxes_f, rois_f, masks_f = pick(all_disp_boxes), pick(all_disp_rois), pick(all_disp_masks)

# labels for plotting (append M2 where present)
labels_plot = [
    f"{all_m1_labels[i]} ({m2_labels_by_idx_all[i]})" if i in m2_labels_by_idx_all else all_m1_labels[i]
    for i in keep
]

col_main, col_side = st.columns([2, 1], gap="small")
with col_main:
    rev = hashlib.md5(f"{file_hash}-{len(boxes_f)}-M1M2-{sel}-annot{int(st.session_state.annotate_mode)}-pend{len(st.session_state.pending_disp_boxes)}".encode()).hexdigest()
    fig = build_main_figure(disp_img, boxes_f, labels_plot, uirev=rev)

    if st.session_state.annotate_mode:
        # annotate mode: use Streamlit chart with box selection callback
        def _on_box_select():
            st_state = st.session_state.get("main_chart_state")
            if not st_state or not getattr(st_state, "selection", None) or not st_state.selection.box:
                return
            for box in st_state.selection.box:
                x0, x1 = box["x"]; y0, y1 = box["y"]
                x = min(x0, x1); y = min(y0, y1); w = abs(x1 - x0); h = abs(y1 - y0)
                # ignore tiny boxes
                if w < 5 or h < 5:
                    continue
                st.session_state.pending_disp_boxes.append((x, y, w, h))
        st.plotly_chart(
            fig,
            key="main_chart_state",
            selection_mode="box",
            on_select=_on_box_select,
            use_container_width=True,
        )
        st.caption("Annotate mode is ON — draw boxes; dashed = pending. Use **Commit pending** above to add them.")
    else:
        sel_evt = plotly_events(fig, click_event=True, hover_event=False, key="det_plot")
        if st.session_state.ignore_next_sel:
            sel_evt = []
            st.session_state.ignore_next_sel = False
        if sel_evt:
            curve = sel_evt[0].get("curveNumber")
            if curve is not None and curve != st.session_state.selected_idx:
                st.session_state.selected_idx = curve

with col_side:
    if st.session_state.annotate_mode:
        st.info("Annotate mode active — existing boxes are disabled. Turn OFF to click and inspect ROIs.")
    else:
        if st.session_state.selected_idx is not None:
            if st.button("Close ROI"):
                st.session_state.selected_idx = None
                st.session_state.ignore_next_sel = True
        idx = st.session_state.selected_idx
        if idx is None or idx >= len(rois_f):
            st.write("Click a box to select ROI")
        else:
            orig_i = keep[idx]
            roi_img, mask_img = all_disp_rois[orig_i], all_disp_masks[orig_i]
            show_mask = st.session_state.get("show_overlay_mask", True)
            st.image(
                overlay_mask(roi_img, mask_img) if show_mask else roi_img,
                caption=(
                    f"ROI #{idx} | M1: {all_m1_labels[orig_i]}"
                    + (f" | M2: {m2_labels_by_idx_all[orig_i]}" if orig_i in m2_labels_by_idx_all else "")
                ),
                use_container_width=True,
            )

            # — Label editing —
            base_len = len(st.session_state.base_m1_labels)
            is_base = orig_i < base_len
            m1_current = (st.session_state.base_m1_labels[orig_i] if is_base else st.session_state.custom_m1_labels[orig_i - base_len])
            m2_current = m2_labels_by_idx_all.get(orig_i)

            m1_new = st.selectbox("M1 label", M1_LABELS, index=M1_LABELS.index(m1_current) if m1_current in M1_LABELS else 0, key=f"m1_edit_{file_hash}_{orig_i}")
            # update if changed
            if m1_new != m1_current:
                if is_base:
                    st.session_state.base_m1_labels[orig_i] = m1_new
                else:
                    st.session_state.custom_m1_labels[orig_i - base_len] = m1_new
                # if moved away from M2-eligible, clear m2
                if m1_new not in {"Cixiidae", "Planthopper"}:
                    if is_base:
                        st.session_state.base_m2_map.pop(orig_i, None)
                    else:
                        st.session_state.custom_m2_labels[orig_i - base_len] = None
                st.session_state.ignore_next_sel = True
                st.experimental_rerun()

            # show M2 dropdown only if this ROI currently has M2 label
            if m2_current is not None:
                # build current value's index
                idx_m2 = M2_LABELS.index(m2_current) if m2_current in M2_LABELS else 0
                m2_new = st.selectbox("M2 label", M2_LABELS, index=idx_m2, key=f"m2_edit_{file_hash}_{orig_i}")
                if m2_new != m2_current:
                    if is_base:
                        st.session_state.base_m2_map[orig_i] = m2_new
                    else:
                        st.session_state.custom_m2_labels[orig_i - base_len] = m2_new
                    st.session_state.ignore_next_sel = True
                    st.experimental_rerun()

            # Per-ROI morphology (from combined df calculated below)
            # We’ll render after df_morph_all is computed.

# =====================================================
# Stats & summaries (computed on combined data)
# =====================================================

# recompute combined labels map after any edits
all_m1_labels  = list(st.session_state.base_m1_labels) + list(st.session_state.custom_m1_labels)
# refresh m2 map
m2_labels_by_idx_all = dict(st.session_state.base_m2_map)
base_len = len(st.session_state.base_m1_labels)
for i, lab in enumerate(st.session_state.custom_m2_labels):
    if lab:
        m2_labels_by_idx_all[base_len + i] = lab

# morphology for all
if all_disp_masks:
    df_morph_all, summary_all = morph_stage(all_m1_labels, all_disp_masks, dpi)
else:
    df_morph_all, summary_all = pd.DataFrame(), pd.DataFrame()

# M2 subset
summary_m2 = pd.DataFrame()
if m2_labels_by_idx_all:
    idxs2 = sorted(m2_labels_by_idx_all.keys())
    masks2 = [all_disp_masks[i] for i in idxs2 if i < len(all_disp_masks)]
    labels2 = [m2_labels_by_idx_all[i] for i in idxs2 if i < len(all_disp_masks)]
    if masks2:
        _, summary_m2 = morph_stage(labels2, masks2, dpi)

# per-ROI morphology table in side panel (if a ROI is open)
if st.session_state.selected_idx is not None and not df_morph_all.empty:
    idx = st.session_state.selected_idx
    if 0 <= idx < len(keep):
        orig_i = keep[idx]
        if 0 <= orig_i < len(df_morph_all):
            drow = df_morph_all.iloc[orig_i]
            per_roi = pd.DataFrame(
                {
                    "Value": [
                        drow.area_mm2,
                        drow.major_mm,
                        drow.minor_mm,
                        drow.aspect_ratio,
                        drow.circularity,
                        drow.solidity,
                        drow.extent,
                        drow.eccentricity,
                        drow.orientation,
                    ]
                },
                index=[
                    "Area (mm²)",
                    "Major (mm)",
                    "Minor (mm)",
                    "Aspect",
                    "Circularity",
                    "Solidity",
                    "Extent",
                    "Eccentricity",
                    "Orientation (rad)",
                ],
            )
            with col_side:
                st.markdown("**Morphology (this ROI)**")
                st.dataframe(per_roi)

# counts and summaries
col_counts_left, col_counts_right = st.columns(2, gap="small")
with col_counts_left:
    st.markdown("**Detections per label (M1)**")
    counts = pd.Series(all_m1_labels).value_counts().sort_index()
    if not counts.empty:
        st.plotly_chart(_bar_figure(counts), use_container_width=True)
    else:
        st.write("No detections.")

with col_counts_right:
    st.markdown("**Detections per label (M2 subset)**")
    if m2_labels_by_idx_all:
        m2_counts = pd.Series(list(m2_labels_by_idx_all.values())).value_counts().sort_index()
        st.plotly_chart(_bar_figure(m2_counts), use_container_width=True)
    else:
        st.write("No M2 refinements.")

col_morph_left, col_morph_right = st.columns(2, gap="small")
with col_morph_left:
    st.markdown("**Morphology summary (M1 labels)**")
    st.dataframe(summary_all.set_index("label") if not summary_all.empty else pd.DataFrame())

with col_morph_right:
    st.markdown("**Morphology summary (M2 subset)**")
    st.dataframe(summary_m2.set_index("label") if not summary_m2.empty else pd.DataFrame())

# =====================================================
# Export (ZIP/CSV + Excel) — reflects edits and custom ROIs
# =====================================================
with st.expander("Export results", expanded=False):
    # Per-instance table
    inst_rows = []
    # base boxes back-computed from disp_boxes_base/scale, but export in original coords when possible
    # Build combined original coords list: base (from boxes_xywh) + custom
    all_orig_boxes = list(boxes_xywh) + list(st.session_state.custom_boxes)
    for i in range(len(all_orig_boxes)):
        b = all_orig_boxes[i]
        x, y, w, h = b
        lab_m1 = all_m1_labels[i]
        lab_m2 = m2_labels_by_idx_all.get(i)
        inst_rows.append({
            "idx": i,
            "m1": lab_m1,
            "m2": lab_m2,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        })
    df_instances = pd.DataFrame(inst_rows)

    # Write ZIP (CSVs)
    import zipfile
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("instances.csv", df_instances.to_csv(index=False))
        zf.writestr("morph_all.csv", df_morph_all.to_csv(index=False))
        zf.writestr("summary_m1.csv", summary_all.to_csv(index=False))
        zf.writestr("summary_m2.csv", summary_m2.to_csv(index=False))
    st.download_button("Download ZIP (CSVs)", data=zip_buffer.getvalue(), file_name=f"results_{file_hash}.zip")

    # Excel (multi-sheet)
    xls_buffer = BytesIO()
    with pd.ExcelWriter(xls_buffer, engine="openpyxl") as writer:
        df_instances.to_excel(writer, index=False, sheet_name="instances")
        (df_morph_all if not df_morph_all.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="morph_all")
        (summary_all if not summary_all.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="summary_m1")
        (summary_m2 if not summary_m2.empty else pd.DataFrame()).to_excel(writer, index=False, sheet_name="summary_m2")
    st.download_button("Download Excel", data=xls_buffer.getvalue(), file_name=f"results_{file_hash}.xlsx")
