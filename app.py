"""
Confidence-gated temporal fusion: one-page demo and results.

Repo layout (unzip app_bundle.zip from the notebook next to this file):
    app.py, requirements.txt, packages.txt
    models/   variantA_V3.pt, variantB_best.pt, baseline2_unet_400.pt, fog_classifier.json
    results/  results_fog_fix_hybrid.json, results_day14_master.json, results_day15_ablations.json,
              results_extended_test.json
    examples/<scene id>/ ref.png, warped_prev1.png, warped_prev2.png, mask1.png, mask2.png, clean.png, meta.json

All preprocessing matches the notebook: RGB order, values / 255, clamp to [0, 1] before uint8.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
DEMO_ORDER = ["004624", "003320", "006833", "001140", "003311"]       # shown first in the scene list
NAMES = {"B1_raw": "Raw degraded frame (B1)", "B2_unet": "Single-frame U-Net (B2)",
         "B3_equal_avg": "Equal-weight average (B3)", "VariantA": "Variant A: confidence-gated fusion",
         "VariantB": "Variant B: detection-aware fine-tuning", "RestoreFuse_all": "Restore-then-fuse on every scene",
         "Hybrid_route": "Hybrid (final system)", "Hybrid_RestoreFuse": "Hybrid with restore-then-fuse on fog",
         "clean_upper": "Clean image (upper bound)"}
MODELS, RESULTS, EXAMPLES = HERE / "models", HERE / "results", HERE / "examples"
torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Model code, copied from the final notebook (Day 7, Day 8 fix, Day 9d)
# ---------------------------------------------------------------------------
def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.ReLU(True))


def _lum(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def quality_cues(x, k=15):
    y = _lum(x)
    m = F.avg_pool2d(y, k, 1, k // 2, count_include_pad=False)
    v = (F.avg_pool2d(y * y, k, 1, k // 2, count_include_pad=False) - m * m).clamp_min(0)
    return torch.cat([m, v.sqrt()], 1)


def img_feats(x):
    lum = _lum(x)[:, 0]
    dx = (lum[:, :, 1:] - lum[:, :, :-1]).abs().mean((1, 2))
    dy = (lum[:, 1:, :] - lum[:, :-1, :]).abs().mean((1, 2))
    mn = x.min(1)[0].mean((1, 2))
    rg = (x.max(1)[0] - x.min(1)[0]).mean((1, 2))
    return torch.stack([lum.mean((1, 2)), lum.flatten(1).std(1, unbiased=False), dx + dy, mn, rg], 1)


class ConfidenceGatingNetV2(nn.Module):
    def __init__(self, base=32, bound=3.0):
        super().__init__()
        self.bound = bound
        cin = 9 + 6 + 2
        self.e1, self.e2, self.e3 = conv_block(cin, base), conv_block(base, base * 2), conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.glob = nn.Sequential(nn.Linear(base * 4, base * 4), nn.ReLU(True))
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2);     self.d1 = conv_block(base * 2, base)
        self.logits = nn.Conv2d(base, 3, 1)


class GateV3(ConfidenceGatingNetV2):
    """F = sum_k C_k I_k, C = softmax(image-level logits + bounded per-pixel correction). Order: t-2, t-1, t."""
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))   # real values come from the checkpoint
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.sel = nn.Sequential(nn.Linear(15, 32), nn.ReLU(True), nn.Linear(32, 3))

    def image_logits(self, ref, w1, w2):
        f = torch.cat([img_feats(w2), img_feats(w1), img_feats(ref)], 1)
        return self.sel((f - self.mu) / self.sd)

    def forward(self, ref, w1, w2, m1, m2):
        cues = torch.cat([quality_cues(w2), quality_cues(w1), quality_cues(ref)], 1)
        x = torch.cat([ref, w1, w2, cues, m1, m2], 1)
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        e3 = e3 + self.glob(e3.mean((2, 3)))[:, :, None, None]
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        local = 2.0 * torch.tanh(self.logits(d1) / 2.0)
        g = self.image_logits(ref, w1, w2)
        C = torch.softmax(g[:, :, None, None] + local, dim=1)
        return C[:, 0:1] * w2 + C[:, 1:2] * w1 + C[:, 2:3] * ref, C


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(True))
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True))
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(True))
        self.out = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 3, 3, padding=1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.dec(e2)
        if d.shape[-2:] != e1.shape[-2:]:
            d = F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([d, e1], 1)) + x


# ---------------------------------------------------------------------------
# Loading (cached; strict=True so a wrong checkpoint fails loudly instead of giving garbage)
# ---------------------------------------------------------------------------
def _gate(path):
    m = GateV3(np.zeros(15), np.ones(15))
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)   # restores mu / sd too
    return m.eval()


@st.cache_resource
def load_models():
    a = _gate(MODELS / "variantA_V3.pt")
    b = _gate(MODELS / "variantB_best.pt") if (MODELS / "variantB_best.pt").exists() else None
    u = SmallUNet()
    u.load_state_dict(torch.load(MODELS / "baseline2_unet_400.pt", map_location="cpu"), strict=True)
    clf_path = MODELS / "fog_classifier.json"
    clf = json.load(open(clf_path)) if clf_path.exists() else None
    return a, b, u.eval(), clf


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rgb_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]


def mask_tensor(gray):
    return torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]


def pad_to(t, mult, mode):
    h, w = t.shape[-2:]
    return F.pad(t, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), mode=mode)


def to_rgb_uint8(t):
    return (t[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def frame_feats(x):                       # same as the notebook; x: (3,H,W) RGB in [0,1]
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    grad = np.abs(np.diff(lum, axis=1)).mean() + np.abs(np.diff(lum, axis=0)).mean()
    return [lum.mean(), lum.std(), grad, x.min(0).mean(), (x.max(0) - x.min(0)).mean()]


def fog_probability(clf, ref, w1, w2):
    x = np.array(frame_feats(w2[0].numpy()) + frame_feats(w1[0].numpy()) + frame_feats(ref[0].numpy()))
    z = float(((x - np.array(clf["mean"])) / np.array(clf["scale"])) @ np.array(clf["coef"]) + clf["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def with_boxes(rgb, yolo):
    if yolo is None: return rgb
    out = yolo(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), verbose=False)[0].plot()
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def run_all(ref_b, w1_b, w2_b, m1_g, m2_g):
    net_a, net_b, unet, clf = load_models()
    H, W = ref_b.shape[:2]
    fit = lambda im, interp: im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H), interpolation=interp)
    w1_b, w2_b = fit(w1_b, cv2.INTER_AREA), fit(w2_b, cv2.INTER_AREA)
    m1_g, m2_g = fit(m1_g, cv2.INTER_NEAREST), fit(m2_g, cv2.INTER_NEAREST)

    ref, w1, w2 = rgb_tensor(ref_b), rgb_tensor(w1_b), rgb_tensor(w2_b)
    m1, m2 = mask_tensor(m1_g), mask_tensor(m2_g)
    crop = lambda t: t[:, :, :H, :W]
    p8 = lambda t: pad_to(t, 8, "replicate")
    restore = lambda t: crop(unet(p8(t))).clamp(0, 1)

    fused_a, C = net_a(p8(ref), p8(w1), p8(w2), p8(m1), p8(m2))
    fused_a, C = crop(fused_a), crop(C)
    out = {
        "Raw degraded frame (B1)": to_rgb_uint8(ref),
        "Single-frame U-Net (B2)": to_rgb_uint8(restore(ref)),
        "Equal-weight average (B3)": np.clip((ref_b.astype(np.float32) + w1_b + w2_b) / 3, 0, 255).round()
                                        .astype(np.uint8)[:, :, ::-1].copy(),
        "Variant A: confidence-gated fusion": to_rgb_uint8(fused_a),
    }
    if net_b is not None:
        p32 = lambda t: pad_to(t, 32, "reflect")
        fused_b, _ = net_b(p32(ref), p32(w1), p32(w2), p32(m1), p32(m2))
        out["Variant B: detection-aware fine-tuning"] = to_rgb_uint8(crop(fused_b))

    route = None
    if clf is not None:
        p_fog = fog_probability(clf, ref, w1, w2)
        if p_fog >= 0.5:
            hybrid = restore(ref)
            route = ("fog", p_fog, "U-Net restoration, because blending foggy frames cannot remove haze")
        else:
            hybrid = fused_a
            route = ("motion blur", 1 - p_fog, "confidence-gated fusion (Variant A)")
        out["Hybrid (final system)"] = to_rgb_uint8(hybrid)

    lo = torch.minimum(torch.minimum(ref, w1), w2); hi = torch.maximum(torch.maximum(ref, w1), w2)
    sane = ((fused_a < lo - 1e-3) | (fused_a > hi + 1e-3)).float().mean().item() < 0.001
    return out, C[0].numpy(), route, sane


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Confidence-gated temporal fusion", layout="wide")
st.title("Confidence-gated temporal fusion for object detection in fog and motion blur")
st.write("The system aligns the current camera frame with the two frames before it, learns how much to trust "
         "each one, and blends them into a cleaner frame for the object detector.")

missing = [n for n in ("variantA_V3.pt", "baseline2_unet_400.pt") if not (MODELS / n).exists()]
if missing:
    st.error(f"Model files missing from models/: {', '.join(missing)}. Unzip app_bundle.zip into the repo.")
    st.stop()

tab_try, tab_results, tab_how = st.tabs(["Try it", "Results", "How it works"])

def try_tab():
    examples = sorted((p for p in EXAMPLES.iterdir() if p.is_dir()),
                      key=lambda p: (DEMO_ORDER.index(p.name) if p.name in DEMO_ORDER else len(DEMO_ORDER), p.name)
                      ) if EXAMPLES.exists() else []
    modes = (["Test scene from our dataset"] if examples else []) + ["Upload your own aligned frames"]
    mode = st.radio("Input", modes, horizontal=True)
    clean_rgb, meta = None, None

    if mode.startswith("Test scene"):
        def label(p):
            m = json.load(open(p / "meta.json"))
            return f"Scene {p.name}: {m['type']}, current frame {m['severity_t']}"
        pick = st.selectbox("Scene", examples, format_func=label)
        rd = lambda n, flag=cv2.IMREAD_COLOR: cv2.imread(str(pick / n), flag)
        ref_b, w1_b, w2_b = rd("ref.png"), rd("warped_prev1.png"), rd("warped_prev2.png")
        m1_g, m2_g = rd("mask1.png", cv2.IMREAD_GRAYSCALE), rd("mask2.png", cv2.IMREAD_GRAYSCALE)
        clean_rgb = cv2.cvtColor(rd("clean.png"), cv2.COLOR_BGR2RGB)
        meta = json.load(open(pick / "meta.json"))
        st.caption(f"True degradation: {meta['type']}. Severity of t-2, t-1, t: "
                   f"{meta['severity_t2']}, {meta['severity_t1']}, {meta['severity_t']}. "
                   "This scene was never used for training.")
    else:
        st.caption("Use the five files from one aligned3/<scene id> folder made by the notebook.")
        c1, c2, c3, c4, c5 = st.columns(5)
        ups = [c1.file_uploader("Current frame t (ref.png)", type=["png", "jpg"]),
               c2.file_uploader("t-1 aligned (warped_prev1.png)", type=["png", "jpg"]),
               c3.file_uploader("t-2 aligned (warped_prev2.png)", type=["png", "jpg"]),
               c4.file_uploader("Mask t-1 (mask1.png)", type=["png"]),
               c5.file_uploader("Mask t-2 (mask2.png)", type=["png"])]
        if not all(ups):
            st.info("Add all five files to run the models.")
            return
        dec = lambda f, flag: cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), flag)
        ref_b, w1_b, w2_b = (dec(f, cv2.IMREAD_COLOR) for f in ups[:3])
        m1_g, m2_g = (dec(f, cv2.IMREAD_GRAYSCALE) for f in ups[3:])
        if any(x is None for x in (ref_b, w1_b, w2_b, m1_g, m2_g)):
            st.error("One of the files is not a readable image."); return

    show_boxes = st.checkbox("Draw YOLOv8 detections", value=True)
    with st.spinner("Running all models"):
        outputs, C, route, sane = run_all(ref_b, w1_b, w2_b, m1_g, m2_g)
        yolo = load_yolo() if show_boxes else None

    if not sane:
        st.warning("The fused output falls outside the range of its inputs, so the input files or model files are wrong.")
    if route:
        st.success(f"Detected {route[0]} (probability {route[1]:.2f}). Hybrid path used: {route[2]}.")

    names = list(outputs)
    for row in range(0, len(names), 3):
        cols = st.columns(3)
        for col, name in zip(cols, names[row:row + 3]):
            cap = name if clean_rgb is None else f"{name}, PSNR {psnr(outputs[name], clean_rgb):.2f} dB"
            col.image(with_boxes(outputs[name], yolo), caption=cap, use_container_width=True)
    if clean_rgb is not None:
        st.image(with_boxes(clean_rgb, yolo), caption="Clean original (ground truth, for reference only)",
                 width=520)

    st.subheader("How much Variant A trusted each frame")
    cols = st.columns(3)
    for col, k, name in zip(cols, (0, 1, 2), ("t-2", "t-1", "t (current)")):
        cmap = cv2.applyColorMap((C[k] * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        col.image(cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB), caption=f"{name}: mean weight {C[k].mean():.2f}",
                  use_container_width=True)
    st.caption("Brighter means more trusted. At every pixel the three weights add up to 1.")


with tab_try:
    try_tab()

with tab_results:
    load = lambda n: json.load(open(RESULTS / n)) if (RESULTS / n).exists() else None
    pretty = lambda df: df.rename(index=lambda m: NAMES.get(m, m))
    ext, main = load("results_extended_test.json"), load("results_fog_fix_hybrid.json")
    d14, d15 = load("results_day14_master.json"), load("results_day15_ablations.json")

    if ext:
        st.subheader("Headline result: 480 held-out test scenes")
        st.write("Change in car detection (AP50) of the final hybrid system compared with the raw degraded frame, "
                 "with 95% bootstrap confidence intervals.")
        cols = st.columns(3)
        for col, grp in zip(cols, ("all", "fog", "blur")):
            v = ext.get("bootstrap", {}).get(f"all 480 | {grp} | Hybrid_route - B1_raw")
            if v:
                col.metric(f"{grp.capitalize()} scenes", f"{v[0]:+.3f} AP50", f"95% CI [{v[1]:+.3f}, {v[2]:+.3f}]",
                           delta_color="off")
        for name, cap in (("all 480", "All 480 test scenes (80 original + 400 extra)"),
                          ("new 400", "The 400 extra scenes only")):
            if name in ext.get("tables", {}):
                st.markdown(f"**{cap}**")
                st.dataframe(pretty(pd.DataFrame(ext["tables"][name]).T)
                             .rename(columns={"car_ap50": "Car AP50", "psnr": "PSNR (dB)"})
                             .style.format("{:.3f}", na_rep="-"), use_container_width=True)
        st.caption("The 400 extra scenes were never used for training, validation or model selection.")

    st.divider()
    st.write("Original test set: 80 KITTI scenes (39 fog, 41 motion blur), 304 cars and 33 pedestrians. "
             "Detection is YOLOv8n; AP50 is average precision at IoU 0.5.")
    if main:
        st.subheader("All methods by degradation type")
        rows = {m: {"Car AP50 (all)": v["car_ap50"]["all"], "Car AP50 (fog)": v["car_ap50"]["fog"],
                    "Car AP50 (blur)": v["car_ap50"]["blur"], "Car AP50:95": v.get("car_ap50_95"),
                    "Pedestrian AP50": v.get("ped_ap50"), "PSNR all (dB)": v["psnr"]["all"],
                    "PSNR fog": v["psnr"]["fog"], "PSNR blur": v["psnr"]["blur"]} for m, v in main.items()}
        st.dataframe(pretty(pd.DataFrame(rows).T).style.format("{:.3f}", na_rep="-"), use_container_width=True)
    if d14:
        st.subheader("Main comparison with image quality")
        st.dataframe(pretty(pd.DataFrame(d14).T)
                     .rename(columns={"car_ap50": "Car AP50", "car_map": "Car AP50:95", "psnr": "PSNR (dB)", "ssim": "SSIM"})
                     .style.format("{:.3f}", na_rep="-"), use_container_width=True)
    if d15:
        st.subheader("Ablation study")
        st.dataframe(pd.DataFrame(d15).T, use_container_width=True)
        st.caption("Alignment was removed at test time only; the model was not retrained without it.")
    if not (main or d14 or d15 or ext):
        st.info("No result files found in results/.")

with tab_how:
    st.markdown("""
**1. Data.** 399 KITTI driving scenes, each with three consecutive frames (t-2, t-1, t). Every frame is
degraded with synthetic fog or motion blur, at a different strength per frame.

**2. Alignment.** RAFT optical flow warps the two older frames onto the current one. A forward-backward
consistency check masks pixels where the motion estimate is unreliable.

**3. Confidence-gated fusion (Variant A).** A network looks at the three aligned frames and predicts a
trust weight for each one at every pixel. The output is their weighted blend, with weights that add up to 1.

**4. Variant B.** Variant A fine-tuned with extra weight on the regions where objects are.

**5. Degradation-aware hybrid (final system).** A small classifier recognises fog or blur from image
statistics. Foggy scenes go to the U-Net restorer, because blending frames that share the same haze cannot
remove it. Blurred scenes go to Variant A, where frames differ in sharpness and fusion helps. We also tested
restoring each frame and then fusing them; it did worse on fog because alignment errors add artefacts.

**6. Detection and evaluation.** YOLOv8n detects cars and pedestrians on every output. We compare against
the raw frame, a single-frame U-Net and equal averaging, using AP, PSNR, SSIM and bootstrap confidence
intervals, on 80 test scenes and on 400 further held-out scenes.

Trained and tested on synthetic degradation; real fog and blur may behave differently.
""")
