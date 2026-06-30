import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path


# ── CONFIG ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Knee OA Grader",
    page_icon="🦴",
    layout="wide",
)


BASE_DIR      = Path(__file__).resolve().parent.parent
BASELINE_PATH = str(BASE_DIR / "data" / "baseline_resnet50.pth")
XAI_PATH      = str(BASE_DIR / "data" / "xai_guided_resnet50.pth")


IMG_SIZE   = 224
H_BAND_TOP = 105
H_BAND_BOT = 160
CbU_THRESH = 0.40   # ← FIXED: was 0.50


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


KL_GRADES = {
    0: ("Normal",   "No signs of OA.",                    "#2e7d32"),
    1: ("Doubtful", "Possible osteophytes, no JSN.",       "#1565c0"),
    2: ("Minimal",  "Definite osteophytes, possible JSN.", "#f57f17"),
    3: ("Moderate", "Multiple osteophytes, definite JSN.", "#e65100"),
    4: ("Severe",   "Large osteophytes, severe JSN.",      "#b71c1c"),
}


# ── MODEL LOADING ──────────────────────────────────────────────
@st.cache_resource
def load_model(path: str, label: str):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    except Exception as e:
        st.sidebar.error(f"Could not load {label}: {e}")
    model.eval()
    model.to(DEVICE)
    return model


# ── TRANSFORMS ─────────────────────────────────────────────────
_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _tf(img).unsqueeze(0).to(DEVICE)


def compute_jser(cam: np.ndarray) -> float:
    total = cam.sum() + 1e-8
    return float(cam[H_BAND_TOP:H_BAND_BOT, :].sum() / total)


def run_gradcam(model, tensor: torch.Tensor, rgb: np.ndarray):
    target_layer = [model.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layer) as cam_obj:
        raw = cam_obj(input_tensor=tensor)[0]
    raw = np.clip(raw, 0, 1)
    overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, raw, use_rgb=True)
    return raw, overlay


def run_single(model, img: Image.Image):
    """Returns (pred, conf, probs, cam, overlay, jser, faithful, rgb)."""
    image_resized = img.resize((IMG_SIZE, IMG_SIZE))
    rgb    = np.array(image_resized.convert("RGB"))
    tensor = preprocess(image_resized)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred  = int(np.argmax(probs))
    conf  = float(probs[pred])
    cam, overlay = run_gradcam(model, tensor, rgb)
    jser  = compute_jser(cam)
    return pred, conf, probs, cam, overlay, jser, jser >= CbU_THRESH, rgb


# ── BATCH INFERENCE (classification only, no Grad-CAM) ─────────
def batch_classify(model, files, batch_size=32):
    """Fast batched forward pass — returns list of (pred, conf, probs) per file."""
    all_results = []
    imgs_rgb    = []

    for f in files:
        img = Image.open(f).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
        imgs_rgb.append(np.array(img))

    tensors = torch.stack([
        _tf(Image.fromarray(r)) for r in imgs_rgb
    ]).to(DEVICE)

    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch  = tensors[start:start + batch_size]
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            for p in probs:
                pred = int(np.argmax(p))
                all_results.append((pred, float(p[pred]), p))

    return all_results, imgs_rgb


# ── 4-PANEL FIGURE ─────────────────────────────────────────────
def make_figure(rgb, overlay, cam, jser, grade, conf, model_label):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="#aaa")

    axes[0].imshow(rgb); axes[0].axis("off")
    axes[0].set_title("X-Ray", color="white", fontsize=10)
    axes[1].imshow(overlay); axes[1].axis("off")
    axes[1].set_title("Grad-CAM", color="white", fontsize=10)

    hv   = overlay.copy().astype(np.float32)
    mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    mask[H_BAND_TOP:H_BAND_BOT] = 1.0
    hv   = (hv * 0.30 + hv * mask * 0.70).astype(np.uint8)
    axes[2].imshow(hv)
    axes[2].axhline(H_BAND_TOP, color="cyan", lw=1.2, ls="--", alpha=0.8)
    axes[2].axhline(H_BAND_BOT, color="cyan", lw=1.2, ls="--", alpha=0.8)
    axes[2].text(4, H_BAND_TOP - 5, "H-Band", color="cyan", fontsize=7)
    axes[2].axis("off")
    axes[2].set_title("H-Band Region", color="white", fontsize=10)

    bar_color = "#4a9a96" if jser >= CbU_THRESH else "#e65100"
    axes[3].barh(["Outside", "H-Band"], [1 - jser, jser],
                 color=["#444", bar_color], height=0.35)
    axes[3].axvline(CbU_THRESH, color="#aaa", lw=1.2, ls="--",
                    label=f"threshold {CbU_THRESH}")
    axes[3].set_xlim(0, 1)
    axes[3].set_title(f"JSER = {jser:.3f}", color="white", fontsize=10)
    axes[3].legend(fontsize=7, labelcolor="white",
                   facecolor="#1a1a1a", edgecolor="#333", loc="lower right")

    g_name = KL_GRADES[grade][0]
    g_col  = KL_GRADES[grade][2]
    fig.suptitle(
        f"{model_label}  ·  KL-{grade} {g_name}  ·  {conf:.1%} confidence",
        color=g_col, fontsize=12, y=1.02, fontweight="bold"
    )
    plt.tight_layout()
    return fig


# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦴 XAI Knee OA")
    st.caption("DS2026 · Topic 3 · Team 14")
    st.divider()

    selected = st.radio(
        "**Model**",
        options=["Baseline ResNet-50", "XAI-Guided ResNet-50"],
        index=1,
    )
    model_path  = XAI_PATH if "XAI" in selected else BASELINE_PATH
    model_label = selected

    custom_path = st.text_input(
        "Or enter a custom .pth path",
        value="",
        placeholder="Leave blank to use selection above",
    )
    if custom_path.strip():
        model_path  = custom_path.strip()
        model_label = "Custom model"

    st.divider()
    show_raw = st.checkbox("Show raw CAM heatmap", value=False)

    st.divider()
    st.caption(f"🖥️ Device: `{DEVICE}`")
    st.divider()

    st.markdown("**KL Grade Reference**")
    for g, (name, desc, col) in KL_GRADES.items():
        st.markdown(
            f"<span style='color:{col}'>■</span> **{g} – {name}**: _{desc}_",
            unsafe_allow_html=True,
        )

with st.spinner(f"Loading {model_label}..."):
    model = load_model(model_path, model_label)


# ── TABS ───────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🔍 Single Image", "📂 Batch CbU Scanner"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE IMAGE
# ══════════════════════════════════════════════════════════════
with tab_single:
    st.header("Single Image Analysis")
    uploaded = st.file_uploader(
        "Upload knee X-ray (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="single_uploader",
    )

    if uploaded:
        image = Image.open(uploaded)
        with st.spinner("Running inference & Grad-CAM…"):
            pred, conf, probs, cam, overlay, jser, faithful, rgb = run_single(model, image)

        col_img, col_result = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

        with col_result:
            g_name, g_desc, g_col = KL_GRADES[pred]
            st.markdown(
                f"<div style='padding:16px;border-radius:10px;"
                f"background:{g_col}22;border:1px solid {g_col}66;'>"
                f"<h2 style='color:{g_col};margin:0;'>KL Grade {pred} — {g_name}</h2>"
                f"<p style='color:#ccc;margin:4px 0 0;'>{g_desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{conf:.1%}")
            m2.metric("JSER", f"{jser:.3f}")
            m3.metric(
                "Faithfulness",
                "✅ Faithful" if faithful else "⚠️ CbU",
                delta=f"threshold {CbU_THRESH}",
                delta_color="normal" if faithful else "inverse",
            )
            st.write("")
            st.markdown("**Grade probabilities**")
            prob_cols = st.columns(5)
            for g in range(5):
                col = KL_GRADES[g][2]
                prob_cols[g].markdown(
                    f"<div style='text-align:center'>"
                    f"<b style='color:{col};font-size:1.2rem;'>{probs[g]:.1%}</b>"
                    f"<br><span style='font-size:0.72rem;color:#888;'>KL-{g}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                prob_cols[g].progress(float(probs[g]))

        st.divider()
        st.subheader("Saliency & Faithfulness Analysis")
        fig = make_figure(rgb, overlay, cam, jser, pred, conf, model_label)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if show_raw:
            st.subheader("Raw CAM Heatmap")
            fig2, ax = plt.subplots(figsize=(4, 4))
            fig2.patch.set_facecolor("#0e1117")
            ax.imshow(cam, cmap="inferno")
            ax.axhline(H_BAND_TOP, color="cyan", lw=1.2, ls="--")
            ax.axhline(H_BAND_BOT, color="cyan", lw=1.2, ls="--")
            ax.axis("off")
            st.pyplot(fig2, use_container_width=False)
            plt.close(fig2)

        with st.expander("ℹ️ What is JSER?"):
            st.markdown(f"""
**Joint-Space Energy Ratio (JSER)** = fraction of total Grad-CAM saliency inside
the **H-Band** (rows {H_BAND_TOP}–{H_BAND_BOT} of the 224×224 image).

| JSER | Verdict |
|---|---|
| ≥ {CbU_THRESH} | ✅ Faithful — model looks at joint space |
| < {CbU_THRESH} | ⚠️ Correct-but-Unfaithful (CbU) |

This sample: `JSER = {jser:.3f}` → **{"Faithful ✅" if faithful else "CbU ⚠️"}**
            """)
    else:
        st.info("👆 Upload a knee X-ray to get started.", icon="🦴")


# ══════════════════════════════════════════════════════════════
# TAB 2 — BATCH CbU SCANNER (val-split locked)
# ══════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch CbU Scanner")
    st.markdown(
        "Runs on the **exact same val split** as scripts 05 & 06 "
        "(`random.seed(42)`, last 20%). Results are directly comparable to the paper."
    )

    # ── Reproduce val split identically to scripts 01/03/04/05 ──
    @st.cache_data
    def get_val_split(data_root: str):
        import random, os
        random.seed(42)
        all_images = []
        for g in range(5):
            folder = os.path.join(data_root, str(g))
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    all_images.append((os.path.join(folder, fname), g))
        random.shuffle(all_images)
        split = int(0.8 * len(all_images))
        return all_images[split:]   # exact val set

    DATA_ROOT = str(BASE_DIR / "data" / "kneeosteoarthritis" / "data")

    val_data = get_val_split(DATA_ROOT)

    if not val_data:
        st.error(f"Val set is empty — data not found at `{DATA_ROOT}`. "
                 "Check that the dataset is present.")
    else:
        st.caption(f"Val set: **{len(val_data)} images** · "
                   f"threshold: **JSER < {CbU_THRESH}** → CbU")

        col_opts1, col_opts2 = st.columns(2)
        top_n      = col_opts1.slider("Show top N worst CbU cases",
                                      min_value=3, max_value=20, value=6)
        batch_size = col_opts2.select_slider(
            "Batch size (larger = faster on GPU)",
            options=[8, 16, 32, 64], value=32,
        )

        if st.button("▶ Run Val-Set Scan", type="primary"):

            # ── PHASE 1: batched classification ───────────
            status = st.empty()
            status.info("⚡ Phase 1/2 — Batched classification…")

            imgs_rgb, true_labels, file_names = [], [], []
            for path, grade in val_data:
                img = Image.open(path).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
                imgs_rgb.append(np.array(img))
                true_labels.append(grade)
                file_names.append(Path(path).name)

            tensors = torch.stack([
                _tf(Image.fromarray(r)) for r in imgs_rgb
            ]).to(DEVICE)

            cls_results = []
            with torch.no_grad():
                for start in range(0, len(tensors), batch_size):
                    batch  = tensors[start:start + batch_size]
                    logits = model(batch)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()
                    for p in probs:
                        pred = int(np.argmax(p))
                        cls_results.append((pred, float(p[pred]), p))

            # ── PHASE 2: Grad-CAM on all val images ───────
            status.info("🔍 Phase 2/2 — Running Grad-CAM…")
            results      = []
            progress_bar = st.progress(0, text="Running Grad-CAM…")

            for i, ((pred, conf, probs), rgb, true_label, fname) in enumerate(
                zip(cls_results, imgs_rgb, true_labels, file_names)
            ):
                tensor   = preprocess(Image.fromarray(rgb))
                cam, overlay = run_gradcam(model, tensor, rgb)
                jser     = compute_jser(cam)
                correct  = pred == true_label
                results.append({
                    "filename":   fname,
                    "true_label": true_label,
                    "pred":       pred,
                    "conf":       conf,
                    "probs":      probs,
                    "jser":       jser,
                    "correct":    correct,
                    "faithful":   jser >= CbU_THRESH,
                    "overlay":    overlay,
                    "rgb":        rgb,
                    "cam":        cam,
                })
                progress_bar.progress(
                    (i + 1) / len(val_data),
                    text=f"Grad-CAM {i+1}/{len(val_data)}: {fname}"
                )

            progress_bar.empty()
            status.empty()

            # ── SUMMARY (mirrors script 05 console output) ─
            correct_results = [r for r in results if r["correct"]]
            cbu_cases       = [r for r in correct_results if not r["faithful"]]
            total      = len(results)
            n_correct  = len(correct_results)
            n_cbu      = len(cbu_cases)
            n_faithful = n_correct - n_cbu
            cbu_rate   = n_cbu / n_correct if n_correct else 0
            accuracy   = n_correct / total if total else 0

            st.divider()
            st.subheader("Scan Summary — Val Set")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Val images",   total)
            s2.metric("Accuracy",     f"{accuracy:.1%}")
            s3.metric("Correct preds", n_correct)
            s4.metric("CbU cases ⚠️", n_cbu)
            s5.metric("CbU Rate",     f"{cbu_rate:.1%}")

            # ── JSER DISTRIBUTION ──────────────────────────
            fig_dist, ax = plt.subplots(figsize=(8, 2.5))
            fig_dist.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            jsers  = [r["jser"] for r in correct_results]
            colors = ["#e65100" if j < CbU_THRESH else "#4a9a96" for j in jsers]
            sorted_data = sorted(zip(jsers, colors), key=lambda x: x[0])
            ax.bar(range(len(sorted_data)),
                   [x[0] for x in sorted_data],
                   color=[x[1] for x in sorted_data], width=0.8)
            ax.axhline(CbU_THRESH, color="white", lw=1.2, ls="--",
                       label=f"CbU threshold ({CbU_THRESH})")
            ax.set_ylabel("JSER", color="white")
            ax.set_xlabel("Correct predictions (sorted by JSER)", color="white")
            ax.tick_params(colors="#aaa")
            ax.set_title("JSER Distribution — val set correct predictions", color="white")
            ax.legend(labelcolor="white", facecolor="#1a1a1a", edgecolor="#333")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            st.pyplot(fig_dist, use_container_width=True)
            plt.close(fig_dist)

            # ── WORST CbU CASES ────────────────────────────
            st.divider()
            cbu_cases.sort(key=lambda r: r["jser"])

            if not cbu_cases:
                st.success("🎉 No CbU cases — all correct predictions are faithful!")
            else:
                st.subheader(f"All {n_cbu} CbU Cases (lowest JSER first)")
                show_cases = cbu_cases[:top_n]

                for idx, r in enumerate(show_cases):
                    g_name, _, g_col = KL_GRADES[r["pred"]]
                    with st.expander(
                        f"#{idx+1}  {r['filename']}  —  "
                        f"JSER {r['jser']:.3f}  |  "
                        f"True KL-{r['true_label']}  Pred KL-{r['pred']} {g_name}  |  "
                        f"Conf {r['conf']:.1%}",
                        expanded=(idx < 3),
                    ):
                        ec1, ec2 = st.columns([1, 2])
                        with ec1:
                            st.image(r["rgb"], caption=r["filename"],
                                     use_container_width=True)
                        with ec2:
                            fig_c, axs = plt.subplots(1, 3, figsize=(10, 3.5))
                            fig_c.patch.set_facecolor("#0e1117")
                            for ax_ in axs:
                                ax_.set_facecolor("#0e1117")
                                ax_.axis("off")
                            axs[0].imshow(r["rgb"])
                            axs[0].set_title("X-Ray", color="white", fontsize=9)
                            axs[1].imshow(r["overlay"])
                            axs[1].set_title("Grad-CAM", color="white", fontsize=9)
                            hv   = r["overlay"].copy().astype(np.float32)
                            mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
                            mask[H_BAND_TOP:H_BAND_BOT] = 1.0
                            hv   = (hv * 0.30 + hv * mask * 0.70).astype(np.uint8)
                            axs[2].imshow(hv)
                            axs[2].axhline(H_BAND_TOP, color="cyan", lw=1, ls="--")
                            axs[2].axhline(H_BAND_BOT, color="cyan", lw=1, ls="--")
                            axs[2].set_title(
                                f"H-Band  (JSER={r['jser']:.3f})",
                                color="#e65100", fontsize=9
                            )
                            plt.tight_layout()
                            st.pyplot(fig_c, use_container_width=True)
                            plt.close(fig_c)

                            m1c, m2c, m3c = st.columns(3)
                            m1c.metric("JSER", f"{r['jser']:.3f}",
                                       delta=f"{r['jser'] - CbU_THRESH:.3f} vs threshold",
                                       delta_color="inverse")
                            m2c.metric("True label", f"KL-{r['true_label']}")
                            m3c.metric("Prediction",
                                       f"KL-{r['pred']} {g_name}",
                                       delta=f"{r['conf']:.1%} confidence")
        