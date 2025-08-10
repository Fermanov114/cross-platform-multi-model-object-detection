# app.py
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import MODELS, LIVE_INPUT_DIR, LOG_FILE
from image_sources import (
    get_uploaded_file,
    list_local_images,
    get_latest_image_from_server,
    get_all_new_images_from_server,
)
from processor import process_image_dual

st.set_page_config(page_title="Multi-Model Multi-Platform Detection", layout="wide")
st.title("Multi-Model Multi-Platform Detection Comparison")

# ===== Sidebar: select image (existing) =====
st.sidebar.header("Select Image")
def _load_existing():
    return list_local_images()

existing = _load_existing()
selected_image_path = None
if existing:
    chosen_name = st.sidebar.selectbox("Existing images (live_input/)", existing, index=0)
    selected_image_path = str(LIVE_INPUT_DIR / chosen_name)
else:
    st.sidebar.info("No images found in live_input. Upload or fetch from server.")

# ===== Sidebar: actions (upload / fetch) =====
st.sidebar.header("Actions")
uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded:
    try:
        saved_path = get_uploaded_file(uploaded)
        st.sidebar.success(f"Uploaded: {Path(saved_path).name}")
        selected_image_path = saved_path
        existing = _load_existing()
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

if st.sidebar.button("Fetch Latest Image"):
    try:
        latest_path = get_latest_image_from_server()
        st.sidebar.success(f"Downloaded: {Path(latest_path).name}")
        selected_image_path = latest_path
        existing = _load_existing()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch latest image: {e}")

if st.sidebar.button("Fetch All New Images"):
    try:
        new_files = get_all_new_images_from_server()
        if new_files:
            st.sidebar.success(f"Downloaded {len(new_files)} new images")
            selected_image_path = new_files[-1]
        else:
            st.sidebar.info("No new images found on server.")
        existing = _load_existing()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch new images: {e}")

# ===== Sidebar: model selection =====
st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Select Model", MODELS)

# ===== Sidebar: log download / clear =====
st.sidebar.header("Logs")
if Path(LOG_FILE).exists():
    with open(LOG_FILE, "rb") as f:
        st.sidebar.download_button(
            label="Download Detection Log CSV",
            data=f,
            file_name="detections.csv",
            mime="text/csv",
        )
else:
    st.sidebar.info("No detection logs available.")

if st.sidebar.button("🗑️ Clear Detection Logs"):
    try:
        if Path(LOG_FILE).exists():
            os.remove(LOG_FILE)
        st.sidebar.success("Detection logs cleared.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to clear logs: {e}")

# ===== Main: run detection & show results =====
if selected_image_path:
    st.subheader(f"Detection Results — {Path(selected_image_path).name}")

    if st.button("Run Detection"):
        try:
            cpu_path, gpu_path, cpu_metrics, gpu_metrics = process_image_dual(
                selected_image_path, model_name
            )

            # ---- CPU panel ----
            st.markdown("### CPU")
            c_img, c_data = st.columns([2, 1], vertical_alignment="top")
            with c_img:
                st.image(str(cpu_path), use_container_width=True, caption=f"{model_name} — CPU")
            with c_data:
                st.markdown("#### Detection Results")
                for det in cpu_metrics.get("detections_list", []):
                    st.write(f"- **{det['class']}** ({det['confidence']*100:.1f}%) {det['bbox']}")
                st.markdown("#### Performance")
                st.metric("Inference Time (ms)", f"{cpu_metrics['inference_ms']:.1f}")
                st.metric("FPS", f"{cpu_metrics['fps']:.2f}")
                st.metric("CPU / Memory", f"{cpu_metrics['cpu_percent']}% / {cpu_metrics['mem_percent']}%")

            st.divider()

            # ---- GPU panel ----
            st.markdown("### GPU")
            if gpu_path and gpu_metrics:
                g_img, g_data = st.columns([2, 1], vertical_alignment="top")
                with g_img:
                    st.image(str(gpu_path), use_container_width=True, caption=f"{model_name} — GPU")
                with g_data:
                    st.markdown("#### Detection Results")
                    for det in gpu_metrics.get("detections_list", []):
                        st.write(f"- **{det['class']}** ({det['confidence']*100:.1f}%) {det['bbox']}")
                    st.markdown("#### Performance")
                    st.metric("Inference Time (ms)", f"{gpu_metrics['inference_ms']:.1f}")
                    st.metric("FPS", f"{gpu_metrics['fps']:.2f}")
                    st.metric("CPU / Memory", f"{gpu_metrics['cpu_percent']}% / {gpu_metrics['mem_percent']}%")
            else:
                st.warning("GPU is not available on this machine or GPU inference was skipped.")
        except Exception as e:
            st.error(f"Detection failed: {e}")

# =======================================================================
# =============== Cross-Platform Comparison section =====================
# =======================================================================
st.header("Cross-Platform Comparison")

st.markdown(
    "- Compare the **same image** across **multiple platforms**.\n"
    "- Upload external CSV(s) from other devices (e.g., FPGA, Jetson, Raspberry Pi). "
    "Required columns: `platform, model, inference_ms, fps` and either `image_sha` or `image_name`."
)

# ---- CSV uploader (multi) ----
ext_csv_files = st.file_uploader(
    "Upload external CSV files", type=["csv"], accept_multiple_files=True, key="ext_csvs"
)

# ---------- utilities ----------
REQ_COLS = ["image_name", "image_sha", "platform", "model", "inference_ms", "fps"]

def _read_csv_any(path_or_bytes) -> pd.DataFrame:
    # robust loader: tolerate bad lines / encodings / mixed columns
    read_kwargs = dict(on_bad_lines="skip", engine="python")
    try_orders = [
        lambda x: pd.read_csv(x, **read_kwargs),
        lambda x: pd.read_csv(x, encoding="utf-8-sig", **read_kwargs),
        lambda x: pd.read_csv(x, encoding="latin-1", **read_kwargs),
    ]
    for fn in try_orders:
        try:
            if isinstance(path_or_bytes, (str, Path)):
                df = fn(path_or_bytes)
            else:
                df = fn(io.BytesIO(path_or_bytes.getvalue()))
            break
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize columns to lowercase and strip
    df.columns = [str(c).strip().lower() for c in df.columns]
    # standardize key columns, fill missing
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("image_name", "image_sha", "platform", "model") else 0.0
    # default platform if missing/empty (external CSV)
    df["platform"] = df["platform"].replace("", "External")
    # keep only meaningful columns if exist
    keep = [
        "timestamp", "image_name", "image_sha", "platform", "model",
        "inference_ms", "fps", "cpu_percent", "mem_percent", "rss_mb", "detections"
    ]
    for c in keep:
        if c not in df.columns:
            # create missing numeric columns
            df[c] = 0 if c in ("inference_ms", "fps", "cpu_percent", "mem_percent", "rss_mb", "detections") else ""
    # coerce numeric
    for c in ["inference_ms", "fps", "cpu_percent", "mem_percent", "rss_mb", "detections"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df[keep]

def _load_all_logs() -> pd.DataFrame:
    frames = []
    # local log
    if Path(LOG_FILE).exists():
        frames.append(_read_csv_any(LOG_FILE))
    # external logs
    for f in ext_csv_files or []:
        frames.append(_read_csv_any(f))
    if not frames:
        return pd.DataFrame(columns=[
            "timestamp","image_name","image_sha","platform","model",
            "inference_ms","fps","cpu_percent","mem_percent","rss_mb","detections"
        ])
    df = pd.concat(frames, ignore_index=True)
    return df.fillna("")

df_all = _load_all_logs()

# ---- choose the target image for "same image" filtering ----
default_img_name = Path(selected_image_path).name if selected_image_path else ""
img_name = st.text_input("Image name to compare (fallback when image_sha is missing):", value=default_img_name)

# allow direct SHA filtering (most accurate)
img_sha = st.text_input("Image SHA (optional, auto-logged locally):", value="")

# ---- filtering ----
def _filter_same_image(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if img_sha:
        sub = df[df["image_sha"] == img_sha]
        if len(sub) > 0:
            return sub
    if img_name:
        return df[df["image_name"] == img_name]
    return df.iloc[0:0]

sub = _filter_same_image(df_all)

# ---- user options ----
metric = st.selectbox("Metric to compare", ["inference_ms", "fps"])
group_cols = st.multiselect("Group by", ["platform", "model"], default=["platform", "model"])

if st.button("Build Comparison"):
    if df_all.empty:
        st.warning("No logs found. Run a local detection or upload CSV files first.")
    elif sub.empty:
        st.warning("No matching rows for this image. "
                   "Run detection locally (to create a SHA), or upload CSV with image_name/image_sha.")
    else:
        # table
        show_cols = ["timestamp","image_name","image_sha","platform","model","inference_ms","fps","detections"]
        show_cols = [c for c in show_cols if c in sub.columns]
        st.markdown("#### Matched rows")
        st.dataframe(sub[show_cols].sort_values(by=metric, ascending=(metric=="inference_ms")))

        # aggregate (if duplicates)
        gb = sub.groupby(group_cols, dropna=False, as_index=False)[metric].mean()
        st.markdown(f"#### {metric} by {', '.join(group_cols)} (mean)")
        # plot
        fig = plt.figure(figsize=(8, 4.5))
        x_labels = gb.apply(lambda r: " / ".join([str(r[c]) for c in group_cols]), axis=1)
        plt.bar(x_labels, gb[metric].values)
        plt.ylabel(metric)
        plt.xlabel(" / ".join(group_cols))
        plt.title(f"{metric} comparison for the same image")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig, clear_figure=True)
