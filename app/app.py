import os
import sys
import argparse
import time
from datetime import datetime

import torch
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict    import CIFAR10Predictor, INFER_TRANSFORM
from src.data_utils import CLASSES, get_dataloaders, denormalise
from src.utilities      import get_device

# ── Globals ───────────────────────────────────────────────────────────────────
DEVICE     = get_device()
PREDICTOR  = None
prediction_history = []

CLASS_META = {
    "airplane":   {"emoji": "✈️",  "color": "#60A5FA", "desc": "Fixed-wing aircraft"},
    "automobile": {"emoji": "🚗",  "color": "#F87171", "desc": "Passenger cars & vehicles"},
    "bird":       {"emoji": "🐦",  "color": "#34D399", "desc": "Any species of bird"},
    "cat":        {"emoji": "🐱",  "color": "#FBBF24", "desc": "Domestic cats"},
    "deer":       {"emoji": "🦌",  "color": "#A78BFA", "desc": "Deer & similar wildlife"},
    "dog":        {"emoji": "🐶",  "color": "#F472B6", "desc": "Domestic dogs"},
    "frog":       {"emoji": "🐸",  "color": "#2DD4BF", "desc": "Frogs & amphibians"},
    "horse":      {"emoji": "🐴",  "color": "#FB923C", "desc": "Horses & ponies"},
    "ship":       {"emoji": "🚢",  "color": "#818CF8", "desc": "Ships & vessels"},
    "truck":      {"emoji": "🚚",  "color": "#86EFAC", "desc": "Heavy trucks & lorries"},
}

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #080C14 !important;
    color: #E2E8F0 !important;
    min-height: 100vh !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 28px 24px !important;
}

/* Hide Gradio chrome */
footer, .built-with, .svelte-byatnx { display: none !important; }
.gr-prose h1, .gr-prose p { margin: 0 !important; }

/* ── Hero ──────────────────────────────────────────────────────────────── */
#hero-wrap {
    position: relative;
    background: linear-gradient(135deg, #0D1829 0%, #0D1117 60%, #120D24 100%);
    border: 1px solid #1E2D45;
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 20px;
    overflow: hidden;
}
#hero-wrap::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(99,102,241,0.10) 0%, transparent 65%);
    pointer-events: none;
}
#hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 65%);
    pointer-events: none;
}
#hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #F1F5F9 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    line-height: 1.15;
}
#hero-sub {
    color: #64748B;
    font-size: 0.92rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 560px;
    margin-bottom: 22px;
}
#badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.hbadge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.01em;
}
.hb-green  { background: rgba(52,211,153,0.10); color: #34D399; border: 1px solid rgba(52,211,153,0.25); }
.hb-indigo { background: rgba(129,140,248,0.10); color: #818CF8; border: 1px solid rgba(129,140,248,0.25); }
.hb-violet { background: rgba(167,139,250,0.10); color: #A78BFA; border: 1px solid rgba(167,139,250,0.25); }
.hb-amber  { background: rgba(251,191,36,0.10);  color: #FBBF24; border: 1px solid rgba(251,191,36,0.25); }

/* ── Panel base ────────────────────────────────────────────────────────── */
.panel {
    background: #0D1117;
    border: 1px solid #1E2D45;
    border-radius: 16px;
    padding: 22px;
}
.panel-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3D5066;
    margin-bottom: 16px;
}

/* ── Upload zone ───────────────────────────────────────────────────────── */
.upload-zone > div {
    background: #0D1117 !important;
    border: 2px dashed #1E2D45 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease !important;
    min-height: 260px !important;
}
.upload-zone > div:hover {
    border-color: #6366F1 !important;
}

/* ── Classify button ───────────────────────────────────────────────────── */
#classify-btn {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    padding: 13px 0 !important;
    width: 100% !important;
    margin-top: 12px !important;
    cursor: pointer !important;
    transition: filter 0.18s ease, transform 0.12s ease !important;
    box-shadow: 0 4px 24px rgba(79,70,229,0.30) !important;
}
#classify-btn:hover  { filter: brightness(1.12) !important; transform: translateY(-1px) !important; }
#classify-btn:active { transform: translateY(0px) !important; filter: brightness(0.97) !important; }

/* ── Result panel ──────────────────────────────────────────────────────── */
#result-out {
    min-height: 380px;
}
.result-card {
    background: #0D1117;
    border: 1px solid #1E2D45;
    border-radius: 16px;
    padding: 28px 26px;
    height: 100%;
}
.result-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3D5066;
    margin-bottom: 10px;
}
.result-emoji  { font-size: 3.2rem; line-height: 1; margin-bottom: 6px; display: block; }
.result-name   {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 4px;
}
.result-desc   { font-size: 0.88rem; color: #64748B; margin-bottom: 18px; }
.conf-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 22px;
}
.pill-high   { background: rgba(52,211,153,0.12); color: #34D399; border: 1px solid rgba(52,211,153,0.28); }
.pill-medium { background: rgba(251,191,36,0.12);  color: #FBBF24; border: 1px solid rgba(251,191,36,0.28); }
.pill-low    { background: rgba(248,113,113,0.12); color: #F87171; border: 1px solid rgba(248,113,113,0.28); }

/* ── Stat boxes ────────────────────────────────────────────────────────── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 22px;
}
.stat-box {
    background: #080C14;
    border: 1px solid #1A2535;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-key {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #3D5066;
}

/* ── Confidence bars ───────────────────────────────────────────────────── */
.bars-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #3D5066;
    margin-bottom: 12px;
}
.bar-row { margin-bottom: 11px; }
.bar-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.bar-cls {
    font-size: 0.84rem;
    font-weight: 500;
    color: #CBD5E1;
    display: flex;
    align-items: center;
    gap: 7px;
}
.bar-cls.top { font-weight: 700; color: #F1F5F9; }
.bar-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748B;
}
.bar-track {
    background: #0A0F1A;
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    border: 1px solid #1A2535;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
}

/* ── Placeholder state ─────────────────────────────────────────────────── */
.placeholder-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 340px;
    gap: 12px;
    color: #2D3F55;
}
.placeholder-icon { font-size: 3.5rem; }
.placeholder-title { font-size: 1rem; font-weight: 600; color: #3D5066; }
.placeholder-hint  { font-size: 0.82rem; color: #2D3F55; text-align: center; line-height: 1.5; }

/* ── History panel ─────────────────────────────────────────────────────── */
.hist-item {
    display: grid;
    grid-template-columns: 28px 1fr auto;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: #080C14;
    border: 1px solid #1A2535;
    border-radius: 10px;
    margin-bottom: 7px;
    transition: border-color 0.15s;
}
.hist-item:hover { border-color: #2D3F55; }
.hist-emoji  { font-size: 1.1rem; text-align: center; }
.hist-body   { min-width: 0; }
.hist-cls    { font-size: 0.82rem; font-weight: 700; color: #CBD5E1; line-height: 1.2; }
.hist-time   { font-size: 0.68rem; color: #3D5066; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.hist-conf   { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; white-space: nowrap; }

/* ── Model info table ──────────────────────────────────────────────────── */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #0F1923;
    gap: 12px;
}
.info-row:last-child { border-bottom: none; padding-bottom: 0; }
.info-key { font-size: 0.78rem; color: #3D5066; white-space: nowrap; }
.info-val { font-size: 0.78rem; font-weight: 500; color: #94A3B8; text-align: right; }
.info-val.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
.info-val.green  { color: #34D399; font-weight: 700; }
.info-val.indigo { color: #818CF8; font-weight: 600; }

/* ── Class chips ───────────────────────────────────────────────────────── */
.chips-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-top: 20px;
}
.chip {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    padding: 14px 8px;
    background: #0D1117;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    transition: border-color 0.15s, background 0.15s;
    cursor: default;
}
.chip:hover { border-color: #2D3F55; background: #111827; }
.chip-emoji { font-size: 1.5rem; line-height: 1; }
.chip-name  { font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: capitalize; }

/* ── Examples ──────────────────────────────────────────────────────────── */
.examples-section { margin-top: 14px; }
.examples-section .examples-header { margin-bottom: 10px; }
.gr-samples { background: transparent !important; border: none !important; }
.gr-samples .grid-wrap { gap: 8px !important; }
.gr-samples img {
    border-radius: 8px !important;
    border: 2px solid #1E2D45 !important;
    transition: border-color 0.15s !important;
}
.gr-samples img:hover { border-color: #6366F1 !important; }

/* ── Footer ────────────────────────────────────────────────────────────── */
#app-footer {
    margin-top: 28px;
    padding: 20px 28px;
    background: #0D1117;
    border: 1px solid #1A2535;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
#footer-author {
    font-size: 0.82rem;
    color: #64748B;
}
#footer-author strong {
    color: #94A3B8;
    font-weight: 600;
}
#footer-author a {
    color: #818CF8;
    text-decoration: none;
    font-weight: 500;
}
#footer-author a:hover { text-decoration: underline; }
#footer-stack {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.stack-tag {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #3D5066;
    background: #080C14;
    border: 1px solid #1A2535;
    border-radius: 6px;
    padding: 3px 9px;
}

/* ── Gradio overrides ──────────────────────────────────────────────────── */
.gap-4 { gap: 16px !important; }
.gr-block { border: none !important; background: transparent !important; padding: 0 !important; }
.wrap { gap: 0 !important; }
"""


# ── HTML builders ─────────────────────────────────────────────────────────────

def hero_html():
    return """
<div id="hero-wrap">
  <div id="hero-title">CIFAR-10 Image Classifier</div>
  <div id="hero-sub">
    A deep learning classifier trained on 50,000 images across 10 categories.
    Upload any image for GPU-accelerated inference — results in milliseconds.
  </div>
  <div id="badge-row">
    <span class="hbadge hb-green">● 95.06% Val Accuracy</span>
    <span class="hbadge hb-indigo">⚡ RTX 5070 · CUDA</span>
    <span class="hbadge hb-violet">🧠 11.17M Params</span>
    <span class="hbadge hb-amber">🔥 Mixed Precision AMP</span>
  </div>
</div>"""


def placeholder_html():
    return """
<div class="result-card">
  <div class="placeholder-wrap">
    <span class="placeholder-icon">🖼️</span>
    <div class="placeholder-title">No image yet</div>
    <div class="placeholder-hint">
      Upload an image on the left or click<br>
      one of the example thumbnails below.
    </div>
  </div>
</div>"""


def result_html(top5, ms):
    cls   = top5[0][0]
    conf  = top5[0][1]
    meta  = CLASS_META.get(cls, {"emoji":"❓","color":"#818CF8","desc":""})

    if conf >= 0.90:
        pill_cls, pill_dot, pill_txt = "pill-high",   "🟢", f"Very High — {conf*100:.1f}%"
    elif conf >= 0.70:
        pill_cls, pill_dot, pill_txt = "pill-medium", "🟡", f"High — {conf*100:.1f}%"
    else:
        pill_cls, pill_dot, pill_txt = "pill-low",    "🔴", f"Low — {conf*100:.1f}%"

    top3_mass = sum(p for _, p in top5[:3]) * 100

    # Confidence bars
    bars = ""
    for i, (c, p) in enumerate(top5):
        m      = CLASS_META.get(c, {"emoji":"❓","color":"#818CF8"})
        pct    = round(p * 100, 1)
        is_top = i == 0
        fill_color = m["color"] if is_top else "#1E2D45"
        bars += f"""
      <div class="bar-row">
        <div class="bar-meta">
          <span class="bar-cls {'top' if is_top else ''}">
            <span>{m['emoji']}</span>{c}
          </span>
          <span class="bar-pct">{pct:.1f}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{pct}%;background:{fill_color};"></div>
        </div>
      </div>"""

    return f"""
<div class="result-card">
  <div class="result-eyebrow">Prediction</div>
  <span class="result-emoji">{meta['emoji']}</span>
  <div class="result-name" style="color:{meta['color']};">{cls.upper()}</div>
  <div class="result-desc">{meta['desc']}</div>
  <div class="conf-pill {pill_cls}">{pill_dot} {pill_txt}</div>

  <div class="stats-row">
    <div class="stat-box">
      <div class="stat-val" style="color:{meta['color']};">{conf*100:.1f}%</div>
      <div class="stat-key">Top-1</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:#94A3B8;">{top3_mass:.1f}%</div>
      <div class="stat-key">Top-3 Mass</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:#94A3B8;">{ms:.0f}ms</div>
      <div class="stat-key">Inference</div>
    </div>
  </div>

  <div class="bars-label">Top-5 Class Probabilities</div>
  {bars}
</div>"""


def history_html(history):
    if not history:
        return """
<div class="panel">
  <div class="panel-label">Recent Predictions</div>
  <div style="text-align:center;padding:24px 0;color:#2D3F55;">
    <div style="font-size:1.8rem;margin-bottom:8px;">🕒</div>
    <div style="font-size:0.8rem;">Predictions will appear here</div>
  </div>
</div>"""

    rows = ""
    for h in reversed(history[-6:]):
        m  = CLASS_META.get(h["cls"], {"emoji":"❓"})
        if h["conf"] >= 0.90:   cc = "#34D399"
        elif h["conf"] >= 0.70: cc = "#FBBF24"
        else:                   cc = "#F87171"
        rows += f"""
    <div class="hist-item">
      <span class="hist-emoji">{m['emoji']}</span>
      <div class="hist-body">
        <div class="hist-cls">{h['cls'].upper()}</div>
        <div class="hist-time">{h['time']}</div>
      </div>
      <span class="hist-conf" style="color:{cc};">{h['conf']*100:.1f}%</span>
    </div>"""

    return f"""
<div class="panel">
  <div class="panel-label">Recent Predictions</div>
  {rows}
</div>"""


def model_info_html():
    return """
<div class="panel" style="margin-top:12px;">
  <div class="panel-label">Model Information</div>
  <div class="info-row">
    <span class="info-key">Architecture</span>
    <span class="info-val">ResNet-style CNN</span>
  </div>
  <div class="info-row">
    <span class="info-key">Parameters</span>
    <span class="info-val mono">11.17 M</span>
  </div>
  <div class="info-row">
    <span class="info-key">Val Accuracy</span>
    <span class="info-val green">95.06%</span>
  </div>
  <div class="info-row">
    <span class="info-key">Training Epochs</span>
    <span class="info-val mono">100</span>
  </div>
  <div class="info-row">
    <span class="info-key">Input Resolution</span>
    <span class="info-val mono">32 × 32 px</span>
  </div>
  <div class="info-row">
    <span class="info-key">Optimizer</span>
    <span class="info-val mono">SGD + Nesterov</span>
  </div>
  <div class="info-row">
    <span class="info-key">LR Schedule</span>
    <span class="info-val mono">CosineAnnealing</span>
  </div>
  <div class="info-row">
    <span class="info-key">Loss Function</span>
    <span class="info-val mono">CrossEntropy (ε=0.1)</span>
  </div>
  <div class="info-row">
    <span class="info-key">Device</span>
    <span class="info-val indigo">RTX 5070 · CUDA</span>
  </div>
</div>"""


def classes_html():
    chips = ""
    for cls in CLASSES:
        m = CLASS_META.get(cls, {"emoji":"❓","color":"#818CF8"})
        chips += f"""
    <div class="chip">
      <span class="chip-emoji">{m['emoji']}</span>
      <span class="chip-name">{cls}</span>
    </div>"""
    return f"""
<div class="panel" style="margin-top:20px;">
  <div class="panel-label">10 Supported Classes</div>
  <div class="chips-grid">{chips}</div>
</div>"""


def footer_html():
    return """
<div id="app-footer">
  <div id="footer-author">
    Designed &amp; built by <strong>Md. Mehenuf Hossain Bhuiyan</strong>
    &nbsp;·&nbsp;
    CIFAR-10 Deep Learning Classifier &nbsp;·&nbsp;
    <a href="https://github.com/mehenuf/cifar10-classifier" target="_blank">
      GitHub Repository ↗
    </a>
  </div>
  <div id="footer-stack">
    <span class="stack-tag">PyTorch</span>
    <span class="stack-tag">Gradio</span>
    <span class="stack-tag">CUDA 13.2</span>
    <span class="stack-tag">Python 3.11</span>
    <span class="stack-tag">ResNet-CNN</span>
  </div>
</div>"""


# ── Inference callback ────────────────────────────────────────────────────────

def classify_image(image: Image.Image):
    global prediction_history

    if image is None:
        return placeholder_html(), history_html(prediction_history)

    try:
        tensor = INFER_TRANSFORM(image.convert("RGB"))

        # Warm-up then timed inference
        PREDICTOR.predict_topk(tensor, k=5)
        t0   = time.perf_counter()
        top5 = PREDICTOR.predict_topk(tensor, k=5)[0]
        ms   = (time.perf_counter() - t0) * 1000

        prediction_history.append({
            "cls":  top5[0][0],
            "conf": top5[0][1],
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        return result_html(top5, ms), history_html(prediction_history)

    except Exception as e:
        err = f"""
<div class="result-card" style="border-color:#3D1515;">
  <div style="color:#F87171;font-size:1rem;font-weight:700;margin-bottom:8px;">
    ⚠️ Error
  </div>
  <div style="color:#64748B;font-size:0.85rem;line-height:1.6;">{str(e)}</div>
</div>"""
        return err, history_html(prediction_history)


# ── Sample images ─────────────────────────────────────────────────────────────

def load_samples(data_dir="./data"):
    try:
        _, _, loader = get_dataloaders(
            data_dir=data_dir, batch_size=64,
            augment=False, num_workers=2
        )
        imgs, lbls = next(iter(loader))
        samples, seen = [], {}
        for img_t, lbl in zip(imgs, lbls):
            cls = CLASSES[lbl.item()]
            if cls in seen:
                continue
            arr = (denormalise(img_t).permute(1,2,0).numpy()*255).astype(np.uint8)
            pil = Image.fromarray(arr).resize((96, 96), Image.NEAREST)
            samples.append(pil)
            seen[cls] = True
            if len(seen) == 10:
                break
        return samples
    except Exception:
        return []


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui(data_dir="./data") -> gr.Blocks:
    samples = load_samples(data_dir)

    with gr.Blocks(css=CSS, title="CIFAR-10 Classifier — Md. Mehenuf Hossain Bhuiyan") as demo:

        # Hero
        gr.HTML(hero_html())

        # ── Main 3-column layout ───────────────────────────────────────────
        with gr.Row(equal_height=True):

            # LEFT — upload + examples
            with gr.Column(scale=4, min_width=280):
                gr.HTML('<div class="panel-label" style="margin-bottom:10px;">Input Image</div>')
                image_input = gr.Image(
                    type         = "pil",
                    label        = "",
                    height       = 268,
                    show_label   = False,
                    elem_classes = ["upload-zone"],
                )
                classify_btn = gr.Button(
                    "🔍  Classify Image",
                    elem_id = "classify-btn",
                )
                if samples:
                    gr.HTML('<div class="panel-label" style="margin:18px 0 10px;">Quick Examples</div>')
                    gr.Examples(
                        examples     = samples,
                        inputs       = image_input,
                        label        = "",
                        examples_per_page = 10,
                    )

            # MIDDLE — result
            with gr.Column(scale=5, min_width=340):
                result_out = gr.HTML(
                    value    = placeholder_html(),
                    elem_id  = "result-out",
                )

            # RIGHT — history + model info
            with gr.Column(scale=3, min_width=240):
                history_out = gr.HTML(history_html([]))
                gr.HTML(model_info_html())

        # Classes row
        gr.HTML(classes_html())

        # Footer
        gr.HTML(footer_html())

        # ── Wire events ────────────────────────────────────────────────────
        classify_btn.click(
            fn      = classify_image,
            inputs  = image_input,
            outputs = [result_out, history_out],
        )
        image_input.change(
            fn      = classify_image,
            inputs  = image_input,
            outputs = [result_out, history_out],
        )

    return demo


# ── Entry ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./checkpoints/best_model.pth")
    p.add_argument("--data-dir",   default="./data")
    p.add_argument("--port",       type=int, default=7860)
    p.add_argument("--share",      action="store_true")
    return p.parse_args()


def main():
    global PREDICTOR
    args      = parse_args()
    PREDICTOR = CIFAR10Predictor(args.checkpoint)

    print(f"\n[app] Launching → http://localhost:{args.port}")
    print(f"[app] Built by  → Md. Mehenuf Hossain Bhuiyan")
    demo = build_ui(args.data_dir)
    demo.launch(
        server_port = args.port,
        share       = args.share,
        show_error  = True,
        inbrowser   = True,
    )


if __name__ == "__main__":
    main()