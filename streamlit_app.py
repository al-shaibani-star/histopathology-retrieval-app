# -*- coding: utf-8 -*-
"""
Streamlit version of the histopathology classification & retrieval web app.

Same six pages as the Flask app (app/app.py):
  1. Retrieval Demo  2. Step-by-Step  3. Overview
  4. Results         5. Comparisons   6. Algorithms

Standalone copy for GitHub / Streamlit Community Cloud: engine.py and
presentation.py sit next to this file. The 1.1 GB model weights are not in the
repository; engine.resolve_weights() downloads them from the Hugging Face repo
named in the HF_MODEL_REPO secret (or uses checkpoints/model_weights.pth if present).

Run:  streamlit run streamlit_app.py
"""
import os
import sys
import io
import csv
import json
import html

import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import presentation as P  # noqa: E402  (tables, facts, algorithms)

RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SAMPLES_DIR = os.path.join(HERE, "samples")
CSS_FILE = os.path.join(HERE, "static", "style.css")
UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

st.set_page_config(page_title="Histopathology Retrieval", page_icon="🔬", layout="wide")

# Streamlit Cloud: pass secrets to engine.resolve_weights() as environment variables
for _key in ("HF_MODEL_REPO", "HF_TOKEN", "MODEL_WEIGHTS"):
    try:
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:  # no secrets file locally
        pass


# ---------------------------------------------------------------- styling
def H(s):
    """Render raw HTML. Lines are stripped so Markdown never turns indented HTML into code."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def inject_css():
    with open(CSS_FILE, encoding="utf-8") as f:
        css = f.read()
    # Streamlit-specific tweaks on top of the Flask stylesheet
    css += """
    .block-container{max-width:1120px;padding-top:4.2rem}
    .topbar{border-radius:12px;margin-bottom:14px}
    .panel h2,.panel h3{color:var(--ink)}
    .samplecap{font-size:11px;text-align:center;color:var(--muted);margin-top:-6px}
    div[data-testid="stImageCaption"]{font-size:12.5px}
    """
    H("<style>" + css + "</style>")


def header():
    H("""
    <header class="topbar"><div class="brand">
    <div class="brand-title">Adaptive Multi-Scale Feature Fusion &amp; Hybrid Representation</div>
    <div class="brand-sub">Histopathological image classification and content-based retrieval</div>
    </div></header>
    """)


def footer():
    H("""<footer class="foot">Manuscript ID computers-4369511 &middot; WSSS4LUAD (4 classes)
    &middot; single 18,496-D descriptor for classification &amp; retrieval</footer>""")


def esc(x):
    return html.escape(str(x))


def bars(rows):
    """rows = [(name, percent)] -> the same bar chart as the Flask app."""
    out = '<div class="bars">'
    for name, p in rows:
        out += ('<div class="barrow"><span class="barname">%s</span>'
                '<span class="bartrack"><span class="barfill" style="width:%.2f%%"></span></span>'
                '<span class="barval">%.1f%%</span></div>' % (esc(name), p, p))
    return out + "</div>"


def table(columns, rows, cls="tbl", hi=()):
    out = '<table class="%s"><thead><tr>%s</tr></thead><tbody>' % (
        cls, "".join("<th>%s</th>" % esc(c) for c in columns))
    for i, row in enumerate(rows):
        out += '<tr class="%s">%s</tr>' % ("rowhi" if i in hi else "",
                                          "".join("<td>%s</td>" % esc(c) for c in row))
    return out + "</tbody></table>"


def kv_table(pairs, cls="tbl kv"):
    return '<table class="%s">%s</table>' % (
        cls, "".join("<tr><th>%s</th><td>%s</td></tr>" % (esc(k), esc(v)) for k, v in pairs))


# ---------------------------------------------------------------- model (cached)
@st.cache_resource(show_spinner="Loading the trained model and gallery index (first time only)...")
def get_engine():
    from engine import get_engine as _get
    return _get()


@st.cache_data(max_entries=32, show_spinner="Running the model...")
def run_analyze(img_bytes):
    return get_engine().analyze(Image.open(io.BytesIO(img_bytes)), k=6)


@st.cache_data(max_entries=32, show_spinner="Running the pipeline step by step...")
def run_steps(img_bytes, name):
    img = Image.open(io.BytesIO(img_bytes))
    img.load()
    return get_engine().analyze_steps(img, k=6, filename=name)


# ---------------------------------------------------------------- inputs
def load_samples():
    items = []
    with open(os.path.join(SAMPLES_DIR, "samples.csv")) as f:
        for row in csv.DictReader(f):
            items.append({"file": row["file"],
                          "label": row["label"].replace("_", "-").title(),
                          "path": os.path.join(SAMPLES_DIR, row["file"])})
    return items


def image_picker(key, button_text):
    """Uploader + sample chips. Returns (bytes, name) of the chosen image, or (None, None)."""
    state = key + "_img"

    def on_upload():
        f = st.session_state.get(key + "_up")
        if f is not None:
            st.session_state[state] = (f.getvalue(), os.path.basename(f.name))
            st.session_state[key + "_new"] = True

    def on_sample(path):
        with open(path, "rb") as fh:
            st.session_state[state] = (fh.read(), os.path.basename(path))
        st.session_state[key + "_new"] = True

    st.file_uploader(button_text, type=UPLOAD_TYPES, key=key + "_up", on_change=on_upload)
    st.caption("or try a sample:")
    samples = load_samples()
    cols = st.columns(len(samples))
    for col, s in zip(cols, samples):
        with col:
            st.image(s["path"], width="stretch")
            st.button(s["label"], key="%s_s_%s" % (key, s["file"]), width="stretch",
                      on_click=on_sample, args=(s["path"],))
    return st.session_state.get(state, (None, None))


def check_image(img_bytes):
    try:
        Image.open(io.BytesIO(img_bytes)).verify()
        return None
    except Exception:
        return "Could not read the uploaded file as an image."


# ================================================================ PAGE 1: demo
def page_demo():
    H("""<section class="panel"><h2>Try the retrieval pipeline</h2>
    <p class="muted">Upload an H&amp;E histopathology patch (or pick a sample). The model runs a single
    forward pass to produce one 18,496-dimensional descriptor, which is used <em>at the same time</em>
    to classify the tissue, explain the decision with Grad-CAM, and retrieve the most similar cases.</p></section>""")

    img_bytes, _ = image_picker("demo", "Choose image & run")
    if img_bytes is None:
        H('<section class="panel placeholder"><p>No image analysed yet — choose a file or a sample above.</p></section>')
        return
    err = check_image(img_bytes)
    if err:
        H('<div class="alert">%s</div>' % err)
        return
    try:
        r = run_analyze(img_bytes)
    except Exception as e:
        H('<div class="alert">Inference failed: %s</div>' % esc(e))
        return

    left, right = st.columns(2, gap="medium")
    with left:
        H("""<section class="panel"><h3>Query &amp; prediction</h3>
        <div class="qrow">
        <figure><img class="tile" src="%s"><figcaption>Uploaded patch</figcaption></figure>
        <figure><img class="tile" src="%s"><figcaption>Grad-CAM saliency</figcaption></figure>
        </div>
        <div class="pred"><div class="pred-main"><span class="pred-label">%s</span>
        <span class="pred-conf">%.1f%% confidence</span></div>%s</div></section>"""
          % (r["query_uri"], r["gradcam"], esc(r["pred_class"]), r["confidence"], bars(r["probs"])))
    with right:
        cards = "".join(
            '<figure class="retcard"><img src="%s"><figcaption><span class="rrank">#%d</span>'
            '<span class="rlabel">%s</span><span class="rsim">%.0f%% sim</span></figcaption></figure>'
            % (x["uri"], x["rank"], esc(x["label"]), x["similarity"]) for x in r["results"])
        H("""<section class="panel"><h3>Top-%d retrieved similar cases</h3>
        <p class="muted small">Ranked by Euclidean distance over the shared descriptor (as in the FAISS index).
        Similarity shown as cosine similarity.</p><div class="retgrid">%s</div></section>"""
          % (len(r["results"]), cards))


# ================================================================ PAGE 2: step-by-step
STEP_TITLES = ["Upload", "Preprocess", "Backbone", "Fusion", "Descriptor", "Search", "Result"]


def step_html(i, s):
    t = s["timings"]
    if i == 0:
        u = s["upload"]
        return """<section class="panel step"><h3><span class="badge">Step 1</span> Image upload</h3>
        <div class="steprow"><figure><img class="tile big" src="%s"><figcaption>Query image</figcaption></figure>
        <div><table class="kv"><tr><th>File</th><td>%s</td></tr><tr><th>Original size</th><td>%s</td></tr>
        <tr><th>Colour mode</th><td>%s</td></tr></table>
        <p class="explain">The user submits a histopathology patch through the upload box (PNG, JPG or TIFF).
        Streamlit receives the file in memory &mdash; nothing is written to disk &mdash; and opens it with Pillow.
        This image is the <b>query</b>: we want to find the most similar cases in the reference database.</p>
        </div></div></section>""" % (u["uri"], esc(u["filename"]), esc(u["size"]), esc(u["mode"]))

    if i == 1:
        p = s["pre"]
        ch = "".join('<figure><img class="tile sm" src="%s"><figcaption>%s channel<br>mean %.2f &rarr; %.2f'
                     '</figcaption></figure>' % (c["uri"], c["name"], c["mean_raw"], c["mean_norm"])
                     for c in p["channels"])
        return """<section class="panel step"><h3><span class="badge">Step 2</span> Preprocessing</h3>
        <div class="flow"><span>Convert to RGB</span><i>&rarr;</i><span>Resize to 224&times;224</span><i>&rarr;</i>
        <span>To tensor [0,1]</span><i>&rarr;</i><span>Normalise (ImageNet mean/std)</span></div>
        <div class="steprow"><figure><img class="tile" src="%s"><figcaption>224 &times; 224 input</figcaption></figure>%s</div>
        <p class="explain">Every image must have the same size and value range as the training images.
        Each channel is standardised with <code>x' = (x &minus; &mu;) / &sigma;</code>. The result is a
        <b>%s</b> fed to the network. Time: %s ms.</p></section>""" % (p["resized_uri"], ch, p["shape"], t["preprocess"])

    if i == 2:
        parts = []
        for j, g in enumerate(s["stages"]):
            parts.append('<figure><img class="tile md" src="%s"><figcaption><b>%s</b><br>%s</figcaption></figure>'
                         % (g["uri"], g["name"], g["shape"]))
            if j < len(s["stages"]) - 1:
                parts.append('<i class="arrow">&rarr;</i>')
        return """<section class="panel step"><h3><span class="badge">Step 3</span> Feature extraction &mdash;
        ConvNeXt V2 + CBAM backbone</h3><div class="steprow">%s</div>
        <p class="explain">The backbone processes the image in four stages. Early stages (large maps) capture fine
        detail such as nuclei and edges; deeper stages (small maps, more channels) capture tissue-level patterns.
        CBAM attention inside each stage emphasises the informative channels and regions.
        Maps show the mean activation (yellow = strong). Shapes are <i>channels &times; height &times; width</i>.</p>
        </section>""" % "".join(parts)

    if i == 3:
        fu = s["fusion"]
        return """<section class="panel step"><h3><span class="badge">Step 4</span> Adaptive weighted multi-scale fusion (neck)</h3>
        <div class="steprow"><div class="grow">%s
        <p class="explain">The four stages are projected to 256 channels, resized to a common 7&times;7 grid and
        combined with <b>learned weights</b> (softmax-normalised, shown above):
        <code>F = &Sigma; w<sub>i</sub> &middot; F<sub>i</sub></code>.
        The network itself learned how much each scale should contribute.</p></div>
        <figure><img class="tile md" src="%s"><figcaption>Fused map<br>%s</figcaption></figure></div></section>""" % (
            bars([(w["stage"], w["w"]) for w in fu["weights"]]), fu["uri"], fu["shape"])

    if i == 4:
        hd = s["head"]
        rows = "".join("<tr><td>%s</td><td>{:,}</td><td>%s</td><td>%.2f</td></tr>".format(p["dim"])
                       % (p["name"], p["what"], p["norm"]) for p in hd["parts"])
        barcode = ('<figure><img class="tile md pix" src="%s"><figcaption>Descriptor as a 136&times;136 '
                   '&ldquo;fingerprint&rdquo;</figcaption></figure>' % hd["barcode_uri"]) if hd["barcode_uri"] else ""
        return """<section class="panel step"><h3><span class="badge">Step 5</span> Hybrid representation &mdash;
        the {total:,}-D descriptor</h3>
        <table class="tbl"><tr><th>Branch</th><th>Dimensions</th><th>Captures</th><th>L2 norm</th></tr>{rows}
        <tr class="sum"><td>Concatenated descriptor</td><td>{total:,}</td><td>one vector for classification &amp; retrieval</td>
        <td>{norm:.2f}</td></tr></table>
        <div class="steprow" style="margin-top:14px">
        <figure><img class="tile md" src="{att}"><figcaption>Attention-branch map</figcaption></figure>{barcode}
        <p class="explain grow">The three branches are concatenated into a single vector &mdash; the image&rsquo;s
        numerical <b>fingerprint</b>. The same vector is used both by the classifier and by the search index.
        Forward pass (steps 3&ndash;5): {fwd} ms.</p></div></section>""".format(
            total=hd["total"], rows=rows, norm=hd["norm"], att=hd["attention_uri"], barcode=barcode, fwd=t["forward"])

    if i == 5:
        se = s["search"]
        dots = "".join('<span class="dot c-%s%s" style="left:%.2f%%" title="%s &middot; d = %.2f"></span>'
                       % (d["cls"], " top" if d["top"] else "", d["x"], d["cls"], d["d"]) for d in se["dots"])
        fp = ""
        if se["top1_barcode"]:
            fp = """<div class="steprow" style="margin-top:14px">
            <figure><img class="tile md pix" src="%s"><figcaption>Query fingerprint</figcaption></figure>
            <i class="arrow">&harr;</i>
            <figure><img class="tile md pix" src="%s"><figcaption>Nearest gallery fingerprint</figcaption></figure>
            <p class="explain grow">Each dot above is one gallery image placed by its distance to the query.
            Similar tissue produces similar fingerprints, so it lands on the left.</p></div>""" % (
                s["head"]["barcode_uri"], se["top1_barcode"])
        return """<section class="panel step"><h3><span class="badge">Step 6</span> Similarity search in the gallery index</h3>
        <p class="explain">The gallery of <b>{n}</b> reference patches was encoded offline by the
        same model and stored as an index ({n} &times; {dim:,}).
        The query descriptor <i>q</i> is compared with every stored descriptor <i>g</i> using the Euclidean distance
        <code>d(q,g) = &radic;&Sigma;(q<sub>j</sub> &minus; g<sub>j</sub>)&sup2;</code> (as in the FAISS IVF-Flat index),
        and the gallery is sorted from nearest to farthest. Search time: {t} ms.</p>
        <div class="strip">{dots}</div>
        <div class="striplbl"><span>nearest (d = {lo:.1f})</span><span>farthest (d = {hi:.1f})</span></div>
        <div class="legend"><span><i class="dot c-normal"></i>Normal</span><span><i class="dot c-stroma"></i>Stroma</span>
        <span><i class="dot c-tumor"></i>Tumor</span><span><i class="dot c-tumor_stroma"></i>Tumor-Stroma</span>
        <span><i class="dot top"></i>selected top-{k}</span></div>{fp}</section>""".format(
            n=se["n_gallery"], dim=se["dim"], t=t["search"], dots=dots, lo=se["min"], hi=se["max"], k=s["k"], fp=fp)

    # i == 6 : result
    r = s["result"]
    cards = "".join(
        '<figure class="retcard%s"><img src="%s"><figcaption><span class="rrank">#%d</span>'
        '<span class="rlabel">%s</span><span class="rsim">d=%.1f</span></figcaption></figure>'
        % (" match" if x["label"] == r["pred_class"] else "", x["uri"], x["rank"], esc(x["label"]), x["distance"])
        for x in r["results"])
    n = len(r["results"])
    if r["agree"] * 2 > n:
        verdict = " &mdash; the retrieval supports the classification."
    else:
        verdict = ". Retrieved classes: " + ", ".join("%s &times;%d" % (esc(l), c) for l, c in r["votes"]) + "."
    return """<section class="panel step"><h3><span class="badge">Step 7</span> Result: retrieved cases,
    classification and explanation</h3><div class="grid2"><div>
    <div class="qrow"><figure><img class="tile" src="%s"><figcaption>Query</figcaption></figure>
    <figure><img class="tile" src="%s"><figcaption>Grad-CAM (why)</figcaption></figure></div>
    <div class="pred"><div class="pred-main"><span class="pred-label">%s</span>
    <span class="pred-conf">%.1f%% confidence</span></div>%s</div></div>
    <div><div class="retgrid">%s</div>
    <p class="explain"><b>%d of %d</b> retrieved cases share the predicted class (green border)%s
    Grad-CAM: %s ms &middot; total: <b>%s ms</b>.</p></div></div></section>""" % (
        s["pre"]["resized_uri"], r["gradcam"], esc(r["pred_class"]), r["confidence"], bars(r["probs"]),
        cards, r["agree"], n, verdict, t["gradcam"], s["total_ms"])


def page_pipeline():
    H("""<section class="panel"><h2>Step-by-step: how an image is retrieved</h2>
    <p class="muted">Upload an H&amp;E patch (or pick a sample). The system then walks through every stage of the
    pipeline &mdash; from the raw upload to the retrieved similar cases &mdash; one step at a time.</p></section>""")

    img_bytes, name = image_picker("pipe", "Upload image & start")
    if img_bytes is None:
        H('<section class="panel placeholder"><p>Upload an image or pick a sample to start the walkthrough.</p></section>')
        return
    err = check_image(img_bytes)
    if err:
        H('<div class="alert">%s</div>' % err)
        return
    try:
        steps = run_steps(img_bytes, name)
    except Exception as e:
        H('<div class="alert">Inference failed: %s</div>' % esc(e))
        return

    ss = st.session_state
    if ss.pop("pipe_new", False) or "pipe_step" not in ss:   # new image -> back to step 1
        ss.pipe_step, ss.pipe_all = STEP_TITLES[0], False

    def move(d):
        i = STEP_TITLES.index(ss.pipe_step or STEP_TITLES[0])
        ss.pipe_step = STEP_TITLES[max(0, min(len(STEP_TITLES) - 1, i + d))]
        ss.pipe_all = False

    def toggle_all():
        ss.pipe_all = not ss.pipe_all

    def picked():
        ss.pipe_all = False

    labels = {t: "%d  %s" % (i + 1, t) for i, t in enumerate(STEP_TITLES)}
    st.segmented_control("Steps", STEP_TITLES, key="pipe_step", format_func=labels.get,
                         label_visibility="collapsed", width="stretch", on_change=picked)
    cur = STEP_TITLES.index(ss.pipe_step) if ss.pipe_step else 0

    c1, c2, c3, _, c4 = st.columns([1.2, 1.4, 1.2, 2, 1.6])
    c1.button("← Previous", on_click=move, args=(-1,), width="stretch", disabled=cur == 0 and not ss.pipe_all)
    c2.markdown("<div style='text-align:center;padding-top:8px' class='muted small'>%s</div>"
                % ("All steps" if ss.pipe_all else "Step %d of %d" % (cur + 1, len(STEP_TITLES))),
                unsafe_allow_html=True)
    c3.button("Next →", type="primary", on_click=move, args=(1,), width="stretch",
              disabled=cur == len(STEP_TITLES) - 1 and not ss.pipe_all)
    c4.button("One step at a time" if ss.pipe_all else "Show all steps", on_click=toggle_all, width="stretch")

    for i in (range(len(STEP_TITLES)) if ss.pipe_all else [cur]):
        H(step_html(i, steps))


# ================================================================ PAGE 3: overview
def figure_panel(fname, caption):
    path = os.path.join(FIG_DIR, fname)
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")


def page_overview():
    H("""<section class="panel"><h2>System overview</h2>
    <p class="muted">A single model produces one 18,496-dimensional descriptor that is used at the
    same time for classification, content-based retrieval, calibration, and Grad-CAM explainability.
    Each block below is tagged with the algorithm that defines it.</p></section>""")
    figure_panel("architecture__algorithms_pipeline.png",
                 "Figure 1 — End-to-end training and downstream inference pipeline of the proposed framework.")

    left, right = st.columns(2, gap="medium")
    with left:
        H('<section class="panel"><h3>Model</h3>%s</section>' % kv_table(P.MODEL_FACTS))
    with right:
        H('<section class="panel"><h3>Dataset</h3>%s</section>' % kv_table(P.DATASET_FACTS))

    cards = "".join('<div class="hlcard"><div class="hltitle">%s</div><div class="hltext">%s</div></div>'
                    % (esc(t), esc(x)) for t, x in P.HIGHLIGHTS)
    H('<section class="panel"><h3>Key contributions</h3><div class="hlgrid">%s</div></section>' % cards)

    H('<section class="panel"><h3>Graphical abstract</h3></section>')
    figure_panel("architecture__graphical_abstract.png",
                 "Figure 2 — Graphical abstract (one descriptor, four outputs, cross-organ transfer).")


# ================================================================ PAGE 4: results
def fmt_pct(x):
    try:
        return "%.2f%%" % (float(x) * 100.0)
    except Exception:
        return "-"


RESULT_SECTIONS = [
    ("Classification", "How well the model separates the four tissue classes.", [
        ("metrics__confusion_matrix.png", "Confusion matrix (normalized)"),
        ("metrics__roc_curve.png", "ROC curves per class (AUC)"),
        ("metrics__precision_recall_curve.png", "Precision-Recall curves"),
        ("metrics__metrics_radar.png", "Per-class metric radar"),
        ("metrics__sensitivity_specificity.png", "Sensitivity vs. specificity"),
        ("metrics__class_distribution.png", "Class distribution"),
    ]),
    ("Retrieval quality", "Ranking quality of the content-based retrieval.", [
        ("advanced__precision_at_k.png", "Precision@K"),
        ("advanced__recall_at_k.png", "Recall@K"),
        ("advanced__ndcg_at_k.png", "NDCG@K"),
        ("advanced__map_score.png", "Mean average precision"),
        ("retrieval__retrieval_label_tumor_0.png", "Retrieval example — Tumor"),
        ("retrieval__retrieval_label_stroma_0.png", "Retrieval example — Stroma"),
        ("retrieval__retrieval_label_normal_0.png", "Retrieval example — Normal"),
        ("retrieval__retrieval_label_tumor_stroma_0.png", "Retrieval example — Tumor-Stroma"),
    ]),
    ("Explainability (Grad-CAM)", "Where the model looks when it decides.", [
        ("explainability__gradcam_label_normal_0.png", "Grad-CAM — Normal"),
        ("explainability__gradcam_label_stroma_0.png", "Grad-CAM — Stroma"),
        ("explainability__gradcam_label_tumor_0.png", "Grad-CAM — Tumor"),
        ("explainability__gradcam_label_tumor_stroma_0.png", "Grad-CAM — Tumor-Stroma"),
    ]),
    ("Training dynamics", "Convergence and generalization behaviour.", [
        ("training__loss_curve.png", "Training / validation loss"),
        ("training__accuracy_curve.png", "Training / validation accuracy"),
        ("training__generalization_gap.png", "Generalization gap"),
        ("training__learning_rate.png", "Learning-rate schedule"),
    ]),
]


def page_results():
    with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
        m = json.load(f)

    def g(k):
        v = m.get(k)
        if isinstance(v, (list, tuple)):
            v = v[-1] if v else None
        return v

    cards = [
        ("Classification accuracy", P.PAPER_ACCURACY, "held-out validation set"),
        ("Macro F1", fmt_pct(g("f1_macro")), "unweighted over 4 classes"),
        ("Top-3 accuracy", fmt_pct(g("top_3_accuracy")), "true class within top 3"),
        ("Cohen's kappa", "%.3f" % g("cohens_kappa"), "agreement with ground truth"),
        ("Weighted kappa", "%.3f" % g("weighted_kappa"), "order-aware agreement"),
        ("Retrieval MAP", "%.4f" % float(g("map_score")), "mean average precision"),
        ("Retrieval MRR", "%.4f" % float(g("mrr")), "mean reciprocal rank"),
        ("Calibration ECE", "%.4f" % float(g("expected_calibration_error")), "lower is better"),
    ]
    H("""<section class="panel"><h2>Quantitative results</h2>
    <p class="muted">Numbers and figures are loaded directly from the model's evaluation output
    (the canonical run reported in the manuscript).</p><div class="cards">%s</div></section>""" % "".join(
        '<div class="card"><div class="card-val">%s</div><div class="card-title">%s</div>'
        '<div class="card-note">%s</div></div>' % (esc(v), esc(t), esc(n)) for t, v, n in cards))

    per_class = [[name, fmt_pct(m.get("precision_label_" + k)), fmt_pct(m.get("recall_label_" + k)),
                  fmt_pct(m.get("f1_label_" + k))]
                 for k, name in [("normal", "Normal"), ("stroma", "Stroma"),
                                 ("tumor", "Tumor"), ("tumor_stroma", "Tumor-Stroma")]]
    pk, rk, nk = m.get("precision_at_k", {}), m.get("recall_at_k", {}), m.get("ndcg_at_k", {})
    ret = [[K, "%.4f" % pk.get(K, 0), "%.4f" % rk.get(K, 0), "%.4f" % nk.get(K, 0)]
           for K in ["1", "3", "5", "10"] if K in pk]
    left, right = st.columns(2, gap="medium")
    with left:
        H('<section class="panel"><h3>Per-class performance</h3>%s</section>'
          % table(["Class", "Precision", "Recall", "F1"], per_class))
    with right:
        H('<section class="panel"><h3>Retrieval @ K</h3>%s</section>'
          % table(["K", "Precision@K", "Recall@K", "NDCG@K"], ret))

    detailed = [
        ("Accuracy", P.PAPER_ACCURACY),
        ("Balanced accuracy", fmt_pct(g("balanced_accuracy"))),
        ("Precision (macro)", fmt_pct(g("precision_macro"))),
        ("Recall (macro)", fmt_pct(g("recall_macro"))),
        ("F1 (macro)", fmt_pct(g("f1_macro"))),
        ("F1 (weighted)", fmt_pct(g("f1_weighted"))),
        ("Cohen's kappa", "%.4f" % g("cohens_kappa")),
        ("Weighted kappa", "%.4f" % g("weighted_kappa")),
        ("Matthews corr. (MCC)", "%.4f" % g("mcc")),
        ("Top-3 accuracy", fmt_pct(g("top_3_accuracy"))),
        ("Log loss", "%.4f" % g("log_loss")),
        ("Brier score", "%.4f" % g("brier_score")),
        ("Expected calibration error", "%.4f" % float(g("expected_calibration_error"))),
    ]
    H('<section class="panel"><h3>All classification &amp; calibration metrics</h3>%s</section>'
      % kv_table(detailed, cls="tbl metricslist"))

    for title, desc, figs in RESULT_SECTIONS:
        figs = [(f, c) for f, c in figs if os.path.exists(os.path.join(FIG_DIR, f))]
        if not figs:
            continue
        H('<section class="panel"><h3>%s</h3><p class="muted small">%s</p></section>' % (title, desc))
        for start in range(0, len(figs), 3):
            cols = st.columns(3)
            for col, (f, c) in zip(cols, figs[start:start + 3]):
                col.image(os.path.join(FIG_DIR, f), caption=c, width="stretch")


# ================================================================ PAGE 5: comparisons
def page_comparisons():
    H("""<section class="panel"><h2>Comparisons &amp; analysis</h2>
    <p class="muted">Baselines, ablation, cross-validation, class balancing, and cross-organ external
    validation — the tables reported in the manuscript (computers-4369511).</p></section>""")
    for t in P.COMPARISON_TABLES:
        H('<section class="panel"><h3>%s</h3></section>' % esc(t["caption"]))
        if t.get("chart") and os.path.exists(os.path.join(FIG_DIR, t["chart"])):
            st.image(os.path.join(FIG_DIR, t["chart"]), width="stretch")
        n = len(t["rows"])
        hi = set()
        if t.get("highlight_last"):
            hi.add(n - 1)
        if t.get("highlight_first"):
            hi.add(0)
        note = '<p class="muted small note">%s</p>' % esc(t["note"]) if t.get("note") else ""
        H('<section class="panel">%s%s</section>' % (table(t["columns"], t["rows"], hi=hi), note))


# ================================================================ PAGE 6: algorithms
def page_algorithms():
    H("""<section class="panel"><h2>Algorithms</h2>
    <p class="muted">Plain-language pseudocode for every component of the framework
    (Chapter 3). Layer settings, dimensions, fusion weights, and index parameters
    follow the actual implementation. The architecture below shows how the algorithms connect.</p></section>""")
    figure_panel("architecture__algorithms_pipeline.png", "Each block is tagged with the algorithm that defines it.")

    for alg in P.ALGORITHMS:
        rows = []
        for indent, marker, text, comment in alg["steps"]:
            pad = 0.2 + indent * 1.5
            if marker is None:
                body = '<span class="sechead">%s</span>' % esc(text)
            else:
                first = text.split(" ")[0]
                if first.lower() in P.CTRL_WORDS:
                    body = '<span class="kw">%s</span>%s' % (esc(first), esc(text[len(first):]))
                else:
                    body = esc(text)
                body = '<span class="mark">%s</span>%s' % (esc(marker), body)
                if comment:
                    body += '<span class="cmt">&#9655; %s</span>' % esc(comment)
            rows.append('<div class="algostep" style="padding-left:%.1fem">%s</div>' % (pad, body))
        H("""<section class="panel algobox"><div class="algohead">Algorithm %d. &nbsp;%s</div>
        <div class="algoio"><span class="iolab">Input:</span> %s</div>
        <div class="algoio botrule"><span class="iolab">Output:</span> %s</div>%s</section>""" % (
            alg["num"], esc(alg["title"]), esc(alg["input"]), esc(alg["output"]), "".join(rows)))


# ================================================================ main
inject_css()
header()
nav = st.navigation([
    st.Page(page_demo, title="Retrieval Demo", default=True),
    st.Page(page_pipeline, title="Step-by-Step", url_path="pipeline"),
    st.Page(page_overview, title="Overview", url_path="overview"),
    st.Page(page_results, title="Results", url_path="results"),
    st.Page(page_comparisons, title="Comparisons", url_path="comparisons"),
    st.Page(page_algorithms, title="Algorithms", url_path="algorithms"),
], position="top")
nav.run()
footer()
