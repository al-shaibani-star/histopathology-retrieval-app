# -*- coding: utf-8 -*-
"""
Inference engine for the retrieval demo.

Loads the trained model once, then provides:
  - classify()  : predicted tissue class + calibrated confidence
  - retrieve()  : top-K visually similar gallery patches (shared descriptor)
  - gradcam()   : class-discriminative saliency overlay

The gallery descriptors are computed once and cached to disk.
"""
import os
import sys
import glob
import csv
import io
import base64

import numpy as np
from PIL import Image

# ---- make the project source importable ----
HERE = os.path.dirname(os.path.abspath(__file__))
DELIV = HERE
SRC = os.path.join(DELIV, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.full_model import create_model

# ---- paths & constants ----
WEIGHTS = os.path.join(DELIV, "checkpoints", "model_weights.pth")
GALLERY_DIR = os.path.join(DELIV, "gallery", "patches")
LABELS_CSV = os.path.join(DELIV, "gallery", "labels.csv")
INDEX_CACHE = os.path.join(DELIV, "gallery", "gallery_index.npz")

CLASS_KEYS = ['normal', 'stroma', 'tumor', 'tumor_stroma']
CLASS_NAMES = {
    'normal': 'Normal',
    'stroma': 'Stroma',
    'tumor': 'Tumor',
    'tumor_stroma': 'Tumor-Stroma',
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _to_data_uri(pil_img, fmt='PNG'):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return "data:image/%s;base64,%s" % (fmt.lower(),
                                        base64.b64encode(buf.getvalue()).decode())


def _thumb(pil_img, max_side):
    img = pil_img.copy()
    img.thumbnail((max_side, max_side))
    return img


def _heat(arr, cmap, size, clip=None):
    """2-D array -> colour-mapped PIL image (min-max normalised).
    clip=(lo, hi) percentiles stretches the contrast of sparse arrays."""
    import cv2
    a = np.asarray(arr, dtype=np.float32)
    if clip:
        a = np.clip(a, *np.percentile(a, clip))
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_NEAREST)
    a = (a * 255).astype(np.uint8)
    if cmap == 'gray':
        return Image.fromarray(a)
    code = {'jet': cv2.COLORMAP_JET, 'viridis': cv2.COLORMAP_VIRIDIS}[cmap]
    return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(a, code), cv2.COLOR_BGR2RGB))


def resolve_weights():
    """Local checkpoint if present; otherwise download it from the Hugging Face
    Hub repo named in the HF_MODEL_REPO environment variable / Streamlit secret."""
    path = os.environ.get('MODEL_WEIGHTS', WEIGHTS)
    if os.path.exists(path):
        return path
    repo = os.environ.get('HF_MODEL_REPO')
    if not repo:
        raise FileNotFoundError(
            "Model weights not found at %s. Put model_weights.pth in checkpoints/ "
            "or set HF_MODEL_REPO to a Hugging Face model repo that contains it." % path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename='model_weights.pth',
                           token=os.environ.get('HF_TOKEN'))


class RetrievalEngine:
    def __init__(self, device=None):
        # CPU by default (the 288M-parameter model can exhaust a small GPU);
        # set MODEL_DEVICE=cuda to opt in
        device = device or os.environ.get('MODEL_DEVICE', 'cpu')
        self.device = torch.device(device)
        self.model = create_model(num_classes=4, model_size='base')
        # mmap: read tensors from the file on demand instead of holding a second
        # in-memory copy of the 1.1 GB checkpoint (matters on small cloud hosts)
        ck = torch.load(resolve_weights(), map_location='cpu', weights_only=False, mmap=True)
        state = ck['model_state_dict'] if 'model_state_dict' in ck else ck
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.epoch = ck.get('epoch')
        self.val_acc = ck.get('val_acc')

        # Grad-CAM hooks on the last backbone stage (512 ch, 7x7)
        self._act = None
        self._grad = None
        target = self.model.backbone.stages[-1]
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

        self._build_or_load_gallery()

    # ---------- hooks ----------
    def _fwd_hook(self, module, inp, out):
        self._act = out.detach()

    def _bwd_hook(self, module, gin, gout):
        self._grad = gout[0].detach()

    # ---------- gallery ----------
    def _forward(self, x, need_grad=False):
        if need_grad:
            return self.model(x)
        with torch.no_grad():
            return self.model(x)

    def _descriptor(self, pil_img):
        x = _transform(pil_img.convert('RGB')).unsqueeze(0).to(self.device)
        out = self._forward(x)
        return out['features'].cpu().numpy()[0].astype(np.float32)

    def _build_or_load_gallery(self):
        labels = {}
        if os.path.exists(LABELS_CSV):
            with open(LABELS_CSV) as f:
                for row in csv.DictReader(f):
                    labels[row['file']] = row['label']

        files = sorted(glob.glob(os.path.join(GALLERY_DIR, "*.png")))
        files = [os.path.basename(f) for f in files]

        if os.path.exists(INDEX_CACHE):
            data = np.load(INDEX_CACHE, allow_pickle=True)
            if list(data['files']) == files:
                self.gallery_desc = data['desc'].astype(np.float32)
                self.gallery_files = list(data['files'])
                self.gallery_labels = list(data['labels'])
                self._norm_gallery()
                return

        desc, labs = [], []
        for fn in files:
            img = Image.open(os.path.join(GALLERY_DIR, fn))
            desc.append(self._descriptor(img))
            labs.append(labels.get(fn, fn.split('__')[0]))
        self.gallery_desc = np.stack(desc).astype(np.float32)
        self.gallery_files = files
        self.gallery_labels = labs
        try:
            np.savez_compressed(INDEX_CACHE, desc=self.gallery_desc,
                                files=np.array(files), labels=np.array(labs))
        except OSError:
            pass  # read-only filesystem on some hosts: keep the index in memory
        self._norm_gallery()

    def _norm_gallery(self):
        n = np.linalg.norm(self.gallery_desc, axis=1, keepdims=True) + 1e-8
        self._gallery_unit = self.gallery_desc / n

    # ---------- public API ----------
    def analyze(self, pil_img, k=6):
        """Full pipeline: classify + descriptor + retrieve + grad-cam."""
        pil_img = pil_img.convert('RGB')
        x = _transform(pil_img).unsqueeze(0).to(self.device)

        # classification + descriptor (with grad enabled for Grad-CAM)
        x.requires_grad_(False)
        out = self.model(x)
        probs = F.softmax(out['logits'], dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        pred_key = CLASS_KEYS[idx.item()]
        descriptor = out['features'].detach().cpu().numpy()[0].astype(np.float32)

        prob_list = [(CLASS_NAMES[CLASS_KEYS[i]], float(probs[i]) * 100.0)
                     for i in range(4)]

        # retrieval (L2 ranking, cosine similarity for display)
        results = self._retrieve(descriptor, k)

        # grad-cam overlay
        cam_uri = self._gradcam(x, idx.item(), pil_img)

        return {
            'pred_class': CLASS_NAMES[pred_key],
            'confidence': float(conf) * 100.0,
            'probs': prob_list,
            'results': results,
            'gradcam': cam_uri,
            'query_uri': _to_data_uri(pil_img.resize((224, 224))),
        }

    def _retrieve(self, descriptor, k):
        d = descriptor.astype(np.float32)
        # L2 distance (as in the FAISS IVFFlat index)
        diff = self.gallery_desc - d[None, :]
        l2 = np.sqrt((diff * diff).sum(axis=1))
        # cosine similarity for a friendly percentage
        du = d / (np.linalg.norm(d) + 1e-8)
        cos = self._gallery_unit @ du
        order = np.argsort(l2)[:k]
        out = []
        for rank, i in enumerate(order, 1):
            fn = self.gallery_files[i]
            out.append({
                'rank': rank,
                'file': fn,
                'label': CLASS_NAMES.get(self.gallery_labels[i], self.gallery_labels[i]),
                'distance': float(l2[i]),
                'similarity': max(0.0, float(cos[i])) * 100.0,
                'uri': self.gallery_uri(fn),
            })
        return out

    def _gradcam(self, x, class_idx, pil_img):
        import cv2
        self.model.zero_grad()
        x2 = x.clone().detach().requires_grad_(True)
        out = self.model(x2)
        score = out['logits'][0, class_idx]
        score.backward()
        act = self._act[0]            # (C,7,7)
        grad = self._grad[0]          # (C,7,7)
        weights = grad.mean(dim=(1, 2))               # (C,)
        cam = torch.relu((weights[:, None, None] * act).sum(0))
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        base = np.array(pil_img.resize((224, 224))).astype(np.float32)
        overlay = (0.45 * heat + 0.55 * base).clip(0, 255).astype(np.uint8)
        return _to_data_uri(Image.fromarray(overlay))

    # ---------- step-by-step walkthrough (for the presentation) ----------
    def analyze_steps(self, pil_img, k=6, filename=None):
        """Run the pipeline stage by stage and return what each stage produced."""
        import time
        timings = {}

        # Step 1 - the raw upload
        orig = pil_img
        step_upload = {
            'filename': filename or 'sample image',
            'size': '%d x %d px' % orig.size,
            'mode': orig.mode,
            'uri': _to_data_uri(_thumb(orig.convert('RGB'), 256)),
        }

        # Step 2 - preprocessing: RGB -> 224x224 -> tensor -> ImageNet normalisation
        t0 = time.perf_counter()
        rgb = orig.convert('RGB')
        resized = rgb.resize((224, 224))
        x = _transform(rgb).unsqueeze(0).to(self.device)
        timings['preprocess'] = (time.perf_counter() - t0) * 1000
        raw = np.asarray(resized, dtype=np.float32) / 255.0
        xn = x[0].cpu().numpy()
        step_pre = {
            'resized_uri': _to_data_uri(resized),
            'shape': 'tensor (1, 3, 224, 224)',
            'channels': [
                {'name': c, 'mean_raw': float(raw[..., i].mean()),
                 'mean_norm': float(xn[i].mean()), 'std_norm': float(xn[i].std()),
                 'uri': _to_data_uri(_heat(xn[i], 'gray', 112))}
                for i, c in enumerate(['Red', 'Green', 'Blue'])
            ],
        }

        # Steps 3-5 - one forward pass, keeping the intermediate tensors
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(x, return_features=True)
        timings['forward'] = (time.perf_counter() - t0) * 1000

        stages = []
        for i, f in enumerate(out['backbone_features']):
            f = f[0].cpu().numpy()
            stages.append({
                'name': 'Stage %d' % (i + 1),
                'shape': '%d x %d x %d' % f.shape,
                'uri': _to_data_uri(_heat(f.mean(0), 'viridis', 140)),
            })

        fw = self.model.neck.get_fusion_weights()
        fw = fw.detach().cpu().numpy().tolist() if fw is not None else []
        neck = out['neck_features'][0].cpu().numpy()
        step_fusion = {
            'weights': [{'stage': 'Stage %d' % (i + 1), 'w': w * 100.0}
                        for i, w in enumerate(fw)],
            'shape': '%d x %d x %d' % neck.shape,
            'uri': _to_data_uri(_heat(neck.mean(0), 'viridis', 140)),
        }

        desc = out['features'][0].cpu().numpy().astype(np.float32)
        g = out['global_features'][0].cpu().numpy()
        s = out['spatial_features'][0].cpu().numpy()
        a = out['attention_features'][0].cpu().numpy()
        att = out['attention_map'][0, 0].cpu().numpy()
        side = int(round(np.sqrt(desc.size)))
        step_head = {
            'parts': [
                {'name': 'Global branch', 'dim': g.size, 'what': 'overall tissue appearance (pooled)', 'norm': float(np.linalg.norm(g))},
                {'name': 'Spatial branch', 'dim': s.size, 'what': 'layout of structures (where things are)', 'norm': float(np.linalg.norm(s))},
                {'name': 'Attention branch', 'dim': a.size, 'what': 'the most informative regions', 'norm': float(np.linalg.norm(a))},
            ],
            'total': desc.size,
            'norm': float(np.linalg.norm(desc)),
            'attention_uri': _to_data_uri(_heat(att, 'jet', 140)),
            'barcode_uri': _to_data_uri(_heat(desc.reshape(side, side), 'viridis', 204, clip=(1, 99)))
                           if side * side == desc.size else None,
        }

        # Step 6 - similarity search over the gallery index
        t0 = time.perf_counter()
        results = self._retrieve(desc, k)
        timings['search'] = (time.perf_counter() - t0) * 1000
        diff = self.gallery_desc - desc[None, :]
        l2 = np.sqrt((diff * diff).sum(axis=1))
        lo, hi = float(l2.min()), float(l2.max())
        top = set(r['file'] for r in results)
        dots = []
        for i in np.argsort(l2):
            fn = self.gallery_files[i]
            dots.append({
                'x': (float(l2[i]) - lo) / (hi - lo + 1e-8) * 100.0,
                'cls': self.gallery_labels[i],
                'top': fn in top,
                'd': float(l2[i]),
            })
        step_search = {
            'n_gallery': len(self.gallery_files),
            'dim': self.gallery_desc.shape[1],
            'min': lo, 'max': hi,
            'dots': dots,
            'top1_barcode': _to_data_uri(_heat(
                self.gallery_desc[int(np.argmin(l2))].reshape(side, side), 'viridis', 204, clip=(1, 99)))
                if side * side == desc.size else None,
        }

        # Step 7 - decision: classification + Grad-CAM + agreement of retrieved cases
        probs = F.softmax(out['logits'], dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        pred = CLASS_NAMES[CLASS_KEYS[idx.item()]]
        t0 = time.perf_counter()
        cam_uri = self._gradcam(x, idx.item(), rgb)
        timings['gradcam'] = (time.perf_counter() - t0) * 1000
        votes = {}
        for r in results:
            votes[r['label']] = votes.get(r['label'], 0) + 1
        step_result = {
            'pred_class': pred,
            'confidence': float(conf) * 100.0,
            'probs': [(CLASS_NAMES[CLASS_KEYS[i]], float(probs[i]) * 100.0) for i in range(4)],
            'gradcam': cam_uri,
            'results': results,
            'votes': sorted(votes.items(), key=lambda kv: -kv[1]),
            'agree': sum(1 for r in results if r['label'] == pred),
        }

        return {
            'upload': step_upload, 'pre': step_pre, 'stages': stages,
            'fusion': step_fusion, 'head': step_head, 'search': step_search,
            'result': step_result, 'k': k,
            'timings': {key: round(v, 1) for key, v in timings.items()},
            'total_ms': round(sum(timings.values()), 1),
        }

    def gallery_uri(self, filename):
        return _to_data_uri(Image.open(os.path.join(GALLERY_DIR, filename)).resize((160, 160)))


# module-level singleton (lazy)
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
    return _engine
