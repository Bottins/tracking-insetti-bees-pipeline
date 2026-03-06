"""
BEES ANALYSIS PIPELINE  â€”  GPU + Parallel edition
===================================================
Pipeline completa per l'analisi video delle api.

Parallelizzazione:
  - Pre-processing: multiprocessing.Pool (CPU-bound I/O video in parallelo)
  - Tracking:       concurrent.futures.ProcessPoolExecutor (un processo per video)
  - Heatmap:        cupy (GPU) con fallback automatico a numpy (CPU)

GPU (NVIDIA RTX 3050 via WSL2 + CUDA):
  - MOG2 background subtraction su GPU con cv2.cuda se disponibile
  - Accumulo heatmap con cupy.histogram2d se disponibile
  - Fallback silenzioso a CPU se CUDA non Ã¨ installato

Struttura output:
  output/
    processed/N/Pre|Post/   video 1080p ritagliati
    processed/O/Pre|Post/
    csv/                    traiettorie CSV per video
    heatmaps/per_video/     heatmap singolo video
    heatmaps/confronto/     Pre vs Post, N vs O, griglia
    tracking_master.csv     dataset completo

Naming convention:
  {tesi}-{conc}({replica}).MOV   =>  N-0,001(1).MOV
  Tesi : N | O
  Conc : 0.001 | 0.01 | 0.1
  Replica: 1 | 2 | 3
  Phase  : Pre | Post   (estratti da ogni video originale)

  Pre  = 30s che precedono il minuto 0:59  (controllo, pre-molecola)
  Post = 60s che seguono il minuto 1:30   (trattamento, post-molecola)
"""

import os
import re
import sys
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# â”€â”€ Probe GPU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import cupy as cp
    # Verifica che le librerie CUDA runtime siano effettivamente disponibili
    import ctypes
    ctypes.CDLL("libnvrtc.so")
    _GPU = cp.cuda.is_available()
except Exception:
    cp   = None
    _GPU = False

try:
    _CUDA_CV = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    _CUDA_CV = False

# I print GPU vengono mostrati solo nel processo principale (in main())

# ===========================================================================
# 0. CONFIGURAZIONE
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = Path(os.getenv("BEES_INPUT_DIR", str(PROJECT_ROOT / "data" / "raw_videos")))
OUTPUT_BASE = Path(os.getenv("BEES_OUTPUT_DIR", str(PROJECT_ROOT / "output")))

# Tracking
GLOBAL_MIN_HITS = 30
GLOBAL_MAX_AGE  = 120
GLOBAL_MIN_AREA = 300
GLOBAL_MAX_AREA = 25000

# Pre-processing
CROP_PERCENT  = 0.10
TARGET_HEIGHT = 1080
TARGET_WIDTH  = 1920

# Finestre temporali (secondi)
PRE_END_SEC    = 59
PRE_DURATION   = 50
POST_START_SEC = 80
POST_DURATION  = 50

# Parallelismo
N_WORKERS = 4   # limitato per evitare OOM con video pesanti

# Heatmap
HEATMAP_BINS = 64

# Subsampling frame per il tracking (1 = tutti, 2 = 1 ogni 2, 3 = 1 ogni 3)
# Non influisce sulla qualitÃ  delle heatmap, solo sulla velocitÃ  di elaborazione
FRAME_STEP = 2

# ===========================================================================
# 1. PARSING NOME FILE
# ===========================================================================

_NAME_RE = re.compile(
    r'^([NO])\s*[-]\s*(\d[\d,]+)\s*\((\d+)\)(?:_(Pre|Post))?',
    re.IGNORECASE
)

def parse_video_name(filename: str) -> dict | None:
    stem = Path(filename).stem
    m = _NAME_RE.match(stem)
    if not m:
        return None
    tesi, conc_raw, replica, phase = m.groups()
    return {
        'tesi':    tesi.upper(),
        'conc':    conc_raw.replace(',', '.'),
        'replica': int(replica),
        'phase':   phase,
        'stem':    stem,
    }

# ===========================================================================
# 2. PRE-PROCESSING  (crop + split Pre/Post + resize 1080p)
# ===========================================================================

def _output_dir(tesi: str, phase: str) -> Path:
    d = OUTPUT_BASE / "processed" / tesi / phase
    d.mkdir(parents=True, exist_ok=True)
    return d

def _build_path(meta: dict, phase: str) -> Path:
    conc_str = meta['conc'].replace('.', ',')
    name = f"{meta['tesi']}-{conc_str}({meta['replica']})_{phase}.mp4"
    return _output_dir(meta['tesi'], phase) / name

def _write_segment(src_path: Path, meta: dict, phase: str,
                   f_start: int, f_end: int, fps: float,
                   orig_w: int, crop_px: int) -> Path:
    """Scrive un segmento (Pre o Post) ritagliato e ridimensionato."""
    out_path = _build_path(meta, phase)
    if out_path.exists():
        return out_path

    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps,
                              (TARGET_WIDTH, TARGET_HEIGHT))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter non aperto per {out_path}")

    for _ in range(f_end - f_start):
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[:, crop_px: orig_w - crop_px]
        resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT),
                              interpolation=cv2.INTER_AREA)
        writer.write(resized)

    writer.release()
    cap.release()

    # Verifica che il file sia stato scritto correttamente
    if not out_path.exists() or out_path.stat().st_size < 1000:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"File output vuoto o corrotto: {out_path}")

    return out_path

def preprocess_video(src_path: Path, meta: dict) -> dict:
    """Ritorna {'Pre': Path, 'Post': Path}."""
    cap   = cv2.VideoCapture(str(src_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    crop_px    = int(orig_w * CROP_PERCENT)
    pre_start  = max(0, int((PRE_END_SEC - PRE_DURATION) * fps))
    pre_end    = int(PRE_END_SEC * fps)
    post_start = int(POST_START_SEC * fps)
    post_end   = min(total, int((POST_START_SEC + POST_DURATION) * fps))

    results = {}
    for phase, (fs, fe) in [('Pre',  (pre_start,  pre_end)),
                              ('Post', (post_start, post_end))]:
        results[phase] = _write_segment(src_path, meta, phase,
                                         fs, fe, fps, orig_w, crop_px)
    return results

# Helper top-level per ProcessPool (deve essere picklable)
def _suppress_stderr():
    """Sopprime l'output spazzatura di ldconfig/OpenCV nei processi figli."""
    import sys, os
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 2)  # reindirizza fd 2 (stderr) a /dev/null

def _preprocess_worker(args):
    _suppress_stderr()
    src_path_str, meta = args
    return meta, preprocess_video(Path(src_path_str), meta)

# ===========================================================================
# 3. KALMAN TRACK
# ===========================================================================

class KalmanTrack:
    def __init__(self, x, y, tid):
        self.id = tid
        self.hits = 1
        self.no_match = 0

        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        kf.R *= 10.
        kf.P[2:,2:] *= 1000.
        kf.Q[2:,2:] *= 0.01
        kf.x[:2] = [[x],[y]]
        self.kf = kf

    def predict(self):   self.kf.predict()

    def update(self, x, y):
        self.kf.update(np.array([[x],[y]], dtype=float))
        self.hits += 1
        self.no_match = 0

    @property
    def pos(self):
        return float(self.kf.x[0][0]), float(self.kf.x[1][0])

# ===========================================================================
# 4. DETECT BLOBS  (CPU o GPU-MOG2)
# ===========================================================================

def _make_fgbg():
    if _CUDA_CV:
        return cv2.cuda.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False)
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False)

def _apply_fgbg(fgbg, gray):
    if _CUDA_CV:
        gpu_gray = cv2.cuda_GpuMat()
        gpu_gray.upload(gray)
        gpu_fg = fgbg.apply(gpu_gray, learningRate=-1)
        return gpu_fg.download()
    return fgbg.apply(gray)

def _detect_blobs(frame, fgbg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg   = _apply_fgbg(fgbg, gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (GLOBAL_MIN_AREA <= area <= GLOBAL_MAX_AREA):
            continue
        x,y,w,h = cv2.boundingRect(c)
        dets.append((x+w//2, y+h//2, area))
    return dets

# ===========================================================================
# 5. TRACKING
# ===========================================================================

def _track_video_impl(video_path: Path) -> pd.DataFrame:
    fgbg     = _make_fgbg()
    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracks   = []
    next_id  = 0
    rows     = []
    fidx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Subsampling: salta i frame intermedi ma aggiorna comunque il Kalman
        if fidx % FRAME_STEP != 0:
            for t in tracks:
                t.predict()
                t.no_match += 1
            tracks = [t for t in tracks if t.no_match <= GLOBAL_MAX_AGE]
            fidx += 1
            continue

        dets = _detect_blobs(frame, fgbg)

        for t in tracks:
            t.predict()

        if tracks and dets:
            pp   = np.array([t.pos for t in tracks])
            dp   = np.array([(d[0], d[1]) for d in dets])
            dist = np.linalg.norm(pp[:,None,:] - dp[None,:,:], axis=2)
            ri, ci = linear_sum_assignment(dist)
            m_t, m_d = set(), set()
            for r, c in zip(ri, ci):
                if dist[r,c] < 80:
                    tracks[r].update(dets[c][0], dets[c][1])
                    m_t.add(r); m_d.add(c)
            for ci2, d in enumerate(dets):
                if ci2 not in m_d:
                    tracks.append(KalmanTrack(d[0], d[1], next_id))
                    next_id += 1
            for ri2, t in enumerate(tracks):
                if ri2 not in m_t:
                    t.no_match += 1
        elif dets:
            for d in dets:
                tracks.append(KalmanTrack(d[0], d[1], next_id))
                next_id += 1

        for t in tracks:
            if t.hits >= GLOBAL_MIN_HITS:
                px, py = t.pos
                rows.append({'frame': fidx, 'track_id': t.id,
                              'x': px, 'y': py, 't': fidx / fps})

        tracks = [t for t in tracks if t.no_match <= GLOBAL_MAX_AGE]
        fidx += 1

    cap.release()

    if not rows:
        return pd.DataFrame(columns=['frame','track_id','x','y','t','video'])

    df = pd.DataFrame(rows)
    df['video'] = video_path.stem
    return df

# Helper top-level per ProcessPool
def _track_worker(vid_path_str: str) -> tuple[str, pd.DataFrame]:
    _suppress_stderr()
    return vid_path_str, _track_video_impl(Path(vid_path_str))

# ===========================================================================
# 6. VELOCITÃ€
# ===========================================================================

def add_speed(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, grp in df.groupby('track_id'):
        grp = grp.sort_values('frame').copy()
        if len(grp) >= 3:
            vx = np.gradient(grp['x'].values, grp['t'].values)
            vy = np.gradient(grp['y'].values, grp['t'].values)
            grp['speed'] = np.sqrt(vx**2 + vy**2)
        else:
            grp['speed'] = 0.0
        parts.append(grp)
    return pd.concat(parts, ignore_index=True) if parts else df.assign(speed=0.0)

# ===========================================================================
# 7. HEATMAP  (GPU con CuPy se disponibile)
# ===========================================================================

def _hist2d_gpu(xs, ys, bins, range_x, range_y):
    xcp = cp.asarray(xs)
    ycp = cp.asarray(ys)
    h, _, _ = cp.histogram2d(xcp, ycp, bins=bins,
                               range=[range_x, range_y])
    return cp.asnumpy(h)

def _hist2d_cpu(xs, ys, bins, range_x, range_y):
    h, _, _ = np.histogram2d(xs, ys, bins=bins,
                               range=[range_x, range_y])
    return h

def _hist2d(xs, ys, bins=HEATMAP_BINS,
            range_x=(0, TARGET_WIDTH), range_y=(0, TARGET_HEIGHT)):
    if _GPU:
        return _hist2d_gpu(xs, ys, bins, range_x, range_y)
    return _hist2d_cpu(xs, ys, bins, range_x, range_y)

def _speed_map_gpu(xs, ys, speeds, bins, xedges, yedges):
    sm = cp.zeros((bins, bins), dtype=cp.float32)
    cm = cp.zeros((bins, bins), dtype=cp.float32)
    xi = cp.clip(cp.searchsorted(cp.asarray(xedges[:-1]), cp.asarray(xs), side='right') - 1, 0, bins-1)
    yi = cp.clip(cp.searchsorted(cp.asarray(yedges[:-1]), cp.asarray(ys), side='right') - 1, 0, bins-1)
    sp = cp.asarray(speeds, dtype=cp.float32)
    for i in range(len(xs)):
        sm[xi[i], yi[i]] += sp[i]
        cm[xi[i], yi[i]] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        res = cp.where(cm > 0, sm / cm, 0)
    return cp.asnumpy(res)

def _speed_map_cpu(xs, ys, speeds, bins, xedges, yedges):
    sm = np.zeros((bins, bins), dtype=np.float32)
    cm = np.zeros((bins, bins), dtype=np.float32)
    xi = np.clip(np.searchsorted(xedges[:-1], xs, side='right') - 1, 0, bins-1)
    yi = np.clip(np.searchsorted(yedges[:-1], ys, side='right') - 1, 0, bins-1)
    np.add.at(sm, (xi, yi), speeds)
    np.add.at(cm, (xi, yi), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cm > 0, sm / cm, 0)

def make_heatmaps(df: pd.DataFrame, title: str, out_path: Path,
                  bins: int = HEATMAP_BINS):
    if df.empty:
        return

    xs = df['x'].values.astype(np.float32)
    ys = df['y'].values.astype(np.float32)

    xedges = np.linspace(0, TARGET_WIDTH,  bins + 1)
    yedges = np.linspace(0, TARGET_HEIGHT, bins + 1)

    pos_map = _hist2d(xs, ys, bins)

    if 'speed' in df.columns:
        sp = df['speed'].values.astype(np.float32)
        if _GPU:
            spd_map = _speed_map_gpu(xs, ys, sp, bins, xedges, yedges)
        else:
            spd_map = _speed_map_cpu(xs, ys, sp, bins, xedges, yedges)
    else:
        spd_map = pos_map.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for ax, data, label, cmap in [
        (axes[0], pos_map.T, 'DensitÃ  di Posizione',       'hot'),
        (axes[1], spd_map.T, 'VelocitÃ  Media (px/s)',       'plasma'),
    ]:
        img = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap,
                        extent=[0, TARGET_WIDTH, 0, TARGET_HEIGHT],
                        interpolation='bilinear')
        plt.colorbar(img, ax=ax, shrink=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [HEATMAP] {out_path.name}")

# ===========================================================================
# 8. HEATMAP MEDIE DI GRUPPO
# ===========================================================================

def compute_group_heatmaps(df: pd.DataFrame, bins: int = HEATMAP_BINS):
    """
    Restituisce (pos_maps, spd_maps):  dict (tesi, conc, phase) -> ndarray
    Ogni mappa Ã¨ la media delle repliche.
    """
    pos_acc = defaultdict(list)
    spd_acc = defaultdict(list)

    xedges = np.linspace(0, TARGET_WIDTH,  bins + 1)
    yedges = np.linspace(0, TARGET_HEIGHT, bins + 1)

    for (vid, tesi, conc, replica, phase), grp in df.groupby(
            ['video', 'tesi', 'conc', 'replica', 'phase']):

        xs = grp['x'].values.astype(np.float32)
        ys = grp['y'].values.astype(np.float32)
        key = (tesi, conc, phase)

        pm = _hist2d(xs, ys, bins)
        pos_acc[key].append(pm)

        if 'speed' in grp.columns:
            sp = grp['speed'].values.astype(np.float32)
            if _GPU:
                sm = _speed_map_gpu(xs, ys, sp, bins, xedges, yedges)
            else:
                sm = _speed_map_cpu(xs, ys, sp, bins, xedges, yedges)
            spd_acc[key].append(sm)

    def _mean(lst):
        return np.mean(np.stack(lst, axis=0), axis=0) if lst else None

    pos_maps = {k: _mean(v) for k, v in pos_acc.items()}
    spd_maps = {k: _mean(v) for k, v in spd_acc.items()}
    return pos_maps, spd_maps

# ===========================================================================
# 9. GRAFICI DI CONFRONTO
# ===========================================================================

def _shared_range(maps, keys):
    """Calcola vmin=0 e vmax comune tra una lista di mappe (per scala colore condivisa)."""
    arrays = [maps[k] for k in keys if k in maps and maps[k] is not None]
    if not arrays:
        return 0, 1
    vmax = max(float(a.max()) for a in arrays)
    return 0, vmax if vmax > 0 else 1

def _imshow_or_empty(ax, data, cmap, extent, vmin=None, vmax=None):
    if data is not None:
        img = ax.imshow(data.T, origin='lower', aspect='auto', cmap=cmap,
                        extent=extent, interpolation='bilinear',
                        vmin=vmin, vmax=vmax)
        plt.colorbar(img, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'Dati assenti', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

_EXTENT = [0, TARGET_WIDTH, 0, TARGET_HEIGHT]

def plot_comparison(pos_maps, spd_maps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys      = list(pos_maps.keys())
    tesi_set  = sorted(set(k[0] for k in keys))
    conc_set  = sorted(set(k[1] for k in keys), key=float)
    phase_set = ['Pre', 'Post']

    map_configs = [
        ('posizione', pos_maps, 'hot',    'DensitÃ  Posizione'),
        ('velocitÃ ',  spd_maps, 'plasma', 'VelocitÃ  Media (px/s)'),
    ]

    # A) Pre vs Post per ogni (tesi, conc)
    for tesi in tesi_set:
        for conc in conc_set:
            for mtype, maps, cmap, label in map_configs:
                pre_k  = (tesi, conc, 'Pre')
                post_k = (tesi, conc, 'Post')
                if pre_k not in maps and post_k not in maps:
                    continue
                vmin, vmax = _shared_range(maps, [pre_k, post_k])
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"Tesi {tesi} | Conc {conc} | {label} â€” Pre vs Post",
                             fontsize=13, fontweight='bold')
                for ax, phase, key in [(axes[0],'Pre',pre_k),(axes[1],'Post',post_k)]:
                    _imshow_or_empty(ax, maps.get(key), cmap, _EXTENT, vmin, vmax)
                    ax.set_title(f"{phase} (media repliche)", fontsize=11)
                    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
                plt.tight_layout()
                fname = f"confronto_PrePost_{tesi}_{conc.replace('.','p')}_{mtype}.png"
                plt.savefig(str(out_dir/fname), dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  [CONFRONTO Pre/Post] {fname}")

    # B) N vs O per ogni (phase, conc)
    for conc in conc_set:
        for phase in phase_set:
            for mtype, maps, cmap, label in map_configs:
                nk = ('N', conc, phase)
                ok = ('O', conc, phase)
                if nk not in maps and ok not in maps:
                    continue
                vmin, vmax = _shared_range(maps, [nk, ok])
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"Fase {phase} | Conc {conc} | {label} â€” N vs O",
                             fontsize=13, fontweight='bold')
                for ax, tesi, key in [(axes[0],'N',nk),(axes[1],'O',ok)]:
                    _imshow_or_empty(ax, maps.get(key), cmap, _EXTENT, vmin, vmax)
                    ax.set_title(f"Tesi {tesi} (media repliche)", fontsize=11)
                    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
                plt.tight_layout()
                fname = f"confronto_NO_{phase}_{conc.replace('.','p')}_{mtype}.png"
                plt.savefig(str(out_dir/fname), dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  [CONFRONTO N/O] {fname}")

    # C) Griglia completa â€” scala comune per tutta la griglia
    n_rows = len(tesi_set) * len(conc_set)
    n_cols = len(phase_set)
    if n_rows == 0:
        return
    for mtype, maps, cmap, label in map_configs:
        vmin, vmax = _shared_range(maps, list(maps.keys()))
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(9*n_cols, 5*n_rows), squeeze=False)
        fig.suptitle(f"Griglia Completa â€” {label}", fontsize=15, fontweight='bold')
        row = 0
        for tesi in tesi_set:
            for conc in conc_set:
                for col, phase in enumerate(phase_set):
                    ax  = axes[row][col]
                    key = (tesi, conc, phase)
                    _imshow_or_empty(ax, maps.get(key), cmap, _EXTENT, vmin, vmax)
                    ax.set_title(f"{tesi} | {conc} | {phase}", fontsize=9)
                    ax.set_xlabel('X'); ax.set_ylabel('Y')
                row += 1
        plt.tight_layout()
        fname = f"griglia_completa_{mtype}.png"
        plt.savefig(str(out_dir/fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  [GRIGLIA] {fname}")

# ===========================================================================
# 10. ANALISI COMPORTAMENTALE
#     Metriche semplici e leggibili per la tesi:
#     - VelocitÃ  media Pre vs Post per ogni tesi e concentrazione
#     - Distanza dal punto di iniezione (centro-basso) Pre vs Post
#     - Delta (Post - Pre) per valutare l'effetto della molecola
# ===========================================================================

# Punto di iniezione: centro-basso del frame (volatile, si diffonde verso l'alto)
INJECTION_X = TARGET_WIDTH  / 2   # centro orizzontale
INJECTION_Y = TARGET_HEIGHT * 0.8  # 80% verso il basso

def behavioral_analysis(df: pd.DataFrame, out_dir: Path):
    """
    Genera grafici comportamentali interpretabili:
    1. VelocitÃ  media Pre vs Post  (per tesi N/O e concentrazione)
    2. Distanza media dal punto di iniezione Pre vs Post
    3. Delta (Post-Pre) normalizzato â€” misura dell'effetto molecola
    4. Tabella riassuntiva CSV
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    conc_order = ['0.001', '0.01', '0.1']
    tesi_colors = {'N': '#2196F3', 'O': '#FF5722'}   # blu=N, arancio=O
    phase_colors = {'Pre': '#78909C', 'Post': '#E53935'}  # grigio=Pre, rosso=Post

    # â”€â”€ Calcola distanza dal punto di iniezione â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = df.copy()
    df['dist_inj'] = np.sqrt((df['x'] - INJECTION_X)**2 +
                              (df['y'] - INJECTION_Y)**2)

    # â”€â”€ Aggregazione corretta a tre livelli â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Livello 1: media per singola ape (track_id) dentro ogni replica
    per_track = df.groupby(['tesi', 'conc', 'replica', 'phase', 'track_id']).agg(
        speed_mean_track = ('speed',    'mean'),
        dist_mean_track  = ('dist_inj', 'mean'),
    ).reset_index()

    # Livello 2: media tra le api dentro ogni replica
    per_replica = per_track.groupby(['tesi', 'conc', 'replica', 'phase']).agg(
        speed_mean_replica = ('speed_mean_track', 'mean'),
        dist_mean_replica  = ('dist_mean_track',  'mean'),
    ).reset_index()

    # Livello 3: media e SEM tra le repliche (n=3)
    def sem(x): return x.std() / np.sqrt(len(x)) if len(x) > 1 else 0.0

    stats = per_replica.groupby(['tesi', 'conc', 'phase']).agg(
        speed_mean  = ('speed_mean_replica', 'mean'),
        speed_sem   = ('speed_mean_replica', sem),
        dist_mean   = ('dist_mean_replica',  'mean'),
        dist_sem    = ('dist_mean_replica',  sem),
        n_repliche  = ('speed_mean_replica', 'count'),
    ).reset_index()

    # Salva tabella
    stats.to_csv(out_dir / "metriche_comportamentali.csv", index=False)
    print(f"  [CSV] metriche_comportamentali.csv")

    # â”€â”€ Calcola Delta Post-Pre â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pre  = stats[stats['phase'] == 'Pre' ].set_index(['tesi','conc'])
    post = stats[stats['phase'] == 'Post'].set_index(['tesi','conc'])
    delta = pd.DataFrame({
        'speed_delta':  post['speed_mean'] - pre['speed_mean'],
        'dist_delta':   post['dist_mean']  - pre['dist_mean'],
    }).reset_index()
    delta.to_csv(out_dir / "delta_effetto_molecola.csv", index=False)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # GRAFICO 1: VelocitÃ  media Pre vs Post â€” N e O affiancati
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("VelocitÃ  Media delle Api: Pre vs Post iniezione molecola",
                 fontsize=14, fontweight='bold')

    for ax, conc in zip(axes, conc_order):
        sub = stats[stats['conc'] == conc]
        x   = np.arange(2)   # N, O
        w   = 0.3

        for ti, tesi in enumerate(['N', 'O']):
            for pi, phase in enumerate(['Pre', 'Post']):
                row = sub[(sub['tesi']==tesi) & (sub['phase']==phase)]
                if row.empty: continue
                val = row['speed_mean'].values[0]
                err = row['speed_sem'].values[0]
                xpos = ti + (pi - 0.5) * w
                color = phase_colors[phase]
                alpha = 0.6 if phase == 'Pre' else 1.0
                bar = ax.bar(xpos, val, w*0.9, color=color, alpha=alpha,
                             label=f"{phase}" if ti == 0 else "")
                ax.errorbar(xpos, val, yerr=err, fmt='none',
                            color='black', capsize=4, linewidth=1.5)

        ax.set_title(f"Concentrazione {conc} mg/L", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tesi N', 'Tesi O'], fontsize=11)
        ax.set_ylabel("VelocitÃ  media (px/s)" if ax == axes[0] else "")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if ax == axes[0]:
            ax.legend(title="Fase", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_dir / "1_velocita_pre_post.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [GRAFICO] 1_velocita_pre_post.png")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # GRAFICO 2: Distanza dal punto di iniezione Pre vs Post
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        f"Distanza media dal punto di iniezione (centro-basso)\nPre vs Post â€” molecola volatile",
        fontsize=13, fontweight='bold')

    for ax, conc in zip(axes, conc_order):
        sub = stats[stats['conc'] == conc]
        for ti, tesi in enumerate(['N', 'O']):
            for pi, phase in enumerate(['Pre', 'Post']):
                row = sub[(sub['tesi']==tesi) & (sub['phase']==phase)]
                if row.empty: continue
                val = row['dist_mean'].values[0]
                err = row['dist_sem'].values[0]
                xpos = ti + (pi - 0.5) * 0.3
                color = phase_colors[phase]
                alpha = 0.6 if phase == 'Pre' else 1.0
                ax.bar(xpos, val, 0.27, color=color, alpha=alpha,
                       label=f"{phase}" if ti == 0 else "")
                ax.errorbar(xpos, val, yerr=err, fmt='none',
                            color='black', capsize=4, linewidth=1.5)

        ax.set_title(f"Concentrazione {conc} mg/L", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tesi N', 'Tesi O'], fontsize=11)
        ax.set_ylabel("Distanza media (px)" if ax == axes[0] else "")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if ax == axes[0]:
            ax.legend(title="Fase", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_dir / "2_distanza_iniezione_pre_post.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [GRAFICO] 2_distanza_iniezione_pre_post.png")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # GRAFICO 3: Delta (Post - Pre) â€” effetto netto della molecola
    #   Positivo = piÃ¹ veloce / piÃ¹ lontano dopo iniezione  â†’ agitazione/fuga
    #   Negativo = piÃ¹ lento  / piÃ¹ vicino dopo iniezione   â†’ attrazione/torpore
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Effetto netto della molecola  (Î” Post âˆ’ Pre)\n"
                 "Positivo = agitazione/fuga  |  Negativo = attrazione/torpore",
                 fontsize=13, fontweight='bold')

    for ax, metric, ylabel, title in [
        (axes[0], 'speed_delta', 'Î” VelocitÃ  (px/s)',     'Variazione VelocitÃ '),
        (axes[1], 'dist_delta',  'Î” Distanza (px)',        'Variazione Distanza dal punto di iniezione'),
    ]:
        x     = np.arange(len(conc_order))
        w     = 0.3
        for ti, tesi in enumerate(['N', 'O']):
            sub = delta[delta['tesi'] == tesi].set_index('conc')
            vals = [sub.loc[c, metric] if c in sub.index else 0 for c in conc_order]
            bars = ax.bar(x + (ti - 0.5) * w, vals, w * 0.9,
                          label=f"Tesi {tesi}", color=tesi_colors[tesi], alpha=0.85)
            # Evidenzia barre positive/negative
            for bar, v in zip(bars, vals):
                bar.set_edgecolor('darkred' if v > 0 else 'darkblue')
                bar.set_linewidth(1.5)

        ax.axhline(0, color='black', linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c} mg/L" for c in conc_order], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(out_dir / "3_delta_effetto_molecola.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [GRAFICO] 3_delta_effetto_molecola.png")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # GRAFICO 4: Andamento velocitÃ  per concentrazione (dose-risposta)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dose-risposta: VelocitÃ  media in funzione della concentrazione",
                 fontsize=13, fontweight='bold')

    x_vals = [0.001, 0.01, 0.1]
    for ax, phase in zip(axes, ['Pre', 'Post']):
        sub = stats[stats['phase'] == phase]
        for tesi, color in tesi_colors.items():
            ys   = []
            errs = []
            for c in conc_order:
                row = sub[(sub['tesi']==tesi) & (sub['conc']==c)]
                ys.append(row['speed_mean'].values[0] if not row.empty else np.nan)
                errs.append(row['speed_sem'].values[0] if not row.empty else 0)
            ax.errorbar(x_vals, ys, yerr=errs, marker='o', linewidth=2,
                        markersize=7, label=f"Tesi {tesi}", color=color,
                        capsize=5)

        ax.set_xscale('log')
        ax.set_title(f"Fase {phase}", fontsize=12)
        ax.set_xlabel("Concentrazione (mg/L) â€” scala log", fontsize=10)
        ax.set_ylabel("VelocitÃ  media (px/s)" if ax == axes[0] else "")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(out_dir / "4_dose_risposta_velocita.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [GRAFICO] 4_dose_risposta_velocita.png")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # STAMPA RIEPILOGO TESTUALE
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print("\n  â”Œâ”€ RIEPILOGO EFFETTO MOLECOLA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”")
    for _, row in delta.iterrows():
        spd = row['speed_delta']
        dst = row['dist_delta']
        interp_spd = "â†‘ piÃ¹ agitate" if spd > 5 else ("â†“ piÃ¹ lente" if spd < -5 else "â‰ˆ invariate")
        interp_dst = "â†‘ si allontanano" if dst > 20 else ("â†“ si avvicinano" if dst < -20 else "â‰ˆ invariata")
        print(f"  â”‚  Tesi {row['tesi']} | Conc {row['conc']:5s}  â†’  "
              f"Vel {spd:+6.1f} px/s ({interp_spd})  |  "
              f"Dist {dst:+6.1f} px ({interp_dst})")
    print("  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜")


# ===========================================================================
# 11. ANALISI TEMPORALE A BIN DA 10s  (confronti_bin10)
#
#  Per ogni (tesi, conc) genera una figura con 2 righe Ã— 5 colonne:
#    Riga 1 (Pre) : bin t=0-10, 10-20, 20-30, 30-40, 40-50  (s9â†’s59 del video)
#    Riga 2 (Post): bin t=0-10, 10-20, 20-30, 30-40, 40-50  (s80â†’s130 del video)
#  Scala colore condivisa su tutta la figura (10 pannelli).
#  Produce una figura per posizione e una per velocitÃ .
# ===========================================================================

BIN_DURATION = 10   # secondi per bin
N_BINS       = 5    # numero di bin (50s / 10s)

def _bin_heatmap(df_seg: pd.DataFrame, t_start: float, t_end: float,
                 bins: int = HEATMAP_BINS):
    """Heatmap di posizione e velocitÃ  per un bin temporale [t_start, t_end)."""
    sub = df_seg[(df_seg['t'] >= t_start) & (df_seg['t'] < t_end)]
    if sub.empty:
        return None, None

    xs = sub['x'].values.astype(np.float32)
    ys = sub['y'].values.astype(np.float32)

    xedges = np.linspace(0, TARGET_WIDTH,  bins + 1)
    yedges = np.linspace(0, TARGET_HEIGHT, bins + 1)

    pos_map = _hist2d_cpu(xs, ys, bins,
                          range_x=(0, TARGET_WIDTH),
                          range_y=(0, TARGET_HEIGHT))
    if 'speed' in sub.columns:
        spd_map = _speed_map_cpu(xs, ys,
                                 sub['speed'].values.astype(np.float32),
                                 bins, xedges, yedges)
    else:
        spd_map = pos_map.copy()

    return pos_map, spd_map


def _mean_bin_heatmaps(df_group: pd.DataFrame, phase: str,
                       bins: int = HEATMAP_BINS):
    """
    Per una fase (Pre o Post), calcola la media delle repliche per ogni bin.
    Restituisce due liste di N_BINS ndarray: pos_bins, spd_bins
    (None dove non ci sono dati).
    """
    pos_bins = [[] for _ in range(N_BINS)]
    spd_bins = [[] for _ in range(N_BINS)]

    for replica, df_rep in df_group[df_group['phase'] == phase].groupby('replica'):
        for i in range(N_BINS):
            t0 = i * BIN_DURATION
            t1 = t0 + BIN_DURATION
            pm, sm = _bin_heatmap(df_rep, t0, t1, bins)
            if pm is not None:
                pos_bins[i].append(pm)
                spd_bins[i].append(sm)

    def _mean(lst):
        return np.mean(np.stack(lst, axis=0), axis=0) if lst else None

    return [_mean(b) for b in pos_bins], [_mean(b) for b in spd_bins]


def temporal_bin_analysis(df: pd.DataFrame, out_dir: Path,
                          bins: int = HEATMAP_BINS):
    """
    Fase 6: heatmap temporali a bin da 10s.
    Una figura per (tesi, conc, tipo_mappa):
      - 2 righe (Pre / Post) Ã— 5 colonne (bin 0-10s â€¦ 40-50s)
      - Scala colore condivisa su tutti i 10 pannelli
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    conc_set  = sorted(df['conc'].unique(), key=float)
    tesi_set  = sorted(df['tesi'].unique())

    # Etichette asse temporale assoluto
    pre_abs_start  = PRE_END_SEC - PRE_DURATION          # s9
    post_abs_start = POST_START_SEC                       # s80

    map_configs = [
        ('posizione', 'hot',    'DensitÃ  Posizione'),
        ('velocitÃ ',  'plasma', 'VelocitÃ  Media (px/s)'),
    ]

    for tesi in tesi_set:
        for conc in conc_set:
            df_tc = df[(df['tesi'] == tesi) & (df['conc'] == conc)]
            if df_tc.empty:
                continue

            # Calcola heatmap medie per bin (Pre e Post)
            pre_pos,  pre_spd  = _mean_bin_heatmaps(df_tc, 'Pre',  bins)
            post_pos, post_spd = _mean_bin_heatmaps(df_tc, 'Post', bins)

            for mtype, cmap, label in map_configs:
                pre_maps  = pre_pos  if mtype == 'posizione' else pre_spd
                post_maps = post_pos if mtype == 'posizione' else post_spd

                fname = (f"bin10_{tesi}_{conc.replace('.','p')}_{mtype}.png")
                out_path = out_dir / fname
                if out_path.exists():
                    print(f"  [SKIP] {fname} giÃ  presente")
                    continue

                # Scala comune su tutti i 10 pannelli
                all_maps = [m for m in pre_maps + post_maps if m is not None]
                if not all_maps:
                    continue
                vmax = max(float(m.max()) for m in all_maps)
                vmax = vmax if vmax > 0 else 1

                fig, axes = plt.subplots(2, N_BINS,
                                         figsize=(5 * N_BINS, 9),
                                         squeeze=False)
                fig.suptitle(
                    f"Tesi {tesi} | Conc {conc} | {label} â€” evoluzione temporale (bin 10s)\n"
                    f"Riga superiore: Pre (s{pre_abs_start}â†’s{PRE_END_SEC})  |  "
                    f"Riga inferiore: Post (s{post_abs_start}â†’s{post_abs_start + PRE_DURATION})",
                    fontsize=12, fontweight='bold')

                for col in range(N_BINS):
                    t0_abs_pre  = pre_abs_start  + col * BIN_DURATION
                    t0_abs_post = post_abs_start + col * BIN_DURATION

                    for row_idx, (data, t0_abs) in enumerate([
                        (pre_maps[col],  t0_abs_pre),
                        (post_maps[col], t0_abs_post),
                    ]):
                        ax = axes[row_idx][col]
                        phase_label = 'Pre' if row_idx == 0 else 'Post'
                        ax.set_title(
                            f"{phase_label}\ns{t0_abs}â€“s{t0_abs + BIN_DURATION}",
                            fontsize=9)
                        _imshow_or_empty(ax, data, cmap, _EXTENT, vmin=0, vmax=vmax)
                        ax.set_xlabel('X (px)', fontsize=7)
                        ax.set_ylabel('Y (px)' if col == 0 else '', fontsize=7)
                        ax.tick_params(labelsize=6)

                plt.tight_layout()
                plt.savefig(str(out_path), dpi=130, bbox_inches='tight')
                plt.close(fig)
                print(f"  [BIN10] {fname}")


# ===========================================================================
# 10. PIPELINE PRINCIPALE
# ===========================================================================

def main():
    t0 = time.time()
    print("=" * 65)
    print(" BEES ANALYSIS PIPELINE  â€”  GPU + Parallel")
    print(f" Workers CPU: {N_WORKERS}  |  GPU heatmap: {_GPU}  |  GPU MOG2: {_CUDA_CV}")
    print("=" * 65)
    if _GPU:
        print("[GPU] CuPy disponibile â€” heatmap su GPU")
    else:
        print("[CPU] CuPy non trovato â€” heatmap su CPU (numpy)")
    if _CUDA_CV:
        print("[GPU] OpenCV CUDA disponibile â€” MOG2 su GPU")
    else:
        print("[CPU] OpenCV CUDA non trovato â€” MOG2 su CPU")

    src_videos = (sorted(INPUT_DIR.glob("*.[Mm][Oo][Vv]")) +
                  sorted(INPUT_DIR.glob("*.[Mm][Pp][4]"))  +
                  sorted(INPUT_DIR.glob("*.[Aa][Vv][Ii]")))

    if not src_videos:
        print(f"[ERRORE] Nessun video in {INPUT_DIR}")
        return

    print(f"\nTrovati {len(src_videos)} video originali.")

    # Parse metadati
    parsed = []
    for src in src_videos:
        meta = parse_video_name(src.name)
        if meta is None:
            print(f"  [SKIP] nome non riconosciuto: {src.name}")
            continue
        parsed.append((src, meta))

    # â”€â”€ FASE 1: PRE-PROCESSING in parallelo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n--- FASE 1: PRE-PROCESSING  (parallelo, {N_WORKERS} workers) ---")
    args = [(str(src), meta) for src, meta in parsed]

    processed_map = {}   # (tesi, conc, replica, phase) -> (Path, meta)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_preprocess_worker, a): a for a in args}
        for fut in as_completed(futures):
            meta, segs = fut.result()
            print(f"  OK  {meta['tesi']}-{meta['conc']}({meta['replica']})")
            for phase, path in segs.items():
                key = (meta['tesi'], meta['conc'], meta['replica'], phase)
                processed_map[key] = (path, meta)

    if not processed_map:
        print("[ERRORE] Nessun video pre-processato.")
        return

    # â”€â”€ FASE 2: TRACKING in parallelo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n--- FASE 2: TRACKING  (parallelo, {N_WORKERS} workers) ---")
    csv_dir = OUTPUT_BASE / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Prepara lista di video da tracciare (salta se CSV esiste giÃ )
    to_track = []
    pre_loaded = {}
    for (tesi, conc, replica, phase), (vid_path, meta) in sorted(processed_map.items()):
        csv_path = csv_dir / f"tracks_{vid_path.stem}.csv"
        if csv_path.exists():
            pre_loaded[(tesi, conc, replica, phase)] = (vid_path, meta, csv_path)
        else:
            to_track.append((tesi, conc, replica, phase, vid_path, meta, csv_path))

    all_dfs = []

    # Carica CSV giÃ  esistenti
    for (tesi, conc, replica, phase), (vid_path, meta, csv_path) in pre_loaded.items():
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [ERRORE LOAD] {csv_path.name}: {e}")
            continue
        if df.empty or len(df) < 2:
            print(f"  [SKIP] {csv_path.name} Ã¨ vuoto, verrÃ  ri-tracciato")
            to_track.append((tesi, conc, replica, phase, vid_path, meta, csv_path))
            continue
        print(f"  [LOAD] {csv_path.name}  ({len(df)} righe)")
        df['tesi']    = tesi
        df['conc']    = conc
        df['replica'] = replica
        df['phase']   = phase
        df = add_speed(df)
        all_dfs.append(df)

    # Traccia in parallelo i video mancanti
    if to_track:
        vid_paths_str = [str(item[4]) for item in to_track]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_track_worker, p): p for p in vid_paths_str}
            for fut in as_completed(futures):
                vid_str, df = fut.result()
                vid_path = Path(vid_str)
                # Trova l'entry corrispondente
                entry = next(it for it in to_track if str(it[4]) == vid_str)
                tesi, conc, replica, phase = entry[0], entry[1], entry[2], entry[3]
                csv_path = entry[6]

                print(f"  [TRACK] {vid_path.name}  ->  {len(df)} righe")
                if df.empty: continue
                df.to_csv(csv_path, index=False)

                df['tesi']    = tesi
                df['conc']    = conc
                df['replica'] = replica
                df['phase']   = phase
                df = add_speed(df)
                all_dfs.append(df)

    master_csv = OUTPUT_BASE / "tracking_master.csv"

    if not all_dfs:
        # Prova a caricare il master CSV giÃ  esistente
        if master_csv.exists():
            print(f"[INFO] Nessun nuovo tracking â€” carico master esistente: {master_csv}")
            df_all = pd.read_csv(master_csv)
            if df_all.empty:
                print("[ERRORE] tracking_master.csv Ã¨ vuoto.")
                return
            print(f"[INFO] Master caricato: {len(df_all)} righe, colonne: {list(df_all.columns)}")
        else:
            print("[ERRORE] Nessun dato di tracking e nessun master CSV trovato.")
            return
    else:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(master_csv, index=False)
        print(f"\n[INFO] Dataset completo: {len(df_all)} righe  ->  {master_csv}")

    # Verifica che le colonne necessarie esistano
    required = {'tesi', 'conc', 'replica', 'phase', 'x', 'y'}
    missing = required - set(df_all.columns)
    if missing:
        print(f"[ERRORE] Colonne mancanti nel dataset: {missing}")
        print(f"         Colonne presenti: {list(df_all.columns)}")
        return

    print(f"\n[DEBUG] Gruppi trovati nel dataset:")
    for key, grp in df_all.groupby(['tesi','conc','replica','phase']):
        print(f"  {key}  ->  {len(grp)} righe")

    # â”€â”€ FASE 3: HEATMAP PER SINGOLO VIDEO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- FASE 3: HEATMAP PER SINGOLO VIDEO ---")
    hmap_dir = OUTPUT_BASE / "heatmaps" / "per_video"
    hmap_dir.mkdir(parents=True, exist_ok=True)

    # Itera direttamente sui gruppi del dataframe â€” nessuna dipendenza da processed_map
    for (tesi, conc, replica, phase), sub in df_all.groupby(['tesi','conc','replica','phase']):
        vid_stem = sub['video'].iloc[0]
        out = hmap_dir / f"heatmap_{vid_stem}.png"
        if out.exists():
            print(f"  [SKIP] {out.name} giÃ  presente")
            continue
        title = f"Tesi {tesi} | Conc {conc} | Rep {replica} | {phase}"
        try:
            make_heatmaps(sub, title, out)
        except Exception as e:
            print(f"  [ERRORE heatmap] {vid_stem}: {e}")

    # â”€â”€ FASE 4: HEATMAP MEDIE E CONFRONTO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- FASE 4: HEATMAP MEDIE E CONFRONTO ---")
    try:
        pos_maps, spd_maps = compute_group_heatmaps(df_all)
        compare_dir = OUTPUT_BASE / "heatmaps" / "confronto"
        plot_comparison(pos_maps, spd_maps, compare_dir)
    except Exception as e:
        import traceback
        print(f"[ERRORE Fase 4] {e}")
        traceback.print_exc()

    # â”€â”€ FASE 5: ANALISI COMPORTAMENTALE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- FASE 5: ANALISI COMPORTAMENTALE ---")
    try:
        behavioral_analysis(df_all, OUTPUT_BASE / "analisi")
    except Exception as e:
        import traceback
        print(f"[ERRORE Fase 5] {e}")
        traceback.print_exc()

    # â”€â”€ FASE 6: ANALISI TEMPORALE BIN 10s â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- FASE 6: ANALISI TEMPORALE BIN 10s ---")
    try:
        temporal_bin_analysis(df_all, OUTPUT_BASE / "confronti_bin10")
    except Exception as e:
        import traceback
        print(f"[ERRORE Fase 6] {e}")
        traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f" PIPELINE COMPLETATA in {elapsed:.0f}s")
    print("=" * 65)
    print(f" Output: {OUTPUT_BASE}")
    print(f"   processed/{{N,O}}/{{Pre,Post}}/  -> video 1080p ritagliati")
    print(f"   csv/                           -> traiettorie CSV")
    print(f"   heatmaps/per_video/            -> heatmap singolo video")
    print(f"   heatmaps/confronto/            -> Pre/Post, N/O, griglia")
    print(f"   confronti_bin10/               -> evoluzione temporale bin 10s")
    print(f"   tracking_master.csv            -> dataset completo")
    print("=" * 65)


if __name__ == "__main__":
    # Necessario per ProcessPoolExecutor su Windows / WSL
    mp.set_start_method('spawn', force=True)
    main()


