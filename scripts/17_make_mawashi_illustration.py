#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_make_mawashi_illustration.py
===============================
Genera ilustraciones de wireframe 3D (GIF) a partir de un C3D real.

Reutilizable:
  make_wireframe_gif(c3d_file, out_gif, endpoint_marker, axes_limits=None,
                     title_fn=None)
    - stick figure 3D (marcadores anatómicos, pelvis, torso, cabeza)
    - puntos en los marcadores (mismo color de la línea)
    - trayectoria del marcador efectuador (puño/patada) en rojo
    - GRID FIJO (set_xlim/ylim/zlim constantes) para que la figura no crezca
      ni decrezca entre frames
  scene(c3d_file, endpoint_marker)  -> dict (coord/edges/t)
  extent_of(scene)                  -> bbox para grid fijo por GIF
  plot_velocities(c3d_file, out_png, markers) -> gráfica de velocidades

Solo LEE C3D (no lo modifica) y escribe imágenes. No toca datos ni pipeline.
Invocación de demo (comportamiento previo conservado):
  main() -> images/s04_mawashi_wireframe.gif + images/s04_mawashi_leg_velocities.png

Los GIF por atleta×técnica se generan con scripts/22_build_athlete_gallery.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import ezc3d

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
C3D_FILE = ROOT / "atletas" / "B0400" / "2017-03-28-B0400-S04" / \
    "2017-03-28-B0400-S04-E01-T01.c3d"
OUT_DIR = ROOT / "images"

SMOOTH_WIN = 15
LINE_COLOR = "#1f4e79"
TRAJ_COLOR = "#c0392b"


def _idx(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError:
        return -1


def _speed(pts, i, rate):
    traj = pts[:3, i, :].astype(float).copy()
    v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
    k = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    return np.convolve(v, k, mode="same")


def scene(c3d_file: Path, endpoint_marker: str) -> dict:
    """Scene del stick figure sobre la ventana centrada en el pico del endpoint."""
    c = ezc3d.c3d(str(c3d_file))
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"]

    ie = _idx(labels, endpoint_marker)
    if ie < 0:
        raise RuntimeError(f"marcador {endpoint_marker} no disponible")
    v = _speed(pts, ie, rate)
    peak = int(np.argmax(v))
    n = pts.shape[2]
    w0, w1 = max(0, peak - 45), min(n, peak + 70)
    t = np.arange(w0, w1) / rate

    def node(name, fallback=()):
        i = _idx(labels, name)
        if i >= 0:
            return pts[:3, i, w0:w1] * 1e-3
        for f in fallback:
            i = _idx(labels, f)
            if i >= 0:
                return pts[:3, i, w0:w1] * 1e-3
        return None

    rasi, lasi = node("RASI"), node("LASI")
    rpsi, lpsi = node("RPSI"), node("LPSI")
    present = [x for x in (rasi, lasi, rpsi, lpsi) if x is not None]
    pelvis = np.mean(present, axis=0) if present else None

    rsho, lsho = node("RSHO"), node("LSHO")
    mshoulder = (rsho + lsho) / 2 if (rsho is not None and lsho is not None) \
        else (rsho if rsho is not None else None)
    hd = [x for x in (node("RFHD"), node("LFHD"), node("RBHD"), node("LBHD"))
          if x is not None]
    head = np.mean(hd, axis=0) if hd else None

    nodes = {}
    for side in ("R", "L"):
        for part in ("THI", "KNE", "TIB", "ANK", "HEE", "TOE",
                     "SHO", "ELB", "WRB", "FIN"):
            x = node(f"{side}{part}")
            if x is not None:
                nodes[f"{side}{part}"] = x

    edges = []
    if pelvis is not None:
        for side in ("R", "L"):
            if f"{side}THI" in nodes:
                edges.append(("PELVIS", f"{side}THI"))
            for a, b in ((f"{side}THI", f"{side}KNE"), (f"{side}KNE", f"{side}TIB"),
                         (f"{side}TIB", f"{side}ANK"), (f"{side}ANK", f"{side}HEE"),
                         (f"{side}ANK", f"{side}TOE")):
                if a in nodes and b in nodes:
                    edges.append((a, b))
    if mshoulder is not None:
        edges.append(("PELVIS", "MSHOULDER"))
    if head is not None and mshoulder is not None:
        edges.append(("MSHOULDER", "HEAD"))
    for side in ("R", "L"):
        if mshoulder is not None and f"{side}SHO" in nodes:
            edges.append(("MSHOULDER", f"{side}SHO"))
        for a, b in ((f"{side}SHO", f"{side}ELB"), (f"{side}ELB", f"{side}WRB"),
                     (f"{side}WRB", f"{side}FIN")):
            if a in nodes and b in nodes:
                edges.append((a, b))

    coord = {"PELVIS": pelvis} if pelvis is not None else {}
    coord.update(nodes)
    if mshoulder is not None:
        coord["MSHOULDER"] = mshoulder
    if head is not None:
        coord["HEAD"] = head

    return {"coord": coord, "edges": edges, "t": t, "w0": w0,
            "endpoint": coord.get(endpoint_marker),
            "endpoint_marker": endpoint_marker, "rate": rate}


def extent_of(sc: dict) -> tuple:
    """Bounding box (x0,x1,y0,y1,z0,z1) sobre TODA la ventana (grid fijo)."""
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for _, arr in sc["coord"].items():
        mins = np.minimum(mins, arr.min(axis=1))
        maxs = np.maximum(maxs, arr.max(axis=1))
    pad = 0.12 * float((maxs - mins).max())
    return (float(mins[0]) - pad, float(maxs[0]) + pad,
            float(mins[1]) - pad, float(maxs[1]) + pad,
            float(mins[2]) - pad, float(maxs[2]) + pad)


def _optimize_gif(gif_path: Path) -> None:
    from PIL import Image
    im = Image.open(gif_path)
    imgs = []
    try:
        n = 0
        while True:
            fr = im.convert("P", palette=Image.ADAPTIVE, colors=128).copy()
            imgs.append(fr)
            im.seek(n + 1)
            n += 1
    except EOFError:
        pass
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 optimize=True, duration=100, loop=0)


def make_wireframe_gif(c3d_file: Path, out_gif: Path, endpoint_marker: str,
                       axes_limits=None, title_fn=None, scene_dict=None,
                       frames_target: int = 24) -> None:
    """Genera un GIF wireframe 3D con trayectoria del endpoint (GRID fijo).

    scene_dict: escena ya calculada (scene()) para evitar reabrir el C3D.
    frames_target: nº aproximado de frames (render ligero para galerías grandes).
    """
    sc = scene_dict if scene_dict is not None else scene(c3d_file, endpoint_marker)
    if axes_limits is None:
        axes_limits = extent_of(sc)

    names = list(sc["coord"].keys())
    idx_of = {n: i for i, n in enumerate(names)}
    edge_idx = [(idx_of[a], idx_of[b]) for a, b in sc["edges"]
                if a in idx_of and b in idx_of]
    nw = len(sc["t"])
    step = max(1, round(nw / frames_target))
    frames = range(0, nw, step)

    fig = plt.figure(figsize=(2.9, 2.9), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    _title = title_fn or (lambda local_t, mk: f"t = {local_t:.2f} s")

    def draw(f):
        local = int(f)
        ax.clear()
        pts = {n: sc["coord"][n][:, local] for n in names}
        for i, j in edge_idx:
            pa, pb = pts[names[i]], pts[names[j]]
            if pa is None or pb is None:
                continue
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                    color=LINE_COLOR, lw=2.0, alpha=0.95)
        xs = [pts[n][0] for n in names if pts[n] is not None]
        ys = [pts[n][1] for n in names if pts[n] is not None]
        zs = [pts[n][2] for n in names if pts[n] is not None]
        if xs:
            ax.scatter(xs, ys, zs, color=LINE_COLOR, s=7, depthshade=False)
        if sc["endpoint"] is not None and local + 1 > 0:
            ex = sc["endpoint"][0, :local + 1]
            ey = sc["endpoint"][1, :local + 1]
            ez = sc["endpoint"][2, :local + 1]
            ax.plot(ex, ey, ez, color=TRAJ_COLOR, lw=2.2, ls="--", alpha=0.9)
            ax.scatter([ex[-1]], [ey[-1]], [ez[-1]], color=TRAJ_COLOR, s=24)
        ax.set_xlim(axes_limits[0], axes_limits[1])
        ax.set_ylim(axes_limits[2], axes_limits[3])
        ax.set_zlim(axes_limits[4], axes_limits[5])
        ax.set_xlabel("m"); ax.set_ylabel("m"); ax.set_zlabel("m")
        ax.set_title(_title(sc["t"][local], endpoint_marker), fontsize=8)

    ani = FuncAnimation(fig, draw, frames=frames, interval=90)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=PillowWriter(fps=8))
    plt.close(fig)
    _optimize_gif(out_gif)


def plot_velocities(c3d_file: Path, out_png: Path, markers,
                    title: str = "") -> None:
    c = ezc3d.c3d(str(c3d_file))
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"]
    ie = _idx(labels, markers[0])
    if ie < 0:
        raise RuntimeError(f"marcador {markers[0]} no disponible")
    v = _speed(pts, ie, rate)
    peak = int(np.argmax(v))
    n = pts.shape[2]
    w0, w1 = max(0, peak - 45), min(n, peak + 70)
    t = np.arange(w0, w1) / rate
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for mk in markers:
        i = _idx(labels, mk)
        if i >= 0:
            ax.plot(t, _speed(pts, i, rate)[w0:w1], lw=1.6, label=mk)
    ax.set_xlabel("tiempo [s]"); ax.set_ylabel("velocidad [m/s]")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_wireframe_gif(C3D_FILE, OUT_DIR / "s04_mawashi_wireframe.gif", "RTOE")
    plot_velocities(C3D_FILE, OUT_DIR / "s04_mawashi_leg_velocities.png",
                    ("RTOE", "RANK", "RHEE", "RKNE"),
                    "Velocidad de los puntos de la pierna durante el "
                    "Mawashi-Geri jodan (B0400 · S04 · E01-T01)")
    print(f"[17] generado: {OUT_DIR / 's04_mawashi_wireframe.gif'}")
    print(f"[17] generado: {OUT_DIR / 's04_mawashi_leg_velocities.png'}")


if __name__ == "__main__":
    main()