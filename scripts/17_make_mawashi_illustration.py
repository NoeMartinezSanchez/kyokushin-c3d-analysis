#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_make_mawashi_illustration.py
===============================
Genera la ilustración de la pestaña de bienvenida del dashboard (S04 — Mawashi
Geri jodan) a partir del C3D real de B0400-S04-E01-T01:

  images/s04_mawashi_wireframe.gif        # stick figure 3D + trayectoria roja del pie
  images/s04_mawashi_leg_velocities.png   # velocidad de puntos de la pierna derecha

Solo LEE el C3D (no lo modifica) y escribe en images/. No toca datos ni pipeline.
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c = ezc3d.c3d(str(C3D_FILE))
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    pts = c["data"]["points"]  # (4, n_points, n_frames), mm

    # ---- ventana de la patada (pico de velocidad del pie derecho) ----
    irtoe = _idx(labels, "RTOE")
    if irtoe < 0:
        raise RuntimeError("RTOE no disponible en el archivo")
    v = _speed(pts, irtoe, rate)
    peak = int(np.argmax(v))
    n_frames = pts.shape[2]
    w0, w1 = max(0, peak - 45), min(n_frames, peak + 70)
    t = np.arange(w0, w1) / rate

    # ---- nodos del stick figure (3D, mm -> m) ----
    def node(name, fallback=()):
        i = _idx(labels, name)
        if i >= 0:
            return pts[:3, i, w0:w1] * 1e-3
        for f in fallback:
            i = _idx(labels, f)
            if i >= 0:
                return pts[:3, i, w0:w1] * 1e-3
        return None

    rasi = node("RASI"); lasi = node("LASI")
    rpsi = node("RPSI"); lpsi = node("LPSI")
    pelvis = None
    present = [x for x in (rasi, lasi, rpsi, lpsi) if x is not None]
    if present:
        pelvis = np.mean(present, axis=0)

    rsho = node("RSHO")
    lsho = node("LSHO")
    mshoulder = (rsho + lsho) / 2 if (rsho is not None and lsho is not None) \
        else (rsho if rsho is not None else None)

    hd = [node(n) for n in ("RFHD", "LFHD", "RBHD", "LBHD")]
    hd = [x for x in hd if x is not None]
    head = np.mean(hd, axis=0) if hd else None

    nodes = {}
    for side in ("R", "L"):
        for part in ("THI", "KNE", "TIB", "ANK", "HEE", "TOE",
                     "SHO", "ELB", "WRB", "FIN"):
            x = node(f"{side}{part}")
            if x is not None:
                nodes[f"{side}{part}"] = x

    # edges: (a, b)
    edges = []
    if pelvis is not None:
        for side in ("R", "L"):
            if f"{side}THI" in nodes:
                edges.append(("PELVIS", f"{side}THI"))
            for a, b in ((f"{side}THI", f"{side}KNE"),
                         (f"{side}KNE", f"{side}TIB"),
                         (f"{side}TIB", f"{side}ANK"),
                         (f"{side}ANK", f"{side}HEE"),
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
        for a, b in ((f"{side}SHO", f"{side}ELB"),
                     (f"{side}ELB", f"{side}WRB"),
                     (f"{side}WRB", f"{side}FIN")):
            if a in nodes and b in nodes:
                edges.append((a, b))

    coord = {"PELVIS": pelvis} if pelvis is not None else {}
    coord.update(nodes)
    if mshoulder is not None:
        coord["MSHOULDER"] = mshoulder
    if head is not None:
        coord["HEAD"] = head

    def frame_xyz(i):
        return {k: v[:, i] for k, v in coord.items()}

    # ---- A) gráfica de velocidades ----
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for mk in ("RTOE", "RANK", "RHEE", "RKNE"):
        i = _idx(labels, mk)
        if i >= 0:
            ax.plot(t, _speed(pts, i, rate)[w0:w1], lw=1.6, label=mk)
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("velocidad [m/s]")
    ax.set_title("Velocidad de los puntos de la pierna durante el "
                 "Mawashi-Geri jodan (B0400 · S04 · E01-T01)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s04_mawashi_leg_velocities.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- B) GIF wireframe 3D (media resolución + optimización) ----
    step = max(1, round((w1 - w0) / 60))
    frames = np.arange(w0, w1, step)
    f3d = plt.figure(figsize=(3.5, 3.5))
    ax3 = f3d.add_subplot(111, projection="3d")
    ax3.grid(False)

    def draw_frame(abs_i):
        local = int(abs_i) - w0
        ax3.clear()
        co = frame_xyz(local)
        for a, b in edges:
            pa, pb = co.get(a), co.get(b)
            if pa is None or pb is None:
                continue
            ax3.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                     color="#1f4e79", lw=2.2, alpha=0.95)
        # trayectoria del pie en rojo (hasta el frame actual)
        toe = coord.get("RTOE")
        if toe is not None and local + 1 > 0:
            x0 = toe[0, :local + 1]
            y0 = toe[1, :local + 1]
            z0 = toe[2, :local + 1]
            ax3.plot(x0, y0, z0, color="#c0392b", lw=2.4, ls="--", alpha=0.9,
                     label="trayectoria del pie")
            ax3.scatter([x0[-1]], [y0[-1]], [z0[-1]], color="#c0392b", s=30)
        ax3.set_title(f"Mawashi-Geri jodan — B0400 (t = {t[local]:.2f} s)")
        ax3.set_xlabel("m"); ax3.set_ylabel("m"); ax3.set_zlabel("m")
        ax3.legend(loc="upper left", fontsize=7)

    ani = FuncAnimation(f3d, draw_frame, frames=frames, interval=90)
    gif_path = OUT_DIR / "s04_mawashi_wireframe.gif"
    ani.save(gif_path, writer=PillowWriter(fps=10))
    plt.close(f3d)

    # optimización: paleta 128 colores + optimización de tamaño (~mitad)
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
    print(f"[17] generado: {OUT_DIR / 's04_mawashi_wireframe.gif'}")
    print(f"[17] generado: {OUT_DIR / 's04_mawashi_leg_velocities.png'}")


if __name__ == "__main__":
    main()