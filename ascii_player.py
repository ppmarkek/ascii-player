"""ascii_player.py - video as coloured ASCII art.

  python ascii_player.py                    built-in 3D Hexcore right in the terminal
  python ascii_player.py --scene bouquet    built-in 3D bouquet
  python ascii_player.py clip.mp4           your own video in the terminal
  python ascii_player.py --export out.mp4   vertical 1080x1920 clip: ASCII + code
  python ascii_player.py clip.mp4 --export out.mp4

Needs numpy, Pillow and ffmpeg in PATH. Press Ctrl+C to leave terminal mode.
"""
import argparse
import builtins
import functools
import io
import keyword
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import tokenize

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
EDGES = "|/-\\"
FFMPEG = shutil.which("ffmpeg")
FONTS = ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc", "LiberationMono-Regular.ttf"]
BOLD = ["consolab.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"]


def load_font(size, bold=False):
    for name in (BOLD + FONTS if bold else FONTS):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


# ---------------------------------------------------------------- frame sources

F32 = np.float32
LEAF = np.array((35, 150, 60), F32)
MOUTH = np.array((0, 0, -1.05), F32)  # neck of the vase: the stems grow from here
SIZE = {"rose": 0.62, "daisy": 0.64, "tulip": 0.74, "ball": 0.3}
KINDS = ("rose", "daisy", "tulip", "rose", "daisy")  # the order a flower is reborn in
COLORS = {  # petal base, edge, heart
    "rose": [((215, 0, 85), (255, 150, 195)), ((170, 0, 15), (255, 60, 50)),
             ((215, 170, 150), (255, 250, 235)), ((230, 60, 30), (255, 175, 120)),
             ((190, 0, 120), (255, 120, 225))],
    "daisy": [((255, 160, 0), (255, 240, 90), (110, 60, 20)),
              ((195, 195, 215), (255, 255, 255), (255, 200, 40)),
              ((230, 70, 0), (255, 175, 60), (90, 40, 10)),
              ((120, 40, 200), (215, 165, 255), (255, 220, 90))],
    "tulip": [((90, 20, 170), (205, 125, 255)), ((230, 30, 20), (255, 220, 60)),
              ((220, 40, 110), (255, 190, 215))],
    "ball": [((30, 60, 220), (175, 215, 255)), ((110, 70, 210), (225, 200, 255)),
             ((120, 170, 230), (255, 255, 255))],
}
VASES = np.array([  # height, foot, belly, where, its width, flare, ribs, colour, band
    (1.40, 0.17, 0.12, 0.30, 0.30, 0.05, 0.00, 70, 125, 240, 205, 228, 255),
    (1.40, 0.30, 0.33, 0.38, 0.24, 0.09, 0.14, 225, 105, 40, 250, 220, 170),
    (1.25, 0.24, 0.46, 0.46, 0.28, 0.04, 0.00, 238, 120, 172, 255, 236, 242),
    (1.15, 0.30, 0.20, 1.00, 0.60, 0.08, 0.22, 238, 238, 242, 70, 120, 230),
    (1.45, 0.20, 0.26, 0.60, 0.20, 0.13, 0.00, 245, 185, 40, 150, 70, 20),
    (1.30, 0.27, 0.38, 0.28, 0.22, 0.15, 0.10, 45, 95, 215, 240, 240, 255),
    (1.20, 0.33, 0.24, 0.50, 0.40, 0.06, 0.00, 150, 215, 190, 235, 70, 100),
], F32)


def ease(x):
    """Smooth step 0..1; works for numbers and arrays alike."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def grid(p, mix):
    """Grid of points (..., u, v, 3) -> points, normals and the share of edge colour."""
    p = np.stack(np.broadcast_arrays(*p), -1).astype(F32)
    n = np.cross(np.gradient(p, axis=-3), np.gradient(p, axis=-2))
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    mix = np.broadcast_to(mix, p.shape[:-1]).astype(F32)
    return p.reshape(-1, 3), n.reshape(-1, 3), mix.reshape(-1)


def rose(opened):
    """Parametric rose: phi is how far it opens, edge is the wave of the petal rim."""
    x = np.linspace(0, 1, 80, dtype=F32)[:, None]
    th = np.linspace(-2 * np.pi, 15 * np.pi, 1100, dtype=F32)[None, :]
    phi = (np.pi / 2) * np.exp(-th / (8 * np.pi)) * (0.3 + 0.52 * opened)
    edge = 1 - 0.5 * (1.25 * (1 - (3.6 * th) % (2 * np.pi) / np.pi) ** 2 - 0.25) ** 2
    y = 1.95653 * x ** 2 * (1.27689 * x - 1) ** 2 * np.sin(phi)
    r = edge * (x * np.sin(phi) + y * np.cos(phi))
    z = edge * (x * np.cos(phi) - y * np.sin(phi))
    return grid((r * np.sin(th), r * np.cos(th), z), x ** 2)


def petals(count, rows, nu, nv):
    """Shared layout of petals in a ring: row number, angle, u along, v across."""
    k = np.arange(count * rows, dtype=F32)[:, None, None]
    row = np.floor(k / count)
    beta = (k + row / rows) * (2 * np.pi / count)
    u = np.linspace(0, 1, nu, dtype=F32)[None, :, None]
    v = np.linspace(-1, 1, nv, dtype=F32)[None, None, :]
    return row, beta, u, v


def daisy(opened):
    row, beta, u, v = petals(13, 2, 40, 13)
    lift = np.radians(80 - 68 * opened - 9 * row)
    r = (0.16 + 0.84 * u) * (0.88 + 0.12 * row)
    side = v * 0.2 * np.abs(np.sin(np.pi * u ** 0.7)) ** 0.6
    out = r * np.cos(lift)
    z = r * np.sin(lift) + 0.08 * v ** 2 - 0.15 * u ** 2 * opened
    p = grid((out * np.cos(beta) - side * np.sin(beta),
              out * np.sin(beta) + side * np.cos(beta), z), u)
    a = np.linspace(0, np.pi / 2, 14, dtype=F32)[:, None]
    b = np.linspace(0, 2 * np.pi, 44, dtype=F32)[None, :]
    core = grid((0.2 * np.sin(a) * np.cos(b), 0.2 * np.sin(a) * np.sin(b),
                 0.1 * np.cos(a) + 0.03), -1.0)
    return tuple(np.concatenate(pair) for pair in zip(p, core))


def tulip(opened):
    row, beta, u, v = petals(3, 2, 60, 30)
    lean = np.radians(6 + 34 * opened * (1 - 0.35 * row))
    bulge = 0.36 * np.sin(np.pi * u * 0.62) ** 0.8 * (1 - 0.12 * row)
    side = v * 0.36 * np.abs(np.sin(np.pi * u ** 0.8)) ** 0.7
    out = bulge * np.cos(lean) + u * np.sin(lean) - 0.35 * v * side
    z = u * np.cos(lean) - bulge * np.sin(lean)
    return grid((out * np.cos(beta) - side * np.sin(beta),
                 out * np.sin(beta) + side * np.cos(beta), z), u)


def ball(opened):
    a = np.linspace(0.02, np.pi - 0.02, 36, dtype=F32)[:, None]
    b = np.linspace(0, 2 * np.pi, 56, dtype=F32)[None, :]
    bump = np.abs(np.sin(5 * a) * np.sin(6 * b))  # tiny florets on the cap
    r = (1 + 0.16 * bump) * (0.35 + 0.65 * opened)
    flat = r * np.sin(a)
    return grid((flat * np.cos(b), flat * np.sin(b), r * np.cos(a)), bump)


SHAPES = {"rose": rose, "daisy": daisy, "tulip": tulip, "ball": ball}


@functools.lru_cache(maxsize=None)
def shape(kind, step):
    return SHAPES[kind](step / 20)


def vase(height, base, belly, pos, width, flare, ribs, *paint):
    """The vase is a solid of revolution; rows of VASES blend smoothly together."""
    v = np.linspace(0, 1, 200, dtype=F32)[:, None]
    a = np.linspace(0, 2 * np.pi, 300, dtype=F32)[None, :]
    r = base + belly * np.exp(-((v - pos) / width) ** 2)
    r += flare * np.clip((v - 0.9) / 0.1, 0, 1) ** 2
    r = 1.2 * (r - 0.04 * np.exp(-((v - 0.85) / 0.07) ** 2))
    pts, nrm, _ = grid((r * np.cos(a), r * np.sin(a), MOUTH[2] + height * (v - 1)), 0)
    band = ((np.abs(v - 0.585) < 0.035) * np.ones_like(a))[..., None]
    body = np.array(paint[:3], F32) * (1 - ribs + ribs * np.cos(12 * a))[..., None]
    rgb = np.where(band > 0, np.array(paint[3:], F32), body)
    return pts, nrm, rgb.reshape(-1, 3).astype(F32)


def leaf(azim, length):
    u = np.linspace(0, 1, 110, dtype=F32)[:, None]
    v = np.linspace(-1, 1, 44, dtype=F32)[None, :]
    side = v * 0.26 * length * np.abs(np.sin(np.pi * u)) ** 0.7 * (1 - 0.3 * u)
    out = 0.1 + u * length
    up = (0.75 * u - 0.6 * u ** 2) * length + 0.3 * np.abs(side)
    ca, sa = math.cos(azim), math.sin(azim)
    pts, nrm, _ = grid((out * ca - side * sa, out * sa + side * ca, up), 0)
    rgb = LEAF * (0.55 + 0.45 * np.abs(v) * np.ones_like(u))[..., None]
    return pts, nrm, rgb.reshape(-1, 3).astype(F32)


def basis(axis, spin):
    """Rotation matrix: the flower's local z axis points along axis."""
    z = axis / np.linalg.norm(axis)
    x = np.cross((1.0, 0, 0) if abs(z[2]) > 0.95 else (0, 0, 1.0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(spin), math.sin(spin)
    return np.stack([x * c + y * s, y * c - x * s, z], axis=1).astype(F32)


class Scene:
    """Shared by the 3D scenes: camera, light and a z-buffer from a point cloud."""

    def __init__(self, w, h):
        self.w, self.h, self.size = w, h, min(w, h)
        self.light = np.array((-0.5, -0.6, 0.62), F32) / 0.997
        half = self.light + np.array((0, -1, 0), F32)
        self.half = half / np.linalg.norm(half)

    @staticmethod
    def camera(a, e):
        """Rotation of the world into camera axes (right, depth, up): azimuth and tilt."""
        ca, sa, ce, se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
        return np.array([[ca, -sa, 0], [sa * ce, ca * ce, -se], [sa * se, ca * se, ce]],
                        F32)

    def shade(self, nrm, rgb, rot):
        """Two-sided Lambert light plus a highlight."""
        n = nrm @ rot.T
        n *= np.where(n[:, 1:2] > 0, -1.0, 1.0)  # the normal always faces the camera
        lit = np.clip(n @ self.light, 0, 1)
        shine = np.clip(n @ self.half, 0, 1) ** 24
        return rgb * (0.35 + 0.65 * lit)[:, None] + 120 * shine[:, None]

    def splat(self, cam, rgb, matte, zoom, low=0.0, back=None):
        """Perspective and a z-buffer of points; matte = 0 for those that glow."""
        k = self.size * zoom * 9.0 / (9.0 + cam[:, 1])
        xi = (self.w / 2 + cam[:, 0] * k).astype(np.int32)
        yi = (self.h / 2 + low - cam[:, 2] * k).astype(np.int32)
        ok = (xi >= 0) & (xi < self.w - 1) & (yi >= 0) & (yi < self.h - 1)
        depth = cam[ok, 1]
        order = np.argsort(-depth)  # far points first, near ones paint over them
        flat = (yi[ok] * self.w + xi[ok])[order]
        flat = np.stack([flat, flat + 1, flat + self.w, flat + self.w + 1], 1).ravel()
        buf = np.zeros((self.h * self.w, 4), F32)
        if back is not None:  # backdrop: the glow seen behind every point
            buf[:, :3] = back.reshape(-1, 3)
        zbuf = np.full(self.h * self.w, depth.max(), F32)
        buf[flat] = np.repeat(np.column_stack([rgb, matte])[ok][order], 4, axis=0)
        zbuf[flat] = np.repeat(depth[order], 4)

        # relief: deeper than the neighbours is darker, sticking out is lighter
        z = Image.fromarray(zbuf.reshape(self.h, self.w))
        soft = z.resize((self.w // 6, self.h // 6), Image.BOX)
        soft = soft.resize(z.size, Image.BILINEAR)
        relief = np.clip((np.asarray(soft) - np.asarray(z)) * 5.0, -0.75, 0.6)
        buf = buf.reshape(self.h, self.w, 4)
        buf = buf[..., :3] * (1 + relief * buf[..., 3])[..., None]
        img = Image.fromarray(np.clip(buf, 0, 255).astype(np.uint8))
        return np.asarray(img.filter(ImageFilter.GaussianBlur(0.6)))


class Bouquet(Scene):
    """A bouquet of point clouds: flowers grow, bloom and are reborn, the vase morphs."""

    def __init__(self, w, h, seed=5):
        super().__init__(w, h)
        rng = random.Random(seed)

        # the first flower stands in the vase from the start, the rest grow one by one
        spots = [("flower", math.acos(1 - 1.1 * i / 10), i * 2.4, 1.15,
                  1.6 + 0.55 * i if i else -1.4) for i in range(11)]
        spots += [("ball", math.radians(30 + 9 * j), 1.1 + j * 2.4, 1.0, 3.5 + 0.6 * j)
                  for j in range(6)]
        self.heads = []
        for kind, polar, az, radius, born in spots:
            flat = radius * math.sin(polar)
            spot = (flat * math.cos(az), flat * math.sin(az),
                    radius * math.cos(polar) - 0.25)
            self.heads.append(dict(
                kind=kind, born=born, spot=spot, scale=rng.uniform(0.92, 1.08),
                life=rng.uniform(2.6, 3.8), spin=rng.uniform(0, 6.28),
                phase=rng.uniform(0, 6.28)))
        self.born = np.array([f["born"] for f in self.heads], F32)
        self.spot = np.array([f["spot"] for f in self.heads], F32)
        self.phase = np.array([f["phase"] for f in self.heads], F32)[:, None]
        jitter = [(rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), 0)
                  for _ in self.heads]
        self.root = MOUTH + np.array(jitter, F32)
        self.leaves = [(leaf(0.3 + i * 0.7, rng.uniform(1.2, 1.5)), 1.0 + 0.4 * i)
                       for i in range(9)]
        self.drops = np.array([[rng.random() for _ in range(3)] for _ in range(1400)], F32)

    def stems(self, t):
        """Stems are Bezier curves from the vase to the flower; they grow with grow."""
        grow = np.array([ease((t - b) / 1.6) for b in self.born], F32)
        sway = np.concatenate([np.sin(1.3 * t + self.phase), np.cos(1.1 * t + self.phase),
                               np.zeros_like(self.phase)], 1)
        end = self.spot + 0.03 * sway
        rise = self.root[:, 2] + (end[:, 2] - self.root[:, 2]) * 0.75
        mid = np.column_stack([end[:, :2] * 0.25, rise])
        g = grow[:, None]
        u = np.linspace(0, 1, 150, dtype=F32)[None, :, None] * g[:, None]
        line = (1 - u) ** 2 * self.root[:, None] + 2 * u * (1 - u) * mid[:, None]
        line = line + u ** 2 * end[:, None]
        tangent = 2 * (1 - g) * (mid - self.root) + 2 * g * (end - mid)
        a = np.linspace(0, 2 * np.pi, 6, endpoint=False, dtype=F32)
        ring = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
        tube = line[grow > 0][:, :, None, :] + 0.026 * ring
        ring = np.broadcast_to(ring, tube.shape).reshape(-1, 3)
        return grow, line[:, -1], tangent, tube.reshape(-1, 3), ring

    def water(self, t):
        """A stream of water from above: glowing drops, dry by the eighth second."""
        fall = (self.drops[:, 0] + t * 0.8) % 1.0
        wobble = np.stack([np.sin(fall * 40 + t * 3), np.cos(fall * 33 + t * 2)], 1)
        pts = np.column_stack([0.05 * wobble * self.drops[:, 1:2], 4.4 - fall * 5.45])
        fade = (1 - ease((t - 6.8) / 1.2)) * (0.6 + 0.4 * self.drops[:, 2:3])
        return pts.astype(F32), np.array((45, 80, 150), F32) * fade

    def frame(self, t):
        grow, tips, tangent, tube, tube_n = self.stems(t)
        rot = self.camera(0.3 + 0.15 * t, 0.3)  # slow fly-around
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        facing = tangent - 1.2 * rot[1]  # the flower faces the stem line and the viewer
        step, part = divmod(t / 1.4, 1)  # the vase keeps flowing into the next one
        old, new = VASES[int(step) % len(VASES)], VASES[int(step + 1) % len(VASES)]
        pot = vase(*(old + (new - old) * ease((part - 0.4) / 0.6)))
        pts, nrm = [tube, pot[0]], [tube_n, pot[1]]
        rgb = [np.broadcast_to(LEAF * 0.8, tube.shape), pot[2]]
        for (p, n, c), born in self.leaves:
            if (g := ease((t - born) / 2.0)) > 0:
                pts.append(MOUTH + p * g), nrm.append(n), rgb.append(c)
        for i, f in enumerate(self.heads):
            if grow[i] <= 0:
                continue
            # a flower lives a few seconds, closes and opens again as another kind
            cycle, age = divmod(t - f["born"], f["life"])
            left, cycle = f["life"] - age, int(cycle)
            kind = KINDS[(i + cycle) % 5] if f["kind"] == "flower" else "ball"
            deep, tip, *core = COLORS[kind][(i + 2 * cycle) % len(COLORS[kind])]
            deep, tip = np.array(deep, F32), np.array(tip, F32)
            opened = ease((age - 0.15) / 1.0) * ease(left / 0.5)
            size = 0.2 + 0.8 * ease(age / 0.5) * ease(left / 0.4)
            size *= SIZE[kind] * f["scale"] * (0.3 + 0.7 * grow[i])
            p, n, mix = shape(kind, round(opened * 20))
            turn = basis(facing[i], f["spin"])
            pts.append(p @ turn.T * size + tips[i])
            nrm.append(n @ turn.T)
            mix = mix[:, None]
            petal = (deep + (tip - deep) * mix) * (0.4 + 0.6 * mix)
            rgb.append(np.where(mix < 0, np.array(core[0] if core else deep, F32), petal))
        pts, nrm, rgb = np.concatenate(pts), np.concatenate(nrm), np.concatenate(rgb)
        drops, glow = self.water(t)  # water glows by itself, light does not affect it
        cam = (np.concatenate([pts, drops]) - np.array((0, 0, -0.4), F32)) @ rot.T
        rgb = np.concatenate([self.shade(nrm, rgb, rot), glow])
        zoom = 0.27 + 0.06 * (1 - ease(t / 8))  # close-up first, then pull back
        low = max(0, self.h - self.w * 1.1) * 0.3  # a tall frame puts the bouquet lower
        return self.splat(cam, rgb, np.arange(len(cam)) < len(pts), zoom, low)


TILE = np.array((14, 24, 86), F32)  # dark faces of the tiles
HEART = np.array([(40, 130, 255), (200, 240, 255)], F32)  # light inside: gap edge, middle
GLOW = np.array((25, 80, 230), F32)  # halo behind the core
RIM = np.array((190, 180, 255), F32)  # tile outlines and pyramid ridges
RUNE = np.array((110, 230, 255), F32)  # runes and lightning


def unit(v):
    v = np.asarray(v, F32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cells(p):
    """Cell pattern 0..1: dark veins with light cells between them."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.abs(np.sin(5 * x + 1) * np.sin(5 * y) * np.sin(5 * z + 2))[:, None] ** 0.4


# runes: polylines split by "|", each point is a pair of digits x, y on a 9 by 9 grid;
# the top looks at the tile edge. Every pyramid carries all four, one rune per face
RUNES = ("48 64 40 24 48",  # diamond
         "17 13 31 43 51 73 77|43 47",  # trident
         "14 26 46 84 42 22 14|46 58|42 50|26 17|22 11",  # leaf with thorns
         "15 48 75|26 20|46 40|66 60")  # arrow of three stripes


def rune(a, b, text):
    """How far the points of a face (a, b from -1 to 1) are from the rune lines."""
    best = np.full(a.shape, 9.0, F32)
    for line in text.split("|"):
        dots = [((int(p[0]) - 4) / 6, (int(p[1]) - 4) / 6) for p in line.split()]
        for (x, y), (p, q) in zip(dots, dots[1:]):
            p, q = p - x, q - y
            along = np.clip(((a - x) * p + (b - y) * q) / (p * p + q * q), 0, 1)
            best = np.minimum(best, np.hypot(a - x - along * p, b - y - along * q))
    return best


def jag(rng, halvings, size, dots):
    """Broken line that starts and ends at zero: each middle is thrown aside, ever less."""
    off = np.zeros((2, 3), F32)
    for _ in range(halvings):
        mid = (off[:-1] + off[1:]) / 2 + rng.normal(0, size, (len(off) - 1, 3))
        off = np.insert(off, np.arange(1, len(off)), mid, 0)
        size *= 0.6
    s = np.linspace(0, len(off) - 1, dots)  # straight runs between the sharp kinks
    return np.stack([np.interp(s, np.arange(len(off)), row) for row in off.T], 1)


class Hexcore(Scene):
    """The Hexcore: a polyhedron of rune pyramids with light in the gaps and lightning."""

    LOOP = 17.0  # seconds per full turn of the camera
    ZOOM = 0.31
    CRACK = 0.045  # half-width of the gap between tiles
    ROUND = 0.08  # corner radius of the triangular tiles
    STRIKES = 160  # lightning strikes per loop

    def __init__(self, w, h, seed=7):
        super().__init__(w, h)
        rng = np.random.default_rng(seed)

        n = max(int(self.size * 0.68), 160)  # shell grid: latitude and longitude
        a = np.linspace(0.01, np.pi - 0.01, n, dtype=F32)[:, None]
        b = np.linspace(0, 2 * np.pi, 2 * n, dtype=F32)[None, :]
        self.x, self.y = np.sin(a) * np.cos(b), np.sin(a) * np.sin(b)
        self.z = np.cos(a) + 0 * b
        self.dirs = dirs = np.stack([self.x, self.y, self.z], -1).reshape(-1, 3)

        # the tiles are faces of a rhombicuboctahedron: 18 squares and 8 triangles
        axes = np.array(list(np.ndindex(3, 3, 3)), F32) - 1
        axes = axes[np.abs(axes).sum(1) > 0]
        tri = np.abs(axes).sum(1) == 3
        self.plane, self.depth = unit(axes), np.where(tri, 1.056, 1.0).astype(F32)
        pull = self.plane / self.depth[:, None]
        gram = pull @ pull.T
        reach = 1 / np.maximum(dirs @ pull.T, 0.05)  # distance to every face
        row, tile = np.arange(len(dirs)), reach.argmin(1)  # the nearest face is the tile
        near = reach[row, tile][:, None]
        slope = near ** 2 * gram[tile, tile][:, None] + reach ** 2 * gram.diagonal()
        slope = np.sqrt(np.maximum(slope - 2 * near * reach * gram[tile], 1e-9))
        seam = np.log(reach / near) / slope  # how far the joint with each neighbour is
        seam[reach > 4 * near] = 9  # faces far away do not count
        seam[row, tile] = 9
        seam -= self.CRACK  # below zero is the gap between tiles
        mid, tri, edge = self.plane[tile], tri[tile], seam.min(1)

        # triangles are flat plates with rounded corners
        gone = np.sqrt((np.maximum(self.ROUND - seam, 0) ** 2).sum(1))
        edge = np.where(tri & (edge < self.ROUND), self.ROUND - gone, edge)

        # squares: corners are cut off, inside the frame sits a pyramid with a cut top;
        # a, b are the coordinates of a point on its own tile, from -1 to 1
        u = unit(np.cross(mid, np.eye(3, dtype=F32)[np.abs(mid).argmin(1)]))
        flat = dirs * near / 0.414  # 0.414 is half the side of a square
        a, b = np.sum(flat * u, 1), np.sum(flat * np.cross(mid, u), 1)
        cut = (1.52 - np.abs(a) - np.abs(b)) * 0.29
        self.tile, seam = tile, np.where(tri, edge, np.minimum(edge, cut))
        lean = np.abs(a) > np.abs(b)  # which pair of the square's sides is closer
        out, along = np.where(lean, a, b), np.where(lean, b, a)
        face = 2 * (out < 0) + ~lean  # number of the pyramid face, going around
        out = np.abs(out)  # 1 at the tile edge, 0 at the top of the pyramid
        self.point = np.where(tri, 0, 0.34 * np.clip((0.8 - out) / 0.67, 0, 1)).astype(F32)

        # pyramid markings: a rune on every face, ridges, the foot line and a glowing top
        ink = np.full(len(dirs), 9.0, F32)
        for k, text in enumerate(RUNES):
            mine = ~tri & ((face + tile) % len(RUNES) == k)
            ink[mine] = rune(along[mine] / 0.44, (out[mine] - 0.57) / 0.44, text)
        ink, top = (ink < 0.1)[:, None], (out < 0.13)[:, None]
        trim = (np.abs(np.abs(a) - np.abs(b)) < 0.035) | (np.abs(out - 0.8) < 0.02)
        self.ink = np.where(top, 255, np.where(ink, RUNE, RIM * 0.6)).astype(F32)
        self.sign = (top | ink | trim[:, None]) & (~tri & (seam > 0.02))[:, None]
        self.keep, self.rim = seam > 0, (seam < 0.014)[:, None]

        # the gaps are flooded with light from inside, brighter towards the middle
        self.film = cells(dirs)
        gap = ~self.keep
        mix = np.clip(0.4 * self.film[gap] + 0.1 - seam[gap, None] / 0.15, 0, 1)
        self.glare = (HEART[0] + (HEART[1] - HEART[0]) * mix).astype(F32)

        self.sway = rng.normal(0, 0.09, (2, 26, 3)).astype(F32)  # where each tile tilts
        self.beat = rng.integers(2, 7, (4, 26))  # whole number of swings per loop
        self.lag = rng.uniform(0, 2 * np.pi, (4, 26))

        self.born = rng.uniform(0, self.LOOP, self.STRIKES)  # when each strike flashes
        self.life = rng.uniform(0.12, 0.5, self.STRIKES)  # and how long it lasts

        y, x = np.mgrid[:h, :w]  # backdrop: a narrow halo right around the core
        far = np.hypot(x - w / 2, y - h / 2) / (self.size * self.ZOOM)
        self.halo = (0.22 * GLOW * np.clip((1.62 - far) / 0.32, 0, 1)[..., None] ** 2)

    def bolts(self, t):
        """Lightning: short strikes in random places, a broken channel with forks."""
        arcs, paint = [np.zeros((0, 3), F32)], [np.zeros((0, 3), F32)]
        age = (t - self.born) % self.LOOP / self.life
        for k in np.flatnonzero(age < 1):
            rng = np.random.default_rng(k)  # a strike keeps its channel while it lives
            now = np.random.default_rng((k, round(t * 30)))  # and flickers every frame
            a = unit(rng.normal(size=3))  # it jumps from one spot of the shell to another
            b = unit(a + rng.normal(0, rng.uniform(0.5, 1.1), 3))
            s = np.linspace(0, 1, 257, dtype=F32)[:, None]
            rise = 1.12 + rng.uniform(0.25, 0.4) * (4 * s * (1 - s)) ** 0.6  # arcs over
            path = unit(a + (b - a) * s) * (rise + rng.choice((0, 0, 0.3)) * s)
            path = path + jag(rng, 5, 0.13, len(s))
            lines = [(path, 14, 0.012, 0.9)]
            for i in rng.integers(30, 200, rng.integers(1, 4)):  # thin forks fly outwards
                tip = unit(path[i + 40] - path[i]) + 0.6 * unit(path[i])
                tip = unit(tip + rng.normal(0, 0.5, 3)) * rng.uniform(0.25, 0.6)
                fork = path[i] + tip * s[::4] + jag(rng, 4, 0.06, 65) * s[::4] ** 0.3
                lines.append((fork, 4, 0.005, 0.35))
            fade = min(1, 2.5 * (1 - age[k]))  # a white flash that crumbles into sparks
            for line, copies, blur, white in lines:
                line = np.repeat(line, copies, 0)[now.random(len(line) * copies) < fade]
                arcs.append(line + now.normal(0, blur, line.shape))
                ink = (RUNE + (255 - RUNE) * white * fade) * now.uniform(0.8, 1)
                paint.append(np.broadcast_to(ink, line.shape))
        return np.concatenate(arcs).astype(F32), np.concatenate(paint).astype(F32)

    def frame(self, t):
        t %= self.LOOP
        turn = 2 * math.pi * t / self.LOOP
        rot = self.camera(0.5 + turn, 0.35 + 0.2 * math.sin(2 * turn))

        # the tiles are alive: they tilt, slide in and out, and now and then pop outwards
        swing = np.sin(self.beat * turn + self.lag).astype(F32)
        plane = unit(self.plane + (self.sway * swing[:2, :, None]).sum(0))[self.tile]
        depth = self.depth * (1 + 0.06 * swing[2] + 0.12 * np.maximum(swing[3], 0) ** 4)
        r = depth[self.tile] / np.maximum(np.sum(self.dirs * plane, 1), 0.3)
        r = r * (1 + self.point) * (1 + 0.02 * math.sin(7 * turn))  # pyramids on top
        r = r.reshape(self.z.shape)
        pts, nrm, _ = grid((self.x * r, self.y * r, self.z * r), 0)

        rgb = np.where(self.rim, RIM, self.shade(nrm, TILE, rot))
        rgb = np.where(self.sign, self.ink, rgb)
        # the inner side of the far tiles glows like the light in the gaps
        back = ((nrm @ rot[1]) > 0)[:, None] & ~self.rim
        rgb = np.where(back, (HEART[0] + (HEART[1] - HEART[0]) * self.film) * 0.9, rgb)
        matte = ~(self.rim | back | self.sign)[:, 0]

        arcs, flash = self.bolts(t)
        cloud = [pts[self.keep], pts[~self.keep] * 0.9, arcs]
        paint = [rgb[self.keep], self.glare, flash]
        cam = np.concatenate(cloud) @ rot.T
        lit = np.zeros(len(cam), F32)  # relief shading only on the shell itself
        lit[:self.keep.sum()] = matte[self.keep]
        img = self.splat(cam, np.concatenate(paint), lit, self.ZOOM, 0, self.halo)
        soft = Image.fromarray(img).filter(ImageFilter.GaussianBlur(self.size / 90))
        return np.clip(img + 0.2 * np.asarray(soft, F32), 0, 255).astype(np.uint8)


SCENES = {"hexcore": Hexcore, "bouquet": Bouquet}


def scene(kind, w, h, fps, seconds=None):
    world, i = SCENES[kind](w, h), 0
    while seconds is None or i < seconds * fps:
        yield world.frame(i / fps)
        i += 1


def video(path, w, h, fps, seconds=None, loop=False):
    """Frames from a file via ffmpeg, fitted into w x h with black bars."""
    vf = (f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
    cmd = [FFMPEG, "-v", "error"] + (["-stream_loop", "-1"] if loop else [])
    cmd += ["-i", path] + (["-t", str(seconds)] if seconds else [])
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        while len(buf := proc.stdout.read(w * h * 3)) == w * h * 3:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.kill()
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------- frame -> glyphs

class Renderer:
    def __init__(self, cols, rows, ramp=RAMP, color=True, edges=True, saturation=1.5,
                 pastel=0.0):
        self.cols, self.rows, self.pastel = cols, rows, pastel
        self.ramp, self.chars = ramp, ramp + EDGES
        self.color, self.edges, self.saturation = color, edges, saturation
        self.atlas = None

    def convert(self, frame):
        """RGB frame -> (glyph indices, colour of every glyph)."""
        small = Image.fromarray(frame).resize((self.cols, self.rows), Image.BOX)
        self.small = small if self.color else small.convert("L").convert("RGB")
        rgb = np.asarray(small, np.float32) / 255.0
        lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        n = len(self.ramp)
        dense = 0.6 * lum + 0.4 * rgb.max(axis=2)  # or else red and blue come out sparse
        glyph = np.clip(dense ** 0.9 * (n - 1) + 0.5, 0, n - 1).astype(np.int32)
        glyph[lum < 0.035] = 0

        if self.edges:
            p = np.pad(lum, 1, mode="edge")
            gx = p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
            gx -= p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]
            gy = p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
            gy -= p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
            ang = np.degrees(np.arctan2(gy, gx)) % 180.0
            line = n + ((ang + 22.5) // 45).astype(np.int32) % 4
            mask = (np.hypot(gx, gy) > 2.2) & (lum > 0.06)
            glyph = np.where(mask, line, glyph)

        if not self.color:
            return glyph, np.full(rgb.shape, 255, np.uint8)
        rgb = lum[..., None] + (rgb - lum[..., None]) * self.saturation
        peak = np.maximum(rgb.max(axis=2, keepdims=True), 1e-4)
        rgb = rgb * (peak ** 0.45 / peak)  # lift the dark: glyphs carry the brightness
        rgb += (rgb.max(axis=2, keepdims=True) - rgb) * self.pastel  # closer to white
        return glyph, (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def draw(self, glyph, fg):
        """A string with ANSI colours for the terminal (colour codes only on change)."""
        q = (fg >> 3 << 3).astype(np.int32)
        key = (q[..., 0] << 16 | q[..., 1] << 8 | q[..., 2]).tolist()
        chars, rows = self.chars, []
        for g_row, k_row in zip(glyph.tolist(), key):
            out, last = [], -1
            for g, k in zip(g_row, k_row):
                if g and k != last:
                    last = k
                    out.append(f"\x1b[38;2;{k >> 16};{k >> 8 & 255};{k & 255}m")
                out.append(chars[g])
            rows.append("".join(out))
        return "\x1b[H" + "\n".join(rows) + "\x1b[0m"

    def raster(self, glyph, fg, cw, ch, glow=0.5, under=0.45):
        """The same, but as a picture - for writing to mp4."""
        if self.atlas is None:
            font = load_font(int(ch * 0.9))
            cells = []
            for c in self.chars:
                cell = Image.new("L", (cw, ch))
                pen = ImageDraw.Draw(cell)
                pen.text((cw / 2, ch / 2), c, font=font, fill=255, anchor="mm")
                cells.append(np.asarray(cell, np.float32) / 255.0)
            self.atlas = np.stack(cells)
        px = self.atlas[glyph][..., None] * fg[:, :, None, None, :]
        px = px.transpose(0, 2, 1, 3, 4).reshape(self.rows * ch, self.cols * cw, 3)
        halo = Image.fromarray(px.astype(np.uint8))
        halo = halo.filter(ImageFilter.GaussianBlur(cw * 0.9))
        soft = self.small.resize(halo.size, Image.BICUBIC)  # blurred frame under glyphs
        soft = np.asarray(soft.filter(ImageFilter.GaussianBlur(cw * 1.4)), np.float32)
        soft = np.clip(soft * 1.9 - soft.mean(axis=2, keepdims=True) * 0.9, 0, 255)
        px += np.asarray(halo, np.float32) * glow + soft * under  # a rich dim glow
        return np.clip(px, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- code panel

BG, BAR, FG, GUTTER = (40, 44, 52), (33, 37, 43), (171, 178, 191), (84, 92, 110)
SIDE = (13, 15, 20)  # margins on both sides of the glyph column
KEYWORD, STRING, NUMBER = (198, 120, 221), (152, 195, 121), (209, 154, 102)
COMMENT, FUNC = (92, 99, 112), (97, 175, 239)
BUILTIN, SELF = (86, 182, 194), (224, 108, 117)


def highlight(src, lines):
    colors = [[FG] * len(line) for line in lines]
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, tok in enumerate(toks):
        kind, s, c = tokenize.tok_name[tok.type], tok.string, None
        if kind == "COMMENT":
            c = COMMENT
        elif kind == "STRING" or kind.startswith(("FSTRING", "TSTRING")):
            c = STRING
        elif kind == "NUMBER":
            c = NUMBER
        elif kind == "NAME":
            if keyword.iskeyword(s):
                c = KEYWORD
            elif s == "self":
                c = SELF
            elif toks[i - 1].string in ("def", "class") or toks[i + 1].string == "(":
                c = BUILTIN if hasattr(builtins, s) else FUNC
        if c:
            (r0, c0), (r1, c1) = tok.start, tok.end
            for r in range(r0, r1 + 1):
                a, b = (c0 if r == r0 else 0), (c1 if r == r1 else len(lines[r - 1]))
                colors[r - 1][a:b] = [c] * (b - a)
    return colors


def scopes(lines):
    """For every line - which function it is in (for the window title)."""
    stack, out = [], []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if m := re.match(r"\s*(def|class)\s+(\w+)", line):
                stack.append((indent, m[2] + ("()" if m[1] == "def" else "")))
        out.append(".".join(name for _, name in stack) or "main")
    return out


class CodePanel:
    """Editor window: this very file scrolls by, the current line is highlighted."""

    def __init__(self, w, h, path, fps, speed=6.0, anchor="class Renderer", size=20,
                 bar=46):
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        lines = src.splitlines()
        self.w, self.h, self.bar, self.fps, self.speed = w, h, bar, fps, speed
        self.name, self.scope = os.path.basename(path), scopes(lines)
        self.font, self.lh = load_font(size), int(size * 1.4)
        first = (i for i, s in enumerate(lines) if s.startswith(anchor))
        self.start = next(first, 0)
        step = self.font.getlength("M")
        sheet = Image.new("RGB", (w, self.lh * len(lines)), BG)
        d = ImageDraw.Draw(sheet)
        for i, (line, cols) in enumerate(zip(lines, highlight(src, lines))):
            y = i * self.lh + self.lh / 2
            d.text((step * 4.5, y), str(i + 1), font=self.font, fill=GUTTER, anchor="rm")
            a = 0
            while a < len(line):
                b = a
                while b < len(line) and cols[b] == cols[a]:
                    b += 1
                x = step * (6 + a)
                d.text((x, y), line[a:b], font=self.font, fill=cols[a], anchor="lm")
                a = b
        self.sheet, self.titles = np.asarray(sheet), {}

    def title(self, scope):
        if scope not in self.titles:
            img = Image.new("RGB", (self.w, self.bar), BAR)
            d = ImageDraw.Draw(img)
            for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
                x = 28 + i * 26
                d.ellipse([x - 8, self.bar / 2 - 8, x + 8, self.bar / 2 + 8], fill=c)
            d.text((116, self.bar / 2), f"{self.name}  -  {scope}", font=self.font,
                   fill=FG, anchor="lm")
            self.titles[scope] = np.asarray(img)
        return self.titles[scope]

    def frame(self, i):
        total = len(self.scope)
        cursor = self.start + (i * self.speed / self.fps) % (total - self.start)
        view_h = self.h - self.bar
        top = max(cursor * self.lh - view_h * 0.4, 0)
        top = int(min(top, self.sheet.shape[0] - view_h))
        out = np.empty((self.h, self.w, 3), np.uint8)
        out[: self.bar] = self.title(self.scope[int(cursor)])
        out[self.bar:] = self.sheet[top: top + view_h]
        y = self.bar + int(cursor) * self.lh - top
        band = out[max(y, self.bar): max(y + self.lh, self.bar)]
        band[:] = np.minimum(band.astype(np.int32) + 22, 255)
        return out


# ---------------------------------------------------------------- output

def play(frames, renderer, fps):
    os.system("")  # turns on ANSI sequences in the Windows console
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    start = time.perf_counter()
    try:
        for i, frame in enumerate(frames):
            out.write(renderer.draw(*renderer.convert(frame)))
            out.flush()
            if (wait := start + (i + 1) / fps - time.perf_counter()) > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        out.flush()


def export(frames, renderer, path, fps, cw, ch, speed, anchor, size=(1080, 1920),
           panel_h=800):
    w, h = size
    top = h - panel_h
    x0 = (w - renderer.cols * cw) // 2
    panel = CodePanel(w, h - top, os.path.abspath(__file__), fps, speed, anchor)
    cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "21",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    canvas = np.full((h, w, 3), SIDE, np.uint8)
    canvas[:top, x0: x0 + renderer.cols * cw] = 0
    for i, frame in enumerate(frames):
        glyph, fg = renderer.convert(frame)
        art = renderer.raster(glyph, fg, cw, ch)
        canvas[:len(art), x0: x0 + renderer.cols * cw] = art
        canvas[top:] = panel.frame(i)
        ff.stdin.write(canvas.tobytes())
        if i % fps == 0:
            print(f"\r  frame {i}", end="", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"\r  done: {path} ({i + 1} frames)")


def main():
    ap = argparse.ArgumentParser(description="Video as coloured ASCII art")
    ap.add_argument("video", nargs="?", help="video file; without it - the built-in scene")
    ap.add_argument("--scene", choices=list(SCENES), default="hexcore",
                    help="built-in scene: Hexcore or bouquet")
    ap.add_argument("--export", metavar="OUT.mp4", help="write a vertical clip")
    ap.add_argument("--seconds", type=float, help="duration (a scene in a clip - 17)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cell", type=int, default=7, help="glyph width in the clip, px")
    ap.add_argument("--cols", type=int, help="width in glyphs for the terminal")
    ap.add_argument("--saturation", type=float, default=1.5)
    ap.add_argument("--code-speed", type=float, default=6.0, help="code lines per second")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    args = ap.parse_args()
    if not FFMPEG and (args.video or args.export):
        sys.exit("ffmpeg not found - it is needed to read and write video.")

    if args.export:
        cw, ch = args.cell, round(args.cell * 2.15)
        narrow = not args.video and args.scene == "bouquet"
        wide = 621 if narrow else 1080  # the bouquet gets a narrow middle column
        cols, rows = wide // cw, 1120 // ch
    else:
        cw, ch = 1, 2
        term = shutil.get_terminal_size((100, 40))
        cols, rows = args.cols or term.columns, term.lines - 1
    w, h = cols * 4, round(rows * 4 * ch / cw)  # a glyph is about twice as tall as wide
    seconds = args.seconds or (17 if args.export and not args.video else None)
    if args.video:
        frames = video(args.video, w, h, args.fps, seconds, loop=not args.export)
    else:
        frames = scene(args.scene, w, h, args.fps, seconds)

    renderer = Renderer(cols, rows, color=not args.no_color, edges=not args.no_edges,
                        saturation=args.saturation, pastel=0.3 if args.export else 0.0)
    if args.export:
        own = SCENES[args.scene].__name__ if not args.video else "Renderer"
        export(frames, renderer, args.export, args.fps, cw, ch, args.code_speed,
               f"class {own}")
    else:
        play(frames, renderer, args.fps)


if __name__ == "__main__":
    main()
