#!/usr/bin/env python3
"""
Cinematic orrery of potentially habitable worlds (PHL HWC).

v5 polish pass:
- 1920x1080 @ 30fps, 360 frames (12s loop)
- Multi-band painterly nebula + 8000-star background with twinkle
- Bloom halos on every star and planet (4-pass + 2-pass scatter stacks)
- Galaxy-Sim depth dimming based on stellar distance
- 24-frame trail with alpha falloff
- Real-sky inset (RA x Dec) showing the Kepler primary-field cluster
- No headline; pure-visual coda
"""

from __future__ import annotations

import csv
import html
import math
import random
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "hwc"
OUT_DIR = ROOT / "assets" / "hwc_orrery_v4_3"
OUT_V5 = ROOT / "assets" / "hwc_orrery_v5"
HWC_PAGE = DATA_DIR / "hwc_data_page.html"
HWC_CSV = DATA_DIR / "hwc_potentially_habitable.csv"
NASA_CSV = DATA_DIR / "nasa_pscomppars_hwc.csv"
HWC_URL = "https://phl.upr.edu/hwc/data"
NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


@dataclass
class Planet:
    name: str
    kind: str = ""
    method: str = ""
    mass: float | None = None
    radius: float | None = None
    flux: float | None = None
    tsurf: float | None = None
    period: float | None = None
    distance_ly: float | None = None
    age: float | None = None
    esi: float | None = None
    hostname: str | None = None
    ra: float | None = None
    dec: float | None = None
    dist_pc: float | None = None
    semi: float | None = None
    st_teff: float | None = None
    st_rad: float | None = None


@dataclass
class System:
    host: str
    planets: list[Planet] = field(default_factory=list)
    ra: float | None = None
    dec: float | None = None
    distance_ly: float | None = None
    x: float = 0.0
    y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    depth: float = 1.0  # depth dimming factor (Galaxy-Sim style)


def fetch_text(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000:
        return path.read_text(encoding="utf-8", errors="ignore")
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read().decode("utf-8", errors="ignore")
    path.write_text(data, encoding="utf-8")
    return data


def clean_number(value: str) -> float | None:
    value = html.unescape(value).strip()
    value = re.sub(r"^[~≥><=\s]+", "", value)
    value = value.replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_hwc_table(text: str) -> list[Planet]:
    text = html.unescape(text)
    first = text.index("data.addRows([")
    start = text.index("[", first)
    end = text.index("]);", start)
    block = text[start:end]
    raw_rows = re.findall(r"\[\{ v:.*?\}\](?:,|\n)", block, flags=re.S)
    planets: list[Planet] = []
    for raw in raw_rows:
        cells = re.findall(r'f:"(.*?)"', raw, flags=re.S)
        if len(cells) < 11:
            continue
        name_match = re.search(r">([^<>]+)</a>", cells[0])
        name = html.unescape(name_match.group(1) if name_match else cells[0]).strip()
        planets.append(
            Planet(
                name=name,
                kind=html.unescape(re.sub("<.*?>", "", cells[1])).strip(),
                method=html.unescape(re.sub("<.*?>", "", cells[2])).strip(),
                mass=clean_number(cells[3]),
                radius=clean_number(cells[4]),
                flux=clean_number(cells[5]),
                tsurf=clean_number(cells[6]),
                period=clean_number(cells[7]),
                distance_ly=clean_number(cells[8]),
                age=clean_number(cells[9]),
                esi=clean_number(cells[10]),
            )
        )
    return planets


def read_local_hwc_csv() -> list[Planet]:
    """Fallback: read the previously-cached PHL CSV (used when the live fetch
    fails or returns an unparseable page)."""
    planets: list[Planet] = []
    with HWC_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            planets.append(
                Planet(
                    name=row["name"],
                    kind=row.get("type", ""),
                    method=row.get("method", ""),
                    mass=clean_number(row.get("mass_earth", "") or ""),
                    radius=clean_number(row.get("radius_earth", "") or ""),
                    flux=clean_number(row.get("flux_earth", "") or ""),
                    tsurf=clean_number(row.get("tsurf_k", "") or ""),
                    period=clean_number(row.get("period_days", "") or ""),
                    distance_ly=clean_number(row.get("distance_ly", "") or ""),
                    age=clean_number(row.get("age_gyr", "") or ""),
                    esi=clean_number(row.get("esi", "") or ""),
                )
            )
    return planets


def write_hwc_csv(planets: list[Planet]) -> None:
    HWC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with HWC_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "type",
                "method",
                "mass_earth",
                "radius_earth",
                "flux_earth",
                "tsurf_k",
                "period_days",
                "distance_ly",
                "age_gyr",
                "esi",
            ]
        )
        for p in planets:
            writer.writerow(
                [
                    p.name,
                    p.kind,
                    p.method,
                    p.mass,
                    p.radius,
                    p.flux,
                    p.tsurf,
                    p.period,
                    p.distance_ly,
                    p.age,
                    p.esi,
                ]
            )


def fetch_nasa_rows(names: list[str]) -> dict[str, dict[str, str]]:
    if NASA_CSV.exists() and NASA_CSV.stat().st_size > 200:
        return read_nasa_csv()
    quoted = ", ".join("'" + name.replace("'", "''") + "'" for name in names)
    query = (
        "select pl_name,hostname,ra,dec,sy_dist,pl_orbper,pl_orbsmax,"
        "pl_rade,pl_bmasse,st_teff,st_rad from pscomppars "
        f"where pl_name in ({quoted})"
    )
    url = NASA_TAP + "?" + urllib.parse.urlencode({"query": query, "format": "csv"})
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read().decode("utf-8", errors="ignore")
    NASA_CSV.write_text(data, encoding="utf-8")
    return read_nasa_csv()


def read_nasa_csv() -> dict[str, dict[str, str]]:
    with NASA_CSV.open(newline="", encoding="utf-8") as handle:
        return {row["pl_name"]: row for row in csv.DictReader(handle)}


def merge_nasa(planets: list[Planet], nasa: dict[str, dict[str, str]]) -> None:
    for p in planets:
        row = nasa.get(p.name)
        if not row:
            p.hostname = p.name.rsplit(" ", 1)[0]
            continue
        p.hostname = row.get("hostname") or p.name.rsplit(" ", 1)[0]
        p.ra = clean_number(row.get("ra", ""))
        p.dec = clean_number(row.get("dec", ""))
        p.dist_pc = clean_number(row.get("sy_dist", ""))
        p.period = p.period or clean_number(row.get("pl_orbper", ""))
        p.semi = clean_number(row.get("pl_orbsmax", ""))
        p.radius = p.radius or clean_number(row.get("pl_rade", ""))
        p.mass = p.mass or clean_number(row.get("pl_bmasse", ""))
        p.st_teff = clean_number(row.get("st_teff", ""))
        p.st_rad = clean_number(row.get("st_rad", ""))


def sky_project(ra: float | None, dec: float | None, rng: random.Random) -> tuple[float, float]:
    if ra is None or dec is None:
        return rng.uniform(-5.5, 5.5), rng.uniform(-3.0, 3.0)
    x = ((ra + 180) % 360 - 180) / 180 * 5.7
    y = dec / 90 * 3.1
    return x, y


def build_systems(planets: list[Planet]) -> list[System]:
    groups: dict[str, list[Planet]] = defaultdict(list)
    for p in planets:
        groups[p.hostname or p.name.rsplit(" ", 1)[0]].append(p)
    rng = random.Random(42)
    systems = []
    for host, ps in groups.items():
        ra = next((p.ra for p in ps if p.ra is not None), None)
        dec = next((p.dec for p in ps if p.dec is not None), None)
        dist_pc = next((p.dist_pc for p in ps if p.dist_pc is not None), None)
        fallback_ly = next((p.distance_ly for p in ps if p.distance_ly is not None), None)
        x, y = sky_project(ra, dec, rng)
        dist_ly = (dist_pc * 3.26156 if dist_pc else fallback_ly)
        depth = 1.0 / (1.0 + 0.0008 * (dist_ly or 0.0))
        depth = max(0.55, min(1.0, depth))
        s = System(
            host=host,
            planets=ps,
            ra=ra,
            dec=dec,
            distance_ly=dist_ly,
            x=x,
            y=y,
            target_x=x,
            target_y=y,
            depth=depth,
        )
        systems.append(s)
    relax_systems(systems)
    return sorted(systems, key=lambda s: (-(max((p.esi or 0) for p in s.planets)), s.host))


def relax_systems(systems: list[System]) -> None:
    min_dist = 0.46
    for _ in range(260):
        for s in systems:
            s.x += (s.target_x - s.x) * 0.012
            s.y += (s.target_y - s.y) * 0.012
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.hypot(dx, dy) or 1e-6
                wanted = min_dist + 0.035 * (len(a.planets) + len(b.planets))
                if dist < wanted:
                    push = (wanted - dist) * 0.5
                    ux, uy = dx / dist, dy / dist
                    a.x -= ux * push
                    a.y -= uy * push
                    b.x += ux * push
                    b.y += uy * push
        for s in systems:
            s.x = max(-6.0, min(6.0, s.x))
            s.y = max(-3.35, min(3.35, s.y))


def star_color(teff: float | None) -> tuple[float, float, float]:
    if teff is None:
        return (1.00, 0.902, 0.651)
    if teff < 3200:
        return (1.00, 0.55, 0.42)
    if teff < 3700:
        return (1.00, 0.69, 0.49)
    if teff < 5200:
        return (1.00, 0.83, 0.56)
    if teff < 6200:
        return (1.00, 0.95, 0.78)
    return (0.81, 0.91, 1.00)


def planet_color(p: Planet) -> tuple[float, float, float]:
    # Galaxy-Sim-derived warm/cool palette, 4 categories.
    if p.flux is not None and p.flux > 1.35:
        return (1.00, 0.83, 0.42)  # hot side - warm gold
    if p.flux is not None and p.flux < 0.45:
        return (0.49, 0.91, 1.00)  # cold side - icy cyan
    if "Terran" in p.kind:
        return (0.50, 0.94, 0.74)  # Earth-like - mint
    return (0.80, 0.72, 1.00)      # Superterran/uncertain - lilac


def estimate_semimajor_axis(p: Planet) -> float:
    if p.semi and p.semi > 0:
        return p.semi
    if not p.period:
        return 0.12
    teff = p.st_teff or 3600
    mass = 0.15 if teff < 3200 else 0.35 if teff < 3900 else 0.7 if teff < 5200 else 1.0
    return (mass * (p.period / 365.25) ** 2) ** (1 / 3)


def make_background(rng: np.random.Generator, n: int = 8000) -> dict[str, np.ndarray]:
    x = rng.uniform(-6.6, 6.6, n)
    y = rng.uniform(-3.7, 3.7, n)
    # Power-law size: most are tiny, a few are bright pinpoints.
    size = (rng.power(2.4, n) * 2.6 + 0.3) ** 1.45
    base_alpha = rng.uniform(0.04, 0.55, n)
    twinkle_phase = rng.uniform(0.0, 2 * math.pi, n)
    twinkle_speed = rng.uniform(0.018, 0.06, n)
    twinkle_amp = rng.uniform(0.05, 0.22, n)
    # Faint chromatic shimmer: most stars are white, ~12% subtly tinted.
    base_color = np.full((n, 3), 0.96)
    tint_idx = rng.choice(n, size=int(n * 0.12), replace=False)
    tints = rng.uniform(0.0, 1.0, size=(tint_idx.size, 3))
    # Bias tints toward cyan/gold/magenta of the nebula.
    palette = np.array(
        [(0.82, 0.94, 1.00), (1.00, 0.93, 0.76), (1.00, 0.78, 0.92), (0.80, 0.86, 1.00)]
    )
    pick = rng.integers(0, len(palette), size=tint_idx.size)
    base_color[tint_idx] = 0.55 * palette[pick] + 0.45 * tints
    return {
        "x": x,
        "y": y,
        "size": size,
        "alpha": base_alpha,
        "color": base_color,
        "phase": twinkle_phase,
        "speed": twinkle_speed,
        "amp": twinkle_amp,
    }


def draw_nebula(ax: plt.Axes) -> None:
    """Five painterly bands of low-alpha haze. Polygons stacked to fake blur."""
    bands = [
        {"y0": -0.20, "amp": 0.55, "phase": 0.00, "color": (0.18, 0.62, 1.00), "alpha": 0.055},
        {"y0":  0.18, "amp": 0.42, "phase": 1.60, "color": (0.55, 0.32, 1.00), "alpha": 0.042},
        {"y0":  0.05, "amp": 0.75, "phase": 3.20, "color": (0.05, 0.95, 0.65), "alpha": 0.034},
        {"y0": -0.40, "amp": 0.30, "phase": 0.80, "color": (1.00, 0.65, 0.40), "alpha": 0.045},
        {"y0":  0.55, "amp": 0.50, "phase": 4.10, "color": (0.95, 0.45, 0.85), "alpha": 0.030},
    ]
    xs = np.linspace(-6.6, 6.6, 540)
    for b in bands:
        # Stack 4 polygons of decreasing thickness to simulate gaussian falloff.
        for thickness, fade in [(0.85, 1.0), (0.60, 0.78), (0.38, 0.55), (0.20, 0.32)]:
            mid = b["y0"] + b["amp"] * np.sin(xs * 0.66 + b["phase"])
            top = mid + thickness
            bot = mid - thickness
            verts = np.concatenate(
                [np.column_stack([xs, top]), np.column_stack([xs[::-1], bot[::-1]])]
            )
            poly = Polygon(
                verts,
                closed=True,
                facecolor=b["color"],
                edgecolor="none",
                alpha=b["alpha"] * fade,
                zorder=0.5,
            )
            ax.add_patch(poly)

    # Procedural Milky-Way-arm star cloud: low-alpha smear along a curved path.
    rng = np.random.default_rng(11)
    n = 700
    t = rng.uniform(0, 1, n)
    arm_x = -6.4 + 12.8 * t + rng.normal(0, 0.18, n)
    arm_y = 1.2 * np.sin(t * 3.4 + 0.6) - 0.5 + rng.normal(0, 0.20, n)
    ax.scatter(
        arm_x, arm_y,
        s=rng.uniform(2, 18, n),
        c=[(1.0, 0.93, 0.78)] * n,
        alpha=rng.uniform(0.02, 0.08, n),
        linewidths=0,
        zorder=0.7,
    )


def draw_vignette(ax: plt.Axes) -> None:
    """Radial darkening overlay so the corners feel like deep space."""
    h, w = 480, 800
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(cx, cy)
    alpha = np.clip((r - 0.45) / 0.55, 0, 1) ** 1.6 * 0.55
    rgba = np.zeros((h, w, 4))
    rgba[..., 3] = alpha
    ax.imshow(
        rgba,
        extent=(-6.6, 6.6, -3.7, 3.7),
        origin="lower",
        zorder=18,
        interpolation="bilinear",
    )


def render_orrery(systems: list[System], frames: int, fps: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    bg = make_background(rng, n=8000)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#020410")
    ax.set_facecolor("#020410")
    ax.set_xlim(-6.6, 6.6)
    ax.set_ylim(-3.7, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    draw_nebula(ax)
    bg_scatter = ax.scatter(
        bg["x"], bg["y"], s=bg["size"], c=bg["color"], alpha=bg["alpha"],
        linewidths=0, zorder=1,
    )

    star_artists: list[dict] = []  # 4-pass bloom artists per system
    planet_artists: list[dict] = []
    orbit_artists: list = []
    trail_artists: list[dict] = []
    labels = []

    notable_hosts = {
        "TRAPPIST-1", "TOI-700", "Kepler-186", "Proxima Cen", "LHS 1140",
        "K2-18", "Teegarden's Star", "Kepler-452", "Ross 508",
    }
    top_hosts = {s.host for s in systems[:8]} | notable_hosts

    bloom_passes = [(1.0, 0.96), (3.5, 0.22), (8.0, 0.10), (16.0, 0.04)]
    planet_bloom = [(1.0, 0.95), (3.5, 0.18)]

    for system in systems:
        star_teff = next((p.st_teff for p in system.planets if p.st_teff is not None), None)
        base_size = 26 + 12 * math.sqrt(len(system.planets))
        passes = []
        for s_mult, alpha in bloom_passes:
            sc = ax.scatter(
                [system.x], [system.y],
                s=base_size * s_mult ** 2,
                c=[star_color(star_teff)],
                alpha=alpha * system.depth,
                linewidths=0,
                zorder=6 + (s_mult < 2.0) * 0.5,
            )
            passes.append((sc, alpha))
        star_artists.append({"system": system, "passes": passes})

        if system.host in top_hosts:
            labels.append(
                ax.text(
                    system.x + 0.06, system.y + 0.05, system.host,
                    color="#dceaff", fontsize=7, alpha=0.0, zorder=9,
                    family="serif",
                )
            )

        scale = 0.22 / max(0.08, max(estimate_semimajor_axis(p) for p in system.planets))
        for p in sorted(system.planets, key=estimate_semimajor_axis):
            semi = estimate_semimajor_axis(p)
            r = max(0.045, min(0.28, semi * scale))
            theta = np.linspace(0, 2 * np.pi, 200)
            line, = ax.plot(
                system.x + r * np.cos(theta), system.y + r * np.sin(theta),
                color="#7d8aa8", alpha=0.23 * system.depth, lw=0.55, zorder=4,
            )
            orbit_artists.append(line)

            pcolor = planet_color(p)
            psize = 8 + 5 * min(2.4, p.radius or 1.0)
            dot_passes = []
            for s_mult, alpha in planet_bloom:
                dot = ax.scatter(
                    [system.x + r], [system.y],
                    s=psize * s_mult ** 2,
                    c=[pcolor], alpha=alpha * system.depth,
                    linewidths=0, zorder=7 + (s_mult < 2.0) * 0.5,
                )
                dot_passes.append((dot, alpha))
            phase0 = random.Random(p.name).random() * 2 * math.pi
            planet_artists.append({
                "system": system, "planet": p, "r": r, "phase0": phase0,
                "passes": dot_passes,
            })

            # Trail rendered as LineCollection with per-segment alpha falloff.
            trail_n = 24
            verts = np.zeros((trail_n + 1, 2))
            segs = np.zeros((trail_n, 2, 2))
            lc = LineCollection(
                segs, colors=[pcolor] * trail_n,
                linewidths=np.linspace(1.2, 0.2, trail_n),
                zorder=5,
            )
            lc.set_alpha(None)  # use per-segment colors with alpha
            ax.add_collection(lc)
            trail_artists.append({
                "system": system, "planet": p, "r": r, "phase0": phase0 + 0.2,
                "lc": lc, "n": trail_n, "color": pcolor,
            })

    draw_vignette(ax)

    # ============ Sky-position inset ============
    # Real RA x Dec scatter with the Cygnus cluster boxed.
    inset = fig.add_axes([0.74, 0.04, 0.225, 0.235])  # right-bottom
    inset.set_xlim(360, 0)  # RA goes right-to-left, astro convention
    inset.set_ylim(-90, 90)
    inset.set_facecolor("#040816")
    for spine in inset.spines.values():
        spine.set_color("#22324c")
        spine.set_linewidth(0.7)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(colors="#3b4f74")

    # Faint ecliptic-ish guide arc.
    ras = np.linspace(0, 360, 200)
    inset.plot(ras, 23.4 * np.sin(np.radians(ras)), color="#1a2640", lw=0.6, alpha=0.7)

    ra_pts, dec_pts, esi_pts = [], [], []
    for s in systems:
        if s.ra is None or s.dec is None:
            continue
        ra_pts.append(s.ra)
        dec_pts.append(s.dec)
        esi_pts.append(max((p.esi or 0) for p in s.planets))
    if ra_pts:
        inset.scatter(
            ra_pts, dec_pts, c=esi_pts, cmap="viridis",
            s=8, alpha=0.85, linewidths=0, zorder=4,
        )
    # Highlight the Kepler primary field (~RA 286 deg, Dec +44 deg, ~12 deg radius).
    kepler_box = Rectangle(
        (276, 36), 22, 16, fill=False, edgecolor="#7ee7ff",
        linewidth=0.8, alpha=0.85, zorder=5,
    )
    inset.add_patch(kepler_box)
    inset.text(
        287, 56, "Kepler primary field",
        color="#7ee7ff", fontsize=6.0, ha="center", family="serif", alpha=0.95,
    )
    inset.set_title(
        "real sky  —  RA × Dec",
        color="#9ec3e8", fontsize=7.2, family="serif", pad=3,
    )

    # ============ Captions (no headline; pure-visual coda) ============
    counter = fig.text(
        0.018, 0.025,
        f"{len(systems)} systems  ·  {sum(len(s.planets) for s in systems)} candidate worlds",
        color="#7ee7ff", fontsize=8.5, family="serif", alpha=0.85, zorder=20,
    )
    bias_caption = fig.text(
        0.853, 0.018,
        "clustering = Kepler's primary field, not the universe's",
        color="#9bb1cc", fontsize=6.6, family="serif",
        ha="center", style="italic", alpha=0.92, zorder=20,
    )

    twinkle_phase = bg["phase"]
    twinkle_speed = bg["speed"]
    twinkle_amp = bg["amp"]
    base_alpha = bg["alpha"]

    def update(frame: int):
        phase = 2 * np.pi * frame / frames
        artists = [counter, bias_caption]

        # Twinkle the background field.
        modulated = base_alpha * (1.0 - twinkle_amp + twinkle_amp * np.sin(twinkle_phase + frame * twinkle_speed) * 0.5 + twinkle_amp * 0.5)
        bg_scatter.set_alpha(np.clip(modulated, 0.02, 0.85))
        artists.append(bg_scatter)

        # Fade in labels in the first 10% of the loop.
        for label in labels:
            label.set_alpha(0.78 if frame > frames * 0.10 else frame / max(1, frames * 0.10) * 0.78)
            artists.append(label)

        # Planets advance along orbits. Period drives angular speed (Kepler-ish).
        for art in planet_artists:
            system = art["system"]
            p = art["planet"]
            r = art["r"]
            period = max(4.0, min(280.0, p.period or 40.0))
            speed = 22.0 / math.sqrt(period)
            a = art["phase0"] + phase * speed
            x = system.x + r * math.cos(a)
            y = system.y + r * math.sin(a)
            for sc, _ in art["passes"]:
                sc.set_offsets([[x, y]])
                artists.append(sc)

        # Trails: LineCollection with linearly fading per-segment alpha.
        for art in trail_artists:
            system = art["system"]
            p = art["planet"]
            r = art["r"]
            n = art["n"]
            period = max(4.0, min(280.0, p.period or 40.0))
            speed = 22.0 / math.sqrt(period)
            head = art["phase0"] + phase * speed
            angles = head - np.linspace(0, 0.8, n + 1)
            xs = system.x + r * np.cos(angles)
            ys = system.y + r * np.sin(angles)
            verts = np.column_stack([xs, ys])
            segs = np.stack([verts[:-1], verts[1:]], axis=1)
            art["lc"].set_segments(segs)
            r_, g_, b_ = art["color"]
            falloff = np.linspace(0.55, 0.0, n) * system.depth
            colors = np.column_stack(
                [np.full(n, r_), np.full(n, g_), np.full(n, b_), falloff]
            )
            art["lc"].set_color(colors)
            artists.append(art["lc"])

        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    gif_path = out_dir / "hwc_orrery_v5.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    still_path = out_dir / "hwc_orrery_v5_still.png"
    fig.savefig(still_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(gif_path)
    print(still_path)


def main() -> None:
    try:
        text = fetch_text(HWC_URL, HWC_PAGE)
        planets = extract_hwc_table(text)
        write_hwc_csv(planets)
    except Exception as exc:  # network failure or page format change
        print(f"[warn] live HWC fetch failed ({exc!s}); falling back to cached CSV")
        planets = read_local_hwc_csv()
    nasa = fetch_nasa_rows([p.name for p in planets])
    merge_nasa(planets, nasa)
    systems = build_systems(planets)
    render_orrery(systems, frames=360, fps=30, out_dir=OUT_V5)


if __name__ == "__main__":
    main()
