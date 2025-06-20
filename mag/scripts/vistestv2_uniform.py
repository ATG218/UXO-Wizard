#!/usr/bin/env python3
"""
Magnetic-survey processing with Kalman harmonic filter
=====================================================
"""

import math
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.signal as sig
import scipy.integrate

import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from pyproj import Transformer

# ═══════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═══════════════════════════════════════════════════════════════════
CSV_PATH = Path(
    "/Users/aleksandergarbuz/Documents/SINTEF/data/"
    "20250611_081139_MWALK_#0122_processed_20250616_094044.csv"
)
OUTDIR   = Path("/Users/aleksandergarbuz/Documents/SINTEF/data/081139puniform")
COLUMN   = "R2 [nT]"

# crop (% rows to drop at start/end)
CUT_START = 0.30
CUT_END   = 0.065

# spike detection
MAD_FACTOR    = 5.0
MEDIAN_WINDOW = 301

# symlog scatter settings
LINTHRESH = 50.0

# satellite basemap
SATELLITE_MAP = True
MAP_PROVIDER  = "Esri.WorldImagery"
MAP_ZOOM      = None

# interactive map (Folium)
INTERACTIVE_MAP   = True
MAP_SAMPLE_STRIDE = 20

# field interpolation
INTERPOLATE_FIELD = True
GRID_RES          = 200

# Kalman harmonic filter
USE_KALMAN          = True      # master switch
ROTOR_FREQ_HZ       = 17.0      # 0   -> auto-detect strongest low-freq peak
ESTIMATE_ROTOR_FREQ = True
N_HARMONICS         = 3         # fundamental + 2 harmonics
Q_HARM   = 1e-2                # process-noise variance for each (a,b)
Q_GEO    = 1e-5                # process noise for geology baseline
R_NOISE  = None                # None  → robust estimate from MAD
# ═══════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────
# basemap helpers
# ───────────────────────────────────────────────────────────────────
if SATELLITE_MAP:
    import contextily as ctx
    from xyzservices import TileProvider

    def _lookup_provider(key: str):
        """Return a contextily/xyzservices provider object from dotted key."""
        try:                                # 1️⃣ QuickMapServices list
            return TileProvider.from_qms(key)
        except ValueError:                  # 2️⃣ ctx.providers dotted path
            try:
                return functools.reduce(getattr, key.split("."), ctx.providers)
            except AttributeError:
                valid = sorted(
                    ".".join(p) for p in ctx.providers.flatten().keys()
                )
                raise ValueError(
                    f"Provider '{key}' not found. Try one of:\n  "
                    + "\n  ".join(valid[:40])
                    + "\n  …"
                )

    provider = _lookup_provider(MAP_PROVIDER)

if INTERACTIVE_MAP:
    import folium
    from branca.colormap import LinearColormap as cm_linear

# ensure output directory exists
OUTDIR.mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════
# 1.  LOAD  →  UNIFORM TIMELINE  →  TRIM
# ═══════════════════════════════════════════════════════════════════
print("📖  Reading", CSV_PATH)
df = pd.read_csv(CSV_PATH)

print("🕑  Parsing datetime …")
df["datetime"] = pd.to_datetime(
    df["GPSDate"].astype(str).str.strip()
    + " "
    + df["GPSTime [hh:mm:ss.sss]"].astype(str).str.strip(),
    errors="coerce",
)
if df["datetime"].isna().any():
    raise ValueError("Could not parse GPSDate / GPSTime")

# 1.1  initial time axis from GPS
df["t_s"] = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()

# 1.2  trim survey ends
n_rows = len(df)
lo, hi = int(n_rows * CUT_START), int(n_rows * (1 - CUT_END))
print(f"✂️  Trim → keeping rows {lo}…{hi-1} of {n_rows}")
df = df.iloc[lo:hi].reset_index(drop=True)

# 1.3  find modal sample interval & rebuild uniform timeline
dts, counts = np.unique(np.diff(df["t_s"]), return_counts=True)
dt_common   = dts[np.argmax(counts)]
t_uniform   = np.arange(len(df)) * dt_common
df["t_uniform_s"] = t_uniform

fs = 1.0 / dt_common
print(f"   Uniform Δt = {dt_common:.4f} s   fs ≈ {fs:.2f} Hz")

# 1.4  working vectors
z           = df[COLUMN].astype(float).values
x           = t_uniform               # used in scatter
time_hours  = t_uniform / 3600        # for long plots

# ═══════════════════════════════════════════════════════════════════
# 2.  STATIC SYMLOG SCATTER  (smear-free)
# ═══════════════════════════════════════════════════════════════════
print("🎨  Symlog scatter …")
fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(
    x,
    z,
    c=z,
    s=4,
    cmap="viridis",
    norm=colors.SymLogNorm(linthresh=LINTHRESH, vmin=z.min(), vmax=z.max()),
)
fig.colorbar(sc, label=COLUMN)
ax.set_xlabel("Time since start (s)")
ax.set_ylabel(COLUMN)
ax.set_title(f"{COLUMN} – symlog (uniform timeline)")
fig.tight_layout()
fig.savefig(OUTDIR / "symlog_scatter_uniform.png", dpi=300)
plt.close()

# ───────────────────────────────────────────────────────────────────
# 2b.  OPTIONAL – interpolate residual field (unchanged)
# ───────────────────────────────────────────────────────────────────
if INTERPOLATE_FIELD:
    print("🗺  Gridding residual field …")
    if {"UTM_Easting", "UTM_Northing"}.issubset(df.columns):
        X = df["UTM_Easting"].values
        Y = df["UTM_Northing"].values
        lon = df["Longitude [Decimal Degrees]"]
        lat = df["Latitude [Decimal Degrees]"]
        interp_bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]
        extent = [X.min(), X.max(), Y.min(), Y.max()]
    else:
        X = df["Longitude [Decimal Degrees]"].values
        Y = df["Latitude  [Decimal Degrees]"].values
        interp_bounds = [[Y.min(), X.min()], [Y.max(), X.max()]]
        extent = [X.min(), X.max(), Y.min(), Y.max()]

    xi = np.linspace(X.min(), X.max(), GRID_RES)
    yi = np.linspace(Y.min(), Y.max(), GRID_RES)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((X, Y), z, (Xi, Yi), method="cubic")

    if {"UTM_Easting", "UTM_Northing"}.issubset(df.columns):
        transformer = Transformer.from_crs("epsg:32633", "epsg:4326", always_xy=True)
        LonGrid, LatGrid = transformer.transform(Xi, Yi)
    else:
        LonGrid, LatGrid = Xi, Yi

    field_points = [
        (lat, lon, val)
        for lat, lon, val in zip(LatGrid.ravel(), LonGrid.ravel(), Zi.ravel())
        if not np.isnan(val)
    ]

    interp_png = OUTDIR / "field_interpolated.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Zi, extent=extent, origin="lower", cmap="viridis")
    fig.colorbar(im, label=COLUMN)
    ax.set_title("Interpolated residual field")
    ax.set_xlabel("Easting / Longitude")
    ax.set_ylabel("Northing / Latitude")
    fig.tight_layout()
    fig.savefig(interp_png, dpi=300)
    plt.close()

# ═══════════════════════════════════════════════════════════════════
# 3.  SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════════
print("🚨  Spike detection …")
trend     = sig.medfilt(z, MEDIAN_WINDOW)
resid     = z - trend
mad       = np.median(np.abs(resid - np.median(resid)))
threshold = MAD_FACTOR * mad
idx       = np.where(np.abs(resid) > threshold)[0]

events, spikes = [], pd.DataFrame()
if idx.size:
    cur = [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            events.append(cur)
            cur = [i]
    events.append(cur)

    dt = dt_common
    spikes = pd.DataFrame(
        [
            {
                "start_time": df["datetime"].iloc[e[0]],
                "end_time": df["datetime"].iloc[e[-1]],
                "duration_s": (e[-1] - e[0] + 1) * dt,
                "peak_nT": z[e[np.argmax(np.abs(z[e]))]],
                "peak_time": df["datetime"].iloc[e[np.argmax(np.abs(z[e]))]],
                "latitude": df["Latitude [Decimal Degrees]"].iloc[
                    e[np.argmax(np.abs(z[e]))]
                ],
                "longitude": df["Longitude [Decimal Degrees]"].iloc[
                    e[np.argmax(np.abs(z[e]))]
                ],
                "n_samples": len(e),
            }
            for e in events
        ]
    )
    spikes.to_csv(OUTDIR / "spike_catalogue.csv", index=False)
    print(f"   ↳ {len(spikes)} spikes ➜ spike_catalogue.csv")
else:
    print("   ↳ no spikes detected")

# ═══════════════════════════════════════════════════════════════════
# 4.  MAPS (unchanged except time axis isn’t used here)
# ═══════════════════════════════════════════════════════════════════
print("🗺️  Building maps …")
if not spikes.empty:
    gdf = gpd.GeoDataFrame(
        spikes,
        geometry=gpd.points_from_xy(spikes.longitude, spikes.latitude),
        crs="EPSG:4326",
    )

    # static PNG
    if SATELLITE_MAP:
        g3857 = gdf.to_crs(epsg=3857)
        if MAP_ZOOM is None:
            lat_c = gdf.latitude.mean()
            width_m = g3857.total_bounds[2] - g3857.total_bounds[0]
            z_est = int(
                round(
                    math.log2((156543.0339 * math.cos(math.radians(lat_c)) * 256)
                              / (width_m / 12))
                )
            )
            MAP_ZOOM = max(3, min(19, z_est))
        fig, ax = plt.subplots(figsize=(8, 8))
        g3857.plot(
            ax=ax, column="peak_nT", cmap="viridis", markersize=90, legend=True, alpha=0.95
        )
        try:
            ctx.add_basemap(ax, source=provider, zoom=MAP_ZOOM, crs="epsg:3857")
        except Exception as e:
            print("⚠️  basemap fetch failed:", e)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUTDIR / "spike_map.png", dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(
            ax=ax, column="peak_nT", cmap="viridis", markersize=90, legend=True, alpha=0.95
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUTDIR / "spike_map.png", dpi=300)
        plt.close()

    # interactive Folium map
    if INTERACTIVE_MAP:
        print("   ↳ interactive map …")
        center = [gdf.latitude.mean(), gdf.longitude.mean()]
        fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

        cm = cm_linear(["blue", "white", "red"],
                       vmin=z.min(), vmax=z.max(),
                       caption=f"{COLUMN} (path samples)")
        sample = df.iloc[::MAP_SAMPLE_STRIDE]
        pts = list(
            zip(
                sample["Latitude [Decimal Degrees]"],
                sample["Longitude [Decimal Degrees]"],
                sample[COLUMN],
            )
        )
        folium.PolyLine(
            [(lat, lon) for lat, lon, _ in pts], color="gray", weight=2, opacity=0.4
        ).add_to(fmap)
        for lat, lon, val in pts:
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color=cm(val),
                fill=True,
                fill_opacity=0.7,
                opacity=0.7,
            ).add_to(fmap)
        cm.caption = f"{COLUMN} (path samples)"
        fmap.add_child(cm)

        spike_group = folium.FeatureGroup(name="Spikes")
        for _, row in gdf.iterrows():
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=8,
                color="red",
                fill=True,
                fill_opacity=0.9,
                tooltip=f"{row.peak_time} | {row.peak_nT:+.1f} nT",
            ).add_to(spike_group)

        if INTERPOLATE_FIELD:
            field_fg = folium.FeatureGroup(name="Interpolated field")
            cm_field = cm_linear(
                ["#440154", "#31688e", "#35b779", "#fde725"],
                vmin=Zi[np.isfinite(Zi)].min(),
                vmax=Zi[np.isfinite(Zi)].max(),
                caption="Interpolated field",
            )
            for lat, lon, val in field_points:
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=2,
                    color=cm_field(val),
                    fill=True,
                    fill_opacity=0.8,
                    opacity=0.8,
                ).add_to(field_fg)
            fmap.add_child(field_fg)

        fmap.add_child(spike_group)
        fmap.add_child(folium.LayerControl())
        fmap.save(OUTDIR / "interactive_spike_map.html")

else:
    print("   ↳ no spikes: skipping maps")

# ═══════════════════════════════════════════════════════════════════
# 5.  COMBINED (50 Hz notch + 0.1 Hz HP) FILTER
# ═══════════════════════════════════════════════════════════════════
print("\n🎯 Combined notch + HP filter …")
nyquist = fs / 2

# Step 1 – 50 Hz notch
notch_freq = 50.0
Q = 30
if notch_freq < nyquist:
    b_notch, a_notch = sig.iirnotch(notch_freq, Q, fs)
    z_notch = sig.filtfilt(b_notch, a_notch, z)
    print("   ✅ 50 Hz notch applied")
else:
    z_notch = z.copy()
    print("   ⚠️  50 Hz > Nyquist → skipped")

# Step 2 – 0.1 Hz high-pass
hp_freq = 0.1
if hp_freq < nyquist:
    b_hp, a_hp = sig.butter(4, hp_freq, btype="high", fs=fs)
    z_combined = sig.filtfilt(b_hp, a_hp, z_notch)
    print("   ✅ 0.1 Hz high-pass applied")
else:
    z_combined = z_notch.copy()
    print("   ⚠️  0.1 Hz > Nyquist → skipped")

# ═══════════════════════════════════════════════════════════════════
# 6.  KALMAN HARMONIC FILTER
# ═══════════════════════════════════════════════════════════════════
if USE_KALMAN:
    print("\n🛰️  Kalman harmonic filter …")

    # 6.1  Auto-detect rotor frequency (optional)
    if ROTOR_FREQ_HZ == 0 and ESTIMATE_ROTOR_FREQ:
        f_psd, P_psd = sig.welch(z, fs, nperseg=4096)
        mask = (f_psd > 5) & (f_psd < 60)
        pk_idx, _ = sig.find_peaks(P_psd[mask])
        if len(pk_idx):
            ROTOR_FREQ_HZ = f_psd[mask][pk_idx[np.argmax(P_psd[mask][pk_idx])]]
            print(f"   ↳ detected rotor fundamental ≈ {ROTOR_FREQ_HZ:.2f} Hz")
        else:
            raise RuntimeError("Could not auto-detect rotor frequency")

    omega = 2 * math.pi * ROTOR_FREQ_HZ
    dt = 1.0 / fs

    n_state = 2 * N_HARMONICS + 1  # (a,b) per harmonic + geology baseline
    F = np.eye(n_state)
    for k in range(N_HARMONICS):
        cos_, sin_ = math.cos(omega * (k + 1) * dt), math.sin(omega * (k + 1) * dt)
        idx = 2 * k
        F[idx:idx+2, idx:idx+2] = [[cos_, sin_], [-sin_, cos_]]

    H = np.zeros((1, n_state))
    H[0, ::2] = 1  # pick cosine component of each harmonic
    H[0, -1]  = 1  # geology baseline

    Q = np.diag([Q_HARM]*(2*N_HARMONICS) + [Q_GEO])
    if R_NOISE is None:
        mad = np.median(np.abs(z - np.median(z)))
        R = (mad / 0.6745) ** 2
    else:
        R = R_NOISE

    x_est = np.zeros(n_state)
    P_est = np.eye(n_state) * 1e3

    g_est   = np.empty_like(z)  # geology baseline
    z_kf    = np.empty_like(z)  # clean series

    for i, z_k in enumerate(z):
        # predict
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q

        # update
        S = H @ P_pred @ H.T + R
        K = (P_pred @ H.T) / S
        y = z_k - (H @ x_pred)
        x_est = x_pred + (K.flatten() * y)
        P_est = (np.eye(n_state) - K @ H) @ P_pred

        g_est[i] = x_est[-1]
        harm_sum = (H[:, :-1] @ x_est[:-1]).item()
        z_kf[i] = z_k - harm_sum

else:
    z_kf = None

# ═══════════════════════════════════════════════════════════════════
# 7.  COMPARISON PLOTS (uniform timeline)
# ═══════════════════════════════════════════════════════════════════
print("\n📈  Comparison figures …")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(time_hours, z,          lw=0.6, alpha=0.5, label="Original")
ax.plot(time_hours, z_combined, lw=0.7, alpha=0.8, label="Notch+HP")
if z_kf is not None:
    ax.plot(time_hours, z_kf,   lw=0.9,             label="Kalman")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Magnetic field (nT)")
ax.set_title("Time-series comparison (uniform axis)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR / "kalman_filter_comparison.png", dpi=300)
plt.close()

# PSD comparison
f_raw, P_raw = sig.welch(z,          fs, nperseg=4096)
f_cmb, P_cmb = sig.welch(z_combined, fs, nperseg=4096)
fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(f_raw[1:], P_raw[1:], label="Original",  alpha=0.6)
ax.loglog(f_cmb[1:], P_cmb[1:], label="Notch+HP", alpha=0.8)
if z_kf is not None:
    f_kf, P_kf = sig.welch(z_kf, fs, nperseg=4096)
    ax.loglog(f_kf[1:], P_kf[1:], label="Kalman",  alpha=0.9)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (nT²/Hz)")
ax.set_title("Welch PSD – original vs filters")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR / "psd_kalman_vs_combined.png", dpi=300)
plt.close()

# ═══════════════════════════════════════════════════════════════════
# 8.  UNIFORM-AXIS INTERACTIVE MAP (filtered)
# ═══════════════════════════════════════════════════════════════════
if INTERACTIVE_MAP and z_kf is not None:
    print("🗺️  Regenerating interactive map with Kalman baseline …")
    z_kf_res = z_kf - np.mean(z_kf)

    m_filt = folium.Map(
        location=[df["Latitude [Decimal Degrees]"].mean(),
                  df["Longitude [Decimal Degrees]"].mean()],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    cm_filt = cm_linear(
        ["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=z_kf_res.min(),
        vmax=z_kf_res.max(),
        caption="Kalman residual (nT)",
    )
    subsample = df.iloc[::MAP_SAMPLE_STRIDE]
    for i, row in subsample.iterrows():
        val = z_kf_res[i]
        folium.CircleMarker(
            location=(
                row["Latitude [Decimal Degrees]"],
                row["Longitude [Decimal Degrees]"],
            ),
            radius=3,
            color=cm_filt(val),
            fill=True,
            fill_opacity=0.8,
            opacity=0.8,
        ).add_to(m_filt)
    cm_filt.add_to(m_filt)
    m_filt.get_root().html.add_child(
        folium.Element(
            "<h3 style='text-align:center;font-size:20px'>"
            "Kalman-filtered Magnetic Field Map"
            "</h3>"
        )
    )
    m_filt.save(OUTDIR / "interactive_map_kalman_filtered.html")
    print("   ✅ interactive_map_kalman_filtered.html saved")

# ═══════════════════════════════════════════════════════════════════
print("\n✅  All done – outputs in", OUTDIR)
