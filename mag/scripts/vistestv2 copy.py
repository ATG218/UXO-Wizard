#!/usr/bin/env python3
import math
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.signal as sig
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
from pyproj import Transformer
# -------------------------------------------------------------------
# 🛠  CONFIG  ── edit these eight lines and run the script
# -------------------------------------------------------------------
from pathlib import Path
CSV_PATH       = Path("/Users/aleksandergarbuz/Documents/SINTEF/data/20250611_081139_MWALK_#0122_processed_20250616_094044.csv")
OUTDIR         = Path("/Users/aleksandergarbuz/Documents/SINTEF/data/081139p2")
COLUMN         = "R2 [nT]"

# crop (% rows to drop @ start/end)
CUT_START       = 0.30
CUT_END         = 0.065

# spike detection
MAD_FACTOR      = 5.0
MEDIAN_WINDOW   = 301
SPIKE_DETECTION = True  # New variable to control spike detection

# static scatter
LINTHRESH       = 50.0

# satellite basemap for static PNG
SATELLITE_MAP   = True
MAP_PROVIDER    = "Esri.WorldImagery"
MAP_ZOOM        = None

# interactive map (Folium)
INTERACTIVE_MAP = True
MAP_SAMPLE_STRIDE = 20

# ⇢ FIELD INTERPOLATION (add just below the PSD settings) ─────────────
INTERPOLATE_FIELD = True      # turn gridding on/off
GRID_RES          = 200       # ≈ number of cells per side of the square grid

# ⇢ KALMAN FILTER SETTINGS ─────────────────────────────────────────────
USE_KALMAN           = True       # enable Kalman harmonic filter
ROTOR_FREQ_HZ        = 0          # rotor frequency (0 = auto-detect)
ESTIMATE_ROTOR_FREQ  = True       # auto-detect rotor frequency
N_HARMONICS          = 3          # number of harmonics to model
Q_HARM               = 1e-6       # process noise for harmonics
Q_GEO                = 1e-4       # process noise for geology baseline
R_NOISE              = None       # measurement noise (None = auto-estimate)
# -------------------------------------------------------------------

# basemap helpers
if SATELLITE_MAP:
    import contextily as ctx
    from xyzservices import TileProvider

    def _lookup_provider(key: str):
        """Return a contextily/xyzservices provider object from a dotted key."""
        # 1️⃣ try QuickMapServices list via xyzservices
        try:
            return TileProvider.from_qms(key)
        except ValueError:
            # 2️⃣ fall back to ctx.providers dotted lookup (e.g. "Esri.WorldImagery")
            try:
                return functools.reduce(getattr, key.split('.'), ctx.providers)
            except AttributeError:
                valid = sorted('.'.join(p) for p in ctx.providers.flatten().keys())
                raise ValueError(
                    f"Provider '{key}' not found. Try one of:\n  " +
                    '\n  '.join(valid[:40]) + '\n  …'
                )

    provider = _lookup_provider(MAP_PROVIDER)

# interactive map helpers
if INTERACTIVE_MAP:
    import folium
    from branca.colormap import LinearColormap as cm_linear

# ensure output directory exists
OUTDIR.mkdir(exist_ok=True, parents=True)

# ════════════════════════════════════════════
# 1. LOAD & TRIM
# ════════════════════════════════════════════
print("📖  Reading ", CSV_PATH)
df = pd.read_csv(CSV_PATH)

print("🕑  Parsing datetime …")
df["datetime"] = pd.to_datetime(
    df["GPSDate"].astype(str).str.strip() + " " +
    df["GPSTime [hh:mm:ss.sss]"].astype(str).str.strip(),
    errors="coerce"
)
if df["datetime"].isna().any():
    raise ValueError("Could not parse GPSDate / GPSTime. Check format.")

df["t_s"] = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()

n_rows = len(df)
lo, hi = int(n_rows * CUT_START), int(n_rows * (1 - CUT_END))
print(f"✂️  Trim → keeping rows {lo}…{hi-1} of {n_rows}")
df = df.iloc[lo:hi].reset_index(drop=True)

z  = df[COLUMN].values
x  = df["t_s"].values
fs = 1 / np.median(np.diff(x))

# ════════════════════════════════════════════
# 2. STATIC SCATTER (symlog)
# ════════════════════════════════════════════
print("🎨  Symlog scatter …")
fig, ax = plt.subplots(figsize=(12, 4))
sc = ax.scatter(
    x, z, c=z, s=4,
    cmap="viridis",
    norm=colors.SymLogNorm(linthresh=LINTHRESH, vmin=z.min(), vmax=z.max())
)
fig.colorbar(sc, label=COLUMN)
ax.set_xlabel("Time since start (s)")
ax.set_ylabel(COLUMN)
ax.set_title(f"{COLUMN} – symlog (linthresh={LINTHRESH})")
fig.tight_layout()
fig.savefig(OUTDIR / "symlog_scatter.png", dpi=300)
plt.close()

# ── 2b. OPTIONAL – create a gridded / interpolated residual layer ──────────
if INTERPOLATE_FIELD:
    print("🗺  Gridding residual field …")
    # 3a. choose X/Y coordinates (prefer UTM, fall back to lon/lat)
    if {'UTM_Easting', 'UTM_Northing'}.issubset(df.columns):
        X = df['UTM_Easting'].values
        Y = df['UTM_Northing'].values
        lon = df['Longitude [Decimal Degrees]'];  lat = df['Latitude [Decimal Degrees]']
        interp_bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]   # for Folium
        extent = [X.min(), X.max(), Y.min(), Y.max()]                      # for imshow
    else:
        X = df['Longitude [Decimal Degrees]'].values
        Y = df['Latitude  [Decimal Degrees]'].values
        interp_bounds = [[Y.min(), X.min()], [Y.max(), X.max()]]
        extent = [X.min(), X.max(), Y.min(), Y.max()]

    # 3b. build a regular grid
    xi = np.linspace(X.min(), X.max(), GRID_RES)
    yi = np.linspace(Y.min(), Y.max(), GRID_RES)
    Xi, Yi = np.meshgrid(xi, yi)

    # 3c. interpolate (cubic); mask NaNs for plotting
    Zi = griddata((X, Y), z, (Xi, Yi), method='cubic')

    # --- convert grid to lat/lon so Folium can plot individual points
    if {'UTM_Easting', 'UTM_Northing'}.issubset(df.columns):
        transformer = Transformer.from_crs("epsg:32633",   # ⚑ adjust zone if needed
                                           "epsg:4326",
                                           always_xy=True)
        LonGrid, LatGrid = transformer.transform(Xi, Yi)   # arrays same shape as Zi
    else:
        LonGrid, LatGrid = Xi, Yi                          # already lon/lat

    # flatten & drop NaNs → list of points
    field_points = [
        (lat, lon, val)
        for lat, lon, val in zip(LatGrid.ravel(), LonGrid.ravel(), Zi.ravel())
        if not np.isnan(val)
    ]

    # 3d. save a static PNG so Folium can re-use the same raster
    interp_png = OUTDIR / "field_interpolated.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Zi, extent=extent, origin='lower', cmap='viridis')
    fig.colorbar(im, label=COLUMN)
    ax.set_title("Interpolated residual field")
    ax.set_xlabel("Easting / Longitude")
    ax.set_ylabel("Northing / Latitude")
    fig.tight_layout()
    fig.savefig(interp_png, dpi=300)
    plt.close()

# ════════════════════════════════════════════
# 3. DATA ARTIFACT ANALYSIS
# ════════════════════════════════════════════

print("\n🔬 ANALYZING DATA ARTIFACTS")

# Analyze the raw data for systematic patterns/artifacts
def analyze_data_artifacts(data, time_array, sampling_rate):
    """Analyze raw data for systematic artifacts like instrument lag or drift"""
    
    print("🔍  Analyzing raw data characteristics...")
    
    # Basic statistics
    print(f"   Data range: {np.min(data):.1f} to {np.max(data):.1f} nT")
    print(f"   RMS: {np.std(data):.1f} nT")
    
    # Analyze consecutive differences to detect instrument lag
    diff1 = np.diff(data)
    diff2 = np.diff(data, n=2)
    
    print(f"   1st difference RMS: {np.std(diff1):.1f} nT")
    print(f"   2nd difference RMS: {np.std(diff2):.1f} nT")
    
    # Look for systematic patterns in the differences
    from scipy import signal
    
    # Calculate power spectral density to identify dominant frequencies
    freqs, psd = signal.welch(data, fs=sampling_rate, nperseg=min(4096, len(data)//4))
    
    # Find dominant low-frequency components (< 1 Hz)
    low_freq_mask = freqs < 1.0
    if np.any(low_freq_mask):
        dominant_freq_idx = np.argmax(psd[low_freq_mask])
        dominant_freq = freqs[low_freq_mask][dominant_freq_idx]
        print(f"   Dominant low-frequency component: {dominant_freq:.3f} Hz")
    
    # Check for trend/drift
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(time_array, data)
    print(f"   Linear trend: {slope:.2f} nT/hour (R²={r_value**2:.3f})")
    
    # Analyze autocorrelation to detect systematic patterns
    autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]
    
    # Find lag where autocorrelation drops below threshold
    threshold = 0.1
    significant_lags = np.where(autocorr > threshold)[0]
    if len(significant_lags) > 1:
        lag_time = significant_lags[-1] / sampling_rate
        print(f"   Significant autocorrelation up to: {lag_time:.1f} seconds")
    
    return {
        'psd_freqs': freqs,
        'psd_values': psd,
        'trend_slope': slope,
        'autocorr': autocorr[:min(1000, len(autocorr))]  # Limit length
    }

# Analyze the raw data
artifact_analysis = analyze_data_artifacts(z, x, fs)

# Implement artifact removal preprocessing
def remove_data_artifacts(data, method='polynomial_detrend', order=3):
    """Remove systematic artifacts from magnetometer data"""
    
    print(f"🧹  Removing artifacts using {method}...")
    
    if method == 'polynomial_detrend':
        # Remove polynomial trend
        x_indices = np.arange(len(data))
        poly_coeffs = np.polyfit(x_indices, data, order)
        poly_trend = np.polyval(poly_coeffs, x_indices)
        corrected_data = data - poly_trend
        print(f"   Removed polynomial trend (order {order})")
        
    elif method == 'high_pass_detrend':
        # Use very low-frequency high-pass filter to remove drift
        from scipy.signal import butter, filtfilt
        cutoff_freq = 0.01  # Hz - very low frequency
        if fs/2 > cutoff_freq:
            b, a = butter(2, cutoff_freq / (fs/2), btype='high')
            corrected_data = filtfilt(b, a, data)
            print(f"   Applied high-pass detrend filter ({cutoff_freq} Hz)")
        else:
            print("   ⚠️  Cannot apply high-pass detrend - frequency too low")
            corrected_data = data
            
    elif method == 'median_detrend':
        # Remove slowly varying median trend
        window_size = int(fs * 60)  # 1-minute window
        if window_size < len(data):
            trend = signal.medfilt(data, window_size | 1)  # Ensure odd
            corrected_data = data - trend
            print(f"   Removed median trend (window: {window_size} samples)")
        else:
            print("   ⚠️  Cannot apply median detrend - window too large")
            corrected_data = data
    else:
        print("   ⚠️  Unknown method - no correction applied")
        corrected_data = data
    
    improvement = (np.std(data) - np.std(corrected_data)) / np.std(data) * 100
    print(f"   Artifact removal: {improvement:.1f}% RMS reduction")
    
    return corrected_data

# Apply artifact removal to raw data
print("\n📊  Testing artifact removal methods...")

# Test different detrending methods
z_poly_detrend = remove_data_artifacts(z, 'polynomial_detrend', order=3)
z_hp_detrend = remove_data_artifacts(z, 'high_pass_detrend')
z_median_detrend = remove_data_artifacts(z, 'median_detrend')

# Choose the best method based on RMS reduction
methods = ['polynomial', 'high_pass', 'median']
corrected_data = [z_poly_detrend, z_hp_detrend, z_median_detrend]
rms_values = [np.std(d) for d in corrected_data]

best_method_idx = np.argmin(rms_values)
z_corrected = corrected_data[best_method_idx]
best_method = methods[best_method_idx]

print(f"\n✅  Best artifact removal method: {best_method}")
print(f"   Original RMS: {np.std(z):.1f} nT")
print(f"   Corrected RMS: {np.std(z_corrected):.1f} nT")
print(f"   Improvement: {(np.std(z) - np.std(z_corrected))/np.std(z)*100:.1f}%")

# Replace the original z data with corrected version
z_original = z.copy()  # Keep original for comparison
z = z_corrected

# ════════════════════════════════════════════
# 3. SPIKE CATALOGUE
# ════════════════════════════════════════════
print("🚨  Multi-stage spike detection …")

def detect_spikes(data, median_window=MEDIAN_WINDOW, mad_factor=MAD_FACTOR):
    """Detect spikes using MAD (Median Absolute Deviation) method"""
    trend = sig.medfilt(data, median_window)
    resid = data - trend
    mad = np.median(np.abs(resid - np.median(resid)))
    threshold = mad_factor * mad
    idx = np.where(np.abs(resid) > threshold)[0]
    
    events = []
    if idx.size:
        cur = [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                events.append(cur)
                cur = [i]
        events.append(cur)
    
    return events, idx

def create_spike_dataframe(events, data, times, dataframe):
    """Create spike catalogue DataFrame from detected events"""
    if not events:
        return pd.DataFrame()
    
    dt = np.median(np.diff(times))
    spikes = pd.DataFrame([
        {
            "start_time": dataframe["datetime"].iloc[e[0]],
            "end_time":   dataframe["datetime"].iloc[e[-1]],
            "duration_s": (e[-1] - e[0] + 1) * dt,
            "peak_nT":    data[e[np.argmax(np.abs(data[e]))]],
            "peak_time":  dataframe["datetime"].iloc[e[np.argmax(np.abs(data[e]))]],
            "latitude":   dataframe["Latitude [Decimal Degrees]"].iloc[e[np.argmax(np.abs(data[e]))]],
            "longitude":  dataframe["Longitude [Decimal Degrees]"].iloc[e[np.argmax(np.abs(data[e]))]],
            "n_samples":  len(e)
        }
        for e in events
    ])
    return spikes

# Stage 1: Spike detection on raw data
if SPIKE_DETECTION:
    print("🔍  Stage 1: Spike detection on raw data …")
    events_raw, peaks_raw = detect_spikes(z, median_window=MEDIAN_WINDOW,
                                          mad_factor=MAD_FACTOR)
    spikes_raw = create_spike_dataframe(events_raw, z, x, df)
    print(f"   ↳ {len(spikes_raw)} spikes detected on raw data")
    
    # Initialize spike collections for later stages
    spikes_combined = pd.DataFrame()
    spikes_kalman = pd.DataFrame()
    spikes_field_2d = pd.DataFrame()  # For 2D field spikes
else:
    spikes_raw = pd.DataFrame()
    spikes_combined = pd.DataFrame()
    spikes_kalman = pd.DataFrame()
    spikes_field_2d = pd.DataFrame()

# ════════════════════════════════════════════
# 4. MAPS
# ════════════════════════════════════════════
print("🗺️  Building maps …")
if not spikes_raw.empty:
    gdf = gpd.GeoDataFrame(
        spikes_raw,
        geometry=gpd.points_from_xy(spikes_raw.longitude, spikes_raw.latitude),
        crs="EPSG:4326"
    )

    # ---- static PNG ----
    if SATELLITE_MAP:
        g3857 = gdf.to_crs(epsg=3857)
        if MAP_ZOOM is None:
            lat_c  = gdf.latitude.mean()
            width_m = (g3857.total_bounds[2] - g3857.total_bounds[0])
            z_est  = int(round(math.log2((156543.0339 * math.cos(math.radians(lat_c)) * 256) / (width_m / 12))))
            MAP_ZOOM = max(3, min(19, z_est))
        fig, ax = plt.subplots(figsize=(8, 8))
        g3857.plot(ax=ax, column="peak_nT", cmap="viridis", markersize=90,
                   legend=True, alpha=0.95)

        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=provider, zoom=MAP_ZOOM, crs="EPSG:3857")
        except Exception as e:
            print("⚠️ basemap fetch failed:", e)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUTDIR / "spike_map.png", dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(ax=ax, column="peak_nT", cmap="viridis", markersize=90,
                 legend=True, alpha=0.95)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUTDIR / "spike_map.png", dpi=300)
        plt.close()

    # ---- interactive Folium ----
    if INTERACTIVE_MAP:
        print("   ↳ interactive map …")
        center = [gdf.latitude.mean(), gdf.longitude.mean()]
        fmap = folium.Map(location=center, zoom_start=15, tiles='OpenStreetMap')

        # survey path coloured by residual value (sampled)
        cm = cm_linear(['blue', 'white', 'red'],
                       vmin=z.min(), vmax=z.max(),
                       caption=f"{COLUMN} (path samples)")
        sample = df.iloc[::MAP_SAMPLE_STRIDE]
        pts = list(zip(sample["Latitude [Decimal Degrees]"],
                       sample["Longitude [Decimal Degrees]"],
                       sample[COLUMN]))
        folium.PolyLine([(lat, lon) for lat, lon, _ in pts],
                        color="gray", weight=2, opacity=0.4,
                        tooltip="Survey path").add_to(fmap)
        for lat, lon, val in pts:
            folium.CircleMarker(
                location=(lat, lon), radius=3, color=cm(val),
                fill=True, fill_opacity=0.7, opacity=0.7
            ).add_to(fmap)
        cm.caption = f"{COLUMN} (path samples)"
        fmap.add_child(cm)

        # spikes layer
        spike_group = folium.FeatureGroup(name="Spikes")
        for _, row in gdf.iterrows():
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=8, color="red", fill=True, fill_opacity=0.9,
                tooltip=f"{row.peak_time} | {row.peak_nT:+.1f} nT"
            ).add_to(spike_group)

        if INTERPOLATE_FIELD:
            field_fg = folium.FeatureGroup(name="Interpolated field")
            cm_field = cm_linear(
                ['#440154', '#31688e', '#35b779', '#fde725'],
                vmin=Zi[np.isfinite(Zi)].min(),
                vmax=Zi[np.isfinite(Zi)].max(),
                caption="Interpolated field"
            )
            for lat, lon, val in field_points:
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=2, color=cm_field(val),
                    fill=True, fill_opacity=0.8, opacity=0.8
                ).add_to(field_fg)
            fmap.add_child(field_fg)

        fmap.add_child(spike_group)
        fmap.add_child(folium.LayerControl())
        fmap.save(OUTDIR / "interactive_spike_map.html")

else:
    print("   ↳ no spikes: skipping maps")

# ════════════════════════════════════════════
# 5. COMBINED FILTER & FOLLOW-UP VISUALS
# ════════════════════════════════════════════
print("\n🎯 APPLYING COMBINED FILTER TO DATASET")
nyquist = fs / 2

# Step 1 – 50 Hz notch filter
notch_freq = 50.0  # Hz
Q = 30           # narrow notch
if notch_freq < nyquist:
    b_notch, a_notch = sig.iirnotch(notch_freq, Q, fs)
    z_notch = sig.filtfilt(b_notch, a_notch, z)
    print("   ✅ 50 Hz notch filter applied")
else:
    z_notch = z.copy()
    print("   ⚠️  50 Hz exceeds Nyquist – skipping notch")

# Stage 2: Spike detection on notch filtered data
if SPIKE_DETECTION:
    print("\n🔍  Stage 2: Spike detection on notch filtered data …")
    events_notch, peaks_notch = detect_spikes(z_notch, median_window=MEDIAN_WINDOW,
                                              mad_factor=MAD_FACTOR * 1.2)  # Slightly less sensitive
    spikes_notch = create_spike_dataframe(events_notch, z_notch, x, df)
    print(f"   ↳ {len(spikes_notch)} spikes detected on notch filtered data")

# Step 2 – 0.1 Hz high-pass filter
hp_freq = 0.1   # Hz
order = 4
if hp_freq < nyquist:
    b_hp, a_hp = sig.butter(order, hp_freq, btype='high', fs=fs)
    z_combined = sig.filtfilt(b_hp, a_hp, z_notch)
    print("   ✅ 0.1 Hz high-pass filter applied")
else:
    z_combined = z_notch.copy()
    print("   ⚠️  0.1 Hz exceeds Nyquist – skipping high-pass")

# Stage 3: Spike detection on combined filtered data
if SPIKE_DETECTION:
    print("\n🔍  Stage 3: Spike detection on combined filtered data …")
    events_combined, peaks_combined = detect_spikes(z_combined, 
                                                   median_window=MEDIAN_WINDOW,
                                                   mad_factor=MAD_FACTOR * 2.5)  # Much less sensitive for filtered data
    spikes_combined = create_spike_dataframe(events_combined, z_combined, x, df)
    print(f"   ↳ {len(spikes_combined)} spikes detected on combined filtered data")

# Step 3 – Kalman Filter for harmonic interference removal
if USE_KALMAN:
    print("\n🛰️ Kalman harmonic filter …")

    # 6.1  Estimate rotor frequency if requested
    if ROTOR_FREQ_HZ == 0 and ESTIMATE_ROTOR_FREQ:
        f_psd, P_psd = sig.welch(z, fs, nperseg=4096)
        # look between 5 Hz and 60 Hz for strongest peak
        mask = (f_psd > 5) & (f_psd < 60)
        peak_idx, _ = sig.find_peaks(P_psd[mask])
        if len(peak_idx):
            pk = peak_idx[np.argmax(P_psd[mask][peak_idx])]
            ROTOR_FREQ_HZ = f_psd[mask][pk]
            print(f"   ↳ detected rotor fundamental ≈ {ROTOR_FREQ_HZ:.2f} Hz")
        else:
            print("   ⚠️  Could not detect rotor frequency; using 20 Hz default")
            ROTOR_FREQ_HZ = 20.0

    omega = 2 * np.pi * ROTOR_FREQ_HZ
    dt    = 1 / fs

    # 6.2  Build state‑space matrices
    n_h = N_HARMONICS
    n_state = 2 * n_h + 1   # (a,b) for each harmonic + slow geology baseline g

    # Transition matrix F
    F = np.eye(n_state)
    for k in range(n_h):
        cos_ = math.cos(omega * (k + 1) * dt)
        sin_ = math.sin(omega * (k + 1) * dt)
        idx  = 2 * k
        F[idx:idx+2, idx:idx+2] = [[cos_, sin_], [-sin_, cos_]]
    # (last row/col already 1 for geology random walk)

    # Observation matrix H  (1 × n_state)
    H = np.zeros((1, n_state))
    H[0, ::2] = 1.0          # pick cosine term of each harmonic
    H[0, -1]  = 1.0          # plus geology baseline

    # Process noise Q
    Q = np.diag([Q_HARM] * (2 * n_h) + [Q_GEO])

    # Measurement noise R
    if R_NOISE is None:
        mad = np.median(np.abs(z - np.median(z)))
        R = (mad / 0.6745) ** 2     # robust variance estimate
    else:
        R = R_NOISE

    # 6.3  Run discrete Kalman filter
    x_est = np.zeros(n_state)          # initial state
    P_est = np.eye(n_state) * 1e3      # large initial covariance

    g_est      = np.empty_like(z)      # geology baseline estimate
    clean_est  = np.empty_like(z)      # = measurement - harmonic interference

    for i, z_k in enumerate(z):
        # Predict
        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + Q

        # Update
        S = H @ P_pred @ H.T + R
        K = (P_pred @ H.T) / S        # Kalman gain (n_state × 1)
        y = z_k - (H @ x_pred)         # innovation
        x_est = x_pred + (K * y).flatten()
        P_est = (np.eye(n_state) - K @ H) @ P_pred

        # Save
        g_est[i]     = x_est[-1]
        harm_sum     = (H[:, :-1] @ x_est[:-1]).item()
        clean_est[i] = z_k - harm_sum   # baseline + everything not modelled as harmonic

    # For comparison we'll call g_est the "KF_clean" series
    z_kf = g_est
    print("   ✅ Kalman filter applied")
    
    # Stage 4: Spike detection on Kalman filtered data
    if SPIKE_DETECTION:
        print("\n🔍  Stage 4: Spike detection on Kalman filtered data …")
        events_kalman, peaks_kalman = detect_spikes(z_kf, 
                                                   median_window=MEDIAN_WINDOW,
                                                   mad_factor=MAD_FACTOR * 5.0)  # Very conservative for clean data
        spikes_kalman = create_spike_dataframe(events_kalman, z_kf, x, df)
        print(f"   ↳ {len(spikes_kalman)} spikes detected on Kalman filtered data")
else:
    z_kf = None

# 5a. Summary statistics
print("\n📊 FILTERING RESULTS")
print(f"   Original std     : {np.std(z):.2f} nT")
print(f"   After notch      : {np.std(z_notch):.2f} nT ({(1 - np.std(z_notch)/np.std(z))*100:.1f}% RMS reduction)")
print(f"   Combined (N+HP)  : {np.std(z_combined):.2f} nT ({(1 - np.std(z_combined)/np.std(z))*100:.1f}% RMS reduction)")
if USE_KALMAN and z_kf is not None:
    print(f"   Kalman filtered  : {np.std(z_kf):.2f} nT ({(1 - np.std(z_kf)/np.std(z))*100:.1f}% RMS reduction)")

# Comprehensive filtering comparison with spike detection results
print("   📈 Generating comprehensive comparison plot with spike detection …")

# Create time series comparison plot with spikes
fig, axes = plt.subplots(4 if USE_KALMAN and z_kf is not None else 3, 1, 
                        figsize=(15, 12), sharex=True)
fig.suptitle("Magnetometer Data: Multi-Stage Filtering with Spike Detection", fontsize=16)

# Convert time to hours for better readability
time_hours = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds() / 3600

# Plot 1: Raw data with spikes
axes[0].plot(time_hours, z, 'b-', alpha=0.7, linewidth=0.8, label='Raw data')
if not spikes_raw.empty:
    spike_times_raw = [(spike_time - df['datetime'].iloc[0]).total_seconds() / 3600 
                       for spike_time in spikes_raw['peak_time']]
    spike_values_raw = spikes_raw['peak_nT'].values 
    axes[0].scatter(spike_times_raw, spike_values_raw, color='red', s=50, 
                   zorder=5, label=f'Spikes ({len(spikes_raw)})', marker='^')
axes[0].set_ylabel('Raw Data (nT)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].set_title(f'Raw Data (σ={np.std(z):.1f} nT)')

# Plot 2: Combined filtered data with spikes
axes[1].plot(time_hours, z_combined, 'g-', alpha=0.7, linewidth=0.8, label='Combined filtered')
if not spikes_combined.empty:
    spike_times_combined = [(spike_time - df['datetime'].iloc[0]).total_seconds() / 3600 
                           for spike_time in spikes_combined['peak_time']]
    spike_values_combined = spikes_combined['peak_nT'].values
    axes[1].scatter(spike_times_combined, spike_values_combined, color='orange', s=50,
                   zorder=5, label=f'Spikes ({len(spikes_combined)})', marker='s')
axes[1].set_ylabel('Combined Filtered (nT)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_title(f'Combined Filtered (σ={np.std(z_combined):.1f} nT)')

# Plot 3: Kalman filtered data with spikes (if available)
if USE_KALMAN and z_kf is not None:
    axes[2].plot(time_hours, z_kf, 'm-', alpha=0.7, linewidth=0.8, label='Kalman filtered')
    if not spikes_kalman.empty:
        spike_times_kalman = [(spike_time - df['datetime'].iloc[0]).total_seconds() / 3600 
                             for spike_time in spikes_kalman['peak_time']]
        spike_values_kalman = spikes_kalman['peak_nT'].values
        axes[2].scatter(spike_times_kalman, spike_values_kalman, color='purple', s=50,
                       zorder=5, label=f'Spikes ({len(spikes_kalman)})', marker='D')
    axes[2].set_ylabel('Kalman Filtered (nT)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_title(f'Kalman Filtered (σ={np.std(z_kf):.1f} nT)')
    
    # Plot 4: Spike count comparison
    spike_counts = [len(spikes_raw), len(spikes_combined), len(spikes_kalman)]
    filter_names = ['Raw', 'Combined', 'Kalman']
    colors = ['red', 'orange', 'purple']
    
    axes[3].bar(filter_names, spike_counts, color=colors, alpha=0.7)
    axes[3].set_ylabel('Number of Spikes')
    axes[3].set_title('Spike Detection Comparison Across Filtering Stages')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    # Add spike count labels on bars
    for i, count in enumerate(spike_counts):
        axes[3].text(i, count + max(spike_counts)*0.02, str(count), 
                    ha='center', va='bottom', fontweight='bold')
else:
    # Plot 3: Spike count comparison (without Kalman)
    spike_counts = [len(spikes_raw), len(spikes_combined)]
    filter_names = ['Raw', 'Combined']
    colors = ['red', 'orange']
    
    axes[2].bar(filter_names, spike_counts, color=colors, alpha=0.7)
    axes[2].set_ylabel('Number of Spikes')
    axes[2].set_title('Spike Detection Comparison Across Filtering Stages')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add spike count labels on bars
    for i, count in enumerate(spike_counts):
        axes[2].text(i, count + max(spike_counts)*0.02, str(count), 
                    ha='center', va='bottom', fontweight='bold')

axes[-1].set_xlabel('Time (hours from start)')
plt.tight_layout()
plt.savefig(OUTDIR / "filtering_stages_with_spikes_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Save spike detection results
print("   💾 Saving spike detection results …")
if not spikes_raw.empty:
    spikes_raw.to_csv(OUTDIR / "spikes_raw_data.csv", index=False)
if not spikes_combined.empty:
    spikes_combined.to_csv(OUTDIR / "spikes_combined_filtered.csv", index=False)
if USE_KALMAN and not spikes_kalman.empty:
    spikes_kalman.to_csv(OUTDIR / "spikes_kalman_filtered.csv", index=False)

# 5b. Comprehensive time-series comparison plot
print("   📈 Generating comprehensive comparison plot …")
time_hours = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds() / 3600

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Top plot: All filtering stages
ax1.plot(time_hours, z,          'b-', alpha=0.7, linewidth=0.8, label='Original')
ax1.plot(time_hours, z_notch,    'orange', linewidth=0.8, label='50 Hz notch')
ax1.plot(time_hours, z_combined, 'r-', linewidth=0.8, label='Notch + 0.1 Hz HP')
if USE_KALMAN and z_kf is not None:
    ax1.plot(time_hours, z_kf,   'green', linewidth=0.8, label='Kalman filtered')

ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Magnetic field (nT)")
ax1.set_title("Multi-Stage Filtering Comparison")
ax1.legend()
ax1.grid(alpha=0.3)

# Bottom plot: Residuals (difference from original)
ax2.plot(time_hours, z_notch - z,    'orange', linewidth=0.8, label='Notch residual')
ax2.plot(time_hours, z_combined - z, 'r-', linewidth=0.8, label='Combined residual')
if USE_KALMAN and z_kf is not None:
    ax2.plot(time_hours, z_kf - z,   'green', linewidth=0.8, label='Kalman residual')

ax2.set_xlabel("Time (h)")
ax2.set_ylabel("Residual (nT)")
ax2.set_title("Filtering Residuals (Filtered - Original)")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTDIR / "comprehensive_filter_comparison.png", dpi=300)
plt.close()

# 5c. Interactive time-series comparison
print("   📊 Generating interactive time series comparison …")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots with secondary y-axis for spike counts
    fig = make_subplots(
        rows=4 if USE_KALMAN and z_kf is not None else 3, 
        cols=1,
        shared_xaxes=True,
        subplot_titles=['Raw Data with Spikes', 'Combined Filtered with Spikes', 
                       'Kalman Filtered with Spikes', 'Spike Count Comparison'] if USE_KALMAN and z_kf is not None
                       else ['Raw Data with Spikes', 'Combined Filtered with Spikes', 'Spike Count Comparison'],
        vertical_spacing=0.08
    )
    
    # Convert datetime to a format Plotly can handle
    time_plotly = df['datetime']
    
    # Plot 1: Raw data with spikes
    fig.add_trace(
        go.Scatter(x=time_plotly, y=z, mode='lines', name='Raw data',
                  line=dict(color='blue', width=1), opacity=0.7),
        row=1, col=1
    )
    
    if not spikes_raw.empty:
        fig.add_trace(
            go.Scatter(x=spikes_raw['peak_time'], y=spikes_raw['peak_nT'],
                      mode='markers', name=f'Raw Spikes ({len(spikes_raw)})',
                      marker=dict(color='red', size=8, symbol='triangle-up')),
            row=1, col=1
        )
    
    # Plot 2: Combined filtered data with spikes
    fig.add_trace(
        go.Scatter(x=time_plotly, y=z_combined, mode='lines', name='Combined filtered',
                  line=dict(color='green', width=1), opacity=0.7),
        row=2, col=1
    )
    
    if not spikes_combined.empty:
        fig.add_trace(
            go.Scatter(x=spikes_combined['peak_time'], y=spikes_combined['peak_nT'],
                      mode='markers', name=f'Combined Spikes ({len(spikes_combined)})',
                      marker=dict(color='orange', size=8, symbol='square')),
            row=2, col=1
        )
    
    # Plot 3: Kalman filtered data with spikes (if available)
    if USE_KALMAN and z_kf is not None:
        fig.add_trace(
            go.Scatter(x=time_plotly, y=z_kf, mode='lines', name='Kalman filtered',
                      line=dict(color='magenta', width=1), opacity=0.7),
            row=3, col=1
        )
        
        if not spikes_kalman.empty:
            fig.add_trace(
                go.Scatter(x=spikes_kalman['peak_time'], y=spikes_kalman['peak_nT'],
                          mode='markers', name=f'Kalman Spikes ({len(spikes_kalman)})',
                          marker=dict(color='purple', size=8, symbol='diamond')),
                row=3, col=1
            )
        
        # Plot 4: Spike count comparison
        spike_counts = [len(spikes_raw), len(spikes_combined), len(spikes_kalman)]
        filter_names = ['Raw', 'Combined', 'Kalman']
        colors = ['red', 'orange', 'purple']
        
        fig.add_trace(
            go.Bar(x=filter_names, y=spike_counts, name='Spike Counts',
                  marker=dict(color=colors), opacity=0.7,
                  text=spike_counts, textposition='outside'),
            row=4, col=1
        )
    else:
        # Plot 3: Spike count comparison (without Kalman)
        spike_counts = [len(spikes_raw), len(spikes_combined)]
        filter_names = ['Raw', 'Combined']
        colors = ['red', 'orange']
        
        fig.add_trace(
            go.Bar(x=filter_names, y=spike_counts, name='Spike Counts',
                  marker=dict(color=colors), opacity=0.7,
                  text=spike_counts, textposition='outside'),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Multi-Stage Filtering with Spike Detection Comparison",
        height=1000 if USE_KALMAN and z_kf is not None else 800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Magnetic Field (nT)", row=1, col=1)
    fig.update_yaxes(title_text="Magnetic Field (nT)", row=2, col=1)
    if USE_KALMAN and z_kf is not None:
        fig.update_yaxes(title_text="Magnetic Field (nT)", row=3, col=1)
        fig.update_yaxes(title_text="Number of Spikes", row=4, col=1)
        fig.update_xaxes(title_text="Time", row=4, col=1)
    else:
        fig.update_yaxes(title_text="Number of Spikes", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=1)
    
    # Save interactive plot
    fig.write_html(OUTDIR / "interactive_filtering_with_spikes_comparison.html")
    print("   ✅ Saved interactive_filtering_with_spikes_comparison.html")
    
except ImportError:
    print("   ⚠️  Plotly not available - skipping interactive time series plot")

# 5c. Regenerate interactive map with filtered residuals
if INTERACTIVE_MAP:
    print("   🗺️  Regenerating interactive map for filtered data …")
    
    # Use Kalman-filtered data if available, otherwise use combined filter
    if USE_KALMAN and z_kf is not None:
        z_final = z_kf
        z_final_residual = z_final - np.mean(z_final)
        filter_name = "Kalman"
        print("   ↳ Using Kalman-filtered data for final map")
    else:
        z_final = z_combined
        z_final_residual = z_combined - np.mean(z_combined)
        filter_name = "Combined (Notch + HP)"
        print("   ↳ Using combined-filtered data for final map")

    # Create interpolated grid for final filtered data
    if INTERPOLATE_FIELD:
        print(f"   🗺  Gridding {filter_name.lower()} filtered field …")
        # Use same grid setup as original
        if {'UTM_Easting', 'UTM_Northing'}.issubset(df.columns):
            X = df['UTM_Easting'].values
            Y = df['UTM_Northing'].values
        else:
            X = df['Longitude [Decimal Degrees]'].values
            Y = df['Latitude  [Decimal Degrees]'].values

        # Use same grid as before
        xi = np.linspace(X.min(), X.max(), GRID_RES)
        yi = np.linspace(Y.min(), Y.max(), GRID_RES)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate final filtered data
        Zi_final = griddata((X, Y), z_final_residual, (Xi, Yi), method='cubic')

        # Convert grid to lat/lon for Folium
        if {'UTM_Easting', 'UTM_Northing'}.issubset(df.columns):
            transformer = Transformer.from_crs("epsg:32633", "epsg:4326", always_xy=True)
            LonGrid_final, LatGrid_final = transformer.transform(Xi, Yi)
        else:
            LonGrid_final, LatGrid_final = Xi, Yi

        # Create final field points
        field_points_final = [
            (lat, lon, val)
            for lat, lon, val in zip(LatGrid_final.ravel(), LonGrid_final.ravel(), Zi_final.ravel())
            if not np.isnan(val)
        ]

        # Save static PNG for final filtered field
        interp_final_png = OUTDIR / f"field_interpolated_{filter_name.lower().replace(' ', '_')}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        im = ax.imshow(Zi_final, extent=extent, origin='lower', cmap='viridis')
        fig.colorbar(im, label=f"{filter_name} Filtered {COLUMN}")
        ax.set_title(f"Interpolated {filter_name.lower()} filtered field")
        ax.set_xlabel("Easting / Longitude")
        ax.set_ylabel("Northing / Latitude")
        fig.tight_layout()
        fig.savefig(interp_final_png, dpi=300)
        plt.close()

    # Enhanced interactive map with final filtered data
    m_final = folium.Map(
        location=[df["Latitude [Decimal Degrees]"].mean(),
                  df["Longitude [Decimal Degrees]"].mean()],
        zoom_start=15, tiles='OpenStreetMap'
    )

    # Add interpolated final filtered field if enabled
    if INTERPOLATE_FIELD:
        field_final_fg = folium.FeatureGroup(name=f"Interpolated {filter_name} field")
        cm_field_final = cm_linear(
            ['#440154', '#31688e', '#35b779', '#fde725'],
            vmin=Zi_final[np.isfinite(Zi_final)].min(),
            vmax=Zi_final[np.isfinite(Zi_final)].max(),
            caption=f"Interpolated {filter_name} field"
        )
        for lat, lon, val in field_points_final:
            folium.CircleMarker(
                location=(lat, lon),
                radius=2, color=cm_field_final(val),
                fill=True, fill_opacity=0.6, opacity=0.6
            ).add_to(field_final_fg)
        m_final.add_child(field_final_fg)

    # Add survey path coloured by final filtered residual value
    cm_final = cm_linear(['#440154', '#31688e', '#35b779', '#fde725'],
                        vmin=z_final_residual.min(),
                        vmax=z_final_residual.max(),
                        caption=f"{filter_name} filtered residual (nT)")
    
    # Sample path points
    subsample = df.iloc[::MAP_SAMPLE_STRIDE]
    path_points = []
    for i in subsample.index:
        val = z_final_residual[i]
        lat = df.loc[i, "Latitude [Decimal Degrees]"]
        lon = df.loc[i, "Longitude [Decimal Degrees]"]
        path_points.append((lat, lon))
        folium.CircleMarker(
            location=(lat, lon),
            radius=3, color=cm_final(val), fill=True,
            fill_opacity=0.8, opacity=0.8
        ).add_to(m_final)
    
    # Add path line
    folium.PolyLine(path_points, color="gray", weight=2, opacity=0.4,
                    tooltip="Survey path").add_to(m_final)
    
    cm_final.add_to(m_final)

    # Add spikes from all detection stages to final map
    if not spikes_raw.empty:
        spike_raw_group = folium.FeatureGroup(name="Raw Data Spikes")
        for _, row in spikes_raw.iterrows():
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=8, color="red", fill=True, fill_opacity=0.9,
                tooltip=f"RAW: {row.peak_time} | {row.peak_nT:+.1f} nT"
            ).add_to(spike_raw_group)
        m_final.add_child(spike_raw_group)

    # Add combined filtered spikes to final map
    if not spikes_combined.empty:
        spike_combined_group = folium.FeatureGroup(name="Combined Filtered Spikes")
        for _, row in spikes_combined.iterrows():
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=6, color="orange", fill=True, fill_opacity=0.8,
                tooltip=f"COMBINED: {row.peak_time} | {row.peak_nT:+.1f} nT"
            ).add_to(spike_combined_group)
        m_final.add_child(spike_combined_group)

    # Add Kalman filtered spikes to final map (if available)
    if USE_KALMAN and not spikes_kalman.empty:
        spike_kalman_group = folium.FeatureGroup(name="Kalman Filtered Spikes")
        for _, row in spikes_kalman.iterrows():
            folium.CircleMarker(
                location=(row.latitude, row.longitude),
                radius=4, color="purple", fill=True, fill_opacity=0.7,
                tooltip=f"KALMAN: {row.peak_time} | {row.peak_nT:+.1f} nT"
            ).add_to(spike_kalman_group)
        m_final.add_child(spike_kalman_group)

    # 2D Field Spike Detection on Interpolated Grid
    if SPIKE_DETECTION and INTERPOLATE_FIELD:
        print("   🗺️  2D field spike detection on interpolated grid …")
        
        def detect_2d_field_spikes(grid_data, grid_lon, grid_lat, mad_factor=3.0):
            """Detect anomalies/spikes in 2D interpolated field using spatial statistics"""
            # Flatten valid (non-NaN) grid points
            valid_mask = np.isfinite(grid_data)
            if not np.any(valid_mask):
                return []
            
            valid_values = grid_data[valid_mask]
            valid_lons = grid_lon[valid_mask]
            valid_lats = grid_lat[valid_mask]
            
            # Calculate local median using a sliding window approach
            from scipy.ndimage import median_filter
            
            # Apply median filter to smooth local variations
            filtered_grid = median_filter(grid_data, size=5, mode='constant', cval=np.nan)
            
            # Calculate residuals (difference from local median)
            residuals = grid_data - filtered_grid
            
            # Calculate MAD threshold on valid residuals
            valid_residuals = residuals[valid_mask]
            median_residual = np.nanmedian(valid_residuals)
            mad_residual = np.nanmedian(np.abs(valid_residuals - median_residual))
            
            if mad_residual == 0:
                return []
            
            # Detect spikes using MAD criterion
            threshold = mad_factor * mad_residual
            spike_mask = np.abs(valid_residuals - median_residual) > threshold
            
            # Extract spike locations and values
            spike_lons = valid_lons[spike_mask]
            spike_lats = valid_lats[spike_mask]
            spike_values = valid_values[spike_mask]
            spike_residuals = valid_residuals[spike_mask]
            
            # Create spike list
            field_spikes = []
            for lon, lat, val, res in zip(spike_lons, spike_lats, spike_values, spike_residuals):
                field_spikes.append({
                    'longitude': lon,
                    'latitude': lat,
                    'field_value': val,
                    'residual': res,
                    'spike_type': 'field_anomaly'
                })
            
            return field_spikes
        
        # Detect 2D field spikes on the final filtered grid
        field_spikes = detect_2d_field_spikes(Zi_final, LonGrid_final, LatGrid_final, 
                                             mad_factor=5.0)  # Much more conservative for 2D detection
        
        if field_spikes:
            print(f"   ↳ {len(field_spikes)} field anomalies detected in interpolated grid")
            
            # Convert to DataFrame for consistency
            spikes_field_2d = pd.DataFrame(field_spikes)
            
            # Add field spikes to the interactive map
            if not spikes_field_2d.empty:
                field_spike_group = folium.FeatureGroup(name="Field Anomalies (2D)")
                for _, row in spikes_field_2d.iterrows():
                    # Color based on anomaly strength
                    anomaly_strength = abs(row['residual'])
                    if anomaly_strength > np.percentile([abs(r) for r in spikes_field_2d['residual']], 90):
                        color = 'darkred'
                        radius = 6
                    elif anomaly_strength > np.percentile([abs(r) for r in spikes_field_2d['residual']], 70):
                        color = 'red'
                        radius = 5
                    else:
                        color = 'pink'
                        radius = 4
                        
                    folium.CircleMarker(
                        location=(row.latitude, row.longitude),
                        radius=radius, color=color, fill=True, fill_opacity=0.6,
                        tooltip=f"FIELD ANOMALY: {row.field_value:+.1f} nT (res: {row.residual:+.1f})"
                    ).add_to(field_spike_group)
                m_final.add_child(field_spike_group)
            
            # Save field spike results
            spikes_field_2d.to_csv(OUTDIR / "spikes_field_2d_anomalies.csv", index=False)
            print(f"   💾 Saved field anomalies to spikes_field_2d_anomalies.csv")
        else:
            print("   ↳ no significant field anomalies detected")
            spikes_field_2d = pd.DataFrame()

    # Add layer control
    m_final.add_child(folium.LayerControl())

    # Add title with filtering information
    filter_stages = ["50 Hz notch", "0.1 Hz HP"]
    if USE_KALMAN:
        filter_stages.append("Kalman harmonic filter")
    
    m_final.get_root().html.add_child(
        folium.Element(
            '<h3 style="text-align:center; font-size:20px">'
            f'Enhanced {filter_name} Filtered Magnetic Field Map<br>'
            f'<small>({" + ".join(filter_stages)} + Two-stage spike detection)</small>'
            '</h3>'
        )
    )
    
    # Save with appropriate filename
    final_map_name = f"interactive_map_{filter_name.lower().replace(' ', '_')}_filtered.html"
    m_final.save(OUTDIR / final_map_name)
    print(f"   ✅ Saved {final_map_name}")

print("\n✅  All done – outputs in", OUTDIR)