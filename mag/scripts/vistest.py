#!/usr/bin/env python3
import math 
import functools
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
# -------------------------------------------------------------------
# 🛠  CONFIG  ── edit these eight lines and run the script
# -------------------------------------------------------------------
from pathlib import Path
CSV_PATH       = Path("/Users/aleksandergarbuz/Documents/SINTEF/data/20250611_081139_MWALK_#0122_processed_20250616_094044.csv")  
OUTDIR         = Path("/Users/aleksandergarbuz/Documents/SINTEF/data/081139p")         
COLUMN         = "R2 [nT]"              

# crop (% rows to drop @ start/end)
CUT_START       = 0.30                       
CUT_END         = 0.065                      

# spike detection
MAD_FACTOR      = 5.0                        
MEDIAN_WINDOW   = 301                       

# static scatter
LINTHRESH       = 50.0                       

# satellite basemap for static PNG
SATELLITE_MAP   = True                       
MAP_PROVIDER    = "Esri.WorldImagery"        
MAP_ZOOM        = None                       

# interactive map (Folium)
INTERACTIVE_MAP = True                       
MAP_SAMPLE_STRIDE = 20                       

# PSD
COMPUTE_PSD         = True                   
PSD_NPERSEG         = 4096                  
PEAK_PROMINENCE     = 1e3                  
CHECK_BANDPOWERS    = True                 
BAND_RANGES         = [                     
    (0, 0.5,  "ultra_low"),
    (0.5, 5,  "low"),
    (5, 25,   "mid"),
    (25, 60,  "mains"),
    (60, 200, "high"),
]                

# ⇢ FIELD INTERPOLATION (add just below the PSD settings) ─────────────
INTERPOLATE_FIELD = True      # turn gridding on/off
GRID_RES          = 200       # ≈ number of cells per side of the square grid
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
                raise ValueError(f"Provider '{key}' not found. Try one of:\n  " + '\n  '.join(valid[:40]) + '\n  …')

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
    df["GPSTime [hh:mm:ss.sss]"].astype(str).str.strip(), errors="coerce")
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
fig, ax = plt.subplots(figsize=(12,4))
sc = ax.scatter(x, z, c=z, s=4,
                cmap="viridis",
                norm=colors.SymLogNorm(linthresh=LINTHRESH,
                                        vmin=z.min(), vmax=z.max()))
fig.colorbar(sc, label=COLUMN)
ax.set_xlabel("Time since start (s)")
ax.set_ylabel(COLUMN)
ax.set_title(f"{COLUMN} – symlog (linthresh={LINTHRESH})")
fig.tight_layout()
fig.savefig(OUTDIR/"symlog_scatter.png", dpi=300)
plt.close()

# ── 2b. OPTIONAL – create a gridded / interpolated residual layer ──────────
if INTERPOLATE_FIELD:
    print("🗺  Gridding residual field …")
    # 3a. choose X/Y coordinates (prefer UTM, fall back to lon/lat)
    if {'UTM_Easting', 'UTM_Northing'}.issubset(df.columns):
        X  = df['UTM_Easting' ].values
        Y  = df['UTM_Northing'].values
        lon = df['Longitude [Decimal Degrees]'];  lat = df['Latitude [Decimal Degrees]']
        interp_bounds = [[lat.min(), lon.min()], [lat.max(), lon.max()]]   # for Folium
        extent = [X.min(), X.max(), Y.min(), Y.max()]                      # for imshow
    else:
        X  = df['Longitude [Decimal Degrees]'].values
        Y  = df['Latitude  [Decimal Degrees]'].values
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
    field_points = [(lat, lon, val)
                    for lat, lon, val in zip(LatGrid.ravel(),
                                             LonGrid.ravel(),
                                             Zi.ravel())
                    if not np.isnan(val)]

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
# 3. SPIKE CATALOGUE
# ════════════════════════════════════════════
print("🚨  Spike detection …")
trend     = sig.medfilt(z, MEDIAN_WINDOW)
resid     = z - trend
mad       = np.median(np.abs(resid - np.median(resid)))
threshold = MAD_FACTOR * mad
idx       = np.where(np.abs(resid) > threshold)[0]

events, spikes = [], pd.DataFrame()
if idx.size:
    cur=[idx[0]]
    for i in idx[1:]:
        (cur.append(i) if i==cur[-1]+1 else (events.append(cur),cur:=[i]))
    events.append(cur)
    dt=np.median(np.diff(x))
    spikes=pd.DataFrame([
        {"start_time":df["datetime"].iloc[e[0]],
         "end_time"  :df["datetime"].iloc[e[-1]],
         "duration_s":(e[-1]-e[0]+1)*dt,
         "peak_nT"   :z[e[np.argmax(np.abs(z[e]))]],
         "peak_time" :df["datetime"].iloc[e[np.argmax(np.abs(z[e]))]],
         "latitude"  :df["Latitude [Decimal Degrees]"].iloc[e[np.argmax(np.abs(z[e]))]],
         "longitude" :df["Longitude [Decimal Degrees]"].iloc[e[np.argmax(np.abs(z[e]))]],
         "n_samples" :len(e)} for e in events])
    spikes.to_csv(OUTDIR/"spike_catalogue.csv", index=False)
    print(f"   ↳ {len(spikes)} spikes ➜ spike_catalogue.csv")
else:
    print("   ↳ no spikes detected")

# ════════════════════════════════════════════
# 4. MAPS
# ════════════════════════════════════════════
print("🗺️  Building maps …")
if not spikes.empty:
    gdf = gpd.GeoDataFrame(spikes,
            geometry=gpd.points_from_xy(spikes.longitude, spikes.latitude),
            crs="EPSG:4326")

    # ---- static PNG ----
    if SATELLITE_MAP:
        g3857=gdf.to_crs(epsg=3857)
        if MAP_ZOOM is None:
            lat_c = gdf.latitude.mean()
            width_m = (g3857.total_bounds[2]-g3857.total_bounds[0])
            z_est=int(round(math.log2((156543.0339*math.cos(math.radians(lat_c))*256)/(width_m/12))))
            MAP_ZOOM=max(3,min(19,z_est))
        fig,ax=plt.subplots(figsize=(8,8))
        g3857.plot(ax=ax,column="peak_nT", cmap="viridis",markersize=90,legend=True,alpha=0.95)

        try:
            ctx.add_basemap(ax, source=provider, zoom=MAP_ZOOM, crs="EPSG:3857")
        except Exception as e:
            print("⚠️ basemap fetch failed:",e)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(OUTDIR/"spike_map.png", dpi=300)
        plt.close()
    else:
        fig,ax=plt.subplots(figsize=(6,6))
        gdf.plot(ax=ax,column="peak_nT", cmap="viridis", markersize=90,legend=True,alpha=0.95)
        ax.set_axis_off(); fig.tight_layout(); fig.savefig(OUTDIR/"spike_map.png", dpi=300); plt.close()

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
        pts = list(zip(sample["Latitude [Decimal Degrees]"], sample["Longitude [Decimal Degrees]"], sample[COLUMN]))
        folium.PolyLine([(lat,lon) for lat,lon,_ in pts], color="gray", weight=2, opacity=0.4, tooltip="Survey path").add_to(fmap)
        for lat,lon,val in pts:
            folium.CircleMarker(location=(lat,lon), radius=3, color=cm(val), fill=True, fill_opacity=0.7, opacity=0.7).add_to(fmap)
        cm.caption=f"{COLUMN} (path samples)"; fmap.add_child(cm)
        # spikes layer
        spike_group = folium.FeatureGroup(name="Spikes")
        for _,row in gdf.iterrows():
            folium.CircleMarker(location=(row.latitude,row.longitude), radius=8,
                color="red", fill=True, fill_opacity=0.9,
                tooltip=f"{row.peak_time} | {row.peak_nT:+.1f} nT").add_to(spike_group)

        if INTERPOLATE_FIELD:
            field_fg = folium.FeatureGroup(name="Interpolated field")
            cm_field = cm_linear(['#440154', '#31688e', '#2a788e', '#35b779', '#fde725'], 
                                 vmin=Zi[np.isfinite(Zi)].min(),
                                 vmax=Zi[np.isfinite(Zi)].max(),
                                 caption="Interpolated field")
            for lat, lon, val in field_points:
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=2,               # tweak size if needed
                    color=cm_field(val),
                    fill=True, fill_opacity=0.8, opacity=0.8
                ).add_to(field_fg)
            fmap.add_child(field_fg)
        
        fmap.add_child(spike_group)
        fmap.add_child(folium.LayerControl())
        fmap.save(OUTDIR/"interactive_spike_map.html")

else:
    print("   ↳ no spikes: skipping maps")

# ════════════════════════════════════════════
# 5. PSD DEEP‑DIVE
# ════════════════════════════════════════════
if COMPUTE_PSD:
    print("🔍  PSD analytics …")
    f,Pxx=sig.welch(z, fs, nperseg=PSD_NPERSEG, window='hann', scaling="density")
    # optional clean‑signal PSD (spikes removed)
    clean=z.copy()
    if idx.size:
        clean[idx]=trend[idx]
    f2,Pxx2=sig.welch(clean, fs, nperseg=PSD_NPERSEG, window='hann', scaling="density")

    # log‑log slopes (least‑squares) for ultra‑low <0.1 Hz
    ul_mask=f<0.1
    # Ensure we have enough points and valid data for reliable fit
    f_ul = f[ul_mask][1:]  # skip DC component
    Pxx_ul = Pxx[ul_mask][1:]
    f2_ul = f2[ul_mask][1:]
    Pxx2_ul = Pxx2[ul_mask][1:]
    
    if len(f_ul) >= 3 and np.all(np.isfinite(f_ul)) and np.all(np.isfinite(Pxx_ul)) and np.all(Pxx_ul > 0):
        try:
            slope_raw = np.polyfit(np.log10(f_ul), np.log10(Pxx_ul), 1)[0]
        except np.RankWarning:
            slope_raw = np.nan
        except Exception:
            slope_raw = np.nan
    else:
        slope_raw = np.nan
        
    if len(f2_ul) >= 3 and np.all(np.isfinite(f2_ul)) and np.all(np.isfinite(Pxx2_ul)) and np.all(Pxx2_ul > 0):
        try:
            slope_clean = np.polyfit(np.log10(f2_ul), np.log10(Pxx2_ul), 1)[0]
        except np.RankWarning:
            slope_clean = np.nan
        except Exception:
            slope_clean = np.nan
    else:
        slope_clean = np.nan
        
    print(f"    ◾ low‑freq slope raw  : {slope_raw:.2f}" if not np.isnan(slope_raw) else "    ◾ low‑freq slope raw  : N/A")
    print(f"    ◾ low‑freq slope clean: {slope_clean:.2f}" if not np.isnan(slope_clean) else "    ◾ low‑freq slope clean: N/A")

    # peak list
    peaks,props=sig.find_peaks(Pxx, prominence=PEAK_PROMINENCE)
    print("    prominent peaks (Hz | prom):")
    for pk,prom in zip(f[peaks], props['prominences']):
        print(f"      {pk:6.2f} | {prom:.1f}")

    # band powers
    if CHECK_BANDPOWERS:
        rows=[]
        for lo,hi,lab in BAND_RANGES:
            m=(f>=lo)&(f<=hi)
            m2=(f2>=lo)&(f2<=hi)
            power_raw=scipy.integrate.trapezoid(Pxx[m], f[m])
            power_clean=scipy.integrate.trapezoid(Pxx2[m2], f2[m2])
            rows.append({"band":lab, "lo_Hz":lo, "hi_Hz":hi,
                         "power_raw":power_raw, "power_clean":power_clean})
        bp=pd.DataFrame(rows)
        bp.to_csv(OUTDIR/"psd_bandpowers.csv", index=False)
        print("    bandpower table ➜ psd_bandpowers.csv")

    # plots
    fig,ax=plt.subplots(figsize=(7,4))
    ax.semilogy(f,Pxx,label="raw")
    ax.semilogy(f2,Pxx2,label="clean",linestyle="--")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (nT²/Hz)")
    ax.set_title("Welch PSD – raw vs clean")
    ax.set_xlim(0,fs/2); ax.legend(); fig.tight_layout()
    plt.savefig(OUTDIR/"welch_psd_raw_clean.png", dpi=300); plt.close()

    # ENHANCED PSD ANALYSIS - Detailed frequency characterization
    print("\n    🔬 DETAILED PSD ANALYSIS:")
    
    # 1. Extract ALL significant peaks (lower threshold for comprehensive analysis)
    low_prominence = PEAK_PROMINENCE * 0.1  # 10% of main threshold
    all_peaks, all_props = sig.find_peaks(Pxx, prominence=low_prominence, height=np.percentile(Pxx, 90))
    
    print(f"    📊 Found {len(all_peaks)} significant peaks (prominence > {low_prominence:.1f}):")
    
    # Sort peaks by prominence for analysis
    peak_data = [(f[i], Pxx[i], all_props['prominences'][j]) 
                 for j, i in enumerate(all_peaks)]
    peak_data.sort(key=lambda x: x[2], reverse=True)  # Sort by prominence
    
    # Display top 10 most prominent peaks
    print("    🏔️  TOP 10 PEAKS (frequency | power | prominence):")
    for i, (freq, power, prom) in enumerate(peak_data[:10]):
        print(f"       {i+1:2d}. {freq:8.3f} Hz | {power:12.1f} | {prom:12.1f}")
    
    # 2. Analyze the dominant low-frequency peak (around 0.06 Hz)
    print("\n    🌊 LOW-FREQUENCY ANALYSIS (0-1 Hz):")
    lf_mask = f <= 1.0
    lf_peaks_idx = [i for i in all_peaks if lf_mask[i]]
    
    if lf_peaks_idx:
        lf_peak_data = [(f[i], Pxx[i], all_props['prominences'][np.where(all_peaks == i)[0][0]]) 
                       for i in lf_peaks_idx]
        lf_peak_data.sort(key=lambda x: x[1], reverse=True)  # Sort by power
        
        print(f"       Found {len(lf_peak_data)} low-frequency peaks:")
        for i, (freq, power, prom) in enumerate(lf_peak_data[:5]):
            print(f"       {i+1}. {freq:6.4f} Hz | Power: {power:10.1f} | Prom: {prom:8.1f}")
            
        # Analyze the biggest low-freq peak
        dominant_lf = lf_peak_data[0]
        print(f"       🎯 DOMINANT: {dominant_lf[0]:.4f} Hz accounts for {dominant_lf[1]/np.sum(Pxx[lf_mask])*100:.1f}% of 0-1Hz power")
    
    # 3. Mains frequency analysis (45-55 Hz focusing on 50 Hz)
    print("\n    ⚡ MAINS FREQUENCY ANALYSIS (45-55 Hz):")
    mains_mask = (f >= 45) & (f <= 55)
    mains_peaks_idx = [i for i in all_peaks if mains_mask[i]]
    
    if mains_peaks_idx:
        mains_peak_data = [(f[i], Pxx[i], all_props['prominences'][np.where(all_peaks == i)[0][0]]) 
                          for i in mains_peaks_idx]
        mains_peak_data.sort(key=lambda x: x[1], reverse=True)
        
        print(f"       Found {len(mains_peak_data)} peaks in mains band:")
        for i, (freq, power, prom) in enumerate(mains_peak_data):
            dist_from_50 = abs(freq - 50.0)
            print(f"       {i+1}. {freq:6.2f} Hz | Power: {power:8.1f} | Δ from 50Hz: {dist_from_50:4.2f}")
            
        # Check for 50 Hz harmonic series
        harmonics_50 = [50, 100, 150]
        print("       🎵 50Hz HARMONICS CHECK:")
        for harm in harmonics_50:
            if harm <= f.max():
                harm_idx = np.argmin(np.abs(f - harm))
                harm_power = Pxx[harm_idx]
                print(f"         {harm:3d}Hz: {harm_power:8.1f} nT²/Hz (at {f[harm_idx]:.2f}Hz)")
    else:
        print("       No significant peaks found in 45-55 Hz range")
    
    # 4. High-frequency analysis (>100 Hz)
    print("\n    📡 HIGH-FREQUENCY ANALYSIS (>100 Hz):")
    hf_mask = f > 100
    hf_peaks_idx = [i for i in all_peaks if hf_mask[i]]
    
    if hf_peaks_idx:
        hf_peak_data = [(f[i], Pxx[i], all_props['prominences'][np.where(all_peaks == i)[0][0]]) 
                       for i in hf_peaks_idx]
        hf_peak_data.sort(key=lambda x: x[1], reverse=True)
        
        print(f"       Found {len(hf_peak_data)} high-frequency peaks:")
        for i, (freq, power, prom) in enumerate(hf_peak_data[:5]):
            print(f"       {i+1}. {freq:6.1f} Hz | Power: {power:8.1f} | Prom: {prom:6.1f}")
    else:
        print("       No significant peaks found above 100 Hz")
    
    # 5. Power distribution analysis
    total_power = scipy.integrate.trapezoid(Pxx, f)
    print("\n    ⚖️  POWER DISTRIBUTION:")
    print(f"       Total power: {total_power:.1f} nT²/Hz")
    
    freq_bands = [
        (0, 0.1, "Ultra-low (<0.1Hz)"),
        (0.1, 1, "Very low (0.1-1Hz)"), 
        (1, 10, "Low (1-10Hz)"),
        (10, 50, "Mid (10-50Hz)"),
        (50, 100, "Mains+harmonics (50-100Hz)"),
        (100, f.max(), "High (>100Hz)")
    ]
    
    for lo, hi, label in freq_bands:
        band_mask = (f >= lo) & (f <= hi)
        if np.any(band_mask):
            band_power = scipy.integrate.trapezoid(Pxx[band_mask], f[band_mask])
            band_pct = band_power / total_power * 100
            print(f"       {label:25s}: {band_power:8.1f} nT²/Hz ({band_pct:5.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # DETAILED PSD PLOTS - Multiple frequency-focused visualizations
    # ═══════════════════════════════════════════════════════════════════
    print("\n    📊 GENERATING DETAILED PSD PLOTS...")
    
    # Create a multi-panel figure for detailed PSD analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Detailed PSD Analysis - Multiple Frequency Ranges", fontsize=14)
    
    # Plot 1: Full spectrum (log-log)
    ax = axes[0, 0]
    ax.loglog(f[1:], Pxx[1:], 'b-', alpha=0.7, label='Raw')
    ax.loglog(f2[1:], Pxx2[1:], 'r--', alpha=0.7, label='Clean')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (nT²/Hz)')
    ax.set_title('Full Spectrum (Log-Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Low frequency zoom (0-1 Hz)
    ax = axes[0, 1]
    lf_mask_plot = f <= 1.0
    ax.semilogy(f[lf_mask_plot], Pxx[lf_mask_plot], 'b-', linewidth=2, label='Raw')
    ax.semilogy(f2[lf_mask_plot], Pxx2[lf_mask_plot], 'r--', linewidth=2, label='Clean')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (nT²/Hz)')
    ax.set_title('Low Frequency Detail (0-1 Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark the dominant low-frequency peak
    if lf_peaks_idx:
        dominant_freq = f[lf_peaks_idx[np.argmax([Pxx[i] for i in lf_peaks_idx])]]
        dominant_power = np.max([Pxx[i] for i in lf_peaks_idx])
        ax.plot(dominant_freq, dominant_power, 'ro', markersize=8, label=f'{dominant_freq:.4f} Hz')
        ax.legend()
    
    # Plot 3: Mains frequency zoom (45-55 Hz)
    ax = axes[0, 2]
    mains_mask_plot = (f >= 45) & (f <= 55)
    if np.any(mains_mask_plot):
        ax.plot(f[mains_mask_plot], Pxx[mains_mask_plot], 'b-', linewidth=2, label='Raw')
        ax.plot(f2[mains_mask_plot], Pxx2[mains_mask_plot], 'r--', linewidth=2, label='Clean')
        ax.axvline(50, color='k', linestyle=':', alpha=0.7, label='50 Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (nT²/Hz)')
        ax.set_title('Mains Frequency Detail (45-55 Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark any peaks in mains band
        if mains_peaks_idx:
            for peak_idx in mains_peaks_idx:
                ax.plot(f[peak_idx], Pxx[peak_idx], 'ro', markersize=6)
    else:
        ax.text(0.5, 0.5, 'No data in 45-55 Hz range', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Medium frequency (1-25 Hz)
    ax = axes[1, 0]
    mid_mask_plot = (f >= 1) & (f <= 25)
    ax.semilogy(f[mid_mask_plot], Pxx[mid_mask_plot], 'b-', linewidth=2, label='Raw')
    ax.semilogy(f2[mid_mask_plot], Pxx2[mid_mask_plot], 'r--', linewidth=2, label='Clean')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (nT²/Hz)')
    ax.set_title('Mid Frequency Detail (1-25 Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: High frequency (50-200 Hz)
    ax = axes[1, 1]
    hf_mask_plot = (f >= 50) & (f <= 200)
    if np.any(hf_mask_plot):
        ax.semilogy(f[hf_mask_plot], Pxx[hf_mask_plot], 'b-', linewidth=2, label='Raw')
        ax.semilogy(f2[hf_mask_plot], Pxx2[hf_mask_plot], 'r--', linewidth=2, label='Clean')
        
        # Mark harmonics of 50 Hz
        for harm in [50, 100, 150]:
            if harm <= 200:
                ax.axvline(harm, color='gray', linestyle=':', alpha=0.5, label=f'{harm}Hz' if harm == 50 else '')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (nT²/Hz)')
        ax.set_title('High Frequency Detail (50-200 Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark peaks in high-frequency band
        if hf_peaks_idx:
            hf_in_range = [i for i in hf_peaks_idx if f[i] <= 200]
            for peak_idx in hf_in_range[:5]:  # Mark top 5 peaks
                ax.plot(f[peak_idx], Pxx[peak_idx], 'ro', markersize=4)
    else:
        ax.text(0.5, 0.5, 'No data in 50-200 Hz range', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 6: Peak summary histogram
    ax = axes[1, 2]
    if len(peak_data) > 0:
        # Create frequency bins for peak distribution
        freq_bins = [0, 0.1, 1, 10, 50, 100, f.max()]
        bin_labels = ['<0.1', '0.1-1', '1-10', '10-50', '50-100', f'>100']
        bin_counts = [0] * (len(freq_bins) - 1)
        
        for freq, _, _ in peak_data:
            for i in range(len(freq_bins) - 1):
                if freq_bins[i] <= freq < freq_bins[i + 1]:
                    bin_counts[i] += 1
                    break
        
        bars = ax.bar(range(len(bin_counts)), bin_counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Frequency Range (Hz)')
        ax.set_ylabel('Number of Peaks')
        ax.set_title('Peak Distribution by Frequency Band')
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'No peaks found', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(OUTDIR/"detailed_psd_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("       ✅ Detailed PSD plots saved to detailed_psd_analysis.png")

    # ═══════════════════════════════════════════════════════════════════
    # ADVANCED LOW-FREQUENCY ANALYSIS & FILTERING INVESTIGATION
    # ═══════════════════════════════════════════════════════════════════
    print("\n    🔬 ADVANCED LOW-FREQUENCY ANALYSIS:")
    
    # 1. Time-domain analysis of the 0.061 Hz component
    print("    🕐 TIME-DOMAIN CHARACTERISTICS:")
    
    # Extract time series for analysis
    time_seconds = np.arange(len(z)) / fs
    
    # Check if 0.061 Hz component shows characteristics of drift vs oscillation
    # Bandpass filter around 0.061 Hz to isolate the component
    from scipy.signal import butter, filtfilt, hilbert
    
    # Design very narrow bandpass around 0.061 Hz (±0.01 Hz)
    lowcut = 0.051
    highcut = 0.071
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if high < 1.0:  # Check if frequencies are within Nyquist limit
        try:
            b, a = butter(4, [low, high], btype='band')
            lf_component = filtfilt(b, a, z)
            
            # Check if filter actually isolated meaningful signal
            if np.any(np.abs(lf_component) > 1e-10):  # Avoid division by zero
                # Analyze the isolated low-frequency component
                lf_amplitude = np.abs(hilbert(lf_component))
                lf_phase = np.angle(hilbert(lf_component))
                
                print("       Isolated 0.061Hz component:")
                print(f"         RMS amplitude: {np.sqrt(np.mean(lf_component**2)):.2f} nT")
                print(f"         Peak amplitude: {np.max(np.abs(lf_component)):.2f} nT")
                
                # Safe amplitude stability calculation
                mean_amp = np.mean(lf_amplitude)
                if mean_amp > 1e-10:
                    amp_stability = np.std(lf_amplitude) / mean_amp
                    print(f"         Amplitude stability (std/mean): {amp_stability:.3f}")
                else:
                    print("         Amplitude stability: N/A (too weak signal)")
                
                # Check for phase coherence (consistent oscillation vs random walk)
                phase_diff = np.diff(np.unwrap(lf_phase))
                if len(phase_diff) > 0 and np.std(phase_diff) > 0:
                    phase_coherence = max(0, 1 - np.std(phase_diff) / (2 * np.pi))
                    print(f"         Phase coherence: {phase_coherence:.3f} (1=perfect oscillation, 0=random)")
                else:
                    phase_coherence = 0
                    print("         Phase coherence: N/A (insufficient phase variation)")
            else:
                print("       ⚠️  0.061Hz bandpass filter removed all signal - component too weak to isolate")
                phase_coherence = 0
                
        except Exception as e:
            print(f"       ❌ Failed to isolate 0.061Hz component: {str(e)}")
            phase_coherence = 0
            
    else:
        print("       ❌ 0.061Hz frequency exceeds Nyquist limit")
        phase_coherence = 0
    
    # Drift analysis - check if there's underlying trend
    from scipy import stats
    trend_slope, _, trend_r, trend_p, _ = stats.linregress(time_seconds, z)
    print(f"         Overall drift: {trend_slope*3600:.3f} nT/hour (R²={trend_r**2:.3f}, p={trend_p:.3e})")
    
    # Classification suggestion
    if phase_coherence > 0.7 and trend_r**2 < 0.1:
        classification = "🌊 LIKELY GEOLOGICAL SIGNAL"
    elif trend_r**2 > 0.3 or phase_coherence < 0.3:
        classification = "📉 LIKELY INSTRUMENTAL DRIFT"
    else:
        classification = "❓ MIXED/UNCERTAIN"
    
    print(f"         Classification: {classification}")
    
    # 2. Aliasing Investigation
    print("\n    🔄 ALIASING ANALYSIS:")
    print(f"       Sampling rate: {fs:.3f} Hz")
    print(f"       Nyquist frequency: {fs/2:.3f} Hz")
    
    # Check for potential aliasing signatures
    # Look for suspicious peaks near Nyquist or that could be aliases
    high_freq_mask = f > (fs/2 - 1)  # Near Nyquist
    if np.any(high_freq_mask):
        near_nyquist_power = np.max(Pxx[high_freq_mask])
        print(f"       Power near Nyquist: {near_nyquist_power:.2f} nT²/Hz")
        if near_nyquist_power > np.percentile(Pxx, 95):
            print("       ⚠️  WARNING: High power near Nyquist - possible aliasing")
    
    # Check if 0.061 Hz could be an alias of a higher frequency
    potential_aliases = []
    for n in range(1, 10):  # Check first few aliases
        alias_freq = n * fs - 0.061
        if alias_freq > 0:
            potential_aliases.append(alias_freq)
    
    print("       Potential alias sources for 0.061 Hz:")
    for i, alias in enumerate(potential_aliases[:5]):
        print(f"         {i+1}. {alias:.3f} Hz (could alias to 0.061 Hz)")
    
    # 3. FILTERING EXPERIMENTS
    print("\n    🔧 CONSERVATIVE FILTERING EXPERIMENTS:")
    
    # More targeted, data-preserving filter strategies
    filters_to_test = {
        # TARGETED 50Hz INTERFERENCE REMOVAL
        'notch_50Hz_narrow': {'type': 'notch', 'freq': 50.0, 'Q': 30},     # Very narrow 50Hz notch
        'notch_50Hz_wide': {'type': 'notch', 'freq': 50.0, 'Q': 10},       # Wider 50Hz notch
        'notch_52.6Hz': {'type': 'notch', 'freq': 52.612, 'Q': 20},        # Target the actual peak we found
        
        # AGGRESSIVE LOW-FREQUENCY OPTIONS TO TACKLE 0.061 Hz BEAST
        'highpass_0.08Hz': {'type': 'highpass', 'cutoff': 0.08, 'order': 4}, # Attack the 0.061Hz peak directly
        'highpass_0.1Hz': {'type': 'highpass', 'cutoff': 0.1, 'order': 4},   # Remove all ultra-low freq
        'highpass_0.15Hz': {'type': 'highpass', 'cutoff': 0.15, 'order': 4}, # More aggressive
        'notch_0.061Hz_narrow': {'type': 'notch', 'freq': 0.061, 'Q': 50},   # Very narrow 0.061Hz notch
        'notch_0.061Hz_wide': {'type': 'notch', 'freq': 0.061, 'Q': 20},     # Wider 0.061Hz notch
        
        # GENTLE OPTIONS (for comparison)
        'highpass_0.01Hz': {'type': 'highpass', 'cutoff': 0.01, 'order': 2}, # Remove only DC drift
        'highpass_0.02Hz': {'type': 'highpass', 'cutoff': 0.02, 'order': 2}, # Very gentle low-freq removal
        
        # COMBINATION APPROACHES
        'dual_notch_50_0.061': {'type': 'dual_notch', 'freq1': 50.0, 'Q1': 20, 'freq2': 0.061, 'Q2': 30},
        'aggressive_hp_50notch': {'type': 'combo', 'hp_cutoff': 0.1, 'hp_order': 4, 'notch_freq': 50.0, 'notch_Q': 25},
    }
    
    filtered_data = {}
    filter_performance = {}
    
    for filter_name, filter_params in filters_to_test.items():
        try:
            if filter_params['type'] == 'highpass':
                cutoff_norm = filter_params['cutoff'] / nyquist
                if cutoff_norm < 1.0:
                    b, a = butter(filter_params['order'], cutoff_norm, btype='high')
                    filtered_data[filter_name] = filtfilt(b, a, z)
                    
            elif filter_params['type'] == 'bandpass':
                low_norm = filter_params['low'] / nyquist
                high_norm = filter_params['high'] / nyquist
                if low_norm < high_norm < 1.0:
                    b, a = butter(filter_params['order'], [low_norm, high_norm], btype='band')
                    filtered_data[filter_name] = filtfilt(b, a, z)
                    
            elif filter_params['type'] == 'notch':
                # Design notch filter
                from scipy.signal import iirnotch
                freq_norm = filter_params['freq'] / nyquist
                if freq_norm < 1.0:
                    b, a = iirnotch(freq_norm, filter_params['Q'])
                    filtered_data[filter_name] = filtfilt(b, a, z)
                    
            elif filter_params['type'] == 'dual_notch':
                # Apply two notch filters in sequence
                from scipy.signal import iirnotch
                freq1_norm = filter_params['freq1'] / nyquist
                freq2_norm = filter_params['freq2'] / nyquist
                
                if freq1_norm < 1.0 and freq2_norm < 1.0:
                    # First notch filter
                    b1, a1 = iirnotch(freq1_norm, filter_params['Q1'])
                    temp_filtered = filtfilt(b1, a1, z)
                    # Second notch filter
                    b2, a2 = iirnotch(freq2_norm, filter_params['Q2'])
                    filtered_data[filter_name] = filtfilt(b2, a2, temp_filtered)
                    
            elif filter_params['type'] == 'combo':
                # Combination: gentle high-pass + notch
                # High-pass first
                hp_cutoff_norm = filter_params['hp_cutoff'] / nyquist
                if hp_cutoff_norm < 1.0:
                    b_hp, a_hp = butter(filter_params['hp_order'], hp_cutoff_norm, btype='high')
                    temp_filtered = filtfilt(b_hp, a_hp, z)
                    
                    # Then notch filter
                    from scipy.signal import iirnotch
                    notch_freq_norm = filter_params['notch_freq'] / nyquist
                    if notch_freq_norm < 1.0:
                        b_notch, a_notch = iirnotch(notch_freq_norm, filter_params['notch_Q'])
                        filtered_data[filter_name] = filtfilt(b_notch, a_notch, temp_filtered)
                    
            # Calculate performance metrics if filter was applied
            if filter_name in filtered_data:
                original_power = np.var(z)
                filtered_power = np.var(filtered_data[filter_name])
                power_reduction = (1 - filtered_power/original_power) * 100
                
                # Calculate SNR improvement (assuming high freq is noise)
                signal_band = (f >= 0.1) & (f <= 10)  # Assume this is signal band
                noise_band = (f >= 60) | (f <= 0.02)   # High freq + very low freq as noise
                
                if np.any(signal_band) and np.any(noise_band):
                    f_filt, Pxx_filt = sig.welch(filtered_data[filter_name], fs, nperseg=PSD_NPERSEG, window='hann', scaling="density")
                    signal_mask_filt = (f_filt >= 0.1) & (f_filt <= 10)
                    noise_mask_filt = (f_filt >= 60) | (f_filt <= 0.02)
                    
                    if np.any(signal_mask_filt) and np.any(noise_mask_filt):
                        snr_orig = np.mean(Pxx[signal_band]) / np.mean(Pxx[noise_band])
                        snr_filt = np.mean(Pxx_filt[signal_mask_filt]) / np.mean(Pxx_filt[noise_mask_filt])
                        snr_improvement = 10 * np.log10(snr_filt / snr_orig)
                    else:
                        snr_improvement = np.nan
                else:
                    snr_improvement = np.nan
                
                filter_performance[filter_name] = {
                    'power_reduction': power_reduction,
                    'snr_improvement': snr_improvement,
                    'rms_reduction': (1 - np.sqrt(filtered_power/original_power)) * 100
                }
                
        except Exception as e:
            print(f"       ❌ Failed to apply {filter_name}: {str(e)}")
    
    # Report filter performance
    print("    📊 FILTER PERFORMANCE SUMMARY:")
    for filter_name, performance in filter_performance.items():
        print(f"       {filter_name:20s}:")
        print(f"         Power reduction: {performance['power_reduction']:6.1f}%")
        print(f"         RMS reduction:   {performance['rms_reduction']:6.1f}%")
        if not np.isnan(performance['snr_improvement']):
            print(f"         SNR improvement: {performance['snr_improvement']:6.1f} dB")
        else:
            print(f"         SNR improvement: N/A")
    
    # 4. GENERATE COMPARATIVE FILTERING PLOTS
    print("\n    📊 GENERATING FILTER COMPARISON PLOTS...")
    
    # Create comprehensive filter comparison figure
    n_filters = min(len(filtered_data), 4)  # Limit to 4 best filters
    if n_filters > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Filter Comparison Analysis", fontsize=14)
        
        # Sort filters by power reduction for plotting
        sorted_filters = sorted(filter_performance.items(), 
                              key=lambda x: x[1]['power_reduction'], reverse=True)
        
        # Plot 1: Time domain comparison (first 1000 points)
        ax = axes[0, 0]
        time_plot = time_seconds[:1000]
        ax.plot(time_plot, z[:1000], 'b-', alpha=0.7, label='Original', linewidth=1)
        
        colors = ['red', 'green', 'orange', 'purple']
        for i, (filter_name, _) in enumerate(sorted_filters[:4]):
            if filter_name in filtered_data:
                ax.plot(time_plot, filtered_data[filter_name][:1000], 
                       color=colors[i], alpha=0.8, label=filter_name, linewidth=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnetic Field (nT)')
        ax.set_title('Time Domain Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: PSD comparison (log scale)
        ax = axes[0, 1]
        ax.loglog(f[1:], Pxx[1:], 'b-', alpha=0.7, label='Original', linewidth=2)
        
        for i, (filter_name, _) in enumerate(sorted_filters[:4]):
            if filter_name in filtered_data:
                f_filt, Pxx_filt = sig.welch(filtered_data[filter_name], fs, nperseg=PSD_NPERSEG, window='hann', scaling="density")
                ax.loglog(f_filt[1:], Pxx_filt[1:], color=colors[i], alpha=0.8, 
                         label=filter_name, linewidth=1.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (nT²/Hz)')
        ax.set_title('PSD Comparison (Log-Log)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Low frequency zoom (0-1 Hz)
        ax = axes[1, 0]
        lf_mask = f <= 1.0
        ax.semilogy(f[lf_mask], Pxx[lf_mask], 'b-', alpha=0.7, label='Original', linewidth=2)
        
        for i, (filter_name, _) in enumerate(sorted_filters[:4]):
            if filter_name in filtered_data:
                f_filt, Pxx_filt = sig.welch(filtered_data[filter_name], fs, nperseg=PSD_NPERSEG, window='hann', scaling="density")
                lf_mask_filt = f_filt <= 1.0
                ax.semilogy(f_filt[lf_mask_filt], Pxx_filt[lf_mask_filt], 
                           color=colors[i], alpha=0.8, label=filter_name, linewidth=1.5)
        
        ax.axvline(0.061, color='red', linestyle='--', alpha=0.7, label='0.061 Hz')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (nT²/Hz)')
        ax.set_title('Low Frequency Detail (0-1 Hz)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Filter performance bar chart
        ax = axes[1, 1]
        filter_names = [name[:15] for name, _ in sorted_filters[:4]]  # Truncate names
        power_reductions = [perf['power_reduction'] for _, perf in sorted_filters[:4]]
        
        bars = ax.bar(range(len(filter_names)), power_reductions, 
                     color=['red', 'green', 'orange', 'purple'][:len(filter_names)], alpha=0.7)
        ax.set_xlabel('Filter Type')
        ax.set_ylabel('Power Reduction (%)')
        ax.set_title('Filter Effectiveness')
        ax.set_xticks(range(len(filter_names)))
        ax.set_xticklabels(filter_names, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, value in zip(bars, power_reductions):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(OUTDIR/"filter_comparison_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("       ✅ Filter comparison plots saved to filter_comparison_analysis.png")
    
    # 5. APPLY COMBINED FILTER AND REGENERATE VISUALIZATIONS
    print("\n    🎯 APPLYING COMBINED FILTER TO DATASET:")
    print("       • Step 1: 50Hz notch filter (remove mains interference)")
    print("       • Step 2: 0.1Hz high-pass filter (remove low-frequency drift)")
    
    # Apply 50Hz notch filter first
    nyquist = fs / 2
    notch_freq = 50.0  # Hz
    Q = 30  # Quality factor for narrow notch
    
    if notch_freq < nyquist:
        b_notch, a_notch = sig.iirnotch(notch_freq, Q, fs)
        z_notch = sig.filtfilt(b_notch, a_notch, z)
        print(f"       ✅ 50Hz notch filter applied")
    else:
        z_notch = z.copy()
        print(f"       ⚠️  50Hz exceeds Nyquist frequency, skipping notch filter")
    
    # Apply 0.1Hz high-pass filter to the notch-filtered data
    hp_freq = 0.1  # Hz
    order = 4
    
    if hp_freq < nyquist:
        b_hp, a_hp = sig.butter(order, hp_freq, btype='high', fs=fs)
        z_combined = sig.filtfilt(b_hp, a_hp, z_notch)
        print(f"       ✅ 0.1Hz high-pass filter applied")
    else:
        z_combined = z_notch.copy()
        print(f"       ⚠️  0.1Hz exceeds Nyquist frequency, skipping high-pass filter")
    
    # Generate statistics comparison
    print(f"\n    📊 FILTERING RESULTS:")
    print(f"       Original data:")
    print(f"         • Mean: {np.mean(z):.2f} nT")
    print(f"         • Std:  {np.std(z):.2f} nT") 
    print(f"         • Range: {np.min(z):.2f} to {np.max(z):.2f} nT")
    print(f"       After 50Hz notch:")
    print(f"         • Mean: {np.mean(z_notch):.2f} nT")
    print(f"         • Std:  {np.std(z_notch):.2f} nT")
    print(f"         • RMS reduction: {(1 - np.std(z_notch)/np.std(z))*100:.1f}%")
    print(f"       After combined filter:")
    print(f"         • Mean: {np.mean(z_combined):.2f} nT")
    print(f"         • Std:  {np.std(z_combined):.2f} nT")
    print(f"         • Range: {np.min(z_combined):.2f} to {np.max(z_combined):.2f} nT")
    print(f"         • Total RMS reduction: {(1 - np.std(z_combined)/np.std(z))*100:.1f}%")
    print(f"         • Total power reduction: {(1 - np.var(z_combined)/np.var(z))*100:.1f}%")
    
    # Create comprehensive time-series comparison plot
    print("\n    📊 Generating full time-series comparison...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # Calculate time in hours from start
    time_hours = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds() / 3600
    
    # Plot 1: Full time series comparison
    ax = axes[0]
    ax.plot(time_hours, z, 'b-', alpha=0.7, linewidth=0.8, label='Original Data')
    ax.plot(time_hours, z_notch, 'orange', alpha=0.8, linewidth=0.8, label='50Hz Notch Only')
    ax.plot(time_hours, z_combined, 'r-', alpha=0.9, linewidth=0.8, label='Combined Filter (50Hz + 0.1Hz HP)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Magnetic Field (nT)')
    ax.set_title('Full Dataset: Original vs Filtered Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed section (first 1000 points for detail)
    ax = axes[1]
    zoom_end = min(1000, len(z))
    ax.plot(time_hours[:zoom_end], z[:zoom_end], 'b-', alpha=0.7, linewidth=1, label='Original Data')
    ax.plot(time_hours[:zoom_end], z_notch[:zoom_end], 'orange', alpha=0.8, linewidth=1, label='50Hz Notch Only')
    ax.plot(time_hours[:zoom_end], z_combined[:zoom_end], 'r-', alpha=0.9, linewidth=1, label='Combined Filter')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Magnetic Field (nT)')
    ax.set_title(f'Detailed View: First {zoom_end} Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: What was removed by 50Hz notch
    ax = axes[2]
    difference_notch = z - z_notch
    ax.plot(time_hours, difference_notch, 'orange', alpha=0.7, linewidth=0.8, label='Removed by 50Hz Notch')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Removed Signal (nT)')
    ax.set_title('Signal Removed by 50Hz Notch Filter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: What was removed by combined filter
    ax = axes[3]
    difference_combined = z - z_combined
    ax.plot(time_hours, difference_combined, 'g-', alpha=0.7, linewidth=0.8, label='Removed by Combined Filter')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Removed Signal (nT)')
    ax.set_title('Total Signal Removed by Combined Filter (Original - Filtered)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTDIR/"combined_filter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("       ✅ Combined filter time-series comparison saved to combined_filter_comparison.png")
    
    # Generate residual field time-series plots (before and after filtering)
    print("    📊 Generating residual field time-series plots...")
    
    # Calculate residual fields (remove mean to show relative variations)
    z_residual_original = z - np.mean(z)
    z_residual_filtered = z_combined - np.nanmean(z_combined)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original residual field over time
    ax = axes[0]
    ax.plot(time_hours, z_residual_original, 'b-', alpha=0.7, linewidth=0.8, label='Original Residual Field')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Residual Magnetic Field (nT)')
    ax.set_title('Original Residual Field Over Time (Mean Removed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    ax.text(0.02, 0.98, f'Std: {np.std(z_residual_original):.2f} nT\nRange: {np.max(z_residual_original) - np.min(z_residual_original):.2f} nT', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Filtered residual field over time
    ax = axes[1]
    ax.plot(time_hours, z_residual_filtered, 'r-', alpha=0.7, linewidth=0.8, label='Combined Filtered Residual Field')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Residual Magnetic Field (nT)')
    ax.set_title('Combined Filtered Residual Field Over Time (Mean Removed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    ax.text(0.02, 0.98, f'Std: {np.std(z_residual_filtered):.2f} nT\nRange: {np.max(z_residual_filtered) - np.min(z_residual_filtered):.2f} nT', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Direct comparison of both residual fields
    ax = axes[2]
    ax.plot(time_hours, z_residual_original, 'b-', alpha=0.6, linewidth=0.8, label='Original Residual')
    ax.plot(time_hours, z_residual_filtered, 'r-', alpha=0.8, linewidth=0.8, label='Filtered Residual')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Residual Magnetic Field (nT)')
    ax.set_title('Residual Field Comparison: Original vs Combined Filtered')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add improvement statistics
    rms_improvement = (1 - np.std(z_residual_filtered)/np.std(z_residual_original))*100
    ax.text(0.02, 0.98, f'RMS Reduction: {rms_improvement:.1f}%\nNoise Removed: {np.std(z_residual_original) - np.std(z_residual_filtered):.2f} nT', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTDIR/"residual_field_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("       ✅ Residual field time-series plots saved to residual_field_comparison.png")
    
    # Regenerate interactive map with combined filtered data
    print("    🗺️  Regenerating interactive map with combined filtered data...")
    
    # Create filtered residual field for mapping
    z_combined_residual = z_combined - np.nanmean(z_combined)
    
    # Prepare data for gridding (use correct column names)
    mask_valid = ~np.isnan(z_combined_residual)
    if np.sum(mask_valid) > 10:  # Need enough points for gridding
        
        # Grid the filtered data
        try:
            # Create grid using correct column names and proper coordinate order
            # For folium, we need latitude as rows (y) and longitude as columns (x)
            lat_min, lat_max = df["Latitude [Decimal Degrees]"].min(), df["Latitude [Decimal Degrees]"].max()
            lon_min, lon_max = df["Longitude [Decimal Degrees]"].min(), df["Longitude [Decimal Degrees]"].max()
            
            # Create coordinate arrays - latitude (y) and longitude (x)
            grid_lats = np.linspace(lat_min, lat_max, 100)
            grid_lons = np.linspace(lon_min, lon_max, 100)
            grid_x, grid_y = np.meshgrid(grid_lons, grid_lats)
            
            # Interpolate filtered residual field
            grid_z_filtered = griddata(
                (df["Longitude [Decimal Degrees]"][mask_valid], df["Latitude [Decimal Degrees]"][mask_valid]),
                z_combined_residual[mask_valid],
                (grid_x, grid_y),
                method='linear'
            )
            
            # Create folium map for filtered data
            center_lat = df["Latitude [Decimal Degrees]"].mean()
            center_lon = df["Longitude [Decimal Degrees]"].mean()
            
            m_filtered = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add filtered field as overlay
            from branca.colormap import LinearColormap
            
            # Ensure proper threshold calculation and sorting
            valid_data = grid_z_filtered[~np.isnan(grid_z_filtered)]
            if len(valid_data) > 0:
                vmin_f, vmax_f = np.percentile(valid_data, [5, 95])
                # Ensure thresholds are sorted and different
                if vmin_f >= vmax_f:
                    vmin_f = np.min(valid_data)
                    vmax_f = np.max(valid_data)
                    if vmin_f >= vmax_f:  # All values are the same
                        vmax_f = vmin_f + 1.0
            else:
                vmin_f, vmax_f = -1.0, 1.0  # Default range
            
            # Create viridis colormap with properly sorted thresholds
            viridis_colors = ['#440154', '#404387', '#2a788e', '#22a884', '#7ad151', '#fde725']
            cm_filtered = LinearColormap(viridis_colors, vmin=vmin_f, vmax=vmax_f)
            
            # Add filtered field as interpolated raster (like original map)
            import matplotlib.pyplot as plt
            import matplotlib.cm as mpl_cm
            from matplotlib.colors import Normalize
            
            # Create normalized colormap
            norm = Normalize(vmin=vmin_f, vmax=vmax_f)
            viridis_cmap = mpl_cm.get_cmap('viridis')
            
            # Convert grid to RGBA image and transpose to fix rotation
            rgba_image = viridis_cmap(norm(np.flipud(grid_z_filtered.T)))  # Transpose to fix orientation
            
            # Add raster overlay
            folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.7,
                interactive=True,
                cross_origin=False,
            ).add_to(m_filtered)
            
            # Add colorbar
            cm_filtered.add_to(m_filtered)
            
            # Add some data points for reference (subsample)
            subsample_step = max(1, len(df) // 500)
            for i in range(0, len(df), subsample_step):
                folium.CircleMarker(
                    location=[df["Latitude [Decimal Degrees]"].iloc[i], df["Longitude [Decimal Degrees]"].iloc[i]],
                    radius=2,
                    popup=(f"<b>Filtered:</b> {z_combined[i]:.2f} nT<br>"
                           f"<b>Original:</b> {z[i]:.2f} nT<br>"
                           f"<b>Removed:</b> {z[i] - z_combined[i]:.2f} nT"),
                    color='white',
                    weight=1,
                    fillOpacity=0.3
                ).add_to(m_filtered)
            
            # Add title
            title_html = '''
            <h3 align="center" style="font-size:20px"><b>Combined Filtered Magnetic Field Map (50Hz Notch + 0.1Hz HP)</b></h3>
            '''
            m_filtered.get_root().html.add_child(folium.Element(title_html))
            
            # Save filtered map
            m_filtered.save(OUTDIR/"interactive_map_combined_filtered.html")
            print("       ✅ Combined filtered interactive map saved to interactive_map_combined_filtered.html")
            
        except Exception as e:
            print(f"       ❌ Failed to create filtered map: {str(e)}")
    else:
        print("       ❌ Insufficient valid data points for mapping")
        
else:
    print("🔉  PSD disabled – set COMPUTE_PSD=True → enable")

print("✅  All done – outputs in", OUTDIR)