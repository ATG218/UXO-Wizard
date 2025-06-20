#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the data file
CSV_PATH = '/Users/aleksandergarbuz/Documents/SINTEF/Magnetic_Data/FIELD_DATA_200525/processing_results/20250520_072745_MWALK_#0122_processed_20250603_134940.csv'

# Read the data
print("Reading data file...")
df = pd.read_csv(CSV_PATH)
print(f"Total records: {len(df)}")

# Time information
print("\nTime range in dataset:")
print(f"Start: {df['GPSTime [hh:mm:ss.sss]'].min()}")
print(f"End: {df['GPSTime [hh:mm:ss.sss]'].max()}")

# Sampling the data for inspection
sample_step = len(df) // 20  # Get about 20 samples
print("\nSample of flight data over time:")
sample_df = df.iloc[::sample_step][['GPSTime [hh:mm:ss.sss]', 'Altitude [m]', 'UTM_Easting', 'UTM_Northing', 'R1 [nT]', 'Satellites', 'Quality']]
print(sample_df.reset_index().to_string())

# Calculate the time periods as datetime objects for better analysis
df['GPSTime_dt'] = pd.to_datetime(df['GPSTime [hh:mm:ss.sss]'], format='%H:%M:%S.%f')

# Calculate distance traveled between points
df['UTM_Easting_diff'] = df['UTM_Easting'].diff()
df['UTM_Northing_diff'] = df['UTM_Northing'].diff()
df['Distance'] = np.sqrt(df['UTM_Easting_diff']**2 + df['UTM_Northing_diff']**2)

# Calculate speed (distance per time)
df['Time_diff'] = df['GPSTime_dt'].diff().dt.total_seconds()
df['Speed'] = df['Distance'] / df['Time_diff']

# Now look at specific time ranges mentioned by the user
# Convert time strings to datetime for comparison
time_start = pd.to_datetime('07:41:56.6', format='%H:%M:%S.%f')
time_mid1 = pd.to_datetime('08:02:31.1', format='%H:%M:%S.%f')
time_mid2 = pd.to_datetime('08:14:16.8', format='%H:%M:%S.%f')
time_end = pd.to_datetime('08:38:00.0', format='%H:%M:%S.%f')

# Create a mask for the segments to exclude
exclude_mask = (
    (df['GPSTime_dt'] < time_start) | 
    ((df['GPSTime_dt'] >= time_mid1) & (df['GPSTime_dt'] <= time_mid2)) | 
    (df['GPSTime_dt'] > time_end)
)

# Let's find statistics on key variables for include vs exclude segments
include_df = df[~exclude_mask]
exclude_df = df[exclude_mask]

print("\n\nStatistical comparison between segments to keep vs. exclude:")
print(f"Keep segments:    {len(include_df)} records ({len(include_df)*100/len(df):.1f}% of data)")
print(f"Exclude segments: {len(exclude_df)} records ({len(exclude_df)*100/len(df):.1f}% of data)")

# Key variables to compare
variables = ['Altitude [m]', 'Speed', 'Distance', 'Satellites', 'Quality', 'R1 [nT]', 'R2 [nT]']

print("\nStatistical comparison between segments:")
for var in variables:
    if var in df.columns:
        print(f"\n{var}:")
        print(f"  KEEP   - Mean: {include_df[var].mean():.2f}, Median: {include_df[var].median():.2f}, Std: {include_df[var].std():.2f}")
        print(f"  EXCLUDE - Mean: {exclude_df[var].mean():.2f}, Median: {exclude_df[var].median():.2f}, Std: {exclude_df[var].std():.2f}")

# Plot altitude and speed to visualize changes
plt.figure(figsize=(14, 10))

# Plot 1: Altitude over time
plt.subplot(2, 1, 1)
plt.plot(df['GPSTime_dt'], df['Altitude [m]'], 'b-', alpha=0.5)
plt.axvline(x=time_start, color='r', linestyle='--', label='Cutoff points')
plt.axvline(x=time_mid1, color='r', linestyle='--')
plt.axvline(x=time_mid2, color='r', linestyle='--')
plt.axvline(x=time_end, color='r', linestyle='--')
plt.title('Altitude vs Time')
plt.ylabel('Altitude [m]')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Speed over time
plt.subplot(2, 1, 2)
valid_speed = df['Speed'].replace([np.inf, -np.inf], np.nan).dropna()
plt.plot(df['GPSTime_dt'], df['Speed'].replace([np.inf, -np.inf], np.nan), 'g-', alpha=0.5)
plt.axvline(x=time_start, color='r', linestyle='--', label='Cutoff points')
plt.axvline(x=time_mid1, color='r', linestyle='--')
plt.axvline(x=time_mid2, color='r', linestyle='--')
plt.axvline(x=time_end, color='r', linestyle='--')
plt.title('Speed vs Time')
plt.ylabel('Speed [m/s]')
plt.xlabel('Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/aleksandergarbuz/Documents/SINTEF/Plots/flight_analysis.png')
print("\nAnalysis plot saved to /Users/aleksandergarbuz/Documents/SINTEF/Plots/flight_analysis.png")

# Now, let's look at the unique characteristics that could be used to filter the data
# without hardcoding these specific times

# 1. Altitude changes
altitude_change = df['Altitude [m]'].diff().abs()
print("\nAltitude changes statistics:")
print(f"Overall mean abs change: {altitude_change.mean():.4f} m")
print(f"99th percentile change: {altitude_change.quantile(0.99):.4f} m")

# 2. Speed characteristics
print("\nSpeed statistics:")
print(f"Overall mean speed: {valid_speed.mean():.2f} m/s")
print(f"95th percentile speed: {valid_speed.quantile(0.95):.2f} m/s")

# 3. Direction changes
df['Heading'] = np.arctan2(df['UTM_Northing_diff'], df['UTM_Easting_diff']) * 180 / np.pi
heading_change = df['Heading'].diff().abs()
heading_change = heading_change.where(heading_change <= 180, 360 - heading_change)
print("\nHeading changes statistics:")
print(f"Overall mean heading change: {heading_change.mean():.2f} degrees")
print(f"95th percentile heading change: {heading_change.quantile(0.95):.2f} degrees")

# 4. GPS Quality
print("\nGPS Quality statistics:")
print(f"Overall mean satellites: {df['Satellites'].mean():.2f}")
print(f"Segments to keep - mean satellites: {include_df['Satellites'].mean():.2f}")
print(f"Segments to exclude - mean satellites: {exclude_df['Satellites'].mean():.2f}")

# Create a second figure for spatial analysis
plt.figure(figsize=(10, 10))
# Color points by time
plt.scatter(df['UTM_Easting'], df['UTM_Northing'], c=(df.index), cmap='viridis', 
           alpha=0.5, s=2)

# Mark excluded regions with different colors
plt.scatter(df.loc[df['GPSTime_dt'] < time_start, 'UTM_Easting'], 
           df.loc[df['GPSTime_dt'] < time_start, 'UTM_Northing'], 
           color='red', alpha=0.5, s=3, label='Before 07:41:56.6')

plt.scatter(df.loc[(df['GPSTime_dt'] >= time_mid1) & (df['GPSTime_dt'] <= time_mid2), 'UTM_Easting'], 
           df.loc[(df['GPSTime_dt'] >= time_mid1) & (df['GPSTime_dt'] <= time_mid2), 'UTM_Northing'], 
           color='magenta', alpha=0.5, s=3, label='08:02:31.1 - 08:14:16.8')

plt.scatter(df.loc[df['GPSTime_dt'] > time_end, 'UTM_Easting'], 
           df.loc[df['GPSTime_dt'] > time_end, 'UTM_Northing'], 
           color='orange', alpha=0.5, s=3, label='After 08:38:00')

plt.title('Flight Path with Excluded Regions Highlighted')
plt.xlabel('UTM Easting')
plt.ylabel('UTM Northing')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/Users/aleksandergarbuz/Documents/SINTEF/Plots/flight_path_excluded_regions.png')
print("Spatial analysis plot saved to /Users/aleksandergarbuz/Documents/SINTEF/Plots/flight_path_excluded_regions.png")
print("\nAnalysis complete!")
