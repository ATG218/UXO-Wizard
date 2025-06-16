#!/usr/bin/env python3
"""
TARVA GAMMA1 Sensor Data Visualization Script
=============================================

This script creates comprehensive visualizations for the cleaned sensor data
from the TARVA GAMMA1 dataset, including:
- Environmental data (temperature, humidity, pressure) 
- GPS tracking and altitude data
- Spectrometer readings and radiation counts
- Time-series analysis across all sensors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_sensor_data():
    """Load all sensor data files and return as a dictionary."""
    data_dir = Path("cleaned_data")
    
    sensors = {
        'gps_external': pd.read_csv(data_dir / "gps_external.csv"),
        'gps_internal': pd.read_csv(data_dir / "gps_internal.csv"), 
        'environmental': pd.read_csv(data_dir / "environmental.csv"),
        'altitude': pd.read_csv(data_dir / "altitude.csv"),
        'spectrometer': pd.read_csv(data_dir / "spectrometer.csv"),
        'stabilized_spectrometer': pd.read_csv(data_dir / "stabilized_spectrometer.csv"),
        'sync_data': pd.read_csv(data_dir / "sync_data.csv")
    }
    
    # Convert timestamps to datetime for all datasets
    for name, df in sensors.items():
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            # Convert timestamp to datetime if datetime column doesn't exist
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return sensors

def plot_environmental_data(env_data):
    """Create environmental sensor visualizations."""
    print("Creating environmental sensor plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Environmental Sensor Data Analysis', fontsize=16, fontweight='bold')
    
    # Filter out the metadata row (first row with NaN values)
    env_clean = env_data.dropna(subset=['Temp', 'Hum', 'Press'])
    
    # Temperature over time
    axes[0,0].plot(env_clean['datetime'], env_clean['Temp'], 'b-', linewidth=1.5, alpha=0.8)
    axes[0,0].set_title('Temperature vs Time')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Humidity over time
    axes[0,1].plot(env_clean['datetime'], env_clean['Hum'], 'g-', linewidth=1.5, alpha=0.8)
    axes[0,1].set_title('Humidity vs Time')
    axes[0,1].set_ylabel('Humidity (%)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Pressure over time
    axes[1,0].plot(env_clean['datetime'], env_clean['Press'], 'r-', linewidth=1.5, alpha=0.8)
    axes[1,0].set_title('Atmospheric Pressure vs Time')
    axes[1,0].set_ylabel('Pressure (hPa)')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap
    correlation_data = env_clean[['Temp', 'Hum', 'Press']].corr()
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1,1])
    axes[1,1].set_title('Environmental Data Correlation')
    
    plt.tight_layout()
    plt.savefig('environmental_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_gps_and_altitude(gps_data, altitude_data):
    """Create GPS tracking and altitude visualizations."""
    print("Creating GPS and altitude plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GPS and Altitude Analysis', fontsize=16, fontweight='bold')
    
    # Filter for valid GPS data (non-zero coordinates)
    gps_clean = gps_data[
        (gps_data['Lat'] != 0) & (gps_data['Lon'] != 0) & 
        (gps_data['Lat'].notna()) & (gps_data['Lon'].notna())
    ]
    
    alt_clean = altitude_data.dropna(subset=['Height'])
    
    # GPS trajectory (if we have valid coordinates)
    if len(gps_clean) > 0:
        axes[0,0].plot(gps_clean['Lon'], gps_clean['Lat'], 'bo-', markersize=3, alpha=0.7)
        axes[0,0].set_title('GPS Trajectory')
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        axes[0,0].grid(True, alpha=0.3)
    else:
        axes[0,0].text(0.5, 0.5, 'No valid GPS coordinates\n(Lat/Lon = 0)', 
                      ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('GPS Trajectory - No Valid Data')
    
    # Altitude over time
    axes[0,1].plot(alt_clean['datetime'], alt_clean['Height'], 'purple', linewidth=1.5, alpha=0.8)
    axes[0,1].set_title('Altitude vs Time')
    axes[0,1].set_ylabel('Height (m)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # GPS signal quality (number of satellites)
    if 'noSat' in gps_data.columns:
        axes[1,0].plot(gps_data['datetime'], gps_data['noSat'], 'orange', linewidth=1.5, alpha=0.8)
        axes[1,0].set_title('GPS Signal Quality (Number of Satellites)')
        axes[1,0].set_ylabel('Number of Satellites')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Height distribution histogram
    axes[1,1].hist(alt_clean['Height'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Altitude Distribution')
    axes[1,1].set_xlabel('Height (m)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gps_altitude_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectrometer_data(spec_data):
    """Create spectrometer and radiation analysis visualizations."""
    print("Creating spectrometer analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gamma Ray Spectrometer Analysis', fontsize=16, fontweight='bold')
    
    # Filter out metadata rows
    spec_clean = spec_data.dropna(subset=['Total', 'Countrate'])
    
    # Total counts over time
    axes[0,0].plot(spec_clean['datetime'], spec_clean['Total'], 'red', linewidth=1.5, alpha=0.8)
    axes[0,0].set_title('Total Gamma Ray Counts vs Time')
    axes[0,0].set_ylabel('Total Counts')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Count rate over time
    axes[0,1].plot(spec_clean['datetime'], spec_clean['Countrate'], 'darkred', linewidth=1.5, alpha=0.8)
    axes[0,1].set_title('Count Rate vs Time')
    axes[0,1].set_ylabel('Count Rate (cps)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Livetime vs Realtime
    axes[1,0].scatter(spec_clean['Realtime'], spec_clean['Livetime'], alpha=0.6, s=20)
    axes[1,0].plot([0, spec_clean['Realtime'].max()], [0, spec_clean['Realtime'].max()], 'r--', alpha=0.5)
    axes[1,0].set_title('Livetime vs Realtime')
    axes[1,0].set_xlabel('Realtime (s)')
    axes[1,0].set_ylabel('Livetime (s)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Count rate distribution
    axes[1,1].hist(spec_clean['Countrate'], bins=30, alpha=0.7, edgecolor='black', color='darkred')
    axes[1,1].set_title('Count Rate Distribution')
    axes[1,1].set_xlabel('Count Rate (cps)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectrometer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_data_overview(sensors):
    """Create an overview dashboard of all sensor data."""
    print("Creating data overview dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TARVA GAMMA1 - Complete Sensor Data Overview', fontsize=18, fontweight='bold')
    
    # Data availability timeline
    ax = axes[0,0]
    y_pos = 0
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, (name, data) in enumerate(sensors.items()):
        if 'datetime' in data.columns and len(data.dropna(subset=['datetime'])) > 0:
            clean_data = data.dropna(subset=['datetime'])
            if len(clean_data) > 0:
                ax.scatter(clean_data['datetime'], [y_pos] * len(clean_data), 
                          alpha=0.6, s=1, color=colors[i % len(colors)], label=name)
                y_pos += 1
    
    ax.set_title('Data Availability Timeline')
    ax.set_ylabel('Sensor Type')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Record counts per sensor
    axes[0,1].bar(range(len(sensors)), [len(df) for df in sensors.values()], 
                  color=colors[:len(sensors)])
    axes[0,1].set_title('Record Count by Sensor Type')
    axes[0,1].set_ylabel('Number of Records')
    axes[0,1].set_xticks(range(len(sensors)))
    axes[0,1].set_xticklabels(sensors.keys(), rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Environmental summary (if available)
    if 'environmental' in sensors:
        env_clean = sensors['environmental'].dropna(subset=['Temp', 'Hum', 'Press'])
        if len(env_clean) > 0:
            axes[0,2].plot(env_clean['datetime'], env_clean['Temp'], 'red', label='Temperature', alpha=0.8)
            axes[0,2].set_title('Temperature Trend')
            axes[0,2].set_ylabel('Temperature (°C)')
            axes[0,2].tick_params(axis='x', rotation=45)
            axes[0,2].grid(True, alpha=0.3)
    
    # Radiation level summary (if available)
    if 'spectrometer' in sensors:
        spec_clean = sensors['spectrometer'].dropna(subset=['Countrate'])
        if len(spec_clean) > 0:
            axes[1,0].plot(spec_clean['datetime'], spec_clean['Countrate'], 'darkred', alpha=0.8)
            axes[1,0].set_title('Radiation Count Rate')
            axes[1,0].set_ylabel('Count Rate (cps)')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
    
    # Altitude profile (if available)
    if 'altitude' in sensors:
        alt_clean = sensors['altitude'].dropna(subset=['Height'])
        if len(alt_clean) > 0:
            axes[1,1].plot(alt_clean['datetime'], alt_clean['Height'], 'purple', alpha=0.8)
            axes[1,1].set_title('Altitude Profile')
            axes[1,1].set_ylabel('Height (m)')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
    
    # Data quality summary
    quality_data = []
    quality_labels = []
    for name, data in sensors.items():
        if 'datetime' in data.columns:
            total_records = len(data)
            valid_timestamps = len(data.dropna(subset=['datetime']))
            quality_data.append(valid_timestamps / total_records * 100 if total_records > 0 else 0)
            quality_labels.append(name)
    
    axes[1,2].bar(range(len(quality_data)), quality_data, color=colors[:len(quality_data)])
    axes[1,2].set_title('Data Quality (% Valid Timestamps)')
    axes[1,2].set_ylabel('Valid Data (%)')
    axes[1,2].set_xticks(range(len(quality_labels)))
    axes[1,2].set_xticklabels(quality_labels, rotation=45)
    axes[1,2].set_ylim(0, 100)
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary(sensors):
    """Print a comprehensive data summary."""
    print("\n" + "="*60)
    print("TARVA GAMMA1 DATA ANALYSIS SUMMARY")
    print("="*60)
    
    total_records = sum(len(df) for df in sensors.values())
    print(f"Total Records Across All Sensors: {total_records:,}")
    print(f"Number of Sensor Types: {len(sensors)}")
    
    print("\nPer-Sensor Breakdown:")
    print("-" * 40)
    for name, data in sensors.items():
        valid_timestamps = len(data.dropna(subset=['datetime'])) if 'datetime' in data.columns else 0
        print(f"{name.upper():25} | {len(data):6,} records | {valid_timestamps:6,} valid timestamps")
    
    # Analyze time range
    all_timestamps = []
    for data in sensors.values():
        if 'datetime' in data.columns:
            timestamps = data.dropna(subset=['datetime'])['datetime']
            all_timestamps.extend(timestamps.tolist())
    
    if all_timestamps:
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        duration = max_time - min_time
        print(f"\nData Collection Period:")
        print(f"Start: {min_time}")
        print(f"End:   {max_time}")
        print(f"Duration: {duration}")
    
    print("\n" + "="*60)

def main():
    """Main function to run all visualizations."""
    print("Starting TARVA GAMMA1 sensor data visualization...")
    print("Loading sensor data...")
    
    # Load all sensor data
    sensors = load_sensor_data()
    
    # Print data summary
    print_data_summary(sensors)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Environmental data analysis
    if 'environmental' in sensors:
        plot_environmental_data(sensors['environmental'])
    
    # GPS and altitude analysis
    if 'gps_external' in sensors and 'altitude' in sensors:
        plot_gps_and_altitude(sensors['gps_external'], sensors['altitude'])
    
    # Spectrometer analysis
    if 'spectrometer' in sensors:
        plot_spectrometer_data(sensors['spectrometer'])
    
    # Complete overview dashboard
    plot_data_overview(sensors)
    
    print("\nVisualization complete! Generated files:")
    print("- environmental_analysis.png")
    print("- gps_altitude_analysis.png") 
    print("- spectrometer_analysis.png")
    print("- complete_overview.png")
    print("\nAll plots have been displayed and saved to PNG files.")

if __name__ == "__main__":
    main() 