import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import RANSACRegressor
import argparse
import os

class FlightAnomalyDetector:
    def __init__(self, csv_file):
        """Initialize the anomaly detector with flight data."""
        self.csv_file = csv_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and prepare flight data."""
        print(f"Loading data from {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        
        # Sort by timestamp to ensure chronological order
        if 'timestamp' in self.df.columns:
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            print(f"Loaded {len(self.df)} data points sorted chronologically")
        else:
            print(f"Warning: No timestamp column found. Loaded {len(self.df)} data points")
            
        # Add sequential index for reference
        self.df['point_index'] = range(len(self.df))
        
    def calculate_movement_metrics(self):
        """Calculate movement-related metrics to identify anomalies."""
        # Calculate distances between consecutive points
        self.df['lat_diff'] = self.df['lat'].diff()
        self.df['lon_diff'] = self.df['lon'].diff()
        
        # Calculate Euclidean distance between consecutive points
        self.df['step_distance'] = np.sqrt(
            self.df['lat_diff']**2 + self.df['lon_diff']**2
        )
        
        # Calculate speed and acceleration if timestamp is available
        if 'timestamp' in self.df.columns:
            self.df['time_diff'] = self.df['timestamp'].diff() / 1000.0  # Convert to seconds
            # Avoid division by zero
            self.df['speed'] = self.df['step_distance'] / (self.df['time_diff'] + 1e-6)
            self.df['acceleration'] = self.df['speed'].diff() / (self.df['time_diff'] + 1e-6)
        
        # Calculate bearing (direction) changes
        self.df['bearing'] = np.arctan2(self.df['lat_diff'], self.df['lon_diff']) * 180 / np.pi
        self.df['bearing_change'] = self.df['bearing'].diff().abs()
        
        # Handle bearing wraparound
        self.df['bearing_change'] = np.where(
            self.df['bearing_change'] > 180,
            360 - self.df['bearing_change'],
            self.df['bearing_change']
        )
        
    def detect_velocity_anomalies(self):
        """Detect anomalies based on sudden velocity changes (RTK timeouts)."""
        anomalies = []
        
        if 'speed' not in self.df.columns:
            print("Warning: No timestamp data available for velocity analysis")
            return anomalies
        
        # Calculate velocity statistics
        valid_speeds = self.df['speed'].dropna()
        if len(valid_speeds) == 0:
            return anomalies
            
        speed_median = valid_speeds.median()
        speed_mad = np.median(np.abs(valid_speeds - speed_median))  # Median Absolute Deviation
        
        # Use MAD-based outlier detection (more robust than standard deviation)
        speed_threshold = speed_median + 5 * speed_mad  # 5 MAD threshold
        
        print(f"Speed analysis: median={speed_median:.6f}, MAD={speed_mad:.6f}, threshold={speed_threshold:.6f}")
        
        # Find points with excessive speed
        speed_outliers = self.df[self.df['speed'] > speed_threshold]
        
        for idx, row in speed_outliers.iterrows():
            anomalies.append({
                'index': idx,
                'type': 'velocity_spike',
                'value': row['speed'],
                'threshold': speed_threshold,
                'severity': row['speed'] / speed_threshold,
                'description': f"Sudden velocity spike: {row['speed']:.6f} (>{speed_threshold:.6f})"
            })
        
        return anomalies
    
    def detect_position_jumps(self):
        """Detect sudden position jumps that are inconsistent with drone movement."""
        anomalies = []
        
        # Calculate rolling statistics for step distance
        window_size = 10  # Look at 10-point moving window
        self.df['distance_rolling_median'] = self.df['step_distance'].rolling(window=window_size, center=True).median()
        self.df['distance_rolling_mad'] = self.df['step_distance'].rolling(window=window_size, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        
        # Detect jumps using local statistics
        for idx in range(window_size//2, len(self.df) - window_size//2):
            current_distance = self.df.loc[idx, 'step_distance']
            local_median = self.df.loc[idx, 'distance_rolling_median']
            local_mad = self.df.loc[idx, 'distance_rolling_mad']
            
            if pd.notna(current_distance) and pd.notna(local_median) and pd.notna(local_mad) and local_mad > 0:
                # Use local MAD-based threshold
                threshold = local_median + 4 * local_mad
                
                if current_distance > threshold:
                    anomalies.append({
                        'index': idx,
                        'type': 'position_jump',
                        'value': current_distance,
                        'threshold': threshold,
                        'severity': current_distance / threshold,
                        'description': f"Position jump: {current_distance:.6f} (local threshold: {threshold:.6f})"
                    })
        
        return anomalies
    
    def detect_temporal_inconsistencies(self):
        """Detect temporal inconsistencies that might indicate RTK timeouts."""
        anomalies = []
        
        if 'timestamp' not in self.df.columns:
            return anomalies
        
        # Check for irregular time intervals
        time_diffs = self.df['time_diff'].dropna()
        if len(time_diffs) == 0:
            return anomalies
            
        time_median = time_diffs.median()
        time_mad = np.median(np.abs(time_diffs - time_median))
        
        # Detect time gaps that might indicate data loss
        time_threshold = time_median + 3 * time_mad
        
        time_outliers = self.df[self.df['time_diff'] > time_threshold]
        
        for idx, row in time_outliers.iterrows():
            anomalies.append({
                'index': idx,
                'type': 'time_gap',
                'value': row['time_diff'],
                'threshold': time_threshold,
                'severity': row['time_diff'] / time_threshold,
                'description': f"Time gap: {row['time_diff']:.2f}s (expected: ~{time_median:.2f}s)"
            })
        
        return anomalies
    
    def detect_bearing_anomalies(self):
        """Detect sudden bearing changes that might indicate RTK jumps."""
        anomalies = []
        
        if 'bearing_change' not in self.df.columns:
            return anomalies
        
        # Only look at significant bearing changes (not normal turns)
        bearing_changes = self.df['bearing_change'].dropna()
        if len(bearing_changes) == 0:
            return anomalies
        
        # Use a high threshold - only catch really sudden direction changes
        bearing_threshold = 60  # degrees - sudden direction change
        
        sudden_turns = self.df[self.df['bearing_change'] > bearing_threshold]
        
        for idx, row in sudden_turns.iterrows():
            # Only flag if it's also associated with a distance anomaly
            if 'step_distance' in self.df.columns and pd.notna(row['step_distance']):
                distance_severity = row['step_distance'] / self.df['step_distance'].median()
                if distance_severity > 2:  # Only if distance is also anomalous
                    anomalies.append({
                        'index': idx,
                        'type': 'bearing_jump',
                        'value': row['bearing_change'],
                        'threshold': bearing_threshold,
                        'severity': distance_severity,
                        'description': f"Sudden bearing change: {row['bearing_change']:.1f}° with distance anomaly"
                    })
        
        return anomalies
    
    def detect_straight_line_deviations(self):
        """Detect points that deviate from straight survey lines using smart segmentation."""
        anomalies = []
        
        if len(self.df) < 10:
            return anomalies
        
        # First, identify straight line segments by looking for consistent bearing
        segments = self._identify_straight_segments()
        
        print(f"Identified {len(segments)} straight line segments for analysis")
        
        for segment in segments:
            if segment['length'] < 5:  # Skip very short segments
                continue
                
            # Analyze deviations within this straight segment
            segment_anomalies = self._detect_deviations_in_straight_segment(segment)
            anomalies.extend(segment_anomalies)
        
        return anomalies
    
    def _identify_straight_segments(self):
        """Identify straight line segments in the flight path."""
        segments = []
        
        # Calculate smoothed bearing to reduce noise
        window = 5
        self.df['bearing_smooth'] = self.df['bearing'].rolling(window=window, center=True).median()
        
        current_start = 0
        current_bearing = None
        bearing_tolerance = 20  # degrees
        
        for i in range(window, len(self.df) - window):
            current_smooth_bearing = self.df.loc[i, 'bearing_smooth']
            
            if pd.isna(current_smooth_bearing):
                continue
                
            if current_bearing is None:
                current_bearing = current_smooth_bearing
                current_start = i
                continue
            
            # Check if bearing has changed significantly
            bearing_diff = abs(current_smooth_bearing - current_bearing)
            if bearing_diff > 180:  # Handle wraparound
                bearing_diff = 360 - bearing_diff
                
            if bearing_diff > bearing_tolerance:
                # End current segment if it's long enough
                if i - current_start >= 10:
                    segments.append({
                        'start': current_start,
                        'end': i - 1,
                        'bearing': current_bearing,
                        'length': i - current_start
                    })
                
                # Start new segment
                current_bearing = current_smooth_bearing
                current_start = i
        
        # Add final segment
        if len(self.df) - current_start >= 10:
            segments.append({
                'start': current_start,
                'end': len(self.df) - 1,
                'bearing': current_bearing,
                'length': len(self.df) - current_start
            })
        
        return segments
    
    def _detect_deviations_in_straight_segment(self, segment):
        """Detect deviations from straightness within a segment."""
        anomalies = []
        start_idx, end_idx = segment['start'], segment['end']
        
        if end_idx - start_idx < 5:
            return anomalies
        
        # Get segment data
        seg_data = self.df.iloc[start_idx:end_idx + 1].copy()
        
        # Method 1: Perpendicular distance from ideal line
        x1, y1 = seg_data['lon'].iloc[0], seg_data['lat'].iloc[0]
        x2, y2 = seg_data['lon'].iloc[-1], seg_data['lat'].iloc[-1]
        
        # Calculate perpendicular distances
        distances = []
        for i, (_, row) in enumerate(seg_data.iterrows()):
            if i == 0 or i == len(seg_data) - 1:  # Skip endpoints
                distances.append(0)
                continue
                
            x0, y0 = row['lon'], row['lat']
            
            # Perpendicular distance formula
            if x2 != x1 or y2 != y1:
                dist = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            else:
                dist = 0
            distances.append(dist)
        
        # Use conservative threshold for line deviations
        if len(distances) > 2:
            distances_array = np.array(distances)
            median_dist = np.median(distances_array[distances_array > 0])
            mad_dist = np.median(np.abs(distances_array - median_dist))
            
            # Conservative threshold - only catch significant deviations
            threshold = median_dist + 2.5 * mad_dist
            
            for i, (global_idx, dist) in enumerate(zip(seg_data.index, distances)):
                if i == 0 or i == len(seg_data) - 1:  # Skip endpoints
                    continue
                    
                if dist > threshold and dist > 0:
                    # Additional check: make sure this isn't near a legitimate turn
                    if not self._is_near_turn(global_idx):
                        anomalies.append({
                            'index': global_idx,
                            'type': 'line_deviation',
                            'value': dist,
                            'threshold': threshold,
                            'severity': dist / threshold if threshold > 0 else 1,
                            'description': f"Line deviation: {dist:.6f} from straight path (threshold: {threshold:.6f})"
                        })
        
        return anomalies
    
    def _is_near_turn(self, idx, window=5):
        """Check if a point is near a turn (where deviations are expected)."""
        if 'bearing_change' not in self.df.columns:
            return False
            
        # Check if there are significant bearing changes nearby
        start = max(0, idx - window)
        end = min(len(self.df), idx + window + 1)
        
        nearby_bearing_changes = self.df.loc[start:end, 'bearing_change'].dropna()
        if len(nearby_bearing_changes) == 0:
            return False
            
        # If there are significant bearing changes nearby, this might be a legitimate turn
        max_bearing_change = nearby_bearing_changes.max()
        return max_bearing_change > 30  # degrees
    
    def analyze_anomalies(self):
        """Perform focused anomaly analysis targeting RTK timeout behavior."""
        print("Analyzing flight path for RTK timeout anomalies...")
        
        # Calculate movement metrics
        self.calculate_movement_metrics()
        
        # Run different anomaly detection methods
        velocity_anomalies = self.detect_velocity_anomalies()
        position_anomalies = self.detect_position_jumps()
        temporal_anomalies = self.detect_temporal_inconsistencies()
        bearing_anomalies = self.detect_bearing_anomalies()
        straight_line_deviations = self.detect_straight_line_deviations()
        
        # Combine all anomalies
        all_anomalies = velocity_anomalies + position_anomalies + temporal_anomalies + bearing_anomalies + straight_line_deviations
        
        # Remove duplicates (same index with different detection methods)
        unique_anomalies = {}
        for anomaly in all_anomalies:
            idx = anomaly['index']
            if idx not in unique_anomalies:
                unique_anomalies[idx] = anomaly
            else:
                # Keep the one with higher severity
                if anomaly['severity'] > unique_anomalies[idx]['severity']:
                    unique_anomalies[idx] = anomaly
        
        final_anomalies = list(unique_anomalies.values())
        
        # Print summary
        print(f"Found {len(final_anomalies)} genuine anomalies")
        anomaly_types = {}
        for anomaly in final_anomalies:
            atype = anomaly['type']
            anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        print("Anomaly types detected:")
        for atype, count in anomaly_types.items():
            print(f"  - {atype}: {count} points")
        
        return {
            'anomalies': final_anomalies,
            'velocity_anomalies': velocity_anomalies,
            'position_anomalies': position_anomalies,
            'temporal_anomalies': temporal_anomalies,
            'bearing_anomalies': bearing_anomalies,
            'straight_line_deviations': straight_line_deviations,
            'statistics': self.get_statistics()
        }
    
    def get_statistics(self):
        """Get basic statistics about the flight."""
        stats = {}
        
        if 'step_distance' in self.df.columns:
            valid_distances = self.df['step_distance'].dropna()
            stats['distance'] = {
                'mean': valid_distances.mean(),
                'median': valid_distances.median(),
                'std': valid_distances.std(),
                'max': valid_distances.max(),
                'min': valid_distances.min()
            }
        
        if 'speed' in self.df.columns:
            valid_speeds = self.df['speed'].dropna()
            if len(valid_speeds) > 0:
                stats['speed'] = {
                    'mean': valid_speeds.mean(),
                    'median': valid_speeds.median(),
                    'std': valid_speeds.std(),
                    'max': valid_speeds.max(),
                    'min': valid_speeds.min()
                }
        
        if 'time_diff' in self.df.columns:
            valid_times = self.df['time_diff'].dropna()
            if len(valid_times) > 0:
                stats['time_intervals'] = {
                    'mean': valid_times.mean(),
                    'median': valid_times.median(),
                    'std': valid_times.std(),
                    'max': valid_times.max(),
                    'min': valid_times.min()
                }
        
        return stats
    
    def create_visualization(self, analysis_results, output_file=None):
        """Create an interactive visualization focusing on genuine anomalies."""
        fig = go.Figure()
        
        # Plot the main flight path
        fig.add_trace(go.Scatter(
            x=self.df['lon'],
            y=self.df['lat'],
            mode='markers+lines',
            marker=dict(size=3, color='blue', opacity=0.6),
            line=dict(width=1, color='rgba(0,0,255,0.4)'),
            name='Flight Path',
            text=[f"Point {i}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}" + 
                  (f"<br>Speed: {speed:.2f}" if 'speed' in self.df.columns and not pd.isna(speed) else "") +
                  (f"<br>Distance: {dist:.6f}" if 'step_distance' in self.df.columns and not pd.isna(dist) else "")
                  for i, (lat, lon, speed, dist) in enumerate(zip(
                      self.df['lat'], self.df['lon'], 
                      self.df.get('speed', [np.nan]*len(self.df)),
                      self.df.get('step_distance', [np.nan]*len(self.df))
                  ))],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Highlight anomalies by type
        if analysis_results['anomalies']:
            # Group anomalies by type
            anomaly_types = {}
            for anomaly in analysis_results['anomalies']:
                atype = anomaly['type']
                if atype not in anomaly_types:
                    anomaly_types[atype] = []
                anomaly_types[atype].append(anomaly)
            
            # Different styles for each anomaly type
            type_styles = {
                'velocity_spike': {'color': 'red', 'symbol': 'x', 'size': 10},
                'position_jump': {'color': 'orange', 'symbol': 'circle-open', 'size': 8},
                'time_gap': {'color': 'purple', 'symbol': 'diamond', 'size': 8},
                'bearing_jump': {'color': 'yellow', 'symbol': 'triangle-up', 'size': 8},
                'line_deviation': {'color': 'green', 'symbol': 'circle', 'size': 8}
            }
            
            for atype, anomalies in anomaly_types.items():
                anomaly_indices = [a['index'] for a in anomalies]
                anomaly_data = self.df.loc[anomaly_indices]
                style = type_styles.get(atype, {'color': 'black', 'symbol': 'circle', 'size': 6})
                
                fig.add_trace(go.Scatter(
                    x=anomaly_data['lon'],
                    y=anomaly_data['lat'],
                    mode='markers',
                    marker=dict(
                        size=style['size'], 
                        color=style['color'], 
                        symbol=style['symbol'], 
                        line=dict(width=2)
                    ),
                    name=f'{atype.replace("_", " ").title()} ({len(anomalies)})',
                    text=[f"{a['description']}<br>Point {a['index']}<br>Severity: {a['severity']:.2f}x"
                          for a in anomalies],
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        # Add start and end markers
        fig.add_trace(go.Scatter(
            x=[self.df['lon'].iloc[0]],
            y=[self.df['lat'].iloc[0]],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='star'),
            text=['START'],
            textposition='top center',
            name='Start'
        ))
        
        fig.add_trace(go.Scatter(
            x=[self.df['lon'].iloc[-1]],
            y=[self.df['lat'].iloc[-1]],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='star'),
            text=['END'],
            textposition='top center',
            name='End'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'RTK Timeout Detection Analysis<br><sub>Total Points: {len(self.df)} | Genuine Anomalies: {len(analysis_results["anomalies"])}</sub>',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            hovermode='closest',
            width=1200,
            height=800,
            showlegend=True
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Visualization saved to: {output_file}")
        
        return fig
    
    def create_cleaned_dataset(self, analysis_results, output_file):
        """Create a cleaned dataset with genuine anomalies removed."""
        indices_to_remove = set()
        
        # Only remove the detected anomalies
        for anomaly in analysis_results['anomalies']:
            indices_to_remove.add(anomaly['index'])
        
        # Create cleaned dataframe
        cleaned_df = self.df[~self.df.index.isin(indices_to_remove)].copy()
        
        # Remove helper columns
        helper_columns = ['point_index', 'lat_diff', 'lon_diff', 'step_distance', 
                         'time_diff', 'speed', 'acceleration', 'bearing', 'bearing_change',
                         'distance_rolling_median', 'distance_rolling_mad', 'bearing_smooth']
        for col in helper_columns:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df.drop(columns=[col])
        
        # Save cleaned dataset
        cleaned_df.to_csv(output_file, index=False)
        
        print(f"\nCleaning Results:")
        print(f"Original data points: {len(self.df)}")
        print(f"Genuine anomalies removed: {len(indices_to_remove)}")
        print(f"Cleaned data points: {len(cleaned_df)}")
        print(f"Data retention: {len(cleaned_df)/len(self.df)*100:.1f}%")
        print(f"Cleaned dataset saved to: {output_file}")
        
        return cleaned_df, indices_to_remove

def main():
    parser = argparse.ArgumentParser(description='Detect genuine RTK timeout anomalies in drone flight paths')
    parser.add_argument('csv_file', help='Path to the CSV file containing flight data')
    parser.add_argument('--output-html', help='Output HTML file for visualization', 
                       default='rtk_timeout_analysis.html')
    parser.add_argument('--output-clean', help='Output CSV file for cleaned data',
                       default='rtk_anomalies_removed.csv')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found")
        return
    
    # Initialize detector and run analysis
    detector = FlightAnomalyDetector(args.csv_file)
    results = detector.analyze_anomalies()
    
    # Print summary
    print(f"\nRTK TIMEOUT DETECTION SUMMARY")
    print("=" * 50)
    print(f"File: {args.csv_file}")
    print(f"Total data points: {len(detector.df)}")
    print(f"Genuine anomalies detected: {len(results['anomalies'])}")
    
    if results['statistics'].get('distance'):
        dist_stats = results['statistics']['distance']
        print(f"\nMovement Statistics:")
        print(f"  Mean step distance: {dist_stats['mean']:.6f}")
        print(f"  Median step distance: {dist_stats['median']:.6f}")
        print(f"  Max step distance: {dist_stats['max']:.6f}")
    
    if results['statistics'].get('speed'):
        speed_stats = results['statistics']['speed']
        print(f"\nSpeed Statistics:")
        print(f"  Mean speed: {speed_stats['mean']:.6f}")
        print(f"  Median speed: {speed_stats['median']:.6f}")
        print(f"  Max speed: {speed_stats['max']:.6f}")
    
    # Create visualization
    fig = detector.create_visualization(results, args.output_html)
    
    # Create cleaned dataset
    cleaned_df, removed_indices = detector.create_cleaned_dataset(results, args.output_clean)
    
    print(f"\nFiles created:")
    print(f"  - Visualization: {args.output_html}")
    print(f"  - Cleaned data: {args.output_clean}")

if __name__ == "__main__":
    main() 