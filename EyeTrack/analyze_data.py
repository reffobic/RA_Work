#!/usr/bin/env python3
"""
Eye Tracking Data Analysis Script
Demonstrates how to view and analyze HDF5 eye tracking data
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def load_hdf5_data(filename):
    """Load eye tracking data from HDF5 file"""
    print(f"Loading data from: {filename}")
    
    with h5py.File(filename, 'r') as hf:
        # Print metadata
        if 'metadata' in hf:
            print("\n=== METADATA ===")
            for key, value in hf['metadata'].attrs.items():
                print(f"{key}: {value}")
        
        # Load all data into pandas DataFrame
        data = {}
        for col in hf.keys():
            if col != 'metadata':
                data[col] = hf[col][:]
        
        df = pd.DataFrame(data)
        print(f"\nLoaded {len(df)} data points")
        print(f"Time span: {df['timestamp'].min():.2f} to {df['timestamp'].max():.2f} seconds")
        
        return df

def analyze_gaze_patterns(df):
    """Analyze gaze patterns and create visualizations"""
    print("\n=== GAZE PATTERN ANALYSIS ===")
    
    # Calculate gaze statistics
    print(f"Average left pupil diameter: {df['left_pupil_diam'].mean():.2f}")
    print(f"Average right pupil diameter: {df['right_pupil_diam'].mean():.2f}")
    print(f"Gaze validity rate: {(df['gaze_valid'].sum() / len(df) * 100):.1f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Eye Tracking Data Analysis', fontsize=16)
    
    # 1. Pupil diameter over time
    axes[0, 0].plot(df['timestamp'], df['left_pupil_diam'], label='Left Eye', alpha=0.7)
    axes[0, 0].plot(df['timestamp'], df['right_pupil_diam'], label='Right Eye', alpha=0.7)
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Pupil Diameter')
    axes[0, 0].set_title('Pupil Diameter Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gaze direction (2D projection)
    valid_gaze = df[df['gaze_valid'] == True]
    if len(valid_gaze) > 0:
        axes[0, 1].scatter(valid_gaze['left_gaze_dir_x'], valid_gaze['left_gaze_dir_y'], 
                          alpha=0.6, s=10, label='Left Eye')
        axes[0, 1].scatter(valid_gaze['right_gaze_dir_x'], valid_gaze['right_gaze_dir_y'], 
                          alpha=0.6, s=10, label='Right Eye')
        axes[0, 1].set_xlabel('Gaze X')
        axes[0, 1].set_ylabel('Gaze Y')
        axes[0, 1].set_title('Gaze Direction Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Eye openness
    axes[1, 0].plot(df['timestamp'], df['left_eye_openness'], label='Left Eye', alpha=0.7)
    axes[1, 0].plot(df['timestamp'], df['right_eye_openness'], label='Right Eye', alpha=0.7)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Eye Openness')
    axes[1, 0].set_title('Eye Openness Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Focus distance
    axes[1, 1].plot(df['timestamp'], df['focus_actor_dist'], alpha=0.7)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Focus Distance')
    axes[1, 1].set_title('Focus Distance Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eye_tracking_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def prepare_ai_data(df):
    """Prepare data for AI/ML analysis"""
    print("\n=== AI DATA PREPARATION ===")
    
    # Create features for AI analysis
    ai_features = df.copy()
    
    # Add derived features
    ai_features['pupil_diameter_diff'] = ai_features['left_pupil_diam'] - ai_features['right_pupil_diam']
    ai_features['gaze_velocity'] = np.gradient(ai_features['avg_pupil_diam'])
    ai_features['focus_velocity'] = np.gradient(ai_features['focus_actor_dist'])
    
    # Create time windows for pattern analysis
    window_size = 50  # 50 data points window
    ai_features['rolling_pupil_mean'] = ai_features['avg_pupil_diam'].rolling(window=window_size).mean()
    ai_features['rolling_pupil_std'] = ai_features['avg_pupil_diam'].rolling(window=window_size).std()
    
    # Remove NaN values
    ai_features = ai_features.dropna()
    
    print(f"AI-ready dataset shape: {ai_features.shape}")
    print("Features available for AI:")
    for col in ai_features.columns:
        print(f"  - {col}")
    
    return ai_features

def export_for_ai(df, output_format='csv'):
    """Export data in AI-friendly format"""
    if output_format == 'csv':
        filename = 'eye_tracking_ai_ready.csv'
        df.to_csv(filename, index=False)
        print(f"Exported AI-ready data to: {filename}")
    elif output_format == 'numpy':
        filename = 'eye_tracking_ai_ready.npz'
        np.savez_compressed(filename, **{col: df[col].values for col in df.columns})
        print(f"Exported AI-ready data to: {filename}")
    elif output_format == 'hdf5':
        filename = 'eye_tracking_ai_ready.h5'
        df.to_hdf(filename, key='data', mode='w', complevel=9)
        print(f"Exported AI-ready data to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze eye tracking HDF5 data')
    parser.add_argument('filename', help='HDF5 file to analyze')
    parser.add_argument('--export', choices=['csv', 'numpy', 'hdf5'], 
                       help='Export AI-ready data in specified format')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_hdf5_data(args.filename)
        
        # Analyze patterns
        if not args.no_plot:
            analyze_gaze_patterns(df)
        
        # Prepare AI data
        ai_data = prepare_ai_data(df)
        
        # Export if requested
        if args.export:
            export_for_ai(ai_data, args.export)
        
        print("\n=== QUICK STATS ===")
        print(f"Total data points: {len(df)}")
        print(f"Valid gaze points: {df['gaze_valid'].sum()}")
        print(f"Average pupil diameter: {df['avg_pupil_diam'].mean():.2f}")
        print(f"Focus distance range: {df['focus_actor_dist'].min():.2f} - {df['focus_actor_dist'].max():.2f}")
        
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found")
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    main()
