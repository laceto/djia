"""Main analysis script for DJIA - DJ Mixing Analytics."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from src.audio_analysis import analyze_track
from src.mixing_metrics import score_track, calculate_overall_score

def extract_track_name(file_path: str) -> str:
    """Extract clean track name from file path."""
    filename = os.path.basename(file_path)
    return filename.replace('.mp3', '').replace('.m4a', '')


def analyze_all_tracks(data_dir: str) -> pd.DataFrame:
    """Analyze all tracks in directory and return DataFrame."""
    results = []

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
        audio_files.extend(Path(data_dir).glob(ext))

    print(f"Found {len(audio_files)} audio files to analyze...")

    for i, file_path in enumerate(sorted(audio_files), 1):
        try:
            full_path = str(file_path.resolve())
            print(f"[{i}/{len(audio_files)}] Analyzing: {os.path.basename(full_path)}")
            features = analyze_track(full_path)
            
            if features:
                track_name = extract_track_name(str(file_path))
                features['track_name'] = track_name
                
                scores = score_track(features)
                features.update(scores)
                
                overall_score = calculate_overall_score(scores)
                features['overall_score'] = overall_score
                
                results.append(features)
                print(f"  [OK] Overall Score: {overall_score:.1f}")
        except Exception as e:
            print(f"  [ERROR] {str(e)[:50]}")
    
    return pd.DataFrame(results)


def generate_dj_setlist(df: pd.DataFrame, num_tracks: int = 15) -> pd.DataFrame:
    """Generate optimized DJ setlist."""
    df = df.sort_values('overall_score', ascending=False)
    
    candidates = df.head(num_tracks * 3).copy()
    
    setlist = []
    tempos = []
    
    for idx, row in candidates.iterrows():
        if len(setlist) >= num_tracks:
            break
        
        tempo = row['tempo']
        
        if len(tempos) == 0:
            setlist.append(row)
            tempos.append(tempo)
        else:
            min_diff = min(abs(tempo - t) for t in tempos)
            
            if min_diff >= 3:
                setlist.append(row)
                tempos.append(tempo)
            elif len(setlist) < num_tracks * 0.3:
                setlist.append(row)
                tempos.append(tempo)
    
    setlist_df = pd.DataFrame(setlist)
    setlist_df = setlist_df.sort_values('tempo').reset_index(drop=True)
    setlist_df['position'] = range(1, len(setlist_df) + 1)
    
    return setlist_df


def create_analysis_report(df: pd.DataFrame, setlist: pd.DataFrame, output_dir: str = 'results'):
    """Create comprehensive analysis report."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'all_tracks_analysis.csv')
    df_export = df[['track_name', 'tempo', 'duration', 'overall_score', 
                     'mixing_quality', 'danceability', 'harmonic_richness', 'production_quality']].copy()
    df_export = df_export.sort_values('overall_score', ascending=False)
    df_export.to_csv(csv_path, index=False)
    print(f"[OK] Full analysis saved: {csv_path}")

    setlist_path = os.path.join(output_dir, 'dj_setlist.csv')
    setlist_export = setlist[['position', 'track_name', 'tempo', 'duration', 'overall_score',
                              'mixing_quality', 'danceability', 'harmonic_richness', 'production_quality']].copy()
    setlist_export.to_csv(setlist_path, index=False)
    print(f"[OK] DJ Setlist saved: {setlist_path}")
    
    stats_path = os.path.join(output_dir, 'analysis_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DJIA - DJ MIXING ANALYTICS REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Tracks Analyzed: {len(df)}\n")
        f.write(f"Average Tempo: {df['tempo'].mean():.1f} BPM\n")
        f.write(f"Tempo Range: {df['tempo'].min():.1f} - {df['tempo'].max():.1f} BPM\n")
        f.write(f"Average Duration: {df['duration'].mean():.1f} seconds ({df['duration'].mean()/60:.1f} min)\n")
        f.write(f"Average Overall Score: {df['overall_score'].mean():.1f}\n\n")
        
        f.write("SCORE DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mixing Quality (avg): {df['mixing_quality'].mean():.1f}\n")
        f.write(f"Danceability (avg): {df['danceability'].mean():.1f}\n")
        f.write(f"Harmonic Richness (avg): {df['harmonic_richness'].mean():.1f}\n")
        f.write(f"Production Quality (avg): {df['production_quality'].mean():.1f}\n\n")
        
        f.write("TOP 10 TRACKS\n")
        f.write("-" * 70 + "\n")
        top10 = df.nlargest(10, 'overall_score')
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            f.write(f"{i:2d}. {row['track_name']}\n")
            f.write(f"    Score: {row['overall_score']:.1f} | Tempo: {row['tempo']:.1f} BPM | Duration: {row['duration']:.0f}s\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDED DJ SETLIST\n")
        f.write("=" * 70 + "\n\n")
        for idx, row in setlist.iterrows():
            f.write(f"{int(row['position']):2d}. {row['track_name']}\n")
            f.write(f"    {row['tempo']:.0f} BPM | {row['duration']:.0f}s | Score: {row['overall_score']:.1f}\n")
    
    print(f"[OK] Statistics saved: {stats_path}")
    
    return csv_path, setlist_path, stats_path


def main():
    """Main execution."""
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("DJIA - DJ MIXING ANALYTICS")
    print("=" * 70 + "\n")
    
    print("Step 1: Analyzing all tracks...")
    df = analyze_all_tracks(data_dir)
    
    if df.empty:
        print("No tracks were successfully analyzed!")
        sys.exit(1)
    
    print(f"\n[OK] Successfully analyzed {len(df)} tracks!\n")

    print("Step 2: Generating DJ setlist...")
    setlist = generate_dj_setlist(df, num_tracks=15)
    print(f"[OK] Generated setlist with {len(setlist)} tracks!\n")
    
    print("Step 3: Creating reports...")
    csv_path, setlist_path, stats_path = create_analysis_report(df, setlist)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70 + "\n")
    print("Top 5 Recommended Tracks:")
    top5 = df.nlargest(5, 'overall_score')
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"{i}. {row['track_name']} ({row['overall_score']:.1f} pts, {row['tempo']:.0f} BPM)")
    
    print("\nDJ Setlist (in tempo progression):")
    for idx, row in setlist.iterrows():
        print(f"{int(row['position']):2d}. {row['track_name']} ({row['tempo']:.0f} BPM, {row['overall_score']:.1f} pts)")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
