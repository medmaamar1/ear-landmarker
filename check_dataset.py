import os
import json
import numpy as np
from collections import Counter

def check_dataset_integrity(data_dir):
    """
    Checks the consistency of landmarks across a dataset of LabelMe JSONs.
    Identifies index shifting, missing shapes, and variable point counts.
    """
    print(f"Analyzing dataset at: {data_dir}\n")
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON files found.")
        return

    stats = {
        'total_files': len(json_files),
        'point_counts': [],
        'missing_shapes': Counter(),
        'shape_point_counts': {}, # label -> list of counts
    }

    for f in json_files:
        path = os.path.join(data_dir, f)
        try:
            with open(path) as jf:
                data = json.load(jf)
                
            labels_found = [s.get('label') for s in data.get('shapes', [])]
            
            # Check for missing expected labels (0, 1, 2, 3)
            for expected in ["0", "1", "2", "3"]:
                if expected not in labels_found:
                    stats['missing_shapes'][expected] += 1
            
            total_pts = 0
            for s in data.get('shapes', []):
                label = s.get('label')
                l_pts = len(s.get('points', []))
                total_pts += l_pts
                
                if label not in stats['shape_point_counts']:
                    stats['shape_point_counts'][label] = []
                stats['shape_point_counts'][label].append(l_pts)
                
            stats['point_counts'].append(total_pts)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # --- REPORT ---
    print(f"Total JSON files scanned: {stats['total_files']}")
    
    unique_counts = set(stats['point_counts'])
    print(f"\n[1] GLOBAL POINT CONSISTENCY:")
    if len(unique_counts) == 1:
        print(f"    SUCCESS: All images have exactly {list(unique_counts)[0]} points.")
    else:
        print(f"    CRITICAL: Found {len(unique_counts)} different versions of total point counts!")
        print(f"    Range: {min(unique_counts)} to {max(unique_counts)} points per image.")
        print(f"    (Common counts: {Counter(stats['point_counts']).most_common(3)})")

    print(f"\n[2] SHAPE BREAKDOWN (Index Shifting Risk):")
    for label, counts in sorted(stats['shape_point_counts'].items()):
        distinct = set(counts)
        if len(distinct) == 1:
            print(f"    Label '{label}': Stable ({list(distinct)[0]} points every time)")
        else:
            print(f"    Label '{label}': UNSTABLE! Varies between {min(distinct)} and {max(distinct)} points.")
            print(f"    (This is why the model is failing visual tests despite low loss!)")

    print(f"\n[3] MISSING SHAPES:")
    if not stats['missing_shapes']:
        print("    None! All images have all shapes.")
    else:
        for label, count in stats['missing_shapes'].items():
            print(f"    Label '{label}' is MISSING in {count} files.")

    print("\nCONCLUSION:")
    if len(unique_counts) > 1 or stats['missing_shapes']:
        print("!!! DATA BUG CONFIRMED !!!")
        print("Your landmarks are 'shifting' indices between images.")
        print("Example: Landmark #25 is an Earlobe in some photos and an Inner fold in others.")
        print("The model is learning an 'average smudge' of these. We must normalize the point counts.")
    else:
        print("Data structure is stable. Problem likely lies in Model Head / Hyperparameters.")

if __name__ == "__main__":
    import sys
    d_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    check_dataset_integrity(d_dir)
