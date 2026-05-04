#!/usr/bin/env python3
"""
Create an evolution animation (GIF) from training visualizations.
Compiles poincare_disk_native.png from each epoch into a single GIF.
"""

import os
import argparse
from pathlib import Path
from PIL import Image

def create_animation(run_dir: str, output_name: str = "evolution.gif", duration: int = 200):
    vis_dir = Path(run_dir) / "visualizations"
    if not vis_dir.exists():
        print(f"Error: Visualization directory not found at {vis_dir}")
        return

    # Find all poincare_disk_native.png files
    image_paths = sorted(list(vis_dir.glob("epoch_*/poincare_disk_native.png")))
    
    if not image_paths:
        print(f"Error: No poincare_disk_native.png files found in {vis_dir}")
        return

    print(f"Found {len(image_paths)} frames. Creating GIF...")
    
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)

    # Save as GIF
    output_path = Path(run_dir) / output_name
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    print(f"Success: Animation saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory")
    parser.add_argument("--output", type=str, default="evolution.gif", help="Output filename")
    parser.add_argument("--duration", type=int, default=200, help="Duration per frame in ms")
    args = parser.parse_args()
    
    create_animation(args.run_dir, args.output, args.duration)
