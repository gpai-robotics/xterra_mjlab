#!/usr/bin/env python3
"""Export unique scenario videos from a benchmark submission with clear, descriptive naming.

Usage:
    python3 scripts/export_unique_scenario_videos.py [OPTIONS]

Options:
    --submission-dir DIR   Path to submission run dir (default: runs/submissions/go2_blind)
    --dest-dir DIR         Target directory for unique videos (default: unique_scenario_videos)
    --move                 Move files instead of copying (default: False, copies safely)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

TERRAIN_DESCRIPTIONS = {
    "flat": "flat_ground",
    "rs1": "slippery_flat",
    "rs2": "rough_slippery_ground",
    "rs3": "slippery_slope",
    "rs4": "rough_slippery_slope",
    "rs5": "stairs",
    "rs6": "repeated_waves",
    "rs7": "curved_hill",
    "rs8": "small_repeated_gaps",
    "rs9": "large_gap",
    "rs10": "high_step",
    "rs11": "marsh_mud",
}


def parse_condition_id(cid: str) -> Tuple[str, str, str, str]:
    """Extract level, terrain_info, geom, friction from condition id."""
    parts = cid.split("-")
    level = parts[0] if len(parts) > 0 else "L?"
    
    # Handle pairwise L2 like L2-rs1-to-rs2-g050
    if "to" in parts:
        try:
            to_idx = parts.index("to")
            t1 = parts[to_idx - 1]
            t2 = parts[to_idx + 1]
            terrain_info = f"{t1}_to_{t2}"
            geom = parts[to_idx + 2] if len(parts) > to_idx + 2 else ""
            frict = parts[to_idx + 3] if len(parts) > to_idx + 3 else ""
            return level, terrain_info, geom, frict
        except Exception:
            pass

    terrain = parts[1] if len(parts) > 1 else ""
    geom = parts[2] if len(parts) > 2 else ""
    frict = parts[3] if len(parts) > 3 else ""
    return level, terrain, geom, frict


def format_scenario_name(cid: str, status: str) -> str:
    """Format a clean, readable video name that sorts by Level and displays outcome."""
    level, terrain, geom, frict = parse_condition_id(cid)
    status_tag = status.upper() if status else "UNKNOWN"
    
    # Resolve human-readable terrain name
    if "_to_" in terrain:
        t1, t2 = terrain.split("_to_")
        desc1 = TERRAIN_DESCRIPTIONS.get(t1, t1)
        desc2 = TERRAIN_DESCRIPTIONS.get(t2, t2)
        terrain_str = f"{t1}-{desc1}_to_{t2}-{desc2}"
    else:
        desc = TERRAIN_DESCRIPTIONS.get(terrain, terrain)
        terrain_str = f"{terrain}_{desc}" if terrain and desc != terrain else (terrain or "flat")
        
    components = [level, status_tag, terrain_str]
    if geom:
        components.append(geom)
    if frict:
        components.append(frict)
        
    return "__".join(components) + ".mp4"


def main():
    parser = argparse.ArgumentParser(description="Export unique scenario videos with clean names")
    parser.add_argument(
        "--submission-dir",
        default="runs/submissions/go2_blind",
        help="Path to submission run dir (default: runs/submissions/go2_blind)",
    )
    parser.add_argument(
        "--dest-dir",
        default="unique_scenario_videos",
        help="Target folder (default: unique_scenario_videos)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move videos instead of copying (default: copy)",
    )
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir).resolve()
    dest_dir = Path(args.dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load condition metadata if available
    conditions_csv = sub_dir / "dashboard" / "conditions.csv"
    condition_meta: Dict[str, Dict[str, str]] = {}
    if conditions_csv.exists():
        with open(conditions_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("id")
                if cid:
                    condition_meta[cid] = row

    # 2. Gather candidate video files
    # Priority is given to real 3D simulation camera recordings in trials/ and extracted_videos/
    # (evidence/videos contains 2D top-down diagrammatic plots, which we avoid).
    search_patterns = [
        str(sub_dir / "extracted_videos" / "*.mp4"),
        str(sub_dir / "trials" / "*" / "video" / "*.mp4"),
    ]
    
    all_videos = []
    for pattern in search_patterns:
        all_videos.extend(glob.glob(pattern))

    print(f"Found {len(all_videos)} video files across submission paths.")

    # 3. Map videos by condition ID
    # A video filename typically looks like <condition_id>-a<attempt>.mp4
    videos_by_condition: Dict[str, Dict[str, str]] = {}
    for vpath in all_videos:
        fname = Path(vpath).name
        stem = Path(vpath).stem
        # Remove attempt suffix like -a01, -a02
        if "-a" in stem:
            cid, attempt = stem.rsplit("-a", 1)
            attempt_str = f"a{attempt}"
        else:
            cid = stem
            attempt_str = "a01"
            
        if cid not in videos_by_condition:
            videos_by_condition[cid] = {}
        videos_by_condition[cid][attempt_str] = vpath

    print(f"Discovered {len(videos_by_condition)} unique scenarios with videos.")

    # 4. For each unique scenario, pick the representative video and export
    copied_count = 0
    records = []
    
    for cid in sorted(videos_by_condition.keys()):
        attempts = videos_by_condition[cid]
        meta = condition_meta.get(cid, {})
        status = meta.get("status", "unknown")
        
        # Pick best video:
        # If passed, try representative_pass attempt first, otherwise earliest attempt
        rep_pass = meta.get("representative_pass", "")
        chosen_attempt = None
        if rep_pass:
            rep_att = rep_pass.split("-")[-1] if "-a" in rep_pass else rep_pass
            if rep_att in attempts:
                chosen_attempt = rep_att
        if not chosen_attempt:
            # Pick a01 if available, otherwise first sorted attempt
            chosen_attempt = "a01" if "a01" in attempts else sorted(attempts.keys())[0]
            
        src_path = attempts[chosen_attempt]
        new_filename = format_scenario_name(cid, status)
        dst_path = dest_dir / new_filename

        action_name = "Moving" if args.move else "Copying"
        try:
            if args.move:
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            copied_count += 1
            records.append({
                "condition_id": cid,
                "status": status,
                "attempt_used": chosen_attempt,
                "filename": new_filename,
                "size_mb": round(os.path.getsize(dst_path) / (1024 * 1024), 2),
            })
        except Exception as err:
            print(f"Error transferring {src_path} -> {dst_path}: {err}", file=sys.stderr)

    # 5. Generate an index Markdown file in dest_dir
    index_md = dest_dir / "INDEX.md"
    with open(index_md, "w") as f:
        f.write("# Unique Scenario Benchmark Videos\n\n")
        f.write(f"Source: `{sub_dir}`\n")
        f.write(f"Total Unique Scenarios Exported: **{copied_count}**\n\n")
        f.write("| Status | Level | Scenario ID | Exported File | Size (MB) |\n")
        f.write("| :---: | :---: | :--- | :--- | :---: |\n")
        for r in records:
            status_icon = "✅ PASSED" if r["status"] == "passed" else ("❌ FAILED" if r["status"] == "failed" else r["status"])
            level = r["condition_id"].split("-")[0]
            f.write(f"| {status_icon} | {level} | `{r['condition_id']}` | [{r['filename']}](./{r['filename']}) | {r['size_mb']} |\n")

    print(f"\nSuccessfully exported {copied_count} unique scenario videos to:")
    print(f"  {dest_dir}")
    print(f"Index created at:")
    print(f"  {index_md}\n")


if __name__ == "__main__":
    main()
