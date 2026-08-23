#!/usr/bin/python3
#the only file to run
from generate import DB_PATH, process_images, ingest_videos, model, preprocess, BASE_PATH
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import torch
from PIL import Image
import sqlite_vec

from generate import (
    BASE_PATH,
    DB_PATH,
    ingest_videos,
    process_images,
    h3population
)


def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ["S", "W"]:
        decimal = -decimal
    return decimal


def gps_to_decimal(gps_str):
    try:
        parts = gps_str.split()
        deg = float(parts[0])
        minutes = float(parts[2].replace("'", ""))
        seconds = float(parts[3].replace('"', ""))
        direction = parts[4]
        return dms_to_decimal(deg, minutes, seconds, direction)
    except Exception:
        return None


def get_location(exif_data):
    """Extracts GPS coordinates and formats them as a GeoJSON Feature."""
    if "GPSLongitude" in exif_data and "GPSLatitude" in exif_data:
        lon = gps_to_decimal(exif_data["GPSLongitude"])
        lat = gps_to_decimal(exif_data["GPSLatitude"])
        if lon is not None and lat is not None:
            geojson_feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
            return geojson_feature, lat, lon
    return None, 0.0, 0.0


def parse_exiftool_json(json_data, target_directory):
    for item in json_data:
        source_file = item.get("SourceFile", "")
        if not source_file:
            continue

        path = Path(source_file).resolve()
        filename = path.name

        if filename.startswith("."):
            continue  # Skip hidden files

        subfolder = path.parent.name
        location_obj, lat, lon = get_location(item)
        location_str = json.dumps(location_obj) if location_obj else ""

        location, lat, lon = get_location(item)
        raw_date = item.get("CreateDate", "")
        created_date = (
            raw_date.split(" ")[0].replace(":", "-") if raw_date else ""
        )
        height = item.get("ExifImageHeight")
        width = item.get("ExifImageWidth")
        
        media_type = "video" if path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v'} else "photo"

        # Calculate path relative to BASE_PATH so 'Samsung' is included in the stored string
        try:
            rel_path = str(path.relative_to(Path(BASE_PATH).resolve()))
        except ValueError:
            # Fallback if outside BASE_PATH: use path relative to parent directory
            rel_path = os.path.join(Path(target_directory).name, path.name)

        yield (
            filename,
            subfolder,
            created_date,
            height,
            width,
            location_str,
            "",  # text placeholder
            lat,
            lon,
            rel_path,
            media_type,
        )


def run_exiftool(directory):
    output_file = os.path.join(directory, "output.json")
    subprocess.run(
        [
            "exiftool",
            "-r",
            "-Make",
            "-CreateDate",
            "-ExifImageHeight",
            "-ExifImageWidth",
            "-GPSLongitude",
            "-GPSLatitude",
            "-j",
            directory,
        ],
        stdout=open(output_file, "w"),
        check=True,
    )



def seed_database_with_exif(json_input, target_directory):
    """Parses ExifTool JSON and inserts rows into SQLite database."""
    conn = sqlite3.connect(DB_PATH)

    with open(json_input) as f:
        exif_data = json.load(f)

    print("Populating database with EXIF metadata...")
    rows = list(parse_exiftool_json(exif_data, target_directory))

    with conn:
        conn.executemany("""
            INSERT OR REPLACE INTO photos 
            (filename, subfolder, date, height, width, location, text, lat, lon, path, media_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    conn.close()
    print(f"Seeded/updated {len(rows)} asset records in SQLite.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="generate", description="Generate metadata")
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--directory", required=False, default=BASE_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    append = args.append
    base_folder = Path(args.directory).resolve()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    print(f"[1/4] Scanning directories and running ExifTool on: {base_folder}")
    run_exiftool(directory=str(base_folder))

    print("[2/4] Seeding raw metadata into SQLite database...")
    raw_json_output = os.path.join(base_folder, "output.json")
    seed_database_with_exif(raw_json_output, base_folder)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        print("[3/4] Initializing AI Models & starting Photo Vector Pipeline...")
        process_images(conn, append=append, base_path=str(base_folder), batch_size=args.batch_size)

        print("[4/4] Starting Video Scene/Face Extraction Pipeline...")
        ingest_videos(conn, append=append, base_path=str(base_folder))
        h3population()
    finally:
        conn.close()

    print("All assets successfully scanned, cataloged, and indexed!")