#!/usr/bin/python3
from generate import DB_PATH, process_images, ingest_videos, model, preprocess, BASE_PATH
import argparse
import json
import os
import subprocess
from pathlib import Path
import time
import sqlite_vec
import clip
import torch
from PIL import Image
import sqlite3
data = []
import pytesseract


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
    if "GPSLongitude" in exif_data and "GPSLatitude" in exif_data:
        lon = gps_to_decimal(exif_data["GPSLongitude"])
        lat = gps_to_decimal(exif_data["GPSLatitude"])
        if lon is not None and lat is not None:
            return [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                },
                lat,
                lon,
            ]
    return ["", 0.0, 0.0]


def get_text_from_image(file_path):
    try:
        return " ".join(
            pytesseract.image_to_string(Image.open(file_path), lang="eng+rus").split()
        )
    except Exception:
        return ""


def parse_exiftool_json(json_data):
    for item in json_data:
        filename = os.path.basename(item.get("SourceFile", ""))
        if filename.startswith("."):
            continue  # skip dot-files

        path = Path(item.get("SourceFile"))
        subfolder = path.parent.name
        file_name = path.name

        location, lat, lon = get_location(item)
        created_date = (
            item.get("CreateDate", "").split(" ")[0] if "CreateDate" in item else ""
        )
        height = item.get("ExifImageHeight")
        width = item.get("ExifImageWidth")
        media_type = "video" if path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'} else "photo"
        row = [
            str(path.relative_to(path.parents[1])).replace(
                " ", "_"
            ),  # relative path with subfolder
            file_name,
            subfolder,
            created_date,
            height,
            width,
            json.dumps(location) if location else "",
            # get_text_from_image(path),
            "",
            lat,
            lon,
            media_type,
        ]
        yield row


def run_exiftool(directory):
    output_file = os.path.join(directory, "output.json")
    with open(output_file, "w") as f:
        subprocess.run(
            [
                "exiftool",
                "-r",  # recursive
                "-Make",
                "-CreateDate",
                "-ExifImageHeight",
                "-ExifImageWidth",
                "-GPSLongitude",
                "-GPSLatitude",
                "-j",  # JSON output
                directory,
            ],
            stdout=f,
        )


def seed_database_with_exif(json_input):
    """Parses Exiftool JSON and provisions the base database rows."""
    conn = sqlite3.connect(DB_PATH)
    
    with open(json_input) as f:
        exif_data = json.load(f)
        
    print("Populating database with EXIF metadata...")
    for row in parse_exiftool_json(exif_data):
        rel_path, file_name, subfolder, created_date, height, width, loc_json, _, lat, lon, media_type = row
        
        conn.execute("""
            INSERT OR IGNORE INTO photos 
            (filename, subfolder, date, height, width, location, text, lat, lon, path, media_type)
            VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)
        """, (file_name, subfolder, created_date, height, width, loc_json, lat, lon, rel_path, media_type))
        
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="generate", description="Generate metadata")
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--text", required=False)
    # group.add_argument("--image", required=False)
    # group.add_argument("--file", required=False)
    
    parser.add_argument(
        "--append", action="store_true", default=False
    )
    parser.add_argument("--directory", required=False, default = BASE_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    append = args.append
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    print(f"using {device}")
    device = torch.device(device)
    # model, preprocess = clip.load("ViT-B/32", device=device)
    images = []
    base_folder = Path(args.directory)
    print(f"[1/4] Scanning directories and running ExifTool on: {base_folder}")
    run_exiftool(directory=args.directory)
    
    print("[2/4] Seeding raw metadata into SQLite database...")
    raw_json_output = os.path.join(args.directory, "output.json")
    seed_database_with_exif(raw_json_output)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        print("[3/4] Initializing AI Models & starting Photo Vector Pipeline...")
        process_images(conn, append=append,base_path=base_folder, batch_size=args.batch_size)
        
        print("[4/4] Starting Video Scene/Face Extraction Pipeline...")
        ingest_videos(conn, append=append, base_path=base_folder)
    finally:
        conn.close()
    print("All assets successfully scanned, cataloged, and indexed!")