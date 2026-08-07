#!/usr/bin/python3
import argparse
import json
from pathlib import Path
import os
import subprocess
import h3
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from tqdm import tqdm
import json
import sqlite3
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import sys
import unicodedata
current_time = datetime.now()
print(current_time)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
print('using', device)


#PART 2
#.schema DDL
# CREATE TABLE photos (
#             id             INTEGER PRIMARY KEY,
#             filename       TEXT UNIQUE,
#             subfolder      TEXT,
#             date           TEXT,
#             height         INTEGER,
#             width          INTEGER,
#             location       TEXT,
#             text           TEXT,
#             lat            REAL,
#             lon            REAL,
#             path           TEXT,
#             dino_embedding BLOB
#         , city TEXT, country TEXT, day_of_week INTEGER, h3_cell TEXT, embedding_v2 BLOB, media_type TEXT DEFAULT 'photo', video_path TEXT);
# CREATE TABLE faces (
#             id                INTEGER PRIMARY KEY,
#             photo_id          INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
#             facenet_embedding BLOB NOT NULL
#         );
# CREATE INDEX idx_date      ON photos(date);
# CREATE INDEX idx_subfolder ON photos(subfolder);
# CREATE INDEX idx_lat_lon   ON photos(lat, lon);
# CREATE INDEX idx_faces_photo_id ON faces(photo_id);
# CREATE INDEX idx_h3_cell ON photos(h3_cell);
# CREATE TABLE video_frames (
#         id             INTEGER PRIMARY KEY,
#         photo_id       INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
#         frame_index    INTEGER NOT NULL,
#         timestamp_ms   INTEGER,
#         dino_embedding BLOB
#     );
# CREATE INDEX idx_frames_photo_id ON video_frames(photo_id);
# CREATE INDEX idx_city ON photos(city);
# CREATE INDEX idx_country ON photos(country);
# CREATE VIRTUAL TABLE vec_photos USING vec0(
#   id INTEGER PRIMARY KEY,
#   dino_embedding float[768]
# );
# CREATE TABLE IF NOT EXISTS "vec_photos_info" (key text primary key, value any);
# CREATE TABLE IF NOT EXISTS "vec_photos_chunks"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,validity BLOB NOT NULL,rowids BLOB NOT NULL);
# CREATE TABLE sqlite_sequence(name,seq);
# CREATE TABLE IF NOT EXISTS "vec_photos_rowids"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id,chunk_id INTEGER,chunk_offset INTEGER);
# CREATE TABLE IF NOT EXISTS "vec_photos_vector_chunks00"(rowid PRIMARY KEY,vectors BLOB NOT NULL);
# CREATE VIRTUAL TABLE vec_faces USING vec0(
#   id INTEGER PRIMARY KEY,  -- This will match the photo_id or a unique face ID
#   facenet_embedding float[512]
# );
# CREATE TABLE IF NOT EXISTS "vec_faces_info" (key text primary key, value any);
# CREATE TABLE IF NOT EXISTS "vec_faces_chunks"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,validity BLOB NOT NULL,rowids BLOB NOT NULL);
# CREATE TABLE IF NOT EXISTS "vec_faces_rowids"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id,chunk_id INTEGER,chunk_offset INTEGER);
# CREATE TABLE IF NOT EXISTS "vec_faces_vector_chunks00"(rowid PRIMARY KEY,vectors BLOB NOT NULL);

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def find_fuzzy_path(relative_path, base_dir, default_subfolder="Samsung"):
    if not relative_path:
        return None

    # Ensure base_dir and relative_path are plain strings before unicodedata conversion
    base_dir = str(base_dir)
    clean_rel = str(relative_path).strip().lstrip('/')

    clean_rel = unicodedata.normalize('NFC', clean_rel)
    base_dir = unicodedata.normalize('NFC', base_dir)
    #TODO

    # 1. Try direct base path: /Volumes/T7/photos_from_icloud/path
    direct_path = os.path.join(base_dir, clean_rel)
    if os.path.exists(direct_path):
        return direct_path

    # 2. Try inside Samsung subfolder: /Volumes/T7/photos_from_icloud/Samsung/path
    samsung_path = os.path.join(base_dir, default_subfolder, clean_rel)
    if os.path.exists(samsung_path):
        return samsung_path

    # 3. Fuzzy fallback (handles filename variants inside Samsung subfolder)
    path_obj = Path(clean_rel)
    parent_str = str(path_obj.parent) if str(path_obj.parent) != '.' else ''
    filename = path_obj.name

    file_variants = [
        filename,
        filename.replace('_', ' '),
        filename.replace(' ', '_'),
        filename.lower()
    ]

    for f in dict.fromkeys(file_variants):
        # Construct path inside Samsung/parent_dir/file_variant
        candidate = os.path.join(base_dir, default_subfolder, parent_str, f) if parent_str else os.path.join(base_dir, default_subfolder, f)
        if os.path.exists(candidate):
            return candidate

    return None

def get_batch_embeddings(images, model, device):
    """
    Takes a list of PIL images, returns a list of binary blobs.
    """
    tensors = torch.stack([transform(img) for img in images]).to(device)
    
    with torch.no_grad():
        features = model.forward_features(tensors)
        
        embeddings = features['x_norm_clstoken']
        
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        
    return [emb.tobytes() for emb in embeddings]
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
print('using', device)


model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = model.to(device)
model.eval()
#cp /Volumes/T7/photos.db ~/Documents/photos.db

BASE_DIR = "/Volumes/T7/photos_from_icloud"


DB_PATH = "/Users/iamgeorgerieh/Documents/photos.db" 
BASE_PATH = "/Volumes/T7/photos_from_icloud"
MOUNT_PATH = "/Volumes/T7/photos_from_icloud"
THUMBS_DIR = "/Volumes/T7/photos_from_icloud-out/thumbs"
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.heic'}
FRAME_INTERVAL_SEC = 30
BATCH_SIZE = 50

os.makedirs(THUMBS_DIR, exist_ok=True)



def h3population():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT rowid, lat, lon 
        FROM photos 
        WHERE lat IS NOT NULL 
          AND lat IS NOT NULL 
          AND (h3_cell IS NULL OR h3_cell = '' OR h3_cell = 'None');
    """
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        print("Please check if column names match 'latitude', 'longitude', and 'h3_cell'.")
        conn.close()
        return

    if not rows:
        print("✓ All photos with GPS data already have an H3 index populated!")
        conn.close()
        return

    print(f"Found {len(rows)} photos missing an H3 Resolution 7 index. Processing...")

    updated_count = 0
    for rowid, lat, lon in rows:
        try:
            h3_index = h3.latlng_to_cell(float(lat), float(lon), 7)
            
            cursor.execute(
                "UPDATE photos SET h3_cell = ? WHERE rowid = ?", 
                (h3_index, rowid)
            )
            updated_count += 1
            
            if updated_count % 500 == 0:
                print(f"  Indexed {updated_count} photos...")
                
        except Exception as err:
            print(f"  Skipping row {rowid} due to conversion error: {err}")

    conn.commit()
    conn.close()
    print(f"Success! Fixed {updated_count} rows. Database index is now complete.")


    
def get_video_duration(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0

def extract_frame(video_path, timestamp_sec, out_path):
    result = subprocess.run([
        'ffmpeg', '-ss', str(timestamp_sec),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '2',
        '-y', out_path
    ], capture_output=True)
    return result.returncode == 0 and os.path.exists(out_path)

def get_dino_embeddings(pil_images):
    """Processes a list of PIL images and returns a list of binary blobs."""
    tensors = torch.stack([transform(img) for img in pil_images]).to(device)
    with torch.no_grad():
        features = model.forward_features(tensors)
        embeddings = torch.nn.functional.normalize(features['x_norm_clstoken'], dim=-1)
        return [emb.cpu().numpy().astype(np.float32).tobytes() for emb in embeddings]

def embed(img, model, device):
    with torch.no_grad():
        tensor = preprocess(img).unsqueeze(0).to(device)
        feats = model.forward_features(tensor)
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        emb = feats.squeeze(0).cpu().numpy()
    return normalize_vector(emb).astype(np.float32)

def ingest_videos(conn, append, base_path, frame_interval_sec=FRAME_INTERVAL_SEC):
    """
    Ingests and extracts frame embeddings + faces for videos matching base_path.
    """
    base_path = Path(base_path)
    folder_name = base_path.name

    folder_name = os.path.basename(os.path.normpath(base_path))
    path_pattern = f"%{folder_name}%"

    processed_paths = set()
    if append:
        # APPEND MODE: Get paths in THIS directory that ALREADY have embeddings so we can skip them
        rows = conn.execute(
            "SELECT path FROM photos WHERE media_type='video' AND path LIKE ?",
            (path_pattern,)
        ).fetchall()
        processed_paths = {r[0] for r in rows}
    else:
        # RE-INDEX MODE: Clear processed_paths so ALL videos in this directory get re-processed
        print(f"Re-indexing ALL videos matching directory pattern: '{folder_name}'...")
        processed_paths = set()
    # 2. Find all videos recursively inside base_path
    videos = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith('.'):
                continue
            file_path = Path(root) / file
            if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                filename_str = str(file_path)
                # Skip if append mode is active and video is already done
                if append and filename_str in processed_paths:
                    continue
                videos.append(file_path)

    videos = sorted(videos)
    print(f"Videos to process in '{folder_name}': {len(videos)}")
    if not videos:
        print("Nothing to process.")
        return

    inserted_videos = 0
    inserted_frames = 0
    skipped = 0

    for video_path in tqdm(videos, desc="Processing videos"):
        filename = str(video_path)
        duration = get_video_duration(filename)
        if duration == 0:
            skipped += 1
            continue

        subfolder = video_path.parent.name

        try:
            mtime = os.path.getmtime(filename)
            date = datetime.fromtimestamp(mtime).strftime('%Y:%m:%d')
        except Exception:
            date = ""

        # Use an explicit SQLite transaction per video
        with conn:
            # Check if record already exists or insert a new one
            cursor = conn.execute("SELECT id FROM photos WHERE path = ?", (filename,))
            row = cursor.fetchone()

            if row:
                photo_id = row[0]
            else:
                cursor = conn.execute("""
                    INSERT INTO photos
                    (filename, subfolder, date, height, width, location, text, lat, lon, path, dino_embedding, media_type, video_path)
                    VALUES (?,?,?,0,0,'','',0.0,0.0,?,?,?,?)
                """, (filename, subfolder, date, filename, None, 'video', filename))
                photo_id = cursor.lastrowid

            # Frame extraction
            timestamps = list(range(0, int(duration), frame_interval_sec)) or [0]
            frame_metadata = []
            pil_frames = []

            for idx, ts in enumerate(timestamps):
                thumb_name = f"{video_path.stem}_f{idx:04d}_t{ts}.jpg"
                thumb_path = os.path.join(THUMBS_DIR, subfolder, thumb_name)
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)

                if extract_frame(filename, ts, thumb_path):
                    try:
                        # Prevent file handle leaks
                        with Image.open(thumb_path) as img:
                            pil_img = img.convert('RGB')
                            pil_img.load()
                            pil_frames.append(pil_img)
                            frame_metadata.append((photo_id, idx, ts * 1000))
                    except Exception:
                        continue

            if not pil_frames:
                skipped += 1
                continue

            # Generate frame embeddings
            blobs = get_dino_embeddings(pil_frames)
            frame_batch = [
                (meta[0], meta[1], meta[2], blob)
                for meta, blob in zip(frame_metadata, blobs)
            ]

            # Insert frame records
            conn.executemany("""
                INSERT OR IGNORE INTO video_frames (photo_id, frame_index, timestamp_ms, dino_embedding)
                VALUES (?, ?, ?, ?)
            """, frame_batch)
            inserted_frames += len(frame_batch)

            if frame_batch:
                # Set first frame's embedding as the main video thumbnail embedding
                first_frame_blob = frame_batch[0][3]
                conn.execute("UPDATE photos SET embedding_v2 = ? WHERE id = ?", (first_frame_blob, photo_id))
                conn.execute("DELETE FROM vec_photos WHERE id = ?", (photo_id,))
                conn.execute("INSERT INTO vec_photos(id, dino_embedding) VALUES (?, ?)", (photo_id, first_frame_blob))

                # Face Detection per extracted frame
                for img in pil_frames:
                    img_detect = img.copy()
                    img_detect.thumbnail((1024, 1024))
                    faces, probs = mtcnn(img_detect, return_prob=True)

                    if faces is not None and probs is not None:
                        for face_tensor, prob in zip(faces, probs):
                            if prob is None or prob < 0.90:
                                continue
                            with torch.no_grad():
                                face_feat = facenet_model(face_tensor.unsqueeze(0).to(device))
                            face_blob = face_feat[0].cpu().numpy().astype(np.float32).tobytes()

                            cursor = conn.execute(
                                "INSERT INTO faces (photo_id, facenet_embedding) VALUES (?, ?)",
                                (photo_id, face_blob)
                            )
                            conn.execute("DELETE FROM vec_faces WHERE id = ?", (cursor.lastrowid,))
                            conn.execute("INSERT INTO vec_faces(id, facenet_embedding) VALUES (?, ?)", (cursor.lastrowid, face_blob))

            inserted_videos += 1

    # Connection remains open for caller use
    print(f"Done — {inserted_videos} videos, {inserted_frames} frames, {skipped} skipped.")
    
mtcnn = MTclean = MTCNN(keep_all=True, device=torch.device("cpu"))
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

def normalize_vector(v):
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    return (v / norm).tolist()

def process_images(conn, append, base_path, batch_size=32):
    """
    Processes or re-indexes photos in the DB matching target base_path.
    """
    folder_name = os.path.basename(os.path.normpath(base_path))
    path_pattern = f"%{folder_name}%"

    where_clauses = ["media_type='photo'", "path LIKE ?"]
    params = [path_pattern]

    if append:
        where_clauses.append("embedding_v2 IS NULL")
        print(f"Checking for unindexed photos in '{folder_name}'...")
    else:
        print(f"RE-INDEXING ALL photos in directory pattern: '{folder_name}'...")

    query = f"SELECT id, path FROM photos WHERE {' AND '.join(where_clauses)}"
    rows = conn.execute(query, params).fetchall()

    if not rows:
        print(f"No matching photos found in database for '{folder_name}'.")
        return

    missing_count = 0
    print(f"Processing {len(rows)} photos for '{folder_name}'...")

    for i in tqdm(range(0, len(rows), batch_size)):
        batch = rows[i:i+batch_size]
        pil_images, valid_items = [], []

        for photo_id, path in batch:
            # Resolve against BASE_PATH since path string in DB contains 'Samsung/...'
            full_path = find_fuzzy_path(path, BASE_PATH)
            if not full_path or not os.path.exists(full_path):
                missing_count += 1
                continue
            try:
                with Image.open(full_path) as img:
                    pil_img = img.convert("RGB")
                    pil_img.load()
                    pil_images.append(pil_img)
                    valid_items.append((photo_id, path))
            except Exception:
                continue

        if not pil_images:
            continue

        blobs = get_dino_embeddings(pil_images)

        with conn:
            for (db_id, path), blob, img in zip(valid_items, blobs, pil_images):
                conn.execute("UPDATE photos SET embedding_v2 = ? WHERE id = ?", (blob, db_id))
                conn.execute("DELETE FROM vec_photos WHERE id = ?", (db_id,))
                conn.execute("INSERT INTO vec_photos(id, dino_embedding) VALUES (?, ?)", (db_id, blob))

                img_detect = img.copy()
                img_detect.thumbnail((1024, 1024))
                faces, probs = mtcnn(img_detect, return_prob=True)

                if faces is not None and probs is not None:
                    conn.execute("DELETE FROM faces WHERE photo_id = ?", (db_id,))
                    for face_tensor, prob in zip(faces, probs):
                        if prob is None or prob < 0.90:
                            continue
                        with torch.no_grad():
                            face_feat = facenet_model(face_tensor.unsqueeze(0).to(device))
                            face_blob = face_feat[0].cpu().numpy().astype(np.float32).tobytes()

                            cursor = conn.execute(
                                "INSERT INTO faces (photo_id, facenet_embedding) VALUES (?, ?)", 
                                (db_id, face_blob)
                            )
                            conn.execute("DELETE FROM vec_faces WHERE id = ?", (cursor.lastrowid,))
                            conn.execute("INSERT INTO vec_faces(id, facenet_embedding) VALUES (?, ?)", (cursor.lastrowid, face_blob))

    if missing_count > 0:
        print(f"Could not find files on disk for {missing_count} database entries.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        help="Path to image folder",
        default="/Volumes/T7/photos_from_icloud",
    )
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument(
        "--output", default="/Volumes/T7/photos_from_icloud-out/embeddings_new.jsonl"
    )
    parser.add_argument(
        "--append", action="store_true", default=False
    )
    args = parser.parse_args()
    append = args.append
    base_folder = Path(args.directory)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    buffer = []
    i = 0
    conn = sqlite3.connect(DB_PATH)


    conn = sqlite3.connect(DB_PATH)
    try:
        # process_images(conn, append, base_path=base_folder)
        # ingest_videos(conn, append, base_path=base_folder)
        h3population()
    finally:
        conn.close()



current_time = datetime.now()
print(current_time)