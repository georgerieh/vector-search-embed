#!/usr/bin/python3
import argparse
import json
from pathlib import Path
import os
import subprocess

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
DB_PATH = "/Users/iamgeorgerieh/Documents/photos.db" 
#cp /Volumes/T7/photos.db ~/Documents/photos.db

BASE_DIR = "/Volumes/T7/photos_from_icloud"

# conn = sqlite3.connect(DB_PATH)


# rows = conn.execute(
#     "SELECT path FROM photos WHERE path IS NOT NULL AND embedding_v2 IS NULL"
# ).fetchall()

# print(f"{len(rows)} photos to embed")


# batch_size = 64

# for i in tqdm(range(0, len(rows), batch_size)):
#     batch = rows[i:i+batch_size]
#     images, paths = [], []
    
#     for (path,) in batch:
#         try:
#             mac_path = path.replace('/media/georgerieh/T7', '/Volumes/T7')
#             img = Image.open(mac_path).convert("RGB")
#             images.append(img)
#             paths.append(path)
#         except Exception as e:
#             continue
    
#     if not images:
#         continue
    
#     tensors = torch.stack([transform(img) for img in images]).to(device)
#     with torch.no_grad():
#         embeddings = model(tensors)  # returns (B, 768) CLS token
#         embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
#         embeddings = embeddings.cpu().numpy()
    
#     for path, emb in zip(paths, embeddings):
#         conn.execute(
#             "UPDATE photos SET embedding_v2 = ? WHERE path = ?",
#             (json.dumps(emb.tolist()), path)
#         )
    
#     conn.commit()

# conn.close()
# print("Done")


BASE_PATH = "/Volumes/T7/photos_from_icloud"
MOUNT_PATH = "/Volumes/T7/photos_from_icloud"
THUMBS_DIR = "/Volumes/T7/photos_from_icloud-out/thumbs"
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.heic'}
FRAME_INTERVAL_SEC = 30
BATCH_SIZE = 50

os.makedirs(THUMBS_DIR, exist_ok=True)

# def normalize_vector(v):
#     arr = np.array(v, dtype=np.float32)
#     norm = np.linalg.norm(arr)
#     return (arr / norm) if norm != 0 else arr

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

# def load_dino():
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"using {device}", flush=True)
    # model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)
    # model.eval().to(device)
    # preprocess = transforms.Compose([
    #     transforms.Resize(224), transforms.CenterCrop(224),
    #     transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    # ])
    # return model, preprocess, device

def embed(img, model, device):
    with torch.no_grad():
        tensor = preprocess(img).unsqueeze(0).to(device)
        feats = model.forward_features(tensor)
        if feats.ndim == 3:
            feats = feats[:, 0, :]
        emb = feats.squeeze(0).cpu().numpy()
    return normalize_vector(emb).astype(np.float32)

def ingest_videos(reembed):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")

    existing_videos = set(
        r[0] for r in conn.execute(
            "SELECT filename FROM photos WHERE media_type='video'"
        ).fetchall()
    )
    print(f"already ingested videos: {len(existing_videos)}")

    # find all videos
    videos = []
    for subfolder in sorted(Path(BASE_PATH).iterdir()):
        if not subfolder.is_dir():
            continue
        for f in sorted(subfolder.iterdir()):
            if f.name.startswith('.'):
                continue
            if f.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if reembed:
                videos.append(f)
            if not reembed:
                if str(f) not in existing_videos:
                    videos.append(f)

    print(f"videos to process: {len(videos)}")
    if not videos:
        print("nothing to do")
        return

    # model, preprocess, device = load_dino()

    inserted_videos = inserted_frames = skipped = 0

    for video_path in tqdm(videos, desc="videos"):
        duration = get_video_duration(str(video_path))
        if duration == 0:
            skipped += 1
            continue

        subfolder = video_path.parent.name
        file_id = video_path.name
        filename = str(video_path)
        path = filename  

        try:
            mtime = os.path.getmtime(filename)
            from datetime import datetime
            date = datetime.fromtimestamp(mtime).strftime('%Y:%m:%d')
        except:
            date = ""
        
        #CREATE TABLE video_frames (
        #id             INTEGER PRIMARY KEY,
        #photo_id       INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
        #frame_index    INTEGER NOT NULL,
        #timestamp_ms   INTEGER,
        #dino_embedding BLOB
        #);
        #CREATE INDEX idx_frames_photo_id ON video_frames(photo_id);
        
        conn.execute("""
            INSERT OR IGNORE INTO photos
            (filename, subfolder, date, height, width, location, text, lat, lon, path, dino_embedding, media_type, video_path)
            VALUES (?,?,?,0,0,'','',0.0,0.0,?,?,?,?)
        """, (filename, subfolder, date, path, None, 'video', filename))
        conn.commit()

        photo_id = conn.execute(
            "SELECT id FROM photos WHERE filename=?", (filename,)
        ).fetchone()[0]

        timestamps = list(range(0, int(duration), FRAME_INTERVAL_SEC))
        if not timestamps:
            timestamps = [0]

        frame_batch = []
        for i, ts in enumerate(timestamps):
            thumb_name = f"{video_path.stem}_f{i:04d}_t{ts}.jpg"
            thumb_path = os.path.join(THUMBS_DIR, subfolder, thumb_name)
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)

            if not extract_frame(str(video_path), ts, thumb_path):
                continue

            try:
                img = Image.open(thumb_path).convert('RGB')
            except:
                continue

            dino = embed(img, model, device)

            frame_batch.append((
                photo_id, i, ts * 1000,
                dino.tobytes(),
            ))

        if frame_batch:
            conn.executemany("""
                INSERT OR IGNORE INTO video_frames
                (id, photo_id, frame_index, timestamp_ms, dino_embedding)
                VALUES (? ?,?,?,?)
            """, frame_batch)
            conn.commit()
            inserted_frames += len(frame_batch)

        if frame_batch:
            conn.execute(
                "UPDATE photos SET dino_embedding=? WHERE id=?",
                (frame_batch[0][3], photo_id)
            )
            conn.commit()

        inserted_videos += 1

    conn.close()
    print(f"done — {inserted_videos} videos, {inserted_frames} frames, {skipped} skipped")

    
    
mtcnn = MTCNN(keep_all=True, device=device)
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


def process_image(file_path):
    """Compute DINO and FaceNet embeddings for a single image"""
    entry = {"filename": str(file_path), "faces": []}

    img = Image.open(file_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(tensor)
        if feats.ndim == 3:
            dino_feat = feats[:, 0, :] 
        elif feats.ndim == 2:
            dino_feat = feats 
        dino_feat = dino_feat.squeeze(0) 

    entry["dino_embedding"] = normalize_vector(dino_feat.cpu().numpy())

    boxes, probs = mtcnn.detect(img)
    faces = mtcnn(img)  
    if faces is not None and boxes is not None:
        for face_tensor, prob in zip(faces, probs):
            if prob is None or prob < 0.90:  
                continue
            with torch.no_grad():
                face_feat = facenet_model(face_tensor.unsqueeze(0).to(device))
            entry["faces"].append(
                {
                    "confidence": float(prob),
                    "embedding": normalize_vector(face_feat[0].cpu().numpy()),
                }
            )
    return entry


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
        "--verbose", default=False
    )
    args = parser.parse_args()
    reembed = args.verbose
    base_folder = Path(args.directory)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    buffer = []
    i = 0
    processed_files = set()
    if output_path.exists():
        with open(output_path, encoding='utf-8', mode="r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_files.add(obj["filename"])
                except:
                    continue
                
                
    images_to_process = []
    videos_to_process = []
    if not reembed:
        conn = sqlite3.connect(DB_PATH)


        rows = conn.execute(
            "SELECT path FROM photos WHERE path IS NOT NULL AND embedding_v2 IS NULL"
        ).fetchall()

        print(f"{len(rows)} photos to embed")


        batch_size = 64

        for file in rows:
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                images_to_process.append(file)
            # elif file.suffix.lower() in VIDEO_EXTENSIONS:
            #     videos_to_process.append(file)
   
    else:
        for subfolder in base_folder.iterdir():
            if subfolder.is_dir():
                for file in subfolder.iterdir():
                    if (
                            not file.is_file()
                            or file.suffix.lower() not in IMAGE_EXTENSIONS
                            or file.name.startswith(".")
                    ):
                        continue
                    elif str(file) in processed_files:
                        continue
                    elif file.suffix.lower() in [".jpg", ".jpeg"]:
                        images_to_process.append(file)
                    # elif file.suffix.lower() in VIDEO_EXTENSIONS:
                    #     videos_to_process.append(file)
    for file in tqdm(images_to_process + videos_to_process, desc="Processing files"):
        try:
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                entry = process_image(file)
                buffer.append(entry)
                i += 1
            # elif file.suffix.lower() in VIDEO_EXTENSIONS:
            #     entry = embed(file)
            #     buffer.append(entry)
            #     i += 1
    
        except Exception as e:
            print(f"Skipping {file}: {e}")

        if len(buffer) >= args.batch_size:
            with open(output_path, encoding='utf-8', mode="a") as f:
                for e in buffer:
                    f.write(json.dumps(e) + "\n")
            buffer = []

    if buffer:
        with open(output_path, encoding='utf-8', mode="a") as f:
            for e in buffer:
                f.write(json.dumps(e) + "\n")
        print(f"Processed total {i} images")
    ingest_videos(args.verbose)
