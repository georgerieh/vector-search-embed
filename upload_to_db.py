import json
import sqlite3
import numpy as np

DB_PATH = "/media/georgerieh/T7/photos.db"
BASE_PATH = "/media/georgerieh/T7/photos_from_icloud"
BASE_PATH_OUT = f"{BASE_PATH}-out"
MOUNT_PATH = "/Volumes/T7/photos_from_icloud"
BATCH_SIZE = 500

METADATA_COLUMNS = ["rel_path", "file_id", "subfolder", "date", "height",
                    "width", "location", "text", "lat", "lon"]

def normalize_vector(vector):
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm) if norm != 0 else arr

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS photos (
            id             INTEGER PRIMARY KEY,
            filename       TEXT UNIQUE,
            subfolder      TEXT,
            date           TEXT,
            height         INTEGER,
            width          INTEGER,
            location       TEXT,
            text           TEXT,
            lat            REAL,
            lon            REAL,
            path           TEXT,
            dino_embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS faces (
            id                INTEGER PRIMARY KEY,
            photo_id          INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
            facenet_embedding BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_date      ON photos(date);
        CREATE INDEX IF NOT EXISTS idx_subfolder ON photos(subfolder);
        CREATE INDEX IF NOT EXISTS idx_lat_lon   ON photos(lat, lon);
        CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
    """)
    conn.commit()

def emb_key(line):
    """Extract relative path from embeddings line for comparison."""
    record = json.loads(line)
    return record["filename"].replace(f"{MOUNT_PATH}/", "")

def meta_key(line):
    """Extract relative path from metadata line for comparison."""
    row = json.loads(line)
    return row[0]  # already subfolder/file_id

def _flush(conn, batch, inserted_photos, inserted_faces):
    cursor = conn.cursor()
    for item in batch:
        cursor.execute("""
            INSERT OR REPLACE INTO photos
            (filename, subfolder, date, height, width, location, text, lat, lon, path, dino_embedding)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, item["row"])
        photo_id = cursor.lastrowid
        cursor.execute("DELETE FROM faces WHERE photo_id=?", (photo_id,))
        for face_blob in item["faces"]:
            cursor.execute(
                "INSERT INTO faces (photo_id, facenet_embedding) VALUES (?,?)",
                (photo_id, face_blob)
            )
            inserted_faces += 1
        inserted_photos += 1
    conn.commit()
    return inserted_photos, inserted_faces

def ingest():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)

    batch = []
    inserted_photos = inserted_faces = skipped = 0

    meta_file = open(f"{BASE_PATH_OUT}/metadata_sorted.jsonl")
    emb_file  = open(f"{BASE_PATH_OUT}/embeddings_sorted.jsonl")

    meta_line = meta_file.readline()
    emb_line  = emb_file.readline()

    while meta_line and emb_line:
        mk = meta_key(meta_line)
        ek = emb_key(emb_line)

        if mk == ek:
            m   = json.loads(meta_line)
            emb = json.loads(emb_line)

            filename = f"{BASE_PATH}/{m[0]}"
            path     = f"{MOUNT_PATH}/{m[0]}"
            subfolder, file_id = m[2], m[1]

            dino_raw = emb["dino_embedding"]
            dino = normalize_vector(dino_raw if isinstance(dino_raw, list) else [0.0] * 768)

            faces = []
            for face in (emb.get("faces") or []):
                if isinstance(face, dict):
                    vec = face.get("embedding", [])
                    if len(vec) == 512:
                        faces.append(normalize_vector(vec).astype(np.float32).tobytes())
                elif isinstance(face, list) and len(face) == 512:
                    faces.append(normalize_vector(face).astype(np.float32).tobytes())

            location = m[6]
            if isinstance(location, dict):
                location = json.dumps(location)
            elif isinstance(location, str):
                try:
                    parsed = json.loads(location)
                    location = json.dumps(parsed) if isinstance(parsed, dict) else location
                except (json.JSONDecodeError, TypeError):
                    pass

            batch.append({
                "row": (
                    filename,
                    subfolder,
                    str(m[3] or ""),
                    int(m[4] or 0),
                    int(m[5] or 0),
                    location or "",
                    str(m[7] or ""),
                    float(m[8] or 0.0),
                    float(m[9] or 0.0),
                    path,
                    dino.astype(np.float32).tobytes(),
                ),
                "faces": faces,
            })

            if len(batch) >= BATCH_SIZE:
                inserted_photos, inserted_faces = _flush(conn, batch, inserted_photos, inserted_faces)
                print(f"Photos: {inserted_photos}, Faces: {inserted_faces}, Skipped: {skipped}")
                batch = []

            meta_line = meta_file.readline()
            emb_line  = emb_file.readline()

        elif mk < ek:
            # metadata ahead — embedding missing for this photo
            skipped += 1
            meta_line = meta_file.readline()
        else:
            # embedding ahead — metadata missing
            emb_line = emb_file.readline()

    meta_file.close()
    emb_file.close()

    if batch:
        inserted_photos, inserted_faces = _flush(conn, batch, inserted_photos, inserted_faces)

    conn.close()
    print(f"Done — {inserted_photos} photos, {inserted_faces} faces, {skipped} skipped.")

if __name__ == "__main__":
    ingest()