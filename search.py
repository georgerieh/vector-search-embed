import os
import time
import sqlite3
import numpy as np
from urllib.parse import unquote
from PIL import Image as PILImage

DB_PATH = "/media/georgerieh/T7/photos.db"
CHUNK_SIZE = 10_000
MOUNT_PATH = "/Volumes/T7/photos_from_icloud"

_DIR = os.path.dirname(os.path.abspath(__file__))

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def _blob_to_vec(blob, dim):
    if blob and len(blob) == dim * 4:
        return np.frombuffer(blob, dtype=np.float32)
    return None

def _vector_search(conn, dino_query, facenet_query, where_clause="", where_params=()):
    dino_q = np.array(dino_query, dtype=np.float32)
    has_face_query = facenet_query is not None and not np.all(np.array(facenet_query) == 0)
    facenet_q = np.array(facenet_query, dtype=np.float32) if has_face_query else None

    if has_face_query:
        # load all face embeddings for photos that pass the filter, grouped by photo_id
        face_sql = f"""
            SELECT f.photo_id, f.facenet_embedding
            FROM faces f
            JOIN photos p ON p.id = f.photo_id
            {where_clause}
            ORDER BY f.photo_id
        """
        face_rows = conn.execute(face_sql, where_params).fetchall()

        # group faces by photo_id → min L2
        from collections import defaultdict
        photo_face_scores = defaultdict(lambda: float("inf"))
        for photo_id, blob in face_rows:
            vec = _blob_to_vec(blob, 512)
            if vec is not None:
                score = float(np.linalg.norm(vec - facenet_q))
                if score < photo_face_scores[photo_id]:
                    photo_face_scores[photo_id] = score

        if not photo_face_scores:
            return []

        # now fetch only photos that have faces
        photo_ids = list(photo_face_scores.keys())
    else:
        photo_ids = None

    # chunked DINO scan
    if photo_ids is not None:
        # only scan photos with faces — batch IN queries
        all_results = []
        for chunk_start in range(0, len(photo_ids), CHUNK_SIZE):
            chunk_ids = photo_ids[chunk_start:chunk_start + CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk_ids))
            sql = f"""
                SELECT id, path, location, lat, lon, dino_embedding
                FROM photos
                WHERE id IN ({placeholders})
            """
            rows = conn.execute(sql, chunk_ids).fetchall()
            all_results.extend(_score_dino_rows(rows, dino_q, photo_face_scores))
    else:
        # full table scan in chunks
        all_results = []
        sql = f"""
            SELECT id, path, location, lat, lon, dino_embedding
            FROM photos
            {where_clause}
            ORDER BY id
        """
        cursor = conn.execute(sql, where_params)
        while True:
            rows = cursor.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            all_results.extend(_score_dino_rows(rows, dino_q, {}))

    all_results.sort(key=lambda x: x[0])
    all_results = all_results[:50]

    seen = set()
    output = []
    for score, path, location, lat, lon in all_results:
        if path in seen:
            continue
        seen.add(path)
        try:
            timestamp = int(os.path.getmtime(path))
        except OSError:
            timestamp = 0
        output.append({
            "location": location,
            "url": unquote(path).replace(f"{MOUNT_PATH}/", ""),
            "score": round(float(score), 3),
            "lat": lat,
            "lon": lon,
            "timestamp": timestamp,
        })
    return output

def _score_dino_rows(rows, dino_q, face_scores):
    results = []
    for row_id, path, location, lat, lon, dino_blob in rows:
        dino_vec = _blob_to_vec(dino_blob, 768)
        if dino_vec is None:
            continue
        dino_score = float(np.linalg.norm(dino_vec - dino_q))
        face_score = face_scores.get(row_id, 0.0)  # 0.0 when no face query
        results.append((dino_score + face_score, path, location, lat, lon))
    return results

def search_with_images(image, limit, embedding, start_date="", end_date="", 
                       facenet_embedding=None, country="", city="", h3cell=""):
    dino_features = get_image_embedding(embedding) if embedding is not None else None
    rows, stats = _search(dino_features, facenet_embedding, limit=limit,
                          start_date=start_date, end_date=end_date,
                          country=country, city=city, h3cell=h3cell)
    stats["generation_time"] = 0
    return rows, stats

def _search(dino_query, facenet_query, limit=50, start_date="", end_date="",
            country="", city="", h3cell=""):
    conn = get_conn()
    st = time.time()

    has_filters = any([start_date, country, city, h3cell])

    if not dino_query and not has_filters:
        conn.close()
        return [], {"query_time": round(time.time() - st, 3)}

    # build WHERE
    conditions = []
    params = []
    if start_date and end_date:
        conditions.append("date BETWEEN ? AND ?")
        params.extend([start_date, end_date])
    elif start_date:
        conditions.append("date = ?")
        params.append(start_date)
    if country:
        conditions.append("country = ?")
        params.append(country)
    if city:
        conditions.append("city = ?")
        params.append(city)
    if h3cell:
        conditions.append("h3_cell = ?")
        params.append(h3cell)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    where_params = tuple(params)

    if not dino_query:
        # filter-only, no vector search
        sql = f"""
            SELECT path, location, lat, lon 
            FROM photos {where} 
            ORDER BY date DESC LIMIT ?
        """
        rows = conn.execute(sql, where_params + (limit,)).fetchall()
        results = []
        for path, location, lat, lon in rows:
            results.append({
                "location": location,
                "url": unquote(path).replace(f"{MOUNT_PATH}/", ""),
                "score": 0.0,
                "lat": lat,
                "lon": lon,
                "timestamp": 0,
            })
        conn.close()
        return results, {"query_time": round(time.time() - st, 3)}

    # vector search with optional filters
    results = _vector_search(conn, dino_query, facenet_query, where, where_params)
    conn.close()
    return results, {"query_time": round(time.time() - st, 3)}
def get_image_embedding(embedding) -> list:
    return (embedding / np.linalg.norm(embedding)).tolist()


def return_file(search_parser, text, image, table, limit, start_date="", end_date="", embedding=None, facenet_embedding=None):
    limit = limit if limit is not None else 50
    images, stats = [], {}

    if search_parser == "search":
        images, stats = search_with_images(
            image,
            limit,
            embedding,
            start_date=start_date if start_date is not None else "",
            end_date=end_date if end_date is not None else "",
            facenet_embedding=None,
            country=None,
            city=None, 
            h3cell=None
        )

    return {
        "images": images if isinstance(images, list) else [],
        "table": table,
        "search_text": text,
        "source_image": unquote(image).replace(f"{MOUNT_PATH}/", "") if image else "",
        "gen_time": stats.get("generation_time", 0),
        "query_time": stats.get("query_time", 0),
        "start_date": start_date or "",
    }