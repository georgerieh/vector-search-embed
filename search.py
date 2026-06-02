import os
import time
import sqlite3
import numpy as np
from urllib.parse import unquote
from PIL import Image as PILImage
import sqlite_vec
DB_PATH = "/media/georgerieh/T7/photos.db"
CHUNK_SIZE = 10_000
MOUNT_PATH = "/Volumes/T7/photos_from_icloud"
import h3
_DIR = os.path.dirname(os.path.abspath(__file__))
import struct
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def _blob_to_vec(blob, dim):
    if blob and len(blob) == dim * 4:
        return np.frombuffer(blob, dtype=np.float32)
    return None

def _score_dino_rows(rows, dino_q, face_scores):
    results = []
    for row_id, path, location, lat, lon, dino_blob, media_type, video_path in rows:
        dino_vec = _blob_to_vec(dino_blob, 768)
        if dino_vec is None:
            continue
        dino_score = float(np.linalg.norm(dino_vec - dino_q))
        face_score = face_scores.get(row_id, 0.0)
        results.append((dino_score + face_score, path, location, lat, lon, media_type, video_path))
    return results


def search_with_images(image, limit, embedding, start_date="", end_date="", 
                       facenet_embedding=None, country="", city="", h3cell=""):
    dino_features = get_image_embedding(embedding) if embedding is not None else None
    rows, stats = _search(dino_features, facenet_embedding, limit=limit,
                          start_date=start_date, end_date=end_date,
                          country=country, city=city, h3cell=h3cell)
    stats["generation_time"] = 0
    return rows, stats

def _vector_search(conn, dino_query, facenet_query, where_clause="", where_params=()):
    dino_q = np.array(dino_query, dtype=np.float32).tobytes()
    has_face_query = facenet_query is not None and not np.all(np.array(facenet_query) == 0)

    sql_where = f"WHERE {where_clause}" if where_clause else ""

    if has_face_query:
        facenet_q = np.array(facenet_query, dtype=np.float32).tobytes()
        
        sql = f"""
            SELECT p.id, p.path, p.location, p.lat, p.lon, 
                   (vd.distance + vf.distance) as total_score,
                   COALESCE(p.media_type, 'photo'), p.video_path
            FROM photos p
            JOIN (
                SELECT id, distance 
                FROM vec_photos 
                WHERE dino_embedding MATCH ? AND k = 50
            ) vd ON p.id = vd.id
            JOIN faces f ON p.id = f.photo_id
            JOIN (
                SELECT id, distance 
                FROM vec_faces 
                WHERE facenet_embedding MATCH ? AND k = 50
            ) vf ON f.id = vf.id
            {sql_where}
            GROUP BY p.id  
            ORDER BY total_score ASC, p.date DESC
            LIMIT 50
        """
        params = (dino_q, facenet_q) + where_params
    else:
        sql = f"""
            SELECT p.id, p.path, p.location, p.lat, p.lon, v.distance as total_score,
                   COALESCE(p.media_type, 'photo'), p.video_path
            FROM photos p
            JOIN (
                SELECT id, distance 
                FROM vec_photos 
                WHERE dino_embedding MATCH ? AND k = 50
            ) v ON p.id = v.id
            {sql_where} 
            ORDER BY total_score ASC, p.date DESC
            LIMIT 50
        """
        params = (dino_q,) + where_params

    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    
    seen = set()
    output = []
    
    for row_id, path, location, lat, lon, score, media_type, video_path in rows:
        if path in seen:
            continue
        seen.add(path)
        
        try:
            timestamp = int(os.path.getmtime(path))
        except OSError:
            timestamp = 0

        url = unquote(path).replace(f"{MOUNT_PATH}/", "")
        output.append({
            "location": location,
            "url": url,
            "video_url": unquote(video_path).replace(f"{MOUNT_PATH}/", "") if media_type == 'video' and video_path else None,
            "score": round(float(score), 3),
            "lat": lat,
            "lon": lon,
            "timestamp": timestamp,
            "media_type": media_type,
        })
        
        if len(output) >= 500:
            break
            
    return output


def _search(dino_query, facenet_query, limit=50, start_date="", end_date="",
            country="", city="", h3cell=""):
    conn = get_conn()
    st = time.time()

    has_filters = any([start_date, country, city, h3cell])

    if not dino_query and not has_filters:
        conn.close()
        return [], {"query_time": round(time.time() - st, 3)}

    from_clause = "photos p"
    conditions = []
    params = []

    has_face_query = facenet_query is not None and not np.all(np.array(facenet_query) == 0)
    
    if has_face_query:
        facenet_q = np.array(facenet_query, dtype=np.float32).tobytes()
        from_clause = """
            photos p
            JOIN faces f ON p.id = f.photo_id
            JOIN (
                SELECT id, distance 
                FROM vec_faces 
                WHERE facenet_embedding MATCH ? AND k = ?
            ) vf ON f.id = vf.id
        """
        vector_k = max(limit * 5, 1000)
        params.extend([facenet_q, vector_k])
    elif start_date and end_date:
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
        children = list(h3.cell_to_children(h3cell, res=7))
        placeholders = ",".join(["?"] * len(children))
        conditions.append(f"h3_cell IN ({placeholders})")
        params.extend(children)

    where_str = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    if not dino_query:
        sql = f"""
            SELECT p.path, p.location, p.lat, p.lon, p.media_type, p.video_path 
            FROM {from_clause} 
            {where_str}
            GROUP BY p.path
            ORDER BY {"vf.distance ASC" if has_face_query else "p.date DESC"} 
            LIMIT ?
        """
        rows = conn.execute(sql, tuple(params) + (limit,)).fetchall()
        seen = set()
        results = []
        for path, location, lat, lon, media_type, video_path in rows:
            if path in seen:
                continue
            seen.add(path)
            try:
                ts = int(os.path.getmtime(path))
            except OSError:
                ts = 0
            results.append({
                "location": location,
                "url": unquote(path).replace(f"{MOUNT_PATH}/", ""),
                "video_url": unquote(video_path).replace(f"{MOUNT_PATH}/", "") if media_type == 'video' and video_path else None,
                "score": 0.0,
                "lat": lat,
                "lon": lon,
                "timestamp": ts,
                "media_type": media_type,
            })
        conn.close()
        return results, {"query_time": round(time.time() - st, 3)}
    else:
        results = _vector_search(conn, dino_query, facenet_query, where_clause=where, where_params=where_params)
        conn.close()
        return results, {"query_time": round(time.time() - st, 3)}

def get_image_embedding(embedding) -> list:
    return (embedding / np.linalg.norm(embedding)).tolist()


def return_file(search_parser, text, image, table, limit, start_date="", end_date="", embedding=None, facenet_embedding=None, country=None, city=None, h3cell=None):
    limit = limit if limit is not None else 50
    images, stats = [], {}

    if search_parser == "search":
        images, stats = search_with_images(
            image,
            limit,
            embedding,
            start_date=start_date if start_date is not None else "",
            end_date=end_date if end_date is not None else "",
            facenet_embedding=facenet_embedding,
            country=country,
            city=city, 
            h3cell=h3cell
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