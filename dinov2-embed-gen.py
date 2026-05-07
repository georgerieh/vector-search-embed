import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import sqlite3
import os
from pathlib import Path
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
# M1 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
print('using', device)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = model.to(device)
model.eval()
DB_PATH = "/Users/iamgeorgerieh/Documents/photos.db" 
#cp /Volumes/T7/photos.db ~/Documents/photos.db

BASE_DIR = "/Volumes/T7/photos_from_icloud"

conn = sqlite3.connect(DB_PATH)

# add new column
try:
    conn.execute("ALTER TABLE photos ADD COLUMN embedding_v2 BLOB")
    conn.commit()
except:
    pass

rows = conn.execute(
    "SELECT path FROM photos WHERE path IS NOT NULL AND embedding_v2 IS NULL"
).fetchall()

print(f"{len(rows)} photos to embed")

import json
from tqdm import tqdm

batch_size = 64

for i in tqdm(range(0, len(rows), batch_size)):
    batch = rows[i:i+batch_size]
    images, paths = [], []
    
    for (path,) in batch:
        try:
            mac_path = path.replace('/media/georgerieh/T7', '/Volumes/T7')
            img = Image.open(mac_path).convert("RGB")
            images.append(img)
            paths.append(path)
        except Exception as e:
            continue
    
    if not images:
        continue
    
    # inputs = processor(images=images, return_tensors="pt").to(device)
    tensors = torch.stack([transform(img) for img in images]).to(device)
    with torch.no_grad():
        # outputs = model(**inputs)
        embeddings = model(tensors)  # returns (B, 768) CLS token
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        # embeddings = outputs.pooler_output  # (B, 768)
        # embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.cpu().numpy()
    
    for path, emb in zip(paths, embeddings):
        conn.execute(
            "UPDATE photos SET embedding_v2 = ? WHERE path = ?",
            (json.dumps(emb.tolist()), path)
        )
    
    conn.commit()

conn.close()
print("Done")