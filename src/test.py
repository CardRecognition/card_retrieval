# query_demo.py
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import faiss
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

MODEL_ID = "google/siglip-base-patch16-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

INDEX_PATH = Path("data/tcgdex/index/cards_all.faiss")
META_PATH = Path("data/tcgdex/index/cards_all.parquet")

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model.eval().to(DEVICE)
    assert hasattr(model, "get_image_features")
    return processor, model

@torch.no_grad()
def embed_image(img_path: str, processor, model) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt").to(DEVICE, dtype=DTYPE)
    feat = model.get_image_features(**inputs)
    feat = torch.nn.functional.normalize(feat.float(), dim=1)
    return feat.cpu().numpy()

def search(img_path: str, topk=5):
    processor, model = load_model()
    q = embed_image(img_path, processor, model)
    index = faiss.read_index(str(INDEX_PATH))
    D, I = index.search(q, topk)
    meta = pd.read_parquet(META_PATH)
    hits = meta.iloc[I[0]].copy()
    hits["score"] = D[0]
    return hits[["lang","set_id","localId","card_id","name","filepath","score"]]

if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else ".\\test4.jpg"
    res = search(img, topk=5)
    print(res.to_string(index=False))