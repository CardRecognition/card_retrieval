# encode_cards.py
import os
from pathlib import Path
from typing import List, Tuple
import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from transformers import AutoProcessor, AutoModel  # SigLIP via HF

CARDS_DIR = Path("data/tcgdex/cards")
IMAGES_DIR = Path("data/tcgdex/images")
EMB_DIR = Path("data/tcgdex/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

LANGS = [ "zh_tw"]
MODEL_ID = "google/siglip-base-patch16-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 64

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    model.eval().to(DEVICE)
    assert hasattr(model, "get_image_features"), "Le modèle choisi doit exposer get_image_features(images=...)"
    return processor, model

def build_filename(set_id: str, local_id: str, card_id: str) -> str:
    def sanitize(x: str) -> str:
        x = (x or "").strip()
        for ch in '\\/:*?"<>|':
            x = x.replace(ch, "_")
        return "_".join(x.split())
    parts = [sanitize(set_id), sanitize(str(local_id)), sanitize(card_id)]
    return "_".join([p for p in parts if p])

def row_to_image_path(lang_dir: Path, row: pd.Series) -> Path | None:
    base = build_filename(row.get("set_id",""), row.get("localId",""), row.get("card_id",""))
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        matches = list(lang_dir.glob(f"{base}{ext[1:]}")) if "*" not in ext else []
        if not matches:
            matches = glob.glob(str(lang_dir / f"{base}.png")) or \
                      glob.glob(str(lang_dir / f"{base}.jpg")) or \
                      glob.glob(str(lang_dir / f"{base}.jpeg"))
            matches = [Path(m) for m in matches]
        if matches:
            return matches[0]
    wide = list(lang_dir.glob(f"{base}.*"))
    return wide[0] if wide else None

@torch.no_grad()
def encode_images(img_paths: List[Path], processor, model) -> np.ndarray:
    embs_list = []
    for i in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="Encoding", unit="batch"):
        batch_paths = img_paths[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (384, 384), color=0)
            imgs.append(img)
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE, dtype=DTYPE)
        feats = model.get_image_features(**inputs)  # [B, D]
        feats = torch.nn.functional.normalize(feats.float(), dim=1)  # L2 norm → cosine ready
        embs_list.append(feats.cpu().numpy())
    return np.vstack(embs_list) if embs_list else np.zeros((0, 768), dtype=np.float32)

def process_lang(lang: str, processor, model):
    csv_path = CARDS_DIR / f"cards_{lang}.csv"
    lang_dir = IMAGES_DIR / lang
    assert csv_path.exists(), f"CSV manquant: {csv_path}"
    assert lang_dir.exists(), f"Dossier images manquant: {lang_dir}"

    df = pd.read_csv(csv_path, encoding="utf-8")
    if df.empty:
        print(f"[INFO] {lang}: CSV vide.")
        return

    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Index {lang}"):
        p = row_to_image_path(lang_dir, row)
        paths.append(str(p) if p and p.exists() else "")

    df["filepath"] = paths
    df = df[df["filepath"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        print(f"[WARN] {lang}: aucune image trouvée.")
        return

    img_paths = [Path(p) for p in df["filepath"].tolist()]
    embs = encode_images(img_paths, processor, model).astype(np.float32)
    if embs.shape[0] != len(df):
        print(f"[WARN] {lang}: mismatch embeddings/df ({embs.shape[0]} vs {len(df)})")

    np.save(EMB_DIR / f"embeddings_{lang}.npy", embs)
    df.to_parquet(EMB_DIR / f"meta_{lang}.parquet", index=False)
    print(f"[INFO] {lang}: {embs.shape} -> {EMB_DIR}")

def main():
    processor, model = load_model()
    for lang in LANGS:
        process_lang(lang, processor, model)

if __name__ == "__main__":
    main()