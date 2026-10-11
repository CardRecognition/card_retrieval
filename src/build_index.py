# build_index.py
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import faiss

EMB_DIR = Path("data/tcgdex/embeddings")
INDEX_DIR = Path("data/tcgdex/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["fr", "en", "ja", "zh_cn", "zh_tw"]

TEXT_COLS = ["set_id", "localId", "card_id", "name", "filepath", "lang"]

def normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    df[TEXT_COLS] = df[TEXT_COLS].fillna("").astype(str)
    return df

def load_lang(lang: str) -> tuple[np.ndarray, pd.DataFrame]:
    E = np.load(EMB_DIR / f"embeddings_{lang}.npy")
    M = pd.read_parquet(EMB_DIR / f"meta_{lang}.parquet")
    n = min(len(M), E.shape[0])
    E, M = E[:n], M.iloc[:n].reset_index(drop=True)
    M["lang"] = lang
    M = normalize_meta(M)
    return E.astype(np.float32), M

def build_flat_index(embs: np.ndarray) -> faiss.Index:
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))
    return index

def save_index(index: faiss.Index, meta: pd.DataFrame, name: str):
    meta = normalize_meta(meta)
    faiss.write_index(index, str(INDEX_DIR / f"{name}.faiss"))
    meta.to_parquet(INDEX_DIR / f"{name}.parquet", index=False)
    print(f"[INFO] {name}: {index.ntotal} vecteurs")

def main():
    for lang in LANGS:
        try:
            E, M = load_lang(lang)
        except FileNotFoundError:
            print(f"[WARN] pas d'embeddings/meta pour {lang}, skip.")
            continue
        idx = build_flat_index(E)
        save_index(idx, M, f"cards_{lang}")

    Embs: List[np.ndarray] = []
    Metas: List[pd.DataFrame] = []
    for lang in LANGS:
        try:
            E, M = load_lang(lang)
        except FileNotFoundError:
            continue
        Embs.append(E); Metas.append(M)

    if Embs:
        Eall = np.vstack(Embs)
        Mall = pd.concat(Metas, ignore_index=True)
        Mall = normalize_meta(Mall)
        idx_all = build_flat_index(Eall)
        save_index(idx_all, Mall, "cards_all")

if __name__ == "__main__":
    main()