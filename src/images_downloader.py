import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse
import mimetypes
import re
import random

import pandas as pd
import httpx
from tqdm import tqdm

CARDS_DIR = Path("data/tcgdex/cards")
IMAGES_DIR = Path("data/tcgdex/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENCY = 16
REQ_TIMEOUT = 30.0
MAX_RETRIES = 5
BACKOFF_BASE = 0.8
BACKOFF_JITTER = 0.25
HEADERS = {"User-Agent": "card-retrieval/1.0 (+github:...)"}

COL_URL = "image_hd_url"
COL_SET = "set_id"
COL_LOCAL = "localId"
COL_CARD = "card_id"
COL_NAME = "name"

def sanitize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text

def ext_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".jpg") or path.endswith(".jpeg"):
        return ".jpg"
    guess = mimetypes.guess_extension(mimetypes.guess_type(path)[0] or "")
    return guess if guess in (".png", ".jpg", ".jpeg") else ".png"

def build_filename(row: pd.Series) -> str:
    set_id = sanitize(str(row.get(COL_SET, "")))
    local_id = sanitize(str(row.get(COL_LOCAL, "")))
    card_id = sanitize(str(row.get(COL_CARD, "")))
    base = "_".join([p for p in (set_id, local_id, card_id) if p])
    return base or "card"

def list_csvs(cards_dir: Path) -> List[Tuple[str, Path]]:
    out = []
    for csv in sorted(cards_dir.glob("cards_*.csv")):
        lang = csv.stem.replace("cards_", "", 1)
        out.append((lang, csv))
    return out

async def http_get_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client.get(url, timeout=REQ_TIMEOUT)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"Retryable status {r.status_code}", request=r.request, response=r
                )
            r.raise_for_status()
            return r.content
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError):
            if attempt == MAX_RETRIES:
                return None
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER)
            await asyncio.sleep(sleep_s)

async def download_for_lang(lang: str, csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    if df.empty:
        print(f"[INFO] {lang}: CSV vide -> skip.")
        return

    if COL_URL not in df.columns:
        url_col_candidates = [c for c in df.columns if c.lower().startswith("image")]
        if not url_col_candidates:
            print(f"[WARN] {lang}: colonne image introuvable -> skip.")
            return
        url_col = url_col_candidates[0]
    else:
        url_col = COL_URL

    urls = df[url_col].fillna("").astype(str)
    mask = urls.str.len() > 0
    df = df[mask].copy()
    if df.empty:
        print(f"[INFO] {lang}: aucune image non vide -> skip.")
        return

    out_dir = IMAGES_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = df[[COL_SET, COL_LOCAL, COL_CARD, COL_NAME, url_col]].to_dict("records")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    downloaded = 0
    skipped = 0
    failed = 0

    async with httpx.AsyncClient(headers=HEADERS) as client:
        pbar = tqdm(total=len(rows), desc=f"[DL] {lang}", unit="img")
        tasks = []

        async def one(row):
            nonlocal downloaded, skipped, failed

            base_url = row[url_col]
            url = base_url.rstrip("/") + "/high.png"

            base = build_filename(pd.Series(row))
            ext = ext_from_url(url)
            out_path = out_dir / f"{base}{ext}"

            if out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                return

            async with sem:
                data = await http_get_bytes(client, url)

            if not data:
                failed += 1
                return

            try:
                out_path.write_bytes(data)
                downloaded += 1
            except Exception:
                failed += 1

        for r in rows:
            tasks.append(asyncio.create_task(one(r)))

        for fut in asyncio.as_completed(tasks):
            await fut
            pbar.update(1)

        pbar.close()

    print(f"[INFO] {lang}: downloaded={downloaded}, skipped={skipped}, failed={failed}")

async def main():
    csvs = list_csvs(CARDS_DIR)
    if not csvs:
        print(f"[ERROR] Aucun CSV trouvé dans {CARDS_DIR.resolve()}")
        return

    for lang, csv_path in csvs:
        try:
            await download_for_lang(lang, csv_path)
        except Exception as e:
            print(f"[ERROR] {lang}: {e}")

if __name__ == "__main__":
    asyncio.run(main())