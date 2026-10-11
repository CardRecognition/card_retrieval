import asyncio
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import quote
import random

import httpx
import pandas as pd
from tqdm import tqdm

BASE = "https://api.tcgdex.net/v2"

LANGS: List[Tuple[str, str]] = [
    ("fr", "fr"),
    ("en", "en"),
    ("ja", "ja"),
    ("zh_cn", "zh-cn"),
    ("zh_tw", "zh-tw"),
]

OUT_DIR = Path("data/tcgdex/cards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Réseau / perf
MAX_CONCURRENCY = 16
REQ_TIMEOUT = 30.0
MAX_RETRIES = 5
BACKOFF_BASE = 0.8
BACKOFF_JITTER = 0.25
HEADERS = {"User-Agent": "card-retrieval/1.0 (+github:yourrepo)"}

@dataclass
class CardRow:
    card_id: str
    set_id: str
    set_name: str
    localId: str
    name: str
    image_hd_url: str

async def get_json(client: httpx.AsyncClient, url: str) -> Any:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client.get(url, timeout=REQ_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"Retryable status {r.status_code}", request=r.request, response=r
                )
            r.raise_for_status()
            return r.json()

        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 404:
                raise
            if attempt == MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER)
            await asyncio.sleep(sleep_s)

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError):
            if attempt == MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, BACKOFF_JITTER)
            await asyncio.sleep(sleep_s)

async def fetch_cards_for_language(lang_key: str, lang_code: str) -> pd.DataFrame:
    rows: List[CardRow] = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(base_url=BASE, headers=HEADERS) as client:
        sets = await get_json(client, f"/{lang_code}/sets")
        set_infos: List[Tuple[str, str]] = [
            (s.get("id", ""), s.get("name", ""))
            for s in sets
            if s.get("id")
        ]
        print(f"\n[INFO] {lang_key}: {len(set_infos)} sets trouvés (REST).")

        async def fetch_set_cards(sid: str, sname: str):
            sid_enc = quote(sid, safe="")
            url = f"/{lang_code}/sets/{sid_enc}"
            try:
                async with sem:
                    full = await get_json(client, url)
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 404:
                    return
                raise

            for cb in full.get("cards", []):
                rows.append(CardRow(
                    card_id=cb.get("id", ""),
                    set_id=sid,
                    set_name=sname,
                    localId=str(cb.get("localId", "")),
                    name=cb.get("name", ""),
                    image_hd_url=cb.get("image", ""),
                ))

        tasks = [fetch_set_cards(sid, sname) for sid, sname in set_infos]

        pbar = tqdm(total=len(tasks), desc=f"[REST] {lang_key} sets", unit="set")
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            finally:
                pbar.update(1)
        pbar.close()

    df = pd.DataFrame([asdict(r) for r in rows])
    if not df.empty:
        df = df.drop_duplicates(subset=["set_id", "localId", "card_id"])
        df = df.sort_values(["set_id", "localId", "card_id"], kind="stable")

    print(f"[INFO] {lang_key}: {len(df)} cartes (REST).")
    return df

async def main():
    for lang_key, lang_code in LANGS:
        try:
            df = await fetch_cards_for_language(lang_key, lang_code)
        except Exception as e:
            print(f"[ERROR] {lang_key}: échec ({e})")
            continue

        out_csv = OUT_DIR / f"cards_{lang_key}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] {out_csv} ({len(df)} lignes).")


if __name__ == "__main__":
    asyncio.run(main())