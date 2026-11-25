"""Download real foggy / hazy photos from Wikimedia Commons for testing (free licenses).

Writes images plus credits.csv (title, author, license, source page) so the
licensing of every image is recorded.

python scripts/download_test_images.py --out samples/internet --per-category 5
"""
import argparse
import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "FogRemovalTestDownloader/1.0 (dehazing research script; python-urllib)"}

DEFAULT_QUERIES = [
    "foggy street",
    "fog road cars",
    "dense fog city",
    "foggy highway",
    "haze city skyline",
    "smog city",
    "misty forest",
    "fog bridge",
]


def fetch(url, retries=5):
    """GET with polite back-off on HTTP 429 / 5xx."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After", 0) or 0) or 5 * (attempt + 1)
            print(f"  rate limited ({e.code}), waiting {wait}s")
            time.sleep(wait)


def get_json(params):
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(params)))


def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search_files(query, limit, width):
    data = get_json({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 30,
        "prop": "imageinfo", "iiprop": "url|mime|extmetadata", "iiurlwidth": width,
    })
    out = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("mime") != "image/jpeg" or "thumburl" not in info:
            continue
        meta = info.get("extmetadata", {})
        out.append({
            "title": page["title"],
            "url": info["thumburl"],
            "page": info.get("descriptionurl", ""),
            "author": strip_html(meta.get("Artist", {}).get("value")),
            "license": strip_html(meta.get("LicenseShortName", {}).get("value")),
        })
    # keep search ranking (most relevant first)
    order = {p["title"]: p.get("index", 0) for p in data.get("query", {}).get("pages", {}).values()}
    out.sort(key=lambda f: order.get(f["title"], 0))
    return out[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="samples/internet")
    p.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    p.add_argument("--per-query", type=int, default=4)
    p.add_argument("--width", type=int, default=1024, help="download width in pixels")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for q in a.queries:
        try:
            files = search_files(q, a.per_query * 2, a.width)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {q}: {e}")
            continue
        got = 0
        for f in files:
            if got >= a.per_query or f["title"] in seen:
                continue
            slug = re.sub(r"[^A-Za-z0-9]+", "_", f["title"].removeprefix("File:"))[:50].strip("_")
            name = f"{q.replace(' ', '-')}_{slug}.jpg"
            dest = out / name
            try:
                if not dest.exists():
                    dest.write_bytes(fetch(f["url"]))
                    time.sleep(1.0)  # be polite to the servers
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {f['title']}: {e}")
                continue
            seen.add(f["title"])
            got += 1
            rows.append([name, q, f["title"], f["author"], f["license"], f["page"]])
        print(f"{q}: {got} images")

    with open(out / "credits.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "query", "commons_title", "author", "license", "source"])
        w.writerows(rows)
    print(f"done: {len(rows)} images in {out} (credits.csv lists author + license)")


if __name__ == "__main__":
    main()
