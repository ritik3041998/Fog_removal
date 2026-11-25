"""Download fog datasets from Hugging Face mirrors (the official Google Drive / Baidu links are dead).

    pip install huggingface_hub pyarrow
    python download_data.py --root %USERPROFILE%/fogdata              # everything below (~2.1 GB)
    python download_data.py --root ~/fogdata --only reside6k sots

Result:
  <root>/RESIDE-6K/train/{hazy,GT}   6000 synthetic-fog pairs (RESIDE ITS+OTS mix)   -> training
  <root>/RESIDE-6K/test/{hazy,GT}    1000 pairs  (only with --with-6k-test)
  <root>/SOTS/outdoor/{hazy,gt}      500 outdoor pairs, the standard benchmark       -> testing
  <root>/SOTS/indoor/{hazy,gt}       500 indoor pairs
  <root>/NH-HAZE2/...                real (not computer-generated) non-uniform haze pairs -> real-fog test
"""
import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download


def unzip(path, dest):
    print(f"unzipping {os.path.basename(path)} ...")
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


def reside6k_from_parquet(dest, splits):
    """RESIDE-6K as Hugging Face's parquet export: 3 big files instead of 12,000 small ones
    (the per-file download runs at ~3 files/s). Rows keep their filename; label is GT or hazy."""
    import json
    import shutil
    import urllib.request

    import pyarrow.parquet as pq

    api = "https://huggingface.co/api/datasets/desonglll/RESIDE-6K/parquet"
    urls = json.load(urllib.request.urlopen(api))["default"]
    tmp = os.path.join(dest, "_parquet")
    os.makedirs(tmp, exist_ok=True)
    for split in splits:
        for i, url in enumerate(urls[split]):
            local = os.path.join(tmp, f"{split}_{i}.parquet")
            if not os.path.exists(local):
                print(f"downloading RESIDE-6K {split} part {i + 1}/{len(urls[split])} ...")
                with urllib.request.urlopen(url) as r, open(local + ".part", "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 22)
                os.replace(local + ".part", local)
            with open(local, "rb") as fh:             # explicit handle so Windows can delete it after
                pf = pq.ParquetFile(fh)
                meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
                names = meta["info"]["features"]["label"]["names"]      # e.g. ["GT", "hazy"]
                for g in range(pf.metadata.num_row_groups):
                    for row in pf.read_row_group(g).to_pylist():
                        out_dir = os.path.join(dest, split, names[row["label"]])
                        os.makedirs(out_dir, exist_ok=True)
                        with open(os.path.join(out_dir, os.path.basename(row["image"]["path"])), "wb") as f:
                            f.write(row["image"]["bytes"])
                del pf
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "fogdata"))
    ap.add_argument("--only", nargs="*", choices=["reside6k", "sots", "nhhaze2"])
    ap.add_argument("--with-6k-test", action="store_true")
    args = ap.parse_args()
    want = set(args.only or ["reside6k", "sots", "nhhaze2"])
    os.makedirs(args.root, exist_ok=True)

    if "reside6k" in want:
        reside6k_from_parquet(os.path.join(args.root, "RESIDE-6K"),
                              ["train"] + (["test"] if args.with_6k_test else []))
    if "sots" in want:
        z = hf_hub_download("desonglll/reside-SOTS", "SOTS.zip", repo_type="dataset",
                            local_dir=os.path.join(args.root, "_zips"))
        unzip(z, os.path.join(args.root, "SOTS"))
    if "nhhaze2" in want:
        z = hf_hub_download("zhangxiaolin/NH-HAZE2", "data-20250125T101253Z-001.zip", repo_type="dataset",
                            local_dir=os.path.join(args.root, "_zips"))
        unzip(z, os.path.join(args.root, "NH-HAZE2"))
    print("done:", args.root)


if __name__ == "__main__":
    main()
