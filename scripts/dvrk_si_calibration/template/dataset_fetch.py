"""Resumable HTTP-range extraction of selected SutureBot ZIP entries only."""

import argparse
import concurrent.futures
import io
import json
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])),
        )
        r = self.session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def fetch(self, start, length):
        end = min(start + length, self.size) - 1
        if end < start:
            return b""
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=90, stream=True)
        if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            r.close()
            raise OSError(f"Server did not honor bounded Range {start}-{end}: {r.status_code}")
        data = r.content
        if len(data) != end - start + 1:
            raise OSError("Truncated range")
        self.transferred += len(data)
        return data

    def readinto(self, b):
        data = self.fetch(self.pos, len(b))
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def task(tissue, mode):
    survey = json.load(open(ROOT / "survey.json"))
    episodes = [e for e in survey["episodes"] if e["id"].split("/")[0] == tissue]
    reader = RangeReader(f"https://huggingface.co/datasets/jchen396/SutureBot/resolve/main/{tissue}.zip")
    with zipfile.ZipFile(reader) as z:
        names = set(z.namelist())
        wanted = []
        if mode == "csv":
            wanted = [n for n in names if n.endswith("/ee_csv.csv")]
        else:
            for e in episodes:
                for f in e["frames"]:
                    for side in ["left", "right"]:
                        name = f"{e['id']}/{side}_img_dir/frame{f:06d}_{side}.jpg"
                        if name in names:
                            wanted.append(name)
        records = []
        for name in sorted(wanted):
            dest = ROOT / "ref/dataset" / name
            info = z.getinfo(name)
            if dest.exists() and dest.stat().st_size == info.file_size:
                records.append({"path": name, "status": "cached", "size": info.file_size})
                continue
            data = z.read(name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            records.append({"path": name, "status": "downloaded", "size": len(data)})
            if len(records) % 50 == 0:
                print(tissue, mode, len(records), "/", len(wanted), flush=True)
        result = {
            "tissue": tissue,
            "mode": mode,
            "entries": records,
            "wire_bytes": reader.transferred,
            "zip_bytes": reader.size,
        }
        (ROOT / "run" / f"fetch_{tissue}_{mode}.json").write_text(json.dumps(result, indent=2))
        print(tissue, mode, "complete", len(records), "wire_MB", round(reader.transferred / 1e6, 1), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["csv", "images"])
    parser.add_argument("--tissues", nargs="*", default=[f"tissue_{i}" for i in range(1, 11)])
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fs = {pool.submit(task, t, args.mode): t for t in args.tissues}
        for f in concurrent.futures.as_completed(fs):
            try:
                f.result()
            except Exception as ex:
                print("FAILED", fs[f], type(ex).__name__, str(ex), flush=True)
                raise
