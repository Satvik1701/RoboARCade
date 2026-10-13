"""Syncs the runtime assets with the public HF dataset.

Sync the runtime asset tree with the public HF dataset roboarcade/roboarcade-assets: exactly the files an
episode can read (every task's meshes and textures where its scene.xml finds them, the robots, the archived packs
the grader replays), packed as one archive per top-level assets/ folder. The packs themselves (task.json,
scene.xml, manifest.json, rubric, test) are in git under tasks/.

  push — scan that closure, tar each folder's part of it, upload one archive per folder
  pull — download the archives and extract into assets/ (what a fresh clone runs)

Pull needs no login. Push needs a token that can write the dataset: `hf auth login`, or set HF_TOKEN.
"""
import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.tasks import ASSETS, TASKS, asset_dir, load_scene   # noqa: E402

REPO = "roboarcade/roboarcade-assets"


def runtime_files():
    needed = set()
    packs = [p.parent for p in TASKS.glob("*/task.json")] + [p.parent for p in (ASSETS / "replay_packs").rglob("scene.xml")]
    for pack in packs:
        spec = load_scene(pack)
        for asset in (*spec.meshes, *spec.textures, *spec.hfields, *spec.skins):
            if asset.file:
                base = spec.texturedir if asset in spec.textures else spec.meshdir
                path = Path(base, asset.file).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"{pack.name} references {path}")
                needed.add(path)
        mirror = asset_dir(pack)
        needed |= {p.resolve() for p in mirror.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    needed |= {p.resolve() for p in (ASSETS / "robots").rglob("*") if p.is_file()}
    return sorted(p.relative_to(ASSETS.resolve()) for p in needed if p.is_relative_to(ASSETS.resolve()))


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("cmd", choices=["push", "pull"])
if ap.parse_args().cmd == "push":
    files = runtime_files()
    print(f"{len(files)} runtime files, {sum((ASSETS / f).stat().st_size for f in files) / 1e6:.0f} MB", flush=True)
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    stage = ASSETS / ".cache"
    stage.mkdir(exist_ok=True)
    for top in sorted({f.parts[0] for f in files}):
        with tempfile.TemporaryDirectory(dir=stage) as td:
            listing = Path(td) / "files.txt"
            listing.write_text("".join(f"{f}\n" for f in files if f.parts[0] == top))
            tar = Path(td) / f"{top}.tar.gz"
            subprocess.run(["tar", "-I", "pigz", "-cf", tar, "-C", ASSETS, "-T", listing], check=True)
            api.upload_file(path_or_fileobj=tar, path_in_repo=tar.name, repo_id=REPO, repo_type="dataset")
        print(f"pushed {top}", flush=True)
else:
    snap = snapshot_download(REPO, repo_type="dataset")
    for tar in sorted(Path(snap).glob("*.tar.gz")):
        print(f"extracting {tar.name}", flush=True)
        tarfile.open(tar).extractall(ASSETS, filter="data")
