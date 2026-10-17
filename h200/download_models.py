#!/usr/bin/env python3
"""Explicit download of pinned EXL3/Engram assets; default is an offline plan."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "h200-data")),
                        help="parent of models/, cache/, results/ (also settable with DATA_DIR)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print the offline plan (default)")
    mode.add_argument("--download", action="store_true", help="download/resume the pinned assets explicitly")
    parser.add_argument("--verify", action="store_true",
                        help="verify local files against pinned HF hashes; optionally after --download")
    parser.add_argument("--workers", type=int, default=8, help="download workers (default: 8)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.plan and args.verify:
        parser.error("--plan is offline and cannot be combined with --verify")
    return args


def field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def selected_files(info, job):
    files = [f for f in info.siblings
             if any(fnmatch.fnmatchcase(f.rfilename, pattern) for pattern in job["allow_patterns"])]
    tensors = [f for f in files if f.rfilename.endswith(".safetensors")]
    if len(tensors) != job["expected_safetensors_files"]:
        raise RuntimeError(f"Unexpected shard count at pinned revision: {job['repo_id']}")
    if sum(f.size for f in tensors) != job["tensor_bytes"]:
        raise RuntimeError(f"Unexpected checkpoint size at pinned revision: {job['repo_id']}")
    return files


def file_record(f):
    lfs_sha = field(f.lfs, "sha256") if f.lfs else None
    return {"path": f.rfilename, "size": f.size,
            "sha256": lfs_sha, "git_blob_sha1": None if lfs_sha else f.blob_id}


def verify_files(directory, files):
    """Check LFS SHA-256; small non-LFS files use the Git blob SHA-1."""
    for f in files:
        path = directory / f.rfilename
        if not path.is_file() or path.stat().st_size != f.size:
            raise RuntimeError(f"Missing or wrong-size file: {path}")
        record = file_record(f)
        if record["sha256"]:
            expected = record["sha256"]
            digest = hashlib.sha256()
        elif record["git_blob_sha1"]:
            expected = record["git_blob_sha1"]
            digest = hashlib.sha1()
            digest.update(f"blob {f.size}\0".encode())
        else:
            raise RuntimeError(f"Pinned HF metadata has no digest for {f.rfilename}")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Hash mismatch: {path}")
        print(f"Verified {f.rfilename}", flush=True)


def main():
    args = parse_args()
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    model_root = args.data_dir.expanduser().resolve() / "models"
    total = sum(job["tensor_bytes"] for job in pins["models"])
    if not args.download and not args.verify:
        print(json.dumps({"mode": "offline plan", "models_directory": str(model_root),
                          "tensor_bytes": total, "tensor_gib": round(total / 2**30, 2),
                          "extra_space": "Rank-weight and compiler caches require additional storage.",
                          "models": pins["models"]}, indent=2))
        return

    # No external packages or network are needed for --help or the default plan.
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest = []
    for job in pins["models"]:
        info = api.model_info(job["repo_id"], revision=job["revision"], files_metadata=True)
        if info.sha != job["revision"]:
            raise RuntimeError(f"Revision mismatch for {job['repo_id']}")
        files = selected_files(info, job)
        directory = model_root / job["directory"]
        if args.download:
            print(f"Downloading {job['repo_id']} @ {job['revision']}", flush=True)
            snapshot_download(repo_id=job["repo_id"], revision=job["revision"],
                              local_dir=directory, allow_patterns=job["allow_patterns"],
                              max_workers=args.workers)
        if args.verify:
            verify_files(directory, files)
        manifest.append({"repo_id": job["repo_id"], "revision": job["revision"],
                         "directory": job["directory"], "hash_verified": args.verify,
                         "files": [file_record(f) for f in files]})
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Pinned model assets are ready." if args.download else "All selected model assets verified.", flush=True)


if __name__ == "__main__":
    main()
