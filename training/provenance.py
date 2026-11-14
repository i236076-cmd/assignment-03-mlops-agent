"""Helpers that find out which dataset and code version a run is using."""
import subprocess

import yaml

DVC_FILE = "data/dataset.csv.dvc"


def _git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def md5_from_dvc_text(text):
    # a .dvc file is just yaml, the hash of the data is under outs
    info = yaml.safe_load(text)
    return info["outs"][0]["md5"]


def find_version_for_md5(md5):
    """Look through all git tags and return the one whose .dvc file has this hash."""
    tags = _git("tag", "--list") or ""
    for tag in tags.splitlines():
        text = _git("show", f"{tag}:{DVC_FILE}")
        if text and md5_from_dvc_text(text) == md5:
            return tag
    return "untagged"


def get_provenance():
    with open(DVC_FILE) as f:
        md5 = md5_from_dvc_text(f.read())

    dirty = _git("status", "--porcelain", "--", ".", ":(exclude)data")
    return {
        "dvc_file": DVC_FILE,
        "dataset_md5": md5,
        "dataset_version": find_version_for_md5(md5),
        "dataset_commit": _git("log", "-1", "--format=%H", "--", DVC_FILE) or "unknown",
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": "true" if dirty else "false",
    }


if __name__ == "__main__":
    for key, value in get_provenance().items():
        print(f"{key}: {value}")
