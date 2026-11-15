"""DVC tools: dataset provenance and reproducibility. Read only.

Only a short list of approved git and dvc commands can run, and the
arguments are checked first, so the model can never run arbitrary commands.
"""
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from agent.config import PROJECT_ROOT
from agent.schemas import ToolError, ToolRejected, ToolUnavailable
from agent.tools.base import register

DEFAULT_DATASET = "data/dataset.csv"

ALLOWED_COMMANDS = {
    ("git", "tag"),
    ("git", "rev-parse"),
    ("git", "show"),
    ("git", "log"),
    ("git", "ls-files"),
    ("dvc", "status"),
    ("dvc", "remote"),
}
SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def run_command(args, timeout=15):
    """Run an approved command without a shell. Returns (code, stdout, stderr)."""
    if len(args) < 2 or (args[0], args[1]) not in ALLOWED_COMMANDS:
        raise ToolRejected("command not allowed: " + " ".join(args[:2]))

    exe = shutil.which(args[0]) or str(Path(sys.executable).parent / args[0])
    try:
        result = subprocess.run(
            [exe] + list(args[1:]),
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        raise ToolUnavailable("%s is not installed or not on PATH" % args[0])
    except subprocess.TimeoutExpired:
        raise ToolUnavailable("command timed out: %s %s" % (args[0], args[1]))
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check_ref(ref):
    if not ref or not SAFE_REF.match(str(ref)):
        raise ToolError("invalid version name: %r" % (ref,))
    return str(ref)


def dvc_file_for(dataset):
    """data/dataset.csv -> data/dataset.csv.dvc (relative paths only)."""
    dataset = str(dataset or DEFAULT_DATASET).strip()
    path = Path(dataset if dataset.endswith(".dvc") else dataset + ".dvc")
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ToolError("invalid dataset path: %r" % dataset)
    return path.as_posix()


def parse_dvc(text):
    info = yaml.safe_load(text)
    out = info["outs"][0]
    return {"md5": out["md5"], "size": out.get("size"), "path": out.get("path")}


def current_dvc_info(dvc_file):
    path = PROJECT_ROOT / dvc_file
    if not path.exists():
        raise ToolError("%s not found, is this dataset tracked by DVC?" % dvc_file)
    return parse_dvc(path.read_text())


def dvc_info_at(ref, dvc_file):
    """The hash recorded in the .dvc file at a git tag or commit."""
    code, out, err = run_command(["git", "show", "%s:%s" % (check_ref(ref), dvc_file)])
    if code != 0:
        raise ToolError("could not read %s at %s (%s)" % (dvc_file, ref, err or "not found"))
    return parse_dvc(out)


def list_tags():
    code, out, _ = run_command(["git", "tag", "--list"])
    return out.splitlines() if code == 0 else []


def version_for_md5(md5, dvc_file):
    """Find the git tag whose .dvc file recorded this hash."""
    for tag in list_tags():
        try:
            if dvc_info_at(tag, dvc_file)["md5"] == md5:
                return tag
        except ToolError:
            continue
    return "untagged"


def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_object(md5):
    """Where the data object for this hash is stored: local cache and remote."""
    rel = Path("files") / "md5" / md5[:2] / md5[2:]
    cache = PROJECT_ROOT / ".dvc" / "cache" / rel
    places = {"local_cache": cache if cache.exists() else None, "remote": None}

    code, out, _ = run_command(["dvc", "remote", "list"])
    if code == 0:
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].startswith("/"):
                candidate = Path(parts[1]) / rel
                if candidate.exists():
                    places["remote"] = candidate
    return places


def count_rows(path):
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus the csv header


DATASET_PARAM = {
    "type": "string",
    "description": "Path of the tracked data file, default data/dataset.csv",
}


@register(
    "list_tracked_datasets",
    "List the datasets tracked by DVC, with the recorded hash, size and the git tag (version) they match.",
)
def list_tracked_datasets():
    code, out, _ = run_command(["git", "ls-files", "*.dvc"])
    datasets = []
    for rel in out.splitlines() if code == 0 else []:
        info = current_dvc_info(rel)
        datasets.append({
            "dvc_file": rel,
            "data_path": (Path(rel).parent / info["path"]).as_posix(),
            "md5": info["md5"],
            "size_bytes": info["size"],
            "version": version_for_md5(info["md5"], rel),
        })
    return {"count": len(datasets), "datasets": datasets}


@register(
    "get_current_dataset_version",
    "Get the dataset version (git tag), data hash and the git commit that recorded the data currently checked out.",
    {"type": "object", "properties": {"dataset": DATASET_PARAM}},
)
def get_current_dataset_version(dataset=DEFAULT_DATASET):
    dvc_file = dvc_file_for(dataset)
    info = current_dvc_info(dvc_file)

    code, last, _ = run_command(["git", "log", "-1", "--format=%H|%s", "--", dvc_file])
    commit, _, subject = (last if code == 0 else "").partition("|")
    code, head, _ = run_command(["git", "rev-parse", "HEAD"])

    head_md5 = dvc_info_at("HEAD", dvc_file)["md5"]
    return {
        "dataset": dataset,
        "version": version_for_md5(info["md5"], dvc_file),
        "md5": info["md5"],
        "size_bytes": info["size"],
        "commit_that_recorded_it": commit or None,
        "commit_message": subject or None,
        "repo_head_commit": head if code == 0 else None,
        "pointer_differs_from_head": info["md5"] != head_md5,
        "available_versions": sorted(list_tags()),
    }


@register(
    "compare_dataset_versions",
    "Compare two dataset versions (git tags such as v1 and v2): hash, size and row count.",
    {
        "type": "object",
        "properties": {
            "version_a": {"type": "string", "description": "first git tag, for example v1"},
            "version_b": {"type": "string", "description": "second git tag, for example v2"},
            "dataset": DATASET_PARAM,
        },
        "required": ["version_a", "version_b"],
    },
)
def compare_dataset_versions(version_a, version_b, dataset=DEFAULT_DATASET):
    dvc_file = dvc_file_for(dataset)
    result = {"dataset": dataset, "versions": {}}
    for ref in (version_a, version_b):
        info = dvc_info_at(ref, dvc_file)
        places = find_object(info["md5"])
        stored = places["local_cache"] or places["remote"]
        result["versions"][ref] = {
            "md5": info["md5"],
            "size_bytes": info["size"],
            "rows": count_rows(stored) if stored else None,
            "stored_in_cache": places["local_cache"] is not None,
            "stored_in_remote": places["remote"] is not None,
        }

    a, b = result["versions"][version_a], result["versions"][version_b]
    result["identical"] = a["md5"] == b["md5"]
    if a["size_bytes"] is not None and b["size_bytes"] is not None:
        result["size_difference_bytes"] = b["size_bytes"] - a["size_bytes"]
    if a["rows"] is not None and b["rows"] is not None:
        result["row_difference"] = b["rows"] - a["rows"]
    return result


@register(
    "check_data_changed",
    "Check whether the data file on disk still matches the hash DVC recorded for it, and show dvc status.",
    {"type": "object", "properties": {"dataset": DATASET_PARAM}},
)
def check_data_changed(dataset=DEFAULT_DATASET):
    dvc_file = dvc_file_for(dataset)
    info = current_dvc_info(dvc_file)
    data_path = PROJECT_ROOT / Path(dvc_file).parent / info["path"]

    if not data_path.exists():
        state, actual = "missing", None
    else:
        actual = md5_of_file(data_path)
        state = "unchanged" if actual == info["md5"] else "modified"

    code, out, err = run_command(["dvc", "status", dvc_file])
    return {
        "dataset": dataset,
        "state": state,
        "recorded_md5": info["md5"],
        "actual_md5": actual,
        "dvc_status": (out or err)[:500],
    }


@register(
    "explain_dataset_reproduction",
    "Explain, step by step, how to get back a previous dataset version, and check that its data is available. Does not change any files.",
    {
        "type": "object",
        "properties": {
            "version": {"type": "string", "description": "git tag of the version, for example v1"},
            "dataset": DATASET_PARAM,
        },
        "required": ["version"],
    },
)
def explain_dataset_reproduction(version, dataset=DEFAULT_DATASET):
    dvc_file = dvc_file_for(dataset)
    info = dvc_info_at(version, dvc_file)
    places = find_object(info["md5"])

    in_cache = places["local_cache"] is not None
    in_remote = places["remote"] is not None
    if not in_cache and not in_remote:
        raise ToolError(
            "the data for %s (md5 %s) is not in the local cache or the DVC remote, "
            "so it cannot be reproduced" % (version, info["md5"])
        )

    fetch = "dvc checkout %s" % dvc_file if in_cache else "dvc pull %s" % dvc_file
    return {
        "version": version,
        "md5": info["md5"],
        "available_in_local_cache": in_cache,
        "available_in_remote": in_remote,
        "steps": [
            "git checkout %s -- %s" % (version, dvc_file),
            fetch,
            "md5sum %s   (should print %s)" % (dataset, info["md5"]),
        ],
        "to_go_back": [
            "git checkout HEAD -- %s" % dvc_file,
            "dvc checkout %s" % dvc_file,
        ],
        "note": "This tool only explains the steps, it does not change any files.",
    }


@register(
    "get_dataset_for_run",
    "Find which dataset version and git commit were used to train a given MLflow run, "
    "from the provenance recorded in that run. Accepts a run id, a run name or an Airflow run id.",
    {
        "type": "object",
        "properties": {"run": {"type": "string", "description": "MLflow run id, run name or Airflow run id"}},
        "required": ["run"],
    },
)
def get_dataset_for_run(run):
    from agent.tools.mlflow_tools import get_client, resolve_run  # loaded only when needed

    client = get_client()
    found = resolve_run(client, run)
    tags = found.data.tags

    if "dataset_version" not in tags:
        raise ToolError("run %s has no recorded dataset provenance" % found.info.run_id)

    md5 = tags.get("dataset_md5")
    still_matches = None
    if md5:
        still_matches = version_for_md5(md5, tags.get("dvc_file", "data/dataset.csv.dvc")) == tags["dataset_version"]

    return {
        "run_id": found.info.run_id,
        "run_name": tags.get("mlflow.runName"),
        "dataset_version": tags["dataset_version"],
        "dataset_md5": md5,
        "dataset_commit": tags.get("dataset_commit"),
        "code_git_commit": tags.get("git_commit"),
        "code_had_uncommitted_changes": tags.get("git_dirty"),
        "recorded_version_matches_git_tags": still_matches,
        "to_reproduce": "ask for explain_dataset_reproduction with version %s" % tags["dataset_version"],
    }
