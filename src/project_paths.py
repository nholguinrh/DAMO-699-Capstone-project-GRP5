from pathlib import Path

#PROJECT_ROOT = Path(__file__).resolve().parent.parent

def find_project_root(start_path: Path | None = None) -> Path:
    """
    Find the project root by searching upward for a known marker.

    Preferred markers:
    - .git
    - pyproject.toml
    - requirements.txt
    - README.md
    """

    current = (start_path or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    markers = {
        ".git",
        "pyproject.toml",
        "requirements.txt",
        #"README.md",
        ".env"
    }

    for folder in [current, *current.parents]:
        if any((folder / marker).exists() for marker in markers):
            return folder

    # No marker found anywhere above this file. Silently falling back to
    # this file's own folder (src/) would make every downstream path
    # (.env, data/raw, data/processed) silently wrong -- fail loudly
    # instead so the real cause gets fixed, not masked.
    raise FileNotFoundError(
        "Could not locate the project root: none of "
        f"{sorted(markers)} were found in {current} or any parent folder. "
        "Confirm you're running from inside a full clone of the repo "
        "(not a partial copy or a ZIP extracted without .git)."
    )


PROJECT_ROOT = find_project_root()

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def write_data_manifest(paths: list[Path], out: Path) -> dict[str, Any]:
    """
    Record sha256 + row count + date range of every input feeding an output run.
    Ensures that evaluation datasets and benchmarks are provably attributable.
    """
    import hashlib
    import json

    manifest: dict[str, Any] = {}
    for p in paths:
        if not p.exists():
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        rel_path = str(p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p).replace("\\", "/")
        entry: dict[str, Any] = {
            "path": rel_path,
            "sha256": h.hexdigest(),
            "size_bytes": p.stat().st_size,
        }
        if p.suffix.lower() == ".csv":
            try:
                import pandas as pd
                df_temp = pd.read_csv(p)
                entry["row_count"] = len(df_temp)
                if "date" in df_temp.columns:
                    dates = pd.to_datetime(df_temp["date"]).dropna()
                    if not dates.empty:
                        entry["start_date"] = str(dates.min().date())
                        entry["end_date"] = str(dates.max().date())
            except Exception:
                import logging
                logging.getLogger(__name__).warning("Could not read %s for manifest row/date bounds", p, exc_info=True)
                entry["row_count"] = None
        manifest[p.name] = entry

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest