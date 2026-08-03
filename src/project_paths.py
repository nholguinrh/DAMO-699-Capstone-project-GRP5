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

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)