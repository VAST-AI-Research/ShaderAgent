__version__ = "0.1.0"

from pathlib import Path

def find_project_root(marker="pyproject.toml"):
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    return None

ROOT_DIR = find_project_root()
