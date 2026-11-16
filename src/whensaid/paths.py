import os
from pathlib import Path


def home() -> Path:
    p = Path(os.environ.get("WHENSAID_HOME") or Path.home() / ".whensaid")
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return home() / "whensaid.db"


def models_dir() -> Path:
    p = home() / "models"
    p.mkdir(exist_ok=True)
    return p
