"""Initialisation du package API.

Ajoute automatiquement le dossier `src` au chemin Python pour permettre
le chargement de `credit_scoring` sans configuration manuelle de PYTHONPATH.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
