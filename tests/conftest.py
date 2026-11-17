"""Rend les modules du projet (à la racine du dépôt) importables depuis les tests."""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)
