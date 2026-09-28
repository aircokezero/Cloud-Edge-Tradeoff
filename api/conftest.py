import os
import sys

# Ensure api/ (where main.py lives) is importable from api/tests/, regardless
# of how pytest is invoked or what its auto-detected rootdir is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
