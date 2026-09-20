# VTV codec is created by Torsten Weber - github:TWeb79 - 09/2026
"""Make `import vtv` work when running `pytest` from the repository root."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
