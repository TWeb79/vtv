"""Assemble vtv.py (single file) from the tested package modules + new parts."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'package', 'vtv') + os.sep
PARTS = os.path.join(HERE, '..', 'parts') + os.sep
CORE = ['config', 'dct', 'rangecoder', 'warp', 'io', 'trajectory', 'motion', 'syntax', 'recon',
        'bitstream', 'encoder', 'decoder']
TITLES = {
 'config': 'Configuration and constants', 'dct': 'Transform / quantisation', 'rangecoder': 'Entropy coder',
 'warp': 'Geometric transport (warping)', 'io': 'Frames, colour conversion, y4m', 'trajectory': 'Temporal trajectories',
 'motion': 'Global motion estimation (encoder only)', 'syntax': 'Bitstream syntax', 'recon': 'Shared reconstruction',
 'bitstream': 'Container format', 'encoder': 'Encoder', 'decoder': 'Decoder'}

DROP = re.compile(r"^(from \.|import numpy as np$|import math$|import struct$|import zlib$|import re$|"
                  r"from collections import defaultdict$|from fractions import Fraction$|from dataclasses import dataclass$)")

def strip(name, src):
    m = re.match(r'(?:\s*#[^\n]*\n)*\s*"""(.*?)"""\s*\n', src, re.S)   # tolerate leading comment/author lines
    doc = m.group(1).strip() if m else ""
    body = src[m.end():] if m else src
    out, skipping = [], False
    for ln in body.split("\n"):
        if skipping:
            if ")" in ln: skipping = False
            continue
        if DROP.match(ln):
            if ln.startswith("from .") and "(" in ln and ")" not in ln: skipping = True
            continue
        out.append(ln)
    bar = "# " + "=" * 77
    head = [bar, f"# {TITLES[name]}"] + [("# " + l).rstrip() for l in doc.split("\n")] + [bar]
    return "\n".join(head) + "\n" + "\n".join(out).strip("\n") + "\n"

doc = open(PARTS + 'part_doc.txt').read().rstrip("\n")
hdr = doc + '''

import io as _stdio
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zlib
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

DOC = __doc__

'''
parts = [strip(n, open(SRC + n + '.py').read()) for n in CORE]
body = "\n\n".join(parts)
body = body.replace("C420mpeg2", "C420jpeg")          # decoded output is full-range BT.601
tail = "\n\n" + "\n\n".join(open(PARTS + p).read() for p in ('part_video.py', 'part_web.py', 'part_cli.py'))
open(os.path.join(HERE, '..', '..', 'vtv.py'), 'w').write(hdr + body + tail)
print("written", len((hdr + body + tail).splitlines()), "lines")
