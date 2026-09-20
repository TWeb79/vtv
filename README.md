# VTV - Vector-Time Video (research codec, beta 0.2)

One-file research video codec: encoder, decoder, container format, command line,
web app with player and statistics. Needs Python 3 and NumPy
(tested with Python 3.12 / NumPy 2.4). For mp4/mov/mkv input you also need one of:
`ffmpeg` on PATH, `pip install imageio-ffmpeg`, or `pip install opencv-python`.

    python vtv.py selftest                    # verify the file works (about 5 s)
    python vtv.py serve                       # web app: upload, statistics, player
    python vtv.py encode clip.mp4 clip.vtv --qp 30 --width 256 --frames 60
    python vtv.py encode demo demo.vtv        # no video file needed
    python vtv.py info clip.vtv
    python vtv.py decode clip.vtv out.y4m     # or .mp4 (needs ffmpeg) or a folder of PNGs

## Layout
- `vtv.py`   the product. Generated from `dev/` - rebuild with `python dev/scripts/build_single.py`
             (output is byte-identical to the committed file).
- `dev/`     modular sources (`package/`), web/CLI/video/doc parts (`parts/`), build script,
             tests, benchmark and tuning scripts, result logs.
- `STATUS.md` honest status, measurements and known issues - read this first.
