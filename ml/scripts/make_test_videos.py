"""Generate 10 realistic test videos (+ invalid files) with a labelled answer key.  macOS only.

Uses only local resources: macOS wallpaper photographs, Apple Color Emoji artwork, system fonts and
`say` for narration. Output (default data/test_videos/, git-ignored):

    01_team_meeting.mp4 ... 10_old_codec_mpeg4.mp4
    invalid/                   corrupt / empty / wrong-extension files for error cases
    TEST_GUIDE.md              what each video tests, searches + Ask questions with expected answers
    eval_set.json              ground truth for:  python scripts/evaluate.py --eval-set ../data/test_videos/eval_set.json
    manifest.json              exact scene timings

Usage (from ml/, venv active):  python scripts/make_test_videos.py [--out DIR] [--only 1 4 7]
Upload the .mp4 files through the app's Library page; don't copy them into data/videos/.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
WALLPAPERS = Path("/System/Library/Desktop Pictures")
FONT_FILES = {
    "sans": ("/System/Library/Fonts/Helvetica.ttc", 0),
    "bold": ("/System/Library/Fonts/Helvetica.ttc", 1),
    "mono": ("/System/Library/Fonts/Menlo.ttc", 0),
    "hand": ("/System/Library/Fonts/Supplemental/MarkerFelt.ttc", 0),
}
SR = 22050  # narration sample rate
WORK = Path(tempfile.mkdtemp(prefix="vs-testvids-"))

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
_font_cache: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _font_cache:
        path, idx = FONT_FILES[kind]
        try:
            _font_cache[key] = ImageFont.truetype(path, size, index=idx)
        except OSError:
            _font_cache[key] = ImageFont.load_default(size)
    return _font_cache[key]


_emoji_cache: dict = {}


def emoji(ch: str, size: int) -> Image.Image:
    """Apple Color Emoji only renders at 160 px; render once, then scale."""
    key = (ch, size)
    if key not in _emoji_cache:
        base = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
        ImageDraw.Draw(base).text((20, 20), ch, font=ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160), embedded_color=True)
        base = base.crop(base.getbbox())
        scale = size / max(base.size)
        _emoji_cache[key] = base.resize((max(1, round(base.width * scale)), max(1, round(base.height * scale))), Image.LANCZOS)
    return _emoji_cache[key]


def paste_emoji(img: Image.Image, ch: str, size: int, center: tuple[float, float], flip: bool = False) -> None:
    e = emoji(ch, size)
    if flip:
        e = e.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(e, (int(center[0] - e.width / 2), int(center[1] - e.height / 2)), e)


_photo_cache: dict = {}


def photo(name: str) -> Image.Image:
    """A macOS wallpaper photo (thumbnail-resolution for most, full-res for Sonoma)."""
    if name not in _photo_cache:
        candidates = [WALLPAPERS / ".thumbnails" / f"{name}.heic", WALLPAPERS / f"{name}.heic",
                      WALLPAPERS / ".wallpapers" / name / f"{name}.heic"]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            raise SystemExit(f"Wallpaper '{name}' not found on this Mac.")
        dst = WORK / f"{name.replace(' ', '_')}.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", "-Z", "2000", str(src), "--out", str(dst)], check=True, capture_output=True)
        _photo_cache[name] = Image.open(dst).convert("RGB")
    return _photo_cache[name]


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    s = max(w / img.width, h / img.height)
    return img.resize((math.ceil(img.width * s), math.ceil(img.height * s)), Image.LANCZOS)


def ken_burns(name: str, w: int, h: int, pan=(1, 0.5)):
    """Slow pan across a photo: returns a render(t, dur) function."""
    base = cover(photo(name), int(w * 1.15), int(h * 1.15))
    dx, dy = base.width - w, base.height - h

    def render(t: float, dur: float) -> Image.Image:
        p = t / max(dur, 1e-6)
        x = int(dx * (p if pan[0] > 0 else 1 - p) * abs(pan[0]))
        y = int(dy * (p if pan[1] > 0 else 1 - p) * abs(pan[1]))
        return base.crop((x, y, x + w, y + h)).copy()

    return render


def gradient(w: int, h: int, top, bottom) -> Image.Image:
    t = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(top) * (1 - t) + np.array(bottom) * t).astype(np.uint8)
    return Image.fromarray(np.broadcast_to(arr, (h, w, 3)).copy())


def text(d: ImageDraw.ImageDraw, xy, s: str, kind="sans", size=40, fill=(20, 20, 30), anchor="la"):
    d.text(xy, s, font=font(kind, size), fill=fill, anchor=anchor)


def caption(img: Image.Image, s: str) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle((0, h - 70, w, h), fill=(0, 0, 0, 140))
    text(d, (30, h - 35), s, "bold", 30, (255, 255, 255), "lm")


def slide(title: str, bullets=(), em: str | None = None, bg=((245, 247, 252), (225, 232, 245)), w=1280, h=720):
    def render(t, dur):
        img = gradient(w, h, *bg)
        d = ImageDraw.Draw(img)
        dark = sum(bg[0]) < 300  # light text on dark backgrounds
        d.rectangle((0, 0, w, 10), fill=(79, 70, 229))
        text(d, (70, 70), title, "bold", 60, (245, 245, 250) if dark else (20, 20, 30))
        for i, b in enumerate(bullets):
            text(d, (90, 200 + i * 75), b, "sans", 40, (205, 210, 230) if dark else (40, 45, 60))
        if em:
            paste_emoji(img, em, 260 if bullets else 300, (w - 240, h / 2 + 30))
        return img
    render.static = True
    return render


# ---------------------------------------------------------------------------
# Timeline model
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    render: Callable[[float, float], Image.Image]
    narration: str | None = None
    min_dur: float = 8.0
    fps: float | None = None  # per-scene frame rate (variable-frame-rate videos)
    label: str = ""
    queries: list[str] = field(default_factory=list)  # visual searches that should land in this scene


@dataclass
class VideoSpec:
    name: str
    title: str
    tests: str
    scenes: list[Scene]
    size: tuple[int, int] = (1280, 720)
    fps: float = 25
    voice: str | None = None
    audio: bool = True
    codec: str = "h264"
    interval: int = 2
    asks: list[tuple[str, str]] = field(default_factory=list)  # (question, expected answer)


def speak(textstr: str, voice: str | None) -> np.ndarray:
    aiff = WORK / f"say_{abs(hash((textstr, voice)))}.aiff"
    cmd = ["say", "-o", str(aiff)] + (["-v", voice] if voice else []) + [textstr]
    if subprocess.run(cmd, capture_output=True).returncode != 0 and voice:
        subprocess.run(["say", "-o", str(aiff), textstr], check=True)  # voice missing -> default
    with av.open(str(aiff)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = [fr.to_ndarray().reshape(-1) for f in c.decode(audio=0) for fr in rs.resample(f)]
        chunks += [fr.to_ndarray().reshape(-1) for fr in rs.resample(None)]
    return np.concatenate(chunks).astype(np.float32)


def build(spec: VideoSpec, out_dir: Path) -> dict:
    w, h = spec.size
    # 1. narration decides scene lengths
    timeline, t = [], 0.0
    for sc in spec.scenes:
        speech = speak(sc.narration, spec.voice) if (sc.narration and spec.audio) else None
        dur = max(sc.min_dur, (len(speech) / SR + 1.6) if speech is not None else 0)
        timeline.append((sc, round(t, 2), round(t + dur, 2), speech))
        t += dur
    total = t

    path = out_dir / spec.name
    out = av.open(str(path), "w")
    if spec.codec == "mpeg4":  # old MPEG-4 Part 2: OpenCV reads it, Chrome can't play it
        vs = out.add_stream("mpeg4", rate=int(spec.fps))
        vs.bit_rate = 2_000_000
    else:
        vs = out.add_stream("libx264", rate=int(max(spec.fps, max((s.fps or 0) for s in spec.scenes))),
                            options={"crf": "23", "preset": "veryfast"})
    vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
    vs.codec_context.time_base = Fraction(1, 1000)
    aus = None
    if spec.audio:  # every stream must exist before the first packet is written
        aus = out.add_stream("aac", rate=SR)
        aus.layout = "mono"

    # 2. video frames (per-scene fps -> variable frame rate where scenes differ)
    last_pts = -1
    for sc, start, end, _ in timeline:
        fps = sc.fps or spec.fps
        n = max(1, int((end - start) * fps))
        static = getattr(sc.render, "static", False)
        cached = None
        for i in range(n):
            local = i / fps
            pts = int(round((start + local) * 1000))
            if pts <= last_pts:
                continue
            img = cached if (static and cached is not None) else sc.render(local, end - start)
            if static:
                cached = img
            frame = av.VideoFrame.from_image(img.convert("RGB"))
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for p in vs.encode(frame):
                out.mux(p)
            last_pts = pts
    for p in vs.encode():
        out.mux(p)

    # 3. narration track
    if aus is not None:
        buf = np.zeros(int(total * SR) + SR, np.float32)
        for sc, start, end, speech in timeline:
            if speech is not None:
                i = int((start + 0.6) * SR)
                buf[i : i + len(speech)] += speech[: len(buf) - i]
        for i in range(0, int(total * SR), 1024):
            fr = av.AudioFrame.from_ndarray(buf[i : i + 1024].reshape(1, -1), format="flt", layout="mono")
            fr.sample_rate, fr.pts, fr.time_base = SR, i, Fraction(1, SR)
            for p in aus.encode(fr):
                out.mux(p)
        for p in aus.encode():
            out.mux(p)
    out.close()

    return {
        "file": spec.name, "title": spec.title, "tests": spec.tests, "duration_sec": round(total, 1),
        "size": f"{w}x{h}", "audio": spec.audio, "codec": spec.codec, "variable_frame_rate": len({s.fps or spec.fps for s in spec.scenes}) > 1,
        "suggested_interval": spec.interval, "mb": round(path.stat().st_size / 1e6, 1),
        "scenes": [{"start": s, "end": e, "label": sc.label, "narration": sc.narration if spec.audio else None, "queries": sc.queries}
                   for sc, s, e, _ in timeline],
        "asks": [{"question": q, "expected": a} for q, a in spec.asks],
    }


# ---------------------------------------------------------------------------
# Scene builders for specific videos
# ---------------------------------------------------------------------------
def whiteboard(lines, w=1280, h=720):
    def render(t, dur):
        img = Image.new("RGB", (w, h), (236, 236, 232))
        d = ImageDraw.Draw(img)
        d.rectangle((40, 30, w - 40, h - 60), fill=(252, 252, 250), outline=(150, 150, 150), width=10)
        d.rectangle((w // 2 - 200, h - 60, w // 2 + 200, h - 40), fill=(170, 170, 170))  # marker tray
        colors = [(200, 40, 40), (30, 70, 190), (30, 70, 190), (30, 70, 190), (30, 120, 60)]
        for i, ln in enumerate(lines):
            text(d, (110, 80 + i * 105), ln, "hand", 70 if i == 0 else 58, colors[i % len(colors)])
        d.line((110, 170, 520, 178), fill=(200, 40, 40), width=6)  # hand-drawn underline
        return img
    render.static = True
    return render


def bar_chart(title, labels, values, w=1280, h=720):
    def render(t, dur):
        img = gradient(w, h, (250, 250, 252), (236, 240, 248))
        d = ImageDraw.Draw(img)
        text(d, (70, 60), title, "bold", 54)
        x0, y0, top = 160, 620, 190
        d.line((x0, y0, 1150, y0), fill=(40, 40, 40), width=4)
        d.line((x0, y0, x0, top - 20), fill=(40, 40, 40), width=4)
        vmax = max(values)
        for i, (lab, v) in enumerate(zip(labels, values)):
            bx = x0 + 80 + i * 300
            bh = (y0 - top) * v / vmax
            d.rectangle((bx, y0 - bh, bx + 170, y0), fill=(79, 70, 229) if i < len(values) - 1 else (165, 160, 240))
            text(d, (bx + 85, y0 - bh - 30), f"${v}M", "bold", 36, anchor="mm")
            text(d, (bx + 85, y0 + 35), lab, "sans", 34, anchor="mm")
        return img
    render.static = True
    return render


def street(vehicles, w=1280, h=720, crossing=False, extras=()):
    """Road scene; vehicles = [(emoji, size, lane_y, speed_px_s, x0)] moving right-to-left."""
    def render(t, dur):
        img = gradient(w, h, (150, 200, 245), (210, 230, 250))
        d = ImageDraw.Draw(img)
        for i, (bx, bw, bh, col) in enumerate([(0, 180, 260, (120, 120, 135)), (190, 140, 330, (95, 100, 120)), (340, 220, 220, (140, 130, 120)),
                                               (580, 160, 300, (105, 110, 125)), (760, 240, 250, (130, 125, 140)), (1020, 260, 310, (100, 105, 115))]):
            d.rectangle((bx, 330 - bh, bx + bw, 330), fill=col)
            for wy in range(330 - bh + 20, 320, 40):
                for wx in range(bx + 15, bx + bw - 25, 40):
                    d.rectangle((wx, wy, wx + 18, wy + 22), fill=(230, 220, 160))
        d.rectangle((0, 330, w, 380), fill=(170, 170, 170))  # sidewalk
        d.rectangle((0, 380, w, h), fill=(70, 70, 75))  # road
        off = int(t * 120) % 160
        for x in range(-off, w, 160):
            d.rectangle((x, 545, x + 90, 555), fill=(240, 240, 240))
        if crossing:
            for x in range(520, 760, 40):
                d.rectangle((x, 380, x + 22, h), fill=(235, 235, 235))
        for ch, size, cx, cy in extras:
            paste_emoji(img, ch, size, (cx(t) if callable(cx) else cx, cy))
        for ch, size, lane_y, speed, x0 in vehicles:
            x = (x0 - speed * t) % (w + 400) - 200
            paste_emoji(img, ch, size, (x, lane_y))
        return img
    return render


def editor(code: str, typing=True, w=1280, h=720):
    lines = code.split("\n")
    total_chars = len(code)
    kw = {"def", "return", "import", "assert", "from"}

    def render(t, dur):
        img = Image.new("RGB", (w, h), (30, 32, 40))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w, 44), fill=(45, 47, 58))
        for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse((18 + i * 28, 14, 34 + i * 28, 30), fill=c)
        text(d, (w / 2, 22), "calculator.py — VS Code", "sans", 22, (200, 200, 210), "mm")
        d.rectangle((0, 44, 230, h), fill=(37, 39, 48))
        for i, f in enumerate(["calculator.py", "test_calc.py", "README.md"]):
            text(d, (24, 70 + i * 34), f, "sans", 22, (230, 230, 240) if i == 0 else (150, 150, 165))
        shown = int(total_chars * min(1, t / (dur * 0.75))) if typing else total_chars
        remaining = shown
        for i, ln in enumerate(lines):
            vis = ln[: max(0, remaining)]
            remaining -= len(ln) + 1
            y = 80 + i * 40
            text(d, (250, y), f"{i + 1:>2}", "mono", 26, (100, 105, 120))
            x = 310
            for tok in vis.split(" "):
                col = (198, 120, 221) if tok in kw else (152, 195, 121) if tok.startswith(("'", '"')) else (220, 223, 228)
                text(d, (x, y), tok, "mono", 26, col)
                x += d.textlength(tok + " ", font=font("mono", 26))
            if remaining < 0 and typing and int(t * 2) % 2 == 0:
                d.rectangle((x - 12, y, x - 9, y + 30), fill=(230, 230, 230))
                break
        return img
    return render


def terminal(lines_with_colors, w=1280, h=720):
    def render(t, dur):
        img = Image.new("RGB", (w, h), (12, 12, 14))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w, 40), fill=(55, 55, 60))
        text(d, (w / 2, 20), "Terminal — zsh", "sans", 22, (210, 210, 210), "mm")
        n = min(len(lines_with_colors), 1 + int(t / max(dur * 0.6, 0.1) * len(lines_with_colors)))
        for i, (ln, col) in enumerate(lines_with_colors[:n]):
            text(d, (30, 70 + i * 42), ln, "mono", 28, col)
        return img
    return render


def browser(w=1280, h=720):
    def render(t, dur):
        img = Image.new("RGB", (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w, 90), fill=(222, 225, 230))
        d.rounded_rectangle((150, 30, w - 150, 70), 18, fill=(255, 255, 255))
        text(d, (180, 50), "localhost:5000", "sans", 24, (60, 60, 70), "lm")
        scroll = int(min(1, t / dur) * 120)
        d.rectangle((0, 90, w, 300 - scroll), fill=(79, 70, 229))
        text(d, (w / 2, 190 - scroll), "Hello, VisionFind!", "bold", 72, (255, 255, 255), "mm")
        for i in range(5):
            y = 360 + i * 60 - scroll
            d.rounded_rectangle((200, y, 200 + 880 - i * 90, y + 26), 8, fill=(225, 228, 238))
        return img
    return render


def kitchen(items, title=None, w=1280, h=720):
    """Wooden table with emoji items: items = [(emoji, size, x, y)]."""
    def render(t, dur):
        img = Image.new("RGB", (w, h), (150, 100, 60))
        d = ImageDraw.Draw(img)
        for y in range(0, h, 90):
            d.rectangle((0, y, w, y + 86), fill=(160 + (y // 90 % 2) * 12, 108 + (y // 90 % 2) * 8, 66))
            d.line((0, y + 87, w, y + 87), fill=(110, 70, 40), width=3)
        bob = 6 * math.sin(t * 1.5)
        for ch, size, x, y in items:
            paste_emoji(img, ch, size, (x, y + bob))
        if title:
            d2 = ImageDraw.Draw(img, "RGBA")
            d2.rounded_rectangle((40, 30, 40 + 22 * len(title) + 60, 100), 16, fill=(255, 255, 255, 220))
            text(d2, (70, 65), title, "bold", 38, (60, 40, 20), "lm")
        return img
    return render


def over_photo(name, items, w=1280, h=720, pan=(1, 0.3)):
    """Real photo background with animated emoji on top: items = [(emoji, size, x(t), y(t), flip)]."""
    bg = ken_burns(name, w, h, pan)

    def render(t, dur):
        img = bg(t, dur)
        for ch, size, x, y, flip in items:
            paste_emoji(img, ch, size, (x(t) if callable(x) else x, y(t) if callable(y) else y), flip)
        return img
    return render


def with_caption(render, s):
    def r(t, dur):
        img = render(t, dur)
        caption(img, s)
        return img
    return r


def sofa_scene(w=1280, h=720):
    def render(t, dur):
        img = gradient(w, h, (235, 225, 210), (215, 200, 180))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 560, w, h), fill=(140, 110, 80))
        d.rounded_rectangle((250, 300, 1030, 600), 40, fill=(70, 110, 150))
        d.rounded_rectangle((210, 380, 330, 610), 30, fill=(60, 95, 135))
        d.rounded_rectangle((950, 380, 1070, 610), 30, fill=(60, 95, 135))
        d.rounded_rectangle((330, 420, 950, 520), 20, fill=(85, 125, 165))
        paste_emoji(img, "🐈", 230, (640, 400 + 4 * math.sin(t * 2)))
        return img
    return render


def phone_app(w=1280, h=720):
    def render(t, dur):
        img = gradient(w, h, (30, 30, 50), (70, 60, 120))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((470, 40, 810, 690), 50, fill=(15, 15, 20))
        d.rounded_rectangle((488, 60, 792, 670), 38, fill=(248, 248, 252))
        text(d, (640, 110), "AuraSound", "bold", 34, (40, 40, 60), "mm")
        paste_emoji(img, "🎧", 110, (640, 210))
        for i, lab in enumerate(["Bass", "Mid", "Treble"]):
            y = 330 + i * 80
            text(d, (520, y), lab, "sans", 24, (70, 70, 90), "lm")
            d.rounded_rectangle((600, y - 6, 760, y + 6), 6, fill=(220, 220, 235))
            knob = 600 + 160 * (0.5 + 0.4 * math.sin(t * 1.3 + i))
            d.ellipse((knob - 14, y - 14, knob + 14, y + 14), fill=(79, 70, 229))
        d.rounded_rectangle((530, 580, 750, 630), 25, fill=(79, 70, 229))
        text(d, (640, 605), "Find my headphones", "bold", 20, (255, 255, 255), "mm")
        return img
    return render


def field_scene(ball, ground, lines_col, w=1280, h=720):
    def render(t, dur):
        img = gradient(w, h, (140, 190, 240), (190, 220, 250))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 300, w, h), fill=ground)
        d.line((0, 500, w, 500), fill=lines_col, width=6)
        d.ellipse((540, 420, 740, 580), outline=lines_col, width=6)
        if ball == "⚽":
            d.rectangle((900, 200, 1180, 420), outline=(250, 250, 250), width=10)
            for x in range(910, 1180, 30):
                d.line((x, 210, x, 415), fill=(225, 225, 225), width=2)
        if ball == "🏀":
            d.rectangle((1080, 120, 1100, 400), fill=(90, 90, 90))
            d.rectangle((1010, 110, 1170, 210), fill=(250, 250, 250), outline=(200, 40, 40), width=6)
        if ball == "🎾":
            d.rectangle((0, 470, w, 490), fill=(250, 250, 250))
        x = 200 + (t * 140) % 700
        y = 470 - abs(math.sin(t * 3)) * 150
        paste_emoji(img, ball, 130, (x, y))
        return img
    return render


# ---------------------------------------------------------------------------
# The 10 videos
# ---------------------------------------------------------------------------
def specs() -> list[VideoSpec]:
    L = lambda a, b: (lambda t: a + (b - a) * t / 12)  # noqa: E731  linear motion helper over ~12 s
    return [
        VideoSpec(
            "01_team_meeting.mp4", "Team meeting (slides, chart, whiteboard, photo)",
            "Core Ask flow: reading slides and a handwritten whiteboard, a fact that is only spoken, citations, and an honest 'not mentioned'.",
            voice="Samantha", interval=2,
            scenes=[
                Scene(slide("Q3 Planning Meeting", ["Acme Robotics", "October 2026"], "🤖"), label="title slide",
                      narration="Welcome everyone to the Q3 planning meeting for Acme Robotics. I'm Sarah, and today we'll cover results, the roadmap, and next steps.",
                      queries=["a title slide with a robot"]),
                Scene(slide("Agenda", ["1. Q2 results", "2. Product roadmap", "3. Hiring", "4. Action items"]), label="agenda slide",
                      narration="Here is the agenda. First the Q2 results, then the product roadmap, hiring, and finally the action items."),
                Scene(bar_chart("Revenue by quarter", ["Q1", "Q2", "Q3 (forecast)"], [2.1, 2.9, 3.4]), label="revenue bar chart",
                      narration="Revenue grew forty percent in the second quarter, from two point one to two point nine million dollars. We expect three point four million this quarter.",
                      queries=["a bar chart", "a slide with a chart"]),
                Scene(whiteboard(["ACTION ITEMS", "1. Launch beta - Nov 3", "2. Hire 2 engineers", "3. Fix login bug (Raj)"]), label="whiteboard",
                      narration="On the whiteboard I wrote our action items. We launch the beta on November third, we hire two engineers, and Raj will fix the login bug.",
                      queries=["handwritten notes on a whiteboard"]),
                Scene(with_caption(ken_burns("Big Sur Coastline", 1280, 720), "Team offsite - Big Sur"), label="offsite photo (Big Sur coast)",
                      narration="Last month the whole team went on an offsite to the Big Sur coast. Thanks to everyone who came.",
                      queries=["a rocky coastline by the ocean"]),
                Scene(slide("Questions?", [], "🙋"), label="closing slide",
                      narration="That's everything for today. Any questions? Our next meeting is in two weeks."),
            ],
            asks=[("What are the action items on the whiteboard?", "Launch beta Nov 3; hire 2 engineers; fix login bug (Raj). Cites the whiteboard scene."),
                  ("How much did revenue grow in Q2?", "40% (2.1M to 2.9M dollars); from the chart and the narration."),
                  ("Who will fix the login bug?", "Raj."),
                  ("Where was the team offsite?", "The Big Sur coast."),
                  ("What is the CEO's name?", "Not mentioned; the presenter is Sarah but no CEO is named. It should say it can't find this.")],
        ),
        VideoSpec(
            "02_product_demo.mp4", "Product demo (headphones)",
            "'What product is shown at the start?' plus spoken specs (battery, price, ship date).",
            voice="Daniel", interval=2,
            scenes=[
                Scene(slide("AuraSound X1", ["Wireless noise-cancelling", "headphones"], "🎧", bg=((235, 240, 255), (205, 215, 245))), label="product hero",
                      narration="Meet the AuraSound X one, our new wireless noise cancelling headphones.", queries=["headphones"]),
                Scene(slide("Features", ["40-hour battery", "Hi-Res audio", "Active noise cancelling"], "🔋"), label="features",
                      narration="They last forty hours on a single charge, play high resolution audio, and block out background noise."),
                Scene(phone_app(), label="phone app", narration="Use the companion app on your phone to adjust the equaliser, or to find your headphones if you lose them.",
                      queries=["a smartphone showing an app"]),
                Scene(slide("$199", ["Ships November 15", "Order today"], "🎁", bg=((255, 246, 230), (250, 225, 190))), label="price",
                      narration="The AuraSound X one costs one hundred and ninety nine dollars and ships on November fifteenth.",
                      queries=["a gift box"]),
            ],
            asks=[("What product is shown at the start?", "AuraSound X1 wireless noise-cancelling headphones."),
                  ("How long does the battery last?", "40 hours."),
                  ("How much does it cost and when does it ship?", "$199, ships November 15."),
                  ("Is it waterproof?", "Not mentioned; it should say so.")],
        ),
        VideoSpec(
            "03_travel_vlog_big_sur.mp4", "Travel vlog (real photographs)",
            "Visual search on real photos; cross-video results (the coastline also appears in video 01).",
            voice="Karen", interval=2,
            scenes=[
                Scene(ken_burns("Big Sur Coastline", 1280, 720), label="coastline with red flowers",
                      narration="Day one. We started our road trip on the Big Sur coastline. Look at those red flowers on the cliffs.",
                      queries=["a rocky coastline by the ocean"]),
                Scene(ken_burns("Big Sur Road", 1280, 720, (1, -0.4)), label="winding road",
                      narration="Then we drove the winding road through the mountains. Every corner had a new view.",
                      queries=["a winding road through the mountains"]),
                Scene(ken_burns("Big Sur Mountains", 1280, 720), label="misty mountains",
                      narration="From the top you can see layer after layer of misty mountains.", queries=["misty mountain ridges"]),
                Scene(ken_burns("Big Sur Horizon", 1280, 720, (-1, 0.3)), label="rocks in calm water",
                      narration="At the beach there were huge rocks sitting in the calm grey water.", queries=["rocks in calm water"]),
                Scene(ken_burns("Catalina Sunset", 1280, 720), label="sunset above clouds",
                      narration="In the evening we watched the sunset above the clouds.", queries=["a sunset above the clouds"]),
                Scene(ken_burns("Sonoma Horizon", 1280, 720, (1, 0)), label="foggy forest hills",
                      narration="The next morning, fog rolled over the forest hills.", queries=["fog over forested hills"]),
                Scene(ken_burns("Big Sur Night Grasses", 1280, 720), label="blue grass at night",
                      narration="And at night, the grass next to our campsite looked almost blue.", queries=["glowing blue grass at night"]),
            ],
            asks=[("What did they see in the evening?", "A sunset above the clouds."),
                  ("Where did the trip start?", "On the Big Sur coastline (red flowers on the cliffs)."),
                  ("Did they see any animals?", "None visible or mentioned; it should say so.")],
        ),
        VideoSpec(
            "04_city_traffic_no_audio.mp4", "City traffic (no audio track)",
            "Transcription is skipped (no audio); Ask works from frames only. Includes the spec's 'a red car beside a person'.",
            audio=False, interval=1,
            scenes=[
                Scene(street([("🚗", 150, 470, 160, 300), ("🚕", 150, 470, 160, 900), ("🚙", 170, 640, 230, 600)]), min_dur=10,
                      label="cars and a taxi driving", queries=["cars driving on a city street", "a yellow taxi"]),
                Scene(street([("🚌", 260, 450, 110, 900)], extras=[("🚲", 190, L(1300, 300), 610)]), min_dur=10,
                      label="bus and bicycle", queries=["a bus", "a bicycle"]),
                Scene(street([], crossing=True, extras=[("🚦", 170, 440, 250), ("🚶", 130, L(500, 800), 500), ("🚶", 120, L(800, 520), 600)]), min_dur=10,
                      label="traffic light and pedestrians crossing", queries=["a traffic light", "people crossing the street"]),
                Scene(street([], extras=[("🚗", 220, 560, 470), ("🧍", 170, 760, 440)]), min_dur=10,
                      label="red car parked beside a person", queries=["a red car beside a person"]),
            ],
            asks=[("Which vehicles appear in this video?", "Cars (red car, blue SUV), a yellow taxi, a bus, a bicycle."),
                  ("What is the person next to the red car doing?", "Standing beside it (frames only; no transcript)."),
                  ("What does the driver say?", "There is no audio/transcript; it should say it can't hear anything.")],
        ),
        VideoSpec(
            "05_wildlife.mp4", "Wildlife (animals on real landscapes)",
            "Object search ('a dog', 'a sea turtle'); counting from frames ('how many horses?').",
            voice="Moira", interval=2,
            scenes=[
                Scene(over_photo("Sonoma Horizon", [("🐕", 200, L(1100, 250), lambda t: 520 - abs(math.sin(t * 4)) * 40, False)]),
                      label="dog running on hills", narration="A dog runs happily across the green hills.", queries=["a dog running"]),
                Scene(over_photo("Big Sur Mountains", [("🐎", 210, 470, 560, False), ("🐎", 190, 820, 580, True)]),
                      label="two horses below mountains", narration="Two horses graze below the mountains.", queries=["horses"]),
                Scene(over_photo("Catalina Clouds", [("🕊️", 120, L(200, 1100), lambda t: 200 + 30 * math.sin(t * 2), True),
                                                     ("🕊️", 100, L(0, 900), lambda t: 300 + 30 * math.cos(t * 2), True)]),
                      label="birds over the sea", narration="Birds glide over the sea under a stormy sky.", queries=["birds flying over the sea"]),
                Scene(over_photo("Big Sur Waters Edge", [("🐢", 220, L(300, 700), 520, False), ("🐠", 110, L(1100, 600), 380, False),
                                                         ("🐟", 100, L(900, 300), 600, False)]),
                      label="sea turtle and fish", narration="In the clear water we spotted colourful fish and a sea turtle.",
                      queries=["a sea turtle", "fish swimming"]),
                Scene(sofa_scene(), label="cat on a sofa", narration="Back home, the cat was fast asleep on the sofa.", queries=["a cat on a sofa"]),
            ],
            asks=[("What animals were in the water?", "Fish and a sea turtle."),
                  ("How many horses are there?", "Two (visible in the frames)."),
                  ("Where was the cat?", "Asleep on the sofa, at home.")],
        ),
        VideoSpec(
            "06_cooking_pizza.mp4", "Cooking show (pizza)",
            "Step-by-step Ask questions mixing visible items and spoken instructions.",
            voice="Samantha", interval=2,
            scenes=[
                Scene(kitchen([("🍅", 170, 300, 380), ("🧅", 150, 530, 420), ("🧄", 140, 740, 380), ("🧀", 170, 980, 410)], "Ingredients"),
                      label="ingredients", narration="Today we're making a margherita pizza. You need tomatoes, an onion, garlic, and mozzarella cheese.",
                      queries=["tomatoes and vegetables on a table"]),
                Scene(kitchen([("🫓", 300, 640, 400)], "Step 1: Stretch the dough"), label="dough",
                      narration="First, stretch the dough into a circle and spread the tomato sauce on top."),
                Scene(kitchen([("🔥", 260, 640, 400)], "Step 2: Bake 10 min at 250°C"), label="baking (flames)",
                      narration="Bake it for ten minutes at two hundred and fifty degrees Celsius.", queries=["flames"]),
                Scene(kitchen([("🍕", 330, 640, 400)]), label="finished pizza", narration="And here is the finished pizza. Crispy and delicious.",
                      queries=["a pizza"]),
                Scene(kitchen([("☕", 220, 470, 400), ("🍰", 230, 800, 410)], "Dessert"), label="coffee and cake",
                      narration="For dessert, a cup of coffee and a slice of cake.", queries=["a cup of coffee and a slice of cake"]),
            ],
            asks=[("What dish are they cooking?", "A margherita pizza."),
                  ("How long and at what temperature is it baked?", "10 minutes at 250 °C."),
                  ("What ingredients are shown at the start?", "Tomatoes, onion, garlic, mozzarella."),
                  ("What was served for dessert?", "Coffee and a slice of cake.")],
        ),
        VideoSpec(
            "07_coding_screencast_vfr.mp4", "Coding screencast (variable frame rate)",
            "Variable frame rate like a real screen recording (30 fps while typing, 3-6 fps when idle): thumbnails must match the player exactly. Reading on-screen text and errors.",
            voice="Daniel", interval=1, fps=30,
            scenes=[
                Scene(editor("def add(a, b):\n    return a + b\n\n\ndef test_add():\n    assert add(2, 3) == 5"), fps=30, label="code editor (typing)",
                      narration="I'm writing a small function that adds two numbers, plus a test for it.", queries=["a code editor"]),
                Scene(terminal([("$ pytest", (230, 230, 230)), ("============ test session starts ============", (200, 200, 200)),
                                ("collected 3 items", (200, 200, 200)), ("test_calc.py ...                     [100%]", (200, 200, 200)),
                                ("============= 3 passed in 0.12s =============", (98, 210, 84))]), fps=6,
                      label="terminal: 3 passed", narration="Now I run the tests with pytest. All three tests pass.", queries=["a terminal window"]),
                Scene(terminal([("$ python calculator.py", (230, 230, 230)), ("Traceback (most recent call last):", (240, 90, 90)),
                                ('  File "calculator.py", line 9, in <module>', (240, 90, 90)), ("    add(2, 'three')", (240, 90, 90)),
                                ("TypeError: unsupported operand type(s) for +: 'int' and 'str'", (255, 70, 70))]), fps=3,
                      label="red TypeError", narration="Oops. Calling it with a string causes a type error.", queries=["a red error message in a terminal"]),
                Scene(browser(), fps=12, label="browser hello page", narration="Finally, the app runs in the browser and shows the hello page.",
                      queries=["a web browser showing a website"]),
            ],
            asks=[("What error appeared?", "TypeError: unsupported operand type(s) for +: 'int' and 'str' (add(2, 'three'))."),
                  ("What command did they run and how many tests passed?", "pytest; 3 passed."),
                  ("What does the web page say?", "Hello, VisionFind!")],
        ),
        VideoSpec(
            "08_phone_portrait_vfr.mp4", "Phone video (portrait, variable frame rate)",
            "Portrait 720x1280 layout in the player and thumbnails; variable frame rate timestamps.",
            voice="Aman", interval=1, size=(720, 1280), fps=30,
            scenes=[
                Scene(ken_burns("The Beach Light", 720, 1280, (0.5, 1)), fps=30, label="beach (illustration)",
                      narration="We reached the beach just before sunset.", queries=["a beach"]),
                Scene(over_photo("Catalina Shoreline", [("🍦", 330, 360, lambda t: 760 + 10 * math.sin(t * 3), False)], 720, 1280, (0.5, 1)), fps=8,
                      label="ice cream by the sea", narration="Of course we had to get ice cream.", queries=["an ice cream cone"]),
                Scene(ken_burns("Tree Light", 720, 1280, (1, 0.2)), fps=30, label="lone tree on sandstone",
                      narration="On the way back we stopped at these amazing sandstone rocks with a single tree on top.",
                      queries=["a lone tree on sandstone rocks"]),
                Scene(over_photo("Valley Light", [("📸", 260, 360, 640, False)], 720, 1280), fps=8, label="camera",
                      narration="Time for one last photo.", queries=["a camera"]),
            ],
            asks=[("What did they eat?", "Ice cream."), ("What was on top of the sandstone rocks?", "A single tree.")],
        ),
        VideoSpec(
            "09_lecture_solar_system_5min.mp4", "5-minute lecture (solar system)",
            "Long video: longer indexing (watch the stages), many transcript chunks, facts only in speech.",
            voice="Karen", interval=5, fps=12,
            scenes=[
                Scene(slide("The Solar System", ["Lecture 4 - Astronomy 101"], "☀️", bg=((15, 20, 45), (40, 50, 95))), min_dur=43, label="title",
                      narration="Welcome to lecture four of Astronomy one oh one. Today we take a tour of the solar system, starting at the Sun and travelling outward. "
                                "We will look at the Sun itself, the Earth and its Moon, the red planet Mars, the giant ringed planet Saturn, and finally the comets that visit us from the edge of the solar system. "
                                "Please keep your notebooks ready, because there will be a short quiz next week.",
                      queries=["the sun"]),  # the title slide shows a sun too
                Scene(slide("The Sun", ["99.8% of the solar system's mass", "About 4.6 billion years old", "Surface: ~5,500 °C"], "☀️"), min_dur=43, label="the sun",
                      narration="The Sun holds ninety nine point eight percent of all the mass in the solar system. It is about four point six billion years old, "
                                "and its surface temperature is around five thousand five hundred degrees Celsius. Every second it turns millions of tonnes of hydrogen into helium, "
                                "and that energy is what lights and warms our planet.", queries=["the sun"]),
                Scene(slide("Earth and the Moon", ["Moon distance: 384,400 km", "Causes ocean tides", "Same side always faces us"], "🌍"), min_dur=43, label="earth and moon",
                      narration="Next is our home, the Earth, with its Moon. The Moon is about three hundred eighty four thousand four hundred kilometres away. "
                                "Its gravity pulls on our oceans and causes the tides. Because it is tidally locked, the same side of the Moon always faces the Earth.",
                      queries=["the planet earth"]),
                Scene(slide("Mars", ["The red planet", "Iron oxide (rust) on the surface", "Olympus Mons: tallest volcano"], "🔴"), min_dur=43, label="mars",
                      narration="Mars is called the red planet. Its colour comes from iron oxide, which is basically rust, covering its surface. "
                                "Mars is also home to Olympus Mons, the tallest volcano in the solar system, nearly three times the height of Mount Everest."),
                Scene(slide("Saturn", ["Rings of ice and rock", "146 known moons", "Less dense than water"], "🪐"), min_dur=43, label="saturn",
                      narration="Saturn is famous for its beautiful rings, which are made mostly of ice and rock. It has one hundred and forty six known moons. "
                                "Surprisingly, Saturn is less dense than water, so in a big enough bathtub it would float.", queries=["a planet with rings"]),
                Scene(slide("Comets", ["Icy bodies from the outer solar system", "Tails point away from the Sun"], "☄️", bg=((15, 20, 45), (40, 50, 95))), min_dur=43,
                      label="comets",
                      narration="Comets are icy bodies that come from the cold outer edge of the solar system. As a comet gets close to the Sun, the ice turns to gas and forms a glowing tail. "
                                "That tail always points away from the Sun, pushed by the solar wind.", queries=["a comet"]),
                Scene(slide("Summary", ["Sun: 99.8% of the mass", "Moon: 384,400 km away", "Mars: red from iron oxide", "Saturn: rings of ice and rock", "Quiz next Tuesday"], "📝"),
                      min_dur=43, label="summary",
                      narration="To summarise: the Sun holds almost all the mass, the Moon is three hundred eighty four thousand kilometres away, Mars is red because of rust, "
                                "and Saturn's rings are made of ice and rock. The quiz is next Tuesday. Thank you for listening, and see you next week."),
            ],
            asks=[("Why is Mars red?", "Iron oxide (rust) on its surface."),
                  ("How far away is the Moon?", "About 384,400 km."),
                  ("What are Saturn's rings made of?", "Mostly ice and rock."),
                  ("When is the quiz?", "Next Tuesday (spoken in the intro and on the summary slide).")],
        ),
        VideoSpec(
            "10_old_codec_mpeg4.mp4", "Old codec (MPEG-4 Part 2, no audio)",
            "Indexes and searches fine, but Chrome can't play it: the video page should show a clear 'can't be played' message instead of a broken player.",
            audio=False, codec="mpeg4", interval=2,
            scenes=[
                Scene(field_scene("⚽", (60, 150, 60), (240, 240, 240)), min_dur=10, label="soccer ball and goal", queries=["a soccer ball on a field"]),
                Scene(field_scene("🏀", (196, 140, 90), (250, 250, 250)), min_dur=10, label="basketball court", queries=["a basketball"]),
                Scene(field_scene("🎾", (60, 110, 170), (250, 250, 250)), min_dur=10, label="tennis court", queries=["a tennis ball"]),
            ],
            asks=[("Which sports appear in this video?", "Soccer, basketball and tennis (frames only).")],
        ),
    ]


# ---------------------------------------------------------------------------
# Answer key + invalid files
# ---------------------------------------------------------------------------
def write_invalid(out: Path, sample: Path) -> list[tuple[str, str]]:
    inv = out / "invalid"
    inv.mkdir(exist_ok=True)
    (inv / "corrupt_video.mp4").write_bytes(np.random.default_rng(0).integers(0, 256, 200_000, dtype=np.uint8).tobytes())
    (inv / "empty.mp4").write_bytes(b"")
    shutil.copyfile(sample, inv / "wrong_extension.mov")
    (inv / "notes.txt.mp4").write_text("This is a text file pretending to be a video.\n")
    return [("corrupt_video.mp4", "400: could not be opened as a video"), ("empty.mp4", "400: the uploaded file is empty"),
            ("wrong_extension.mov", "rejected in the browser: only .mp4 files are supported"),
            ("notes.txt.mp4", "400: could not be opened as a video")]


def write_guide(out: Path, videos: list[dict], invalid: list[tuple[str, str]]) -> None:
    def ts(s: float) -> str:
        return f"{int(s) // 60:02d}:{int(s) % 60:02d}"

    lines = [
        "# VisionFind test videos", "",
        "Generated by `ml/scripts/make_test_videos.py`. Upload them through the **Library** page (don't copy them into `data/videos/`).", "",
        "| # | File | Length | Audio | Notes | Suggested interval |", "|---|---|---|---|---|---|",
    ]
    for i, v in enumerate(videos, 1):
        notes = ", ".join(n for n, on in [("variable frame rate", v["variable_frame_rate"]), ("portrait", v["size"] == "720x1280"),
                                          ("MPEG-4 Part 2 (won't play in Chrome)", v["codec"] == "mpeg4")] if on) or "-"
        lines.append(f"| {i} | `{v['file']}` | {ts(v['duration_sec'])} | {'speech' if v['audio'] else 'none'} | {notes} | {v['suggested_interval']} s |")
    lines += ["", "## Invalid files (`invalid/`) for error handling", "", "| File | Expected result |", "|---|---|"]
    lines += [f"| `{f}` | {r} |" for f, r in invalid]
    lines += ["", "To test the size limit, restart the ML service with `VS_MAX_UPLOAD_MB=1` and upload any video above 1 MB (expect 'larger than the 1 MB limit').", ""]
    for i, v in enumerate(videos, 1):
        lines += [f"## {i}. {v['title']} - `{v['file']}`", "", f"**Tests:** {v['tests']}", "",
                  "| Time | Scene | Search for (should land here) |", "|---|---|---|"]
        for s in v["scenes"]:
            q = "; ".join(f"\"{x}\"" for x in s["queries"]) or "-"
            lines.append(f"| {ts(s['start'])}-{ts(s['end'])} | {s['label']} | {q} |")
        if v["asks"]:
            lines += ["", "**Ask about this video:**", ""]
            lines += [f"- *{a['question']}* -> {a['expected']}" for a in v["asks"]]
        lines.append("")
    lines += ["## Cross-video searches (Search page, all videos)", "",
              "- \"a rocky coastline by the ocean\" -> both `01_team_meeting` (offsite photo) and `03_travel_vlog_big_sur` (first scene)",
              "- \"a red car beside a person\" -> `04_city_traffic_no_audio` last scene",
              "- \"a whiteboard\" -> `01_team_meeting`; \"a terminal window\" -> `07_coding_screencast_vfr`",
              "- Something not in any video (e.g. \"an elephant\") -> results with low similarity (~0.10-0.15); that's expected", "",
              "## Measure accuracy", "", "```bash", "cd ml && source .venv/bin/activate",
              "python scripts/evaluate.py --eval-set ../data/test_videos/eval_set.json --intervals 1 2 5", "```", ""]
    (out / "TEST_GUIDE.md").write_text("\n".join(lines))


def write_eval_set(out: Path, videos: list[dict]) -> None:
    ev = {"videos": []}
    for v in videos:
        ranges: dict[str, list] = {}
        for s in v["scenes"]:
            for q in s["queries"]:
                ranges.setdefault(q, []).append([s["start"], s["end"]])
        if ranges:
            ev["videos"].append({"path": v["file"], "queries": [{"query": q, "relevant": r} for q, r in ranges.items()]})
    (out / "eval_set.json").write_text(json.dumps(ev, indent=2))


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("This generator needs macOS (say, system fonts, wallpapers, Apple Color Emoji).")
    p = argparse.ArgumentParser(description="Generate VisionFind test videos")
    p.add_argument("--out", type=Path, default=REPO / "data" / "test_videos")
    p.add_argument("--only", type=int, nargs="*", help="video numbers to (re)build, e.g. --only 1 7")
    args = p.parse_args()
    out = args.out.resolve()
    if out == (REPO / "data" / "videos").resolve():
        raise SystemExit("Don't write into data/videos/ (the app manages it). Use the default data/test_videos/.")
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "manifest.json"
    previous = {v["file"]: v for v in json.loads(manifest_path.read_text())["videos"]} if manifest_path.exists() else {}
    videos = []
    for i, spec in enumerate(specs(), 1):
        if args.only and i not in args.only and spec.name in previous:
            videos.append(previous[spec.name])
            continue
        print(f"[{i:>2}/10] {spec.name} …", flush=True)
        info = build(spec, out)
        print(f"        {info['duration_sec']:.0f}s, {info['size']}, {info['mb']} MB", flush=True)
        videos.append(info)

    invalid = write_invalid(out, out / videos[1]["file"])
    manifest_path.write_text(json.dumps({"videos": videos}, indent=2))
    write_eval_set(out, videos)
    write_guide(out, videos, invalid)
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"\nDone: {out}\n  - {len(videos)} videos, invalid/ samples, TEST_GUIDE.md, eval_set.json, manifest.json")


if __name__ == "__main__":
    main()
