"""Video probing, frame decoding and clip export through ffmpeg."""

import json
import shutil
import subprocess
from pathlib import Path


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, check=True)


def probe(path: Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name:format=duration",
                "-of", "json", str(path)]).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "duration": float(data["format"]["duration"]),
        "fps": float(num) / float(den),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "frames": int(stream.get("nb_frames") or 0),
    }


def decode_frames(path: Path, out_dir: Path, fps: float, duration: float, quality: int = 3) -> list[dict]:
    """Decode the video once at `fps`. Frame i is the source frame nearest to t = i / fps."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps}", "-q:v", str(quality),
          str(tmp / "%05d.jpg")])
    frames = []
    for i, src in enumerate(sorted(tmp.glob("*.jpg"))):
        t = round(i / fps, 3)
        if t > duration:
            src.unlink()
            continue
        dst = tmp / f"t{int(round(t * 1000)):07d}.jpg"
        src.rename(dst)
        frames.append({"t": t, "file": dst.name})
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return frames


def export_clip(path: Path, out: Path, start: float, end: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}",
          "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
          "-movflags", "+faststart", str(out)])
