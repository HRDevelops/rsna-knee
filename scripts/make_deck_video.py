# Copyright 2026 Husnain Rasool
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Narrated rehearsal video of the final deck: each slide image with its speaker notes read aloud (edge-tts),
plus an .srt with the script. Needs PowerPoint (slide export), ffmpeg and edge-tts; edge-tts needs internet.

    python scripts/make_deck_video.py [--voice en-US-AndrewNeural] [--rate -5%]

Output (gitignored, it shows competition MRI images): docs/presentation/Team10_FinalPresentation_rehearsal.mp4/.srt
"""
import argparse
import asyncio
import re
import subprocess
from pathlib import Path

import edge_tts
from pptx import Presentation

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "docs/presentation/Team10_FinalPresentation_KneeMRIAbnormalityDetection.pptx"
OUT = ROOT / "docs/presentation/Team10_FinalPresentation_rehearsal"
WORK = ROOT / "artifacts/deck_video"
PAD = 0.8  # seconds of silence before and after each slide's narration


def clean(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)  # drop "[0:00-1:00]" and stage directions
    text = text.replace("β", "beta").replace("~", "about ").replace("≤", "at most ")
    return re.sub(r"\s+", " ", text).strip()


def export_slides():
    ps = (f"$pp = New-Object -ComObject PowerPoint.Application; "
          f"$p = $pp.Presentations.Open('{DECK}', $true, $false, $false); $i = 1; "
          f"foreach ($s in $p.Slides) {{ $s.Export('{WORK}\\slide' + $i + '.png', 'PNG', 1920, 1080); $i++ }}; "
          f"$p.Close(); $pp.Quit()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


async def speak(text, mp3, voice, rate):
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="SentenceBoundary")
    subs = edge_tts.SubMaker()
    with open(mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                subs.feed(chunk)
    return subs


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout)


def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--rate", default="-5%")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    export_slides()
    notes = [clean(s.notes_slide.notes_text_frame.text) for s in Presentation(DECK).slides]

    clips, srt, t0, n = [], [], 0.0, 1
    for i, text in enumerate(notes, 1):
        mp3 = WORK / f"slide{i}.mp3"
        subs = asyncio.run(speak(text, mp3, a.voice, a.rate))
        d = duration(mp3)
        clip = WORK / f"clip{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30", "-i", str(WORK / f"slide{i}.png"),
                        "-i", str(mp3), "-filter_complex",
                        f"[1:a]adelay={int(PAD * 1000)}:all=1,apad=pad_dur={PAD},aresample=48000[a]",
                        "-map", "0:v", "-map", "[a]", "-t", f"{d + 2 * PAD:.3f}", "-c:v", "libx264", "-tune",
                        "stillimage", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(clip)], check=True)
        for cue in subs.cues:  # sentence-level captions, shifted to this slide's start
            s, e = cue.start.total_seconds() + t0 + PAD, cue.end.total_seconds() + t0 + PAD
            srt.append(f"{n}\n{srt_time(s)} --> {srt_time(e)}\n{cue.content}\n")
            n += 1
        clips.append(clip)
        t0 += duration(clip)
        print(f"slide {i}: {d:.1f} s narration", flush=True)

    (WORK / "list.txt").write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    OUT.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(WORK / "list.txt"),
                    "-i", str(OUT.with_suffix(".srt")), "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng", str(OUT.with_suffix(".mp4"))], check=True)
    print(f"wrote {OUT.with_suffix('.mp4').name}: {t0 / 60:.1f} min, {len(clips)} slides")


if __name__ == "__main__":
    main()
