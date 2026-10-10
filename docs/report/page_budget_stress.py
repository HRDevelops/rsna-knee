"""Stress test: replace every notes block by filler prose of a known total word count, compile, report pages.
Used to tell how many words of real prose fit in 6 pages next to the report's floats and references."""
import re
import subprocess
import sys

import pymupdf

WORDS = ("the model attends over adjacent slices and each finding pools evidence from the windows that matter while "
         "validation against noisy labels remains a conservative proxy for performance on the hidden test set so "
         "every comparison uses the same folds and the same bootstrap noise before any claim is made").split()


def filler(n):
    out, i = [], 0
    while len(out) < n:
        out.append(WORDS[i % len(WORDS)])
        i += 1
    s = " ".join(out)
    # break into ~18-word sentences
    toks = s.split()
    sents = [" ".join(toks[k:k + 18]).capitalize() + "." for k in range(0, len(toks), 18)]
    return " ".join(sents)


def run(total_words, tag="stress_fill"):
    src = open("main.tex", encoding="utf-8").read()
    bs = chr(92)
    blocks = re.findall(re.escape(bs + "begin{notes}") + ".*?" + re.escape(bs + "end{notes}"), src, flags=re.S)
    per = total_words // len(blocks)
    for b in blocks:
        src = src.replace(b, filler(per) + "\n", 1)
    open(tag + ".tex", "w", encoding="utf-8").write(src)
    cmd = ["pdflatex", "-interaction=nonstopmode", tag + ".tex"]
    subprocess.run(cmd, capture_output=True, text=True)
    subprocess.run(["bibtex", tag], capture_output=True, text=True)
    for _ in range(2):
        subprocess.run(cmd, capture_output=True, text=True)
    d = pymupdf.open(tag + ".pdf")
    last = d[-1]
    ys = [b[3] for b in last.get_text("blocks")] + [im["bbox"][3] for im in last.get_image_info()]
    return len(d), max(ys) / last.rect.height


for w in [int(a) for a in sys.argv[1:]]:
    n, fill = run(w)
    print(f"{w} words of prose -> {n} pages, last page {fill:.0%} full")
