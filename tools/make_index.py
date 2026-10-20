#!/usr/bin/env python3
"""Regenerate index.html from the lab*/ folders (title = first line of lab README.md)."""
import glob, os, html
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
out = ['<!doctype html><meta charset="utf-8"><title>CV8508 labs</title>',
       '<h1>CV8508: Perceptual Cognition for Computer Vision, labs</h1>']
labs = sorted(glob.glob(os.path.join(root, "lab*")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
for d in labs:
    name = os.path.basename(d); rm = os.path.join(d, "README.md")
    title = open(rm).readline().lstrip("# ").strip() if os.path.exists(rm) else name
    out.append(f"<h2>{html.escape(title)}</h2><ul>")
    for f in sorted(glob.glob(os.path.join(d, "*.ipynb"))):
        out.append(f'<li><a href="{name}/{os.path.basename(f)}">{os.path.basename(f)}</a></li>')
    for f in sorted(glob.glob(os.path.join(d, "data", "*.zip"))):
        out.append(f'<li><a href="{name}/data/{os.path.basename(f)}">data/{os.path.basename(f)}</a></li>')
    out.append("</ul>")
open(os.path.join(root, "index.html"), "w").write("\n".join(out) + "\n")
print("index.html written for", [os.path.basename(d) for d in labs])
