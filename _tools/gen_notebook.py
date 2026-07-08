#!/usr/bin/env python3
"""
Shared notebook generator for minimind-tutorial.
Usage:
  python gen_notebook.py <content_file> <output_dir>

Content file format (delimiter-based, avoids all Python/JSON quoting issues):

  @@NOTEBOOK ch##.ipynb
  @@MD
  (markdown content - any quotes/special chars OK)
  @@CODE
  (code content - any quotes OK)
  @@MD
  (next markdown cell)
  @@NOTEBOOK summary.ipynb
  ...
  @@END
"""
import json, os, sys, uuid


def make_cell(cell_type, source):
    cell = {"cell_type": cell_type, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_and_generate(content_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(content_path, "r", encoding="utf-8") as f:
        raw = f.read()

    notebooks = {}
    current_nb = None
    current_type = None
    current_lines = []

    def flush_cell():
        nonlocal current_type, current_lines
        if current_type and current_lines:
            src = "\n".join(current_lines).strip("\n")
            if src:
                notebooks[current_nb].append(make_cell(current_type, src))
        current_type = None
        current_lines = []

    def flush_notebook():
        nonlocal current_nb
        flush_cell()
        if current_nb and current_nb in notebooks:
            cells = notebooks[current_nb]
            path = os.path.join(output_dir, current_nb)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(make_notebook(cells), f, ensure_ascii=False, indent=1)
            n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
            n_code = sum(1 for c in cells if c["cell_type"] == "code")
            ratio = n_md / max(n_code, 1) if n_code else float("inf")
            print(f"  {current_nb}: {n_md} md + {n_code} code = {ratio:.1f}:1")

    for line in raw.split("\n"):
        if line.startswith("@@NOTEBOOK "):
            flush_notebook()
            current_nb = line[len("@@NOTEBOOK ") :].strip()
            notebooks[current_nb] = []
            current_type = None
            current_lines = []
        elif line.startswith("@@MD"):
            flush_cell()
            current_type = "markdown"
        elif line.startswith("@@CODE"):
            flush_cell()
            current_type = "code"
        elif line.startswith("@@END"):
            break
        else:
            if current_type is not None:
                current_lines.append(line)

    flush_notebook()
    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <content_file> <output_dir>")
        sys.exit(1)
    parse_and_generate(sys.argv[1], sys.argv[2])
