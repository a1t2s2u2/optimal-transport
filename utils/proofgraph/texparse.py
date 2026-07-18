#!/usr/bin/env python3
"""proofgraph 用の最小 TeX パーサ・ヘルパ群。

以前は seminar/site/scripts/tex2md.py を sys.path 越しに import して再利用して
いたが、サイト生成は Node（tex2md.mjs）へ移行したため、proofgraph が必要とする
小さなヘルパだけをここに切り出して self-contained にした。

提供するもの:
  - BLOCK_ENVS / ENV_TO_PREFIX : ブロック環境とラベル接頭辞
  - strip_comments             : 行コメント除去
  - _extract_brace_arg         : 入れ子の波括弧に対応した引数抽出
  - build_label_map / build_chapter_map : ラベル→タイトル, 章ラベル→章題
  - _clean_chapter_title       : \\texorpdfstring 等の表示用整形を除く
"""

import glob
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SEMINAR_DIR = os.path.join(REPO_ROOT, "seminar", "cuturi", "tex")

# Named block environments and their markdown mappings.
# (env_name, container_class, heading_prefix)
BLOCK_ENVS = {
    "definition": ("definition", "Def"),
    "claim":      ("theorem",    "Clm"),
    "theorem":    ("theorem",    "Thm"),
    "proposition":("theorem",    "Prop"),
    "remark":     ("fact",       "Rem"),
    "example":    ("fact accent","Ex"),
    "algorithm":  ("definition", ""),
}

ENV_TO_PREFIX = {
    "definition": "def",
    "claim": "clm",
    "theorem": "thm",
    "proposition": "prop",
    "remark": "rem",
    "example": "ex",
}


def _all_chapter_tex():
    """本編(main/) と 付録(foundations/) の全 tex を出現順に返す。"""
    paths = []
    for sub in ("main", "foundations"):
        paths.extend(glob.glob(os.path.join(SEMINAR_DIR, sub, "*.tex")))
    return sorted(paths)


def strip_comments(line: str) -> str:
    """Remove TeX line comments (% ...), preserving escaped \\%."""
    result = []
    i = 0
    while i < len(line):
        if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
            break
        result.append(line[i])
        i += 1
    return "".join(result).rstrip()


def _extract_brace_arg(text: str, start: int):
    """Extract a brace-balanced {…} argument starting at *start*.
    Returns (content, end_index) or None if *start* is not '{'."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return None


def build_label_map():
    """Scan every TeX chapter file and build a map: 'prefix:label' -> title."""
    label_map = {}
    for tex_path in _all_chapter_tex():
        if not os.path.exists(tex_path):
            continue
        with open(tex_path, "r", encoding="utf-8") as f:
            content = f.read()
        for m in re.finditer(r"\\begin\{(\w+)\}", content):
            env_name = m.group(1)
            if env_name not in ENV_TO_PREFIX:
                continue
            pos = m.end()
            title_result = _extract_brace_arg(content, pos)
            if title_result is None:
                continue
            title, pos = title_result
            label_result = _extract_brace_arg(content, pos)
            if label_result is None:
                continue
            label, _ = label_result
            prefix = ENV_TO_PREFIX[env_name]
            label_map[f"{prefix}:{label}"] = title
    return label_map


def _clean_chapter_title(title: str) -> str:
    r"""\texorpdfstring{A}{B} -> A など、章タイトル中の表示用整形を除く。"""
    nested = r"(?:[^{}]|\{[^{}]*\})*"
    title = re.sub(rf"\\texorpdfstring\{{({nested})\}}\{{{nested}\}}", r"\1", title)
    return title.strip()


def build_chapter_map():
    r"""Scan every TeX chapter file for \chapter{TITLE}\label{ch:...} pairs."""
    chapter_map = {}
    for tex_path in _all_chapter_tex():
        with open(tex_path, "r", encoding="utf-8") as f:
            content = f.read()
        for m in re.finditer(r"\\chapter\{", content):
            res = _extract_brace_arg(content, m.end() - 1)
            if res is None:
                continue
            title, pos = res
            label_m = re.match(r"\s*\\label\{(ch:[^}]*)\}", content[pos:pos + 200])
            if not label_m:
                continue
            chapter_map[label_m.group(1)] = _clean_chapter_title(title)
    return chapter_map
