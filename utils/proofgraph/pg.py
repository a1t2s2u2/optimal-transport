#!/usr/bin/env python3
"""proofgraph の単一エントリポイント。

  cd proofgraph
  uv run pg.py            # 抽出 → 検証 → viewer サーバ起動（既定）
  uv run pg.py build      # 抽出 → 検証 のみ（CI 向け・サーバを起動しない）
  uv run pg.py extract    # tex → out/graph.json
  uv run pg.py validate   # graph.json の健全性検証（エラーで終了コード 1）
  uv run pg.py serve      # ローカルサーバを起動し viewer を案内（--port 8000）

リポジトリルートからは:
  uv run --project proofgraph python proofgraph/pg.py ...

スクリプトは __file__ 基準でパスを解決するので、どのディレクトリから実行してもよい。
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _run(script: str) -> int:
    """同じ Python（uv の venv）で proofgraph スクリプトを実行する。"""
    return subprocess.run([PY, os.path.join(HERE, script)]).returncode


def extract() -> int:
    return _run("extractor.py")


def validate() -> int:
    return _run("validate.py")


def build() -> int:
    rc = extract()
    if rc:
        return rc
    return validate()


def serve(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=HERE)
    httpd = http.server.ThreadingHTTPServer(("", port), handler)
    base = f"http://localhost:{port}/viewer/"
    print(f"viewer  : {base}")
    print(f"サンプル: {base}?data=sample.graph.json")
    print("停止は Ctrl+C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止しました。")
    finally:
        httpd.server_close()


def main() -> None:
    p = argparse.ArgumentParser(description="proofgraph runner")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("extract", help="tex → out/graph.json")
    sub.add_parser("validate", help="graph.json の健全性検証")
    sub.add_parser("build", help="抽出 → 検証（サーバを起動しない）")
    s = sub.add_parser("serve", help="viewer サーバを起動")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("up", help="抽出 → 検証 → サーバ起動（既定）")

    args = p.parse_args()
    cmd = args.cmd or "up"

    if cmd == "extract":
        sys.exit(extract())
    if cmd == "validate":
        sys.exit(validate())
    if cmd == "build":
        sys.exit(build())
    if cmd == "serve":
        serve(args.port)
        return
    if cmd == "up":
        rc = build()
        if rc:
            sys.exit(rc)
        print()
        serve(8000)


if __name__ == "__main__":
    main()
