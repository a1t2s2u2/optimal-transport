#!/usr/bin/env python3
"""graph.json の構造的健全性を検証する（軽量な整合性チェック）。

検査項目:
  1. dangling ref   : 存在しないブロックを指す依存
  2. 循環 / 接地不能 : uses + AND/OR ルートで接地できないノード（循環の徴候）
  3. 空間前提充足   : 弱い空間の結果が、より強い空間を要する結果に依存していないか
  4. ルート健全性   : 証明を持つノードが「健全な（接地可能な）ルート」を最低 1 本持つか

エラー（dangling / 循環）があれば終了コード 1、警告のみなら 0 を返す（CI 用）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from model import DependencyGraph, Node, Route, SpaceLattice

DEFAULT_GRAPH = os.path.join(os.path.dirname(__file__), "out", "graph.json")


def load_graph(path: str) -> DependencyGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    routes_by_node: dict = {}
    for r in data.get("routes", []):
        routes_by_node.setdefault(r["node"], []).append(Route(name=r["route"], deps=list(r["deps"])))
    # uses は edges から復元（kind == 'uses'）
    uses_by_node: dict = {}
    for e in data.get("edges", []):
        if e.get("kind") == "uses":
            uses_by_node.setdefault(e["from"], []).append(e["to"])
    nodes = []
    for n in data["nodes"]:
        nodes.append(Node(
            id=n["id"], env=n["env"], title=n["title"],
            chapter=n.get("chapter", ""), section=n.get("section", ""),
            source_file=n.get("source_file", ""),
            spaces=n.get("spaces", []),
            has_proof=n.get("has_proof", False),
            uses=uses_by_node.get(n["id"], []),
            routes=routes_by_node.get(n["id"], []),
        ))
    lattice = SpaceLattice.load()
    return DependencyGraph(nodes, lattice)


def validate(graph: DependencyGraph) -> int:
    errors = 0
    warnings = 0

    # 1. dangling ref ------------------------------------------------------
    dangling = graph.dangling_refs()
    if dangling:
        errors += len(dangling)
        print(f"[ERROR] dangling refs ({len(dangling)}):")
        for nid, miss in dangling:
            print(f"  {nid} → {miss}（未定義）")

    # 2. 循環 / 接地不能 ---------------------------------------------------
    # 未定義依存を充足扱いにして接地不能を見れば、dangling 由来を除いた
    # “純粋な循環”だけが残る。
    cyclic = graph.ungrounded(ignore_missing=True)
    if cyclic:
        errors += len(cyclic)
        print(f"[ERROR] 接地不能（循環の疑い）({len(cyclic)}):")
        for nid in cyclic:
            print(f"  {nid}")

    # 3. 空間前提充足 ------------------------------------------------------
    violations = graph.space_violations()
    if violations:
        warnings += len(violations)
        print(f"[WARN] 空間前提の違反 ({len(violations)}): 弱い空間の結果が強い空間に依存")
        for nid, dep, dep_spaces in violations:
            node = graph.nodes[nid]
            print(f"  {nid} (space={node.spaces}) → {dep} (要 space={dep_spaces})")

    # 4. ルート健全性 ------------------------------------------------------
    grounded = graph.grounded_set()
    unsound = []
    for nid, node in graph.nodes.items():
        if node.is_axiomatic:
            continue
        if not any(all(d in grounded for d in r.deps) for r in node.routes):
            unsound.append(nid)
    # 既に cyclic で報告したものは除く
    unsound = [n for n in unsound if n not in set(cyclic)]
    if unsound:
        warnings += len(unsound)
        print(f"[WARN] 健全な証明ルートを持たないノード ({len(unsound)}):")
        for nid in unsound:
            print(f"  {nid}")

    # サマリ ---------------------------------------------------------------
    n = len(graph.nodes)
    print("\n--- summary ---")
    print(f"nodes={n}  grounded={len(grounded)}  "
          f"errors={errors}  warnings={warnings}")
    annotated = sum(1 for x in graph.nodes.values() if x.spaces)
    routed = sum(1 for x in graph.nodes.values()
                 if any(r.name != '_auto' for r in x.routes))
    print(f"space-annotated={annotated}  explicit-routes={routed}")
    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", nargs="?", default=DEFAULT_GRAPH)
    args = ap.parse_args()
    graph = load_graph(args.graph)
    sys.exit(validate(graph))


if __name__ == "__main__":
    main()
