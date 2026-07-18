#!/usr/bin/env python3
"""proofgraph のデータモデル: 空間ラティスと AND/OR 依存グラフ。

- SpaceLattice: spaces.yaml を読み、refines の推移閉包で空間の強弱を判定する。
- Node / Route: 1 つの数学ブロック（Def/Prop/Thm/...）とその証明ルート。
- DependencyGraph: ノード集合と、uses（statement 依存）/ proof ルート（AND/OR）依存。

依存の意味論:
  * uses 依存は「常に必要」（AND）。
  * 各ノードは 0 個以上の証明ルートを持ち、ルート内の依存は連言（AND）、
    複数ルートは選言（OR, ＝同値な別証明）。
  * ノードが「接地可能(grounded)」とは、全 uses 依存が接地可能で、かつ
    ルートが無い（公理・定義）か、少なくとも 1 本のルートの全依存が接地可能なこと。
    接地不能なノードは循環または未定義依存に巻き込まれている。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

SPACES_PATH = os.path.join(os.path.dirname(__file__), "spaces.yaml")


class SpaceLattice:
    """空間タグの強弱半順序。refines の推移閉包を保持する。"""

    def __init__(self, spec: dict):
        self.spec = spec
        self.labels = {k: (v or {}).get("label", k) for k, v in spec.items()}
        # 直接の refines 辺
        self._refines = {k: list((v or {}).get("refines", []) or []) for k, v in spec.items()}
        self._closure = {k: self._compute_closure(k) for k in spec}

    @classmethod
    def load(cls, path: str = SPACES_PATH) -> "SpaceLattice":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data["spaces"])

    def _compute_closure(self, space: str) -> set:
        """space が refines するすべての空間（推移的, 自身を含む）。

        返り値は「space が備える構造の集合」。例: closure('normed') ⊇
        {normed, vector, metric, topological, set}。
        """
        seen = set()
        stack = [space]
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            for parent in self._refines.get(s, []):
                if parent not in seen:
                    stack.append(parent)
        return seen

    def known(self, space: str) -> bool:
        return space in self.spec

    def provides(self, tags) -> set:
        """タグ集合が提供する全構造（各タグの閉包の和）。"""
        out: set = set()
        for t in tags:
            out |= self._closure.get(t, {t})
        return out

    def satisfies(self, node_tags, dep_tags) -> bool:
        """node_tags の空間で dep（dep_tags を要する結果）が利用可能か。

        dep の要求する各構造が node の提供構造に含まれれば True。
        含まれない＝「より弱い空間の結果が、より強い空間を要する結果に依存」。
        """
        provided = self.provides(node_tags)
        return all(t in provided for t in dep_tags)

    def lattice_json(self) -> dict:
        """viewer 用に語彙とラベル・refines を返す。"""
        return {
            k: {
                "label": self.labels[k],
                "refines": self._refines.get(k, []),
                "closure": sorted(self._closure[k]),
            }
            for k in self.spec
        }


@dataclass
class Route:
    """証明ルート（連言依存の集合）。name は 'A','B',... または '_auto'。"""

    name: str
    deps: list = field(default_factory=list)


@dataclass
class Node:
    """1 つの数学ブロック。"""

    id: str                      # 例 'thm:sem-kantorovich-duality'
    env: str                     # definition / theorem / proposition / ...
    title: str
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    source_file: str = ""
    spaces: list = field(default_factory=list)   # 空間タグ
    uses: list = field(default_factory=list)     # statement 依存（AND）
    routes: list = field(default_factory=list)   # 証明ルート（OR of AND）: list[Route]
    statement: str = ""          # ブロック本文（statement テキスト、TeX のまま）
    group: str = ""              # 概要グループ（手動の「まとまり」。未指定なら章で代用）
    has_proof: bool = False

    @property
    def is_axiomatic(self) -> bool:
        """定義など、証明ルートを持たないブロック。"""
        return len(self.routes) == 0


class DependencyGraph:
    """ノード集合と依存。AND/OR 接地判定・循環検出を提供する。"""

    def __init__(self, nodes, lattice: SpaceLattice | None = None):
        self.nodes: dict = {n.id: n for n in nodes}
        self.lattice = lattice

    # --- 依存の列挙 -------------------------------------------------------
    def all_dep_ids(self, node: Node) -> set:
        ids = set(node.uses)
        for r in node.routes:
            ids.update(r.deps)
        return ids

    def dangling_refs(self) -> list:
        """存在しない id を指す (node_id, missing_id) の一覧。"""
        out = []
        for nid, node in self.nodes.items():
            for dep in sorted(self.all_dep_ids(node)):
                if dep not in self.nodes:
                    out.append((nid, dep))
        return out

    # --- AND/OR 接地判定（循環検出を兼ねる） ------------------------------
    def grounded_set(self, ignore_missing: bool = False) -> set:
        """接地可能なノード id の集合を fixpoint で計算する。

        ignore_missing=False: 存在しない依存は接地不能として扱う。
        ignore_missing=True : 未定義依存を「充足済み」とみなす。これにより
            dangling 由来の接地不能を除いた“純粋な循環”だけを取り出せる。
        """
        grounded: set = set()
        changed = True
        while changed:
            changed = False
            for nid, node in self.nodes.items():
                if nid in grounded:
                    continue
                if self._is_grounded(node, grounded, ignore_missing):
                    grounded.add(nid)
                    changed = True
        return grounded

    def _dep_ok(self, dep: str, grounded: set, ignore_missing: bool) -> bool:
        if dep in grounded:
            return True
        return ignore_missing and dep not in self.nodes

    def _is_grounded(self, node: Node, grounded: set, ignore_missing: bool = False) -> bool:
        # uses 依存はすべて接地済み（or 無視可能な未定義）でなければならない
        if not all(self._dep_ok(u, grounded, ignore_missing) for u in node.uses):
            return False
        if node.is_axiomatic:
            return True
        # 少なくとも 1 本のルートの全依存が接地済み
        return any(all(self._dep_ok(d, grounded, ignore_missing) for d in r.deps)
                   for r in node.routes)

    def ungrounded(self, ignore_missing: bool = False) -> list:
        """接地不能なノード id。ignore_missing=True なら循環のみが残る。"""
        grounded = self.grounded_set(ignore_missing=ignore_missing)
        return sorted(nid for nid in self.nodes if nid not in grounded)

    # --- 支持集合（到達可能な依存の推移閉包） -----------------------------
    def support_set(self, node_id: str, route: str | None = None) -> set:
        """node を選んだルートで支える全ブロックの推移閉包を返す（node 自身は除く）。

        各ノードについて uses は常に辿り、証明ルートは:
          - 起点 node では指定 route（None なら最初のルート）を辿る
          - 以降のノードでは接地可能な最初のルートを辿る（最小支持の近似）
        未定義依存は無視する。
        """
        start = self.nodes.get(node_id)
        if start is None:
            return set()
        grounded = self.grounded_set()
        support: set = set()
        stack = list(self._deps_for(start, route, grounded))
        while stack:
            nid = stack.pop()
            if nid in support or nid not in self.nodes:
                continue
            support.add(nid)
            stack.extend(self._deps_for(self.nodes[nid], None, grounded))
        support.discard(node_id)
        return support

    def _deps_for(self, node: Node, route: str | None, grounded: set) -> set:
        deps = set(node.uses)
        if node.routes:
            chosen = None
            if route is not None:
                chosen = next((r for r in node.routes if r.name == route), None)
            if chosen is None:
                # 接地可能な最初のルート、無ければ最初のルート
                chosen = next(
                    (r for r in node.routes
                     if all(d in grounded for d in r.deps)),
                    node.routes[0],
                )
            deps.update(chosen.deps)
        return deps

    # --- 空間前提充足 -----------------------------------------------------
    def space_violations(self) -> list:
        """(node_id, dep_id, dep_tags) — 弱い空間の結果が強い空間に依存する違反。

        node と dep の双方に空間タグがある場合のみ検査する。
        """
        if self.lattice is None:
            return []
        out = []
        for nid, node in self.nodes.items():
            if not node.spaces:
                continue
            for dep in sorted(self.all_dep_ids(node)):
                d = self.nodes.get(dep)
                if d is None or not d.spaces:
                    continue
                if not self.lattice.satisfies(node.spaces, d.spaces):
                    out.append((nid, dep, list(d.spaces)))
        return out
