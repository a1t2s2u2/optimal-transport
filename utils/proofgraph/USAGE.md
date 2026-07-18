# proofgraph 起動方法

数学ブロック依存グラフの抽出・検証・可視化の起動手順。依存は
[uv](https://docs.astral.sh/uv/) で管理し（`proofgraph/pyproject.toml`）、
すべて単一の `pg.py` から起動する。

## 前提

| 必要なもの | 確認 |
|---|---|
| uv（`pyyaml` を自動解決） | `uv --version` |
| ブラウザ（viewer 表示用） | — |

初回に依存を取得する（以降の `uv run` は自動同期するため必須ではない）:

```bash
uv sync --project proofgraph
```

仮想環境は `proofgraph/.venv`、バージョンは `uv.lock` で固定される。

---

## いちばん簡単な起動（推奨）

```bash
cd proofgraph
uv run pg.py
```

これだけで **抽出 → 検証 → ローカルサーバ起動** まで通す。ブラウザで
**http://localhost:8000/viewer/** を開く。停止は `Ctrl+C`。

> `pg.py` は `__file__` 基準でパスを解決するので、どのディレクトリから実行してもよい。
> リポジトリルートからは `uv run --project proofgraph python proofgraph/pg.py` とする。

## 個別サブコマンド

```bash
cd proofgraph
uv run pg.py extract     # tex → out/graph.json（出力例: nodes=153 edges=98 space-annotated=8）
uv run pg.py validate    # graph.json の健全性検証（エラーで終了コード 1・CI 用）
uv run pg.py build       # 抽出 → 検証 のみ（サーバを起動しない・CI 用）
uv run pg.py serve        # サーバのみ起動（--port で変更可）
```

- `out/` は `.gitignore` 済み。tex を編集したら `extract` を再実行する。
- viewer は `fetch` 制限のため **必ずサーバ経由**で開く（`file://` 直開きは不可）。

## viewer の操作

- **セマンティックズーム（詳細⇄概要）**: 既定では全ブロックのカードが画面を埋める**詳細表示**で始まる。
  地図のように**ズームアウトすると主要定理（Thm / Prop / Clm）だけの概要図**になり、定義・補題はたたまれ、
  結果どうしの依存だけが**スケルトン辺**で表示される（定義・補題は飛ばす）。ズームインで支える定義・補題が再び現れる。
- **文面カード**: 各ブロックは「種類＋タイトル＋文面冒頭」のカードで表示（枠色＝種類）。
  クリックすると右パネルに**文面全文を数式付き（MathJax）**で表示する。
- **フォーカス（依存を読む主役機能）**: 左の「フォーカス」を `直接` / `推移` にしてノードをクリックすると、
  その**依存先（上流＝青辺）と被依存（下流＝紫辺）だけ**に絞り込んで再配置する。
  「全体表示に戻す」または右パネルのボタンで解除。
- **空間で層別**: 選んだ空間で成り立つ結果のみ強調（強い空間を要する結果は淡色化）。
- **表示**: uses（実線）/ proof（破線）の表示切替。
- ルートの「辺を強調」「支持集合」ボタン: そのルートが支える全ブロックを緑強調。

## サンプルデータ（抽出不要のデモ）

viewer は `?data=` で読み込むグラフを差し替えられる。セミナー抽出結果がなくても、
同梱のサンプルで UI を試せる（左パネル「データ」で切替）:

- **解析学サンプル**（学部の解析学・約 40 ブロック）: `?data=sample.graph.json`
- **リーマン予想マップ**（RH をハブにした全体把握・約 28 ブロック）: `?data=riemann.graph.json`

独自の `graph.json` を `viewer/` 配下に置けば `?data=foo.json` で読める
（スキーマは `out/graph.json` と同じ: nodes / edges / routes / spaces / stats）。

## トラブルシュート

- **viewer が真っ白／データを読み込めない**: `file://` で開いていないか確認。`uv run pg.py` の
  サーバ経由で開く。実データを見るには先に `uv run pg.py extract` を実行しておく。
- **`ModuleNotFoundError: texparse`**: `extractor.py` は同ディレクトリの `texparse.py` を import する。
  `pg.py` 経由なら問題ない。
- **validate がエラー**: 循環は `\blockmeta{route.X=...}` で別証明ルートを明示して解消する
  （[README.md](README.md) の「AND/OR による循環解消」を参照）。
