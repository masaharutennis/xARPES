# venv仮想環境のコマンド

## 起動（アクティベート）
source venv/bin/activate

## 停止（デアクティベート）
deactivate

# ファイル変換コマンド

## .ipynb → .Rmd → .py に変換（venv環境で実行）
source venv/bin/activate && python examples/ipynb2Rmd2py.py

## .Rmd → .ipynb に変換（venv環境で実行）
source venv/bin/activate && python examples/Rmd2ipynb.py
