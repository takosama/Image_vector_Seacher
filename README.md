# Image_vector_Seacher

clipを使用して近傍画像を探索します
操作　マウススクロール拡大縮小　クリック　近傍画像探索（フォルダへ格納）

![F2fQaR5bQAMyHlM](https://github.com/takosama/Image_vector_Seacher/assets/16166677/5b24b8bd-a733-4f42-9b2f-e73fcf1faa08)
![F2fPw8iaAAIbClW](https://github.com/takosama/Image_vector_Seacher/assets/16166677/8af0b2f0-67fd-4256-a808-fbbe483c4f07)

## 安全な保存形式への移行

- 検索結果は毎回 `similar_images/search_<unique>/` に保存します。以前の検索結果・手作業のファイルは削除しません。不要になった結果は自分で整理してください。
- データセットは `dataset.npz` に変更しました。Unicodeファイル名配列と数値ベクトルだけを保存し、`allow_pickle=False` で読み込みます。ファイル名、形状、有限値、展開サイズを検証します。
- 旧 `dataset.pkl` は自動読み込み・自動変換しません。元のPNGを `img/` に用意し、既存のCLIP依存をセットアップした環境で `python vectorize.py` を実行して作り直してください。その後 `python view.py` で表示できます。ベクトル化は外部モデルを取得する場合があります。
- `view.py` / `vectorize.py` のimportだけではUI表示、ファイル読込み、モデル取得は行いません。
- 小型テスト：numpy、Pillow、matplotlib、scikit-learn、pytestを用意して `python -m pytest -q`。生成fixtureだけで往復、危険型/パス拒否、既存ファイル保全、書込失敗、2回連続のGUIクリックを検証します。CLIPモデルのダウンロードは行いません。
- 未確認：実CLIPモデルによる埋め込み生成、Windowsのフォルダ表示、実利用データの検索品質。
