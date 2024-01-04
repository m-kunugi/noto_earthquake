# 説明
国土地理院が公開している学習済みCNNモデルを使用して、建物被害分布を出してみた。

Qiitaの説明記事
https://qiita.com/mkunu/items/b7b33deff174c7127ab4

## 追加ファイル
国土地理院が公開している学習済みモデルは以下から取得して、このREADME.mdと同じ階層に置く。  
https://gisstar.gsi.go.jp/gsi-dataset/99/model-09.zip

## 参考文献
- 令和6年(2024年)能登半島地震に関する情報（国土地理院）
https://www.gsi.go.jp/BOUSAI/20240101_noto_earthquake.html

- 指定した領域のラスタータイルをダウンロードしたりGeoTIFFに変換してくれるPythonパッケージをPyPIで公開しました！（MIERUNE）
https://qiita.com/nokonoko_1203/items/5c32c22b92bb72e7770c

- 地物抽出用推論プログラムと学習済モデル（国土地理院）
https://gisstar.gsi.go.jp/gsi-dataset/99/index.html


# 作成例
左図が災害前、右図が災害後の抽出結果。各図の上側は大規模火災があった輪島市「朝市通り」の周辺である。災害後、建物として検出されない領域が広範囲に存在することが分かる。
![before_after](https://github.com/m-kunugi/noto_earthquake/assets/38278310/c8a61ae5-7d12-48d7-bd91-d437e09c7812)
