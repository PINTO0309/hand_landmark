# hand_landmark

https://github.com/PINTO0309/PINTO_model_zoo

- onnx

  https://github.com/PINTO0309/hand_landmark/releases

- demo

  https://github.com/PINTO0309/hand_landmark/assets/33194443/9e4e188b-5c44-46fc-8328-21ae8a122971


```
z:
  - 手首のキーポイントの位置を0.0とした奥行き方向の数値 (ほぼ0.0だが0.0よりほんの少し大きい値になる)
  - カメラ方向がマイナス、奥方向がプラス
  - 正規化されていない
  - 実測してみた結果 -80.0 〜 +80.0 の範囲で推移する (なお、ほぼ水平の手は検出不可能)
  - カメラからの距離や対象の手のひらのサイズ感には影響を受けない (224x224の固定サイズに伸縮して推論しているため)

  output_image_20231114_173009.png 手首z[0]: 0.0001862049102783203, 中指先端z[12]: 45.90625
  output_image_20231114_173017.png 手首z[0]: 0.00020122528076171875, 中指先端z[12]: 61.59375
  output_image_20231114_173022.png 手首z[0]: 0.0001690387725830078, 中指先端z[12]: 1.0263671875
  output_image_20231114_173026.png 手首z[0]: 0.00021648406982421875, 中指先端z[12]: -73.5625
  output_image_20231114_173031.png 手首z[0]: 0.0002181529998779297, 中指先端z[12]: -65.8125
  output_image_20231114_174046.png 手首z[0]: 0.0002727508544921875, 中指先端z[12]: -79.875
  output_image_20231114_174311.png 手首z[0]: 0.00013566017150878906, 中指先端z[12]: 63.75
```
