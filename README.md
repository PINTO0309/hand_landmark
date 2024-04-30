# hand_landmark

HandLandmark Detection that can be performed only in onnxruntime. Pre-focusing by skeletal detection is not performed. This does not use MediaPipe.

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

- 中指先端z[12]: 45.90625

  ![landmark_screenshot_14 11 2023_1](https://github.com/PINTO0309/hand_landmark/assets/33194443/ff15aa75-fcb2-4cdd-943c-47bcce259020)

- 中指先端z[12]: 61.59375

  ![landmark_screenshot_14 11 2023_2](https://github.com/PINTO0309/hand_landmark/assets/33194443/c6ab9054-d94b-40dc-96f0-5b270c0a63f9)

- 中指先端z[12]: 1.0263671875

  ![landmark_screenshot_14 11 2023_3](https://github.com/PINTO0309/hand_landmark/assets/33194443/bd555156-012c-4fb8-b63e-c8bef328f86f)

- 中指先端z[12]: -73.5625

  ![landmark_screenshot_14 11 2023_4](https://github.com/PINTO0309/hand_landmark/assets/33194443/0647be4a-7977-46e5-b8e6-0f8340655669)

- 中指先端z[12]: -65.8125

  ![landmark_screenshot_14 11 2023_5](https://github.com/PINTO0309/hand_landmark/assets/33194443/7bea94f4-97db-4cb1-adbf-32d8a2c12ee9)

- 中指先端z[12]: -79.875

  ![landmark_screenshot_14 11 2023_6](https://github.com/PINTO0309/hand_landmark/assets/33194443/31a37782-d66c-4fba-b7c8-da5f015bf7c8)

- 中指先端z[12]: 63.75

  ![landmark_screenshot_14 11 2023_7](https://github.com/PINTO0309/hand_landmark/assets/33194443/5f6c3ab4-cafb-46ed-ab86-3a31e32d5024)

![image](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/2c44534a-beb5-40d9-9c11-f97251f395ed)

```
0. wrist
1. thumb_cmc
2. thumb_mcp
3. thumb_ip
4. thumb_tip
5. index_finger_mcp
6. index_finger_pip
7. index_finger_dip
8. index_finger_tip
9. middle_finger_mcp
10. middle_finger_pip
11. middle_finger_dip
12. middle_finger_tip
13. ring_finger_mcp
14. ring_finger_pip
15. ring_finger_dip
16. ring_finger_tip
17. pinky_mcp
18. pinky_pip
19. pinky_dip
20. pinky_tip
```
