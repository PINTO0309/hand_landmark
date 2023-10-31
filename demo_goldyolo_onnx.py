#!/usr/bin/env python

import cv2
import copy
import math
import time
import numpy as np
import onnxruntime
from argparse import ArgumentParser
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass(frozen=False)
class PaddingsAndScale():
    padding_height: int # 224x224へリサイズするときに足された合計パディング幅（奇数になることがある）
    padding_width: int # 224x224へリサイズするときに足された合計パディング高（奇数になることがある）
    scale_height: float # 224x224へリサイズするときの元画像の幅スケール比率
    scale_width: float # 224x224へリサイズするときの元画像の高スケール比率

@dataclass(frozen=False)
class Point3D():
    x: float # 横
    y: float # 縦
    z: float # 前後
    depth: int # 距離

@dataclass(frozen=False)
class Palm():
    cx: float # 手のひら中心X, 元画像上でのスケール値を保持
    cy: float # 手のひら中心Y, 元画像上でのスケール値を保持
    width: float # 手のひら幅, 元画像上でのスケール値を保持
    height: float # 手のひら高さ, 元画像上でのスケール値を保持
    degree: float # 手のひら回転角(degree)

@dataclass(frozen=False)
class Hand():
    cx: float # 中心X
    cy: float # 中心Y
    x1: float # xmin
    y1: float # ymin
    x2: float # xmax
    y2: float # ymax
    vector_x: float # 回転ベクトルX
    vector_y: float # 回転ベクトルY
    vector_z: float # 回転ベクトルZ
    quaternion_x: float # クォータニオンX
    quaternion_y: float # クォータニオンY
    quaternion_z: float # クォータニオンZ
    quaternion_w: float # クォータニオンW
    keypoints: List[Point3D] # キーポイントN点
    palm: Palm # 手のひら

class GoldYOLOONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'gold_yolo_n_hand_post_0333_0.4040_1x3x512x896.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """GoldYOLOONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for GoldYOLO

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]

    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GoldYOLOONNX

        Parameters
        ----------
        image: np.ndarray
            Entire image (RGB)

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self.__preprocess(
            temp_image,
        )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {input_name: inferece_image for input_name in self.input_names},
        )[0]

        # PostProcess
        result_boxes, result_scores = \
            self.__postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes, result_scores

    def __preprocess(
        self,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)\n
            CHW to HWC: (1,2,0)\n
            HWC to HWC: (0,1,2)\n
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Normalization + BGR->RGB
        resized_image = cv2.resize(
            image,
            (
                int(self.input_shapes[0][3]),
                int(self.input_shapes[0][2]),
            )
        )
        resized_image = np.divide(resized_image, 255.0)
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(
            resized_image,
            dtype=np.float32,
        )
        return resized_image

    def __postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: np.ndarray
            Predicted boxes: [N, y1, x1, y2, x2]

        result_scores: np.ndarray
            Predicted box confs: [N, score]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_y1x1y2x2_score: float32[N,7]
        """
        result_boxes = []
        result_scores = []
        if len(boxes) > 0:
            scores = boxes[:, 6:7]
            keep_idxs = scores[:, 0] > self.class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    cx = (box[2] + box[4]) // 2
                    cy = (box[3] + box[5]) // 2
                    w = abs(box[2] - box[4]) * 1.2
                    h = abs(box[3] - box[5]) * 1.2
                    x_min = int(max(cx - w // 2, 0) * image_width / self.input_shapes[0][3])
                    y_min = int(max(cy - h // 2, 0) * image_height / self.input_shapes[0][2])
                    x_max = int(min(cx + w // 2, self.input_shapes[0][3]) * image_width / self.input_shapes[0][3])
                    y_max = int(min(cy + h // 2, self.input_shapes[0][2]) * image_height / self.input_shapes[0][2])
                    result_boxes.append([x_min, y_min, x_max, y_max])
                    result_scores.append(score)

        return np.asarray(result_boxes), np.asarray(result_scores)


class PalmDetectionONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'palm_detection_full_Nx3x192x192_post.onnx',
        class_score_th: Optional[float] = 0.20,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """PalmDetectionONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for PalmDetection

        class_score_th: Optional[float]
            Score threshold. Default: 0.30

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]

    def __call__(
        self,
        image: np.ndarray,
        hand_infos: List[Hand],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PalmDetectionONNX

        Parameters
        ----------
        image: np.ndarray
            全体画像

        hand_infos: List[Hand]
            手の位置情報のリスト

        Returns
        -------
        hand_infos: List[Hand]
            手の回転角セット済みの手の位置情報リスト
        """
        if len(hand_infos) == 0:
            return hand_infos

        temp_image = copy.deepcopy(image)
        temp_hand_infos = copy.deepcopy(hand_infos)

        # PreProcess
        hand_images, hand_192x192_images, normalized_hand_192x192_images = \
            self.__preprocess(
                image=temp_image,
                hand_infos=temp_hand_infos,
            )

        # Inference
        batch_nums, score_cx_cy_w_wristcenterxy_middlefingerxys = \
            self.onnx_session.run(
                self.output_names,
                {input_name: normalized_hand_192x192_images for input_name in self.input_names},
            )

        # PostProcess
        # 手のひらの回転角を算出して手情報にセットする
        hand_infos = \
            self.__postprocess(
                hand_192x192_images=hand_192x192_images,
                hand_infos=temp_hand_infos,
                batch_nums=batch_nums,
                score_cx_cy_w_wristcenterxy_middlefingerxys=score_cx_cy_w_wristcenterxy_middlefingerxys,
            )

        # 元画像スケールの手のひら位置情報のリスト, (デバッグ用途) 回転角をゼロ度に調整した手のひら画像のリスト [N, H, W, 3]
        return hand_infos

    def __preprocess(
        self,
        *,
        image: np.ndarray,
        hand_infos: List[Hand],
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            全体画像

        hand_infos: List[Hand]
            手の位置情報のリスト

        swap: tuple
            HWC to CHW: (2,0,1)\n
            CHW to HWC: (1,2,0)\n
            HWC to HWC: (0,1,2)\n
            CHW to CHW: (0,1,2)

        Returns
        -------
        hand_images: List[np.ndarray]
            元画像からクロップした手の画像のリスト\n
            正方形ではない\n
            uint8 [N,H,W,3]

        hand_192x192_images: np.ndarray
            アスペクトレシオを維持したまま192x192へ収まるようにリサイズしたうえで192x192へパディングした手の画像\n
            uint8 [N,192,192,3]

        normalized_hand_images: np.ndarray
            正規化済み、なおかつ アスペクトレシオを維持したまま192x192へ収まるようにリサイズしたうえで192x192へパディングした手の画像\n
            float32 [N,3,192,192]
        """
        hand_images: List[np.ndarray] = []
        hand_192x192_images: List[np.ndarray] = []
        normalized_hand_192x192_images: List[np.ndarray] = []
        image_height = image.shape[0]
        image_width = image.shape[1]

        for hand_info in hand_infos:
            # 手の検出情報を元にして全体画像から手の画像をクロッピング
            hand_image = \
                image[
                    int(hand_info.y1*image_height):int(hand_info.y2*image_height),
                    int(hand_info.x1*image_width):int(hand_info.x2*image_width),
                    :
                ]

            # 処理対象：クロップした手画像 (元画像スケール)
            # 非ノーマライゼーション
            hand_images.append(hand_image)

            # PalmDetectionモデルの入力サイズ 192x192 へアスペクトレシオを維持しながらリサイズして正方形にパディング
            padded_hand_image, resized_hand_image = \
                keep_aspect_resize_and_pad(
                    image=hand_image,
                    resize_width=int(self.input_shapes[0][3]),
                    resize_height=int(self.input_shapes[0][2]),
                )

            # 処理対象：アスペクトレシオを維持しながら192x192にリサイズとパディングを施した手画像
            # 非ノーマライゼーション
            unnormalized_hand_image = copy.deepcopy(padded_hand_image)
            # [N,192,192,3]
            hand_192x192_images.append(unnormalized_hand_image)

            # 処理対象：アスペクトレシオを維持しながら192x192にリサイズとパディングを施した手画像
            # ノーマライゼーション
            normalized_hand_image = np.divide(padded_hand_image, 255.0)
            normalized_hand_image = normalized_hand_image.transpose(swap)
            normalized_hand_image = \
                np.ascontiguousarray(
                    normalized_hand_image,
                    dtype=np.float32,
                )
            # [N,3,192,192]
            normalized_hand_192x192_images.append(normalized_hand_image)

        return \
            hand_images, \
            np.asarray(hand_192x192_images, dtype=np.uint8), \
            np.asarray(normalized_hand_192x192_images, dtype=np.float32)

    def __postprocess(
        self,
        *,
        hand_192x192_images: List[np.ndarray],
        hand_infos: List[Hand],
        batch_nums: np.ndarray,
        score_cx_cy_w_wristcenterxy_middlefingerxys: np.ndarray,
    ) -> Tuple[List[Hand], np.ndarray]:
        """__postprocess
        手のひらの回転角を算出して手情報にセットする

        Parameters
        ----------
        hand_192x192_images: np.ndarray
            アスペクトレシオを維持したまま192x192へ収まるようにリサイズしたうえで192x192へパディングした手の画像\n
            uint8 [N,192,192,3]

        hand_infos: List[Hand]
            元画像スケールでの手の位置情報のリスト\n
            すべてスケール値で値を保持している (ピクセル座標ではない)

        batch_nums: np.ndarray
            int64 [N, 1]

        score_cx_cy_w_wristcenterxy_middlefingerxys: np.ndarray
            float32 [N, 8]

        Returns
        -------
        hand_infos: List[Hand]
            クロップ済みで回転前の元画像スケールでの手のひら位置情報を計算したPalmをセットしたHandのリスト\n
            すべてスケール値で保持\n
            [cx, cy, width, height, degree]
        """
        keep = score_cx_cy_w_wristcenterxy_middlefingerxys[:, 0] > self.class_score_th
        score_cx_cy_w_wristcenterxy_middlefingerxys = score_cx_cy_w_wristcenterxy_middlefingerxys[keep, :]
        batch_nums = batch_nums[keep, :]

        for hand_192x192_image, hand_info, score_cx_cy_w_wristcenterxy_middlefingerxy \
            in zip(hand_192x192_images, hand_infos, score_cx_cy_w_wristcenterxy_middlefingerxys):

            image_height = hand_192x192_image.shape[0]
            image_width = hand_192x192_image.shape[1]
            score: float = score_cx_cy_w_wristcenterxy_middlefingerxy[0] # スコア
            wh = max(image_width, image_height)
            w: int = int(score_cx_cy_w_wristcenterxy_middlefingerxy[3] * wh) # 192x192画像内を基準とした手のひらの幅, ピクセル座標
            wrist_cx: int = int(score_cx_cy_w_wristcenterxy_middlefingerxy[4] * image_width) # 192x192画像内を基準とした手首座標X, ピクセル座標
            wrist_cy: int = int(score_cx_cy_w_wristcenterxy_middlefingerxy[5] * image_height) # 192x192画像内を基準とした手首座標Y, ピクセル座標
            middlefinger_x: int = int(score_cx_cy_w_wristcenterxy_middlefingerxy[6] * image_width) # 192x192画像内を基準とした中指座標X, ピクセル座標
            middlefinger_y: int = int(score_cx_cy_w_wristcenterxy_middlefingerxy[7] * image_height) # 192x192画像内を基準とした中指座標Y, ピクセル座標

            # ここ以下の処理はスケール値をピクセル座標に変換してから処理する
            if w > 0:
                # HandLandmark Detection 用の正方形画像の生成
                kp02_x = middlefinger_x - wrist_cx
                kp02_y = middlefinger_y - wrist_cy
                rotation = 0.5 * math.pi - math.atan2(-kp02_y, kp02_x) # radians
                rotation = normalize_radians(angle=rotation) # normalized radians
                degree = np.rad2deg(rotation) # radians to degree

                # クロップ済みで回転前の元画像スケールでの手のひら位置情報を保存する
                # palm_info = np.asarray([cx, cy, width, height, degree], dtype=np.float32)
                # cx, cy, width, height はスケール値
                palm = \
                    Palm(
                        cx=hand_info.cx,
                        cy=hand_info.cy,
                        width=abs(hand_info.x2 - hand_info.x1) / 2,
                        height=abs(hand_info.y2 - hand_info.y1) / 2,
                        degree=degree,
                    )
                hand_info.palm = palm

            else:
                # 検出した手のひらのサイズが正常ではない場合はからの手のひら情報を生成する
                palm = \
                    Palm(
                        cx=None,
                        cy=None,
                        width=None,
                        height=None,
                        degree=None,
                    )
                hand_info.palm = palm

        # 元画像スケールの手のひら位置情報, (デバッグ用途)クロップして回転角ゼロ度に調整した手のひら画像のリスト
        return hand_infos


class HandLandmarkDetectionONNX(object):
    def __init__(
        self,
        model_path: Optional[str] = 'hand_landmark_sparse_Nx3x224x224.onnx',
        class_score_th: Optional[float] = 0.20,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """HandLandmarkDetectionONNX

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for HandLandmark Detection

        class_score_th: Optional[float]
            Score threshold. Default: 0.20

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]

    def __call__(
        self,
        image: np.ndarray,
        hand_infos: List[Hand],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """HandLandmarkDetectionONNX

        Parameters
        ----------
        image: np.ndarray
            全体画像

        hand_infos: List[Hand]
            手の位置情報のリスト

        Returns
        -------
        hand_infos: List[Hand]
            手のキーポイントセット済みの手のひら情報リスト
        """
        if len(hand_infos) == 0:
            return hand_infos

        temp_image = copy.deepcopy(image)
        temp_hand_infos = copy.deepcopy(hand_infos)

        # PreProcess
        hand_images, hand_224x224_images, normalized_hand_224x224_images, paddings_and_scales, rec_size_difference_rotation_angles = \
            self.__preprocess(
                image=temp_image,
                hand_infos=temp_hand_infos,
            )

        # 処理対象の手のイメージが１件以上あるときのみ処理する
        if len(normalized_hand_224x224_images) > 0:
            # Inference
            #   xyz_x21s: float32 [hands, 63], xyz*21
            #   hand_scores: float32 [hands, 1]
            #   lefthand_0_or_righthand_1s: float32 [hands, 1]
            xyz_x21s, hand_scores, lefthand_0_or_righthand_1s = \
                self.onnx_session.run(
                    self.output_names,
                    {input_name: normalized_hand_224x224_images for input_name in self.input_names},
                )

            # PostProcess
            hand_infos = \
                self.__postprocess(
                    image=temp_image,
                    hand_224x224_images=hand_224x224_images,
                    hand_infos=temp_hand_infos,
                    xyz_x21s=xyz_x21s,
                    hand_scores=hand_scores,
                    lefthand_0_or_righthand_1s=lefthand_0_or_righthand_1s,
                    paddings_and_scales=paddings_and_scales,
                    rec_size_difference_rotation_angles=rec_size_difference_rotation_angles,
                )

        # 元画像スケールの手のひら位置情報のリスト
        return hand_infos

    def __preprocess(
        self,
        *,
        image: np.ndarray,
        hand_infos: List[Hand],
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[PaddingsAndScale], List[List[int]]]:
        """__preprocess

        Parameters
        ----------
        image: np.ndarray
            全体画像

        hand_infos: List[Hand]
            手の位置情報のリスト

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        hand_images: List[np.ndarray]
            元画像からクロップした手の画像のリスト, 正方形ではない uint8 [N,H,W,3]

        hand_224x224_images: np.ndarray
            アスペクトレシオを維持したまま224x224へ収まるようにリサイズしたうえで224x224へパディングした手の画像 uint8 [N,224,224,3]

        normalized_hand_images: np.ndarray
            正規化済み、なおかつ、アスペクトレシオを維持したまま224x224へ収まるようにリサイズしたうえで224x224へパディングした手の画像 float32 [N,3,224,224]

        paddings_and_scales: List[PaddingsAndScale]
            パディングピクセル数と手画像のスケールレシオ ([H,W,3]から[224,224,3]へ変換したときの変換倍率)
            [[padding_height, padding_width, scale_height, scale_width],[...],[...]]

        rec_size_difference_rotation_angles: List[List[int]]
            矩形クロップをするときに生じた幅と高さのサイズの差分。左右合計差分サイズと上下合計差分サイズ。\n
            最終的に元の座標系に戻すときには各値を２分の１にして減算して使用する。
            [[w,h], [w,h], ...] -> [[6,3], [10,15], ...]
        """
        hand_images: List[np.ndarray] = []
        hand_224x224_images: List[np.ndarray] = []
        normalized_hand_224x224_images: List[np.ndarray] = []
        paddings_and_scales: List[PaddingsAndScale] = []
        rec_size_differences: List[List[int]] = []

        image_height = image.shape[0]
        image_width = image.shape[1]

        for hand_info in hand_infos:
            # 手のひらが未検出の場合は処理をスキップする
            if hand_info == []:
                continue
            if hand_info.palm is None:
                continue

            # 手の検出情報を元にして全体画像から手の画像をクロッピング
            # 非回転の手の画像
            hand_image = \
                image[
                    int(hand_info.y1*image_height):int(hand_info.y2*image_height),
                    int(hand_info.x1*image_width):int(hand_info.x2*image_width),
                    :
                ]

            # 全体画像から１件分の回転角ゼロ度の手の画像をクロップして取得
            rotated_hand_image: List[np.ndarray] = [] # 回転角をゼロ度に調整してクロップした手の画像
            rotated_hand_image, rec_size_difference_rotation_angles = \
                rotate_and_crop_rectangle(
                    image=image,
                    rects=np.asarray(
                        [[
                            hand_info.cx * image_width,
                            hand_info.cy * image_height,
                            abs(hand_info.x2 - hand_info.x1) * image_width,
                            abs(hand_info.y2 - hand_info.y1) * image_height,
                            hand_info.palm.degree,
                        ]]
                    ),
                    update_rects=True,
                )

            if len(rec_size_difference_rotation_angles) == 0:
                continue

            hand_image = rotated_hand_image[0]
            rec_size_difference = rec_size_difference_rotation_angles[0]

            # 処理対象：クロップした手画像 (元画像スケール)
            # 非ノーマライゼーション
            hand_images.append(hand_image)
            rec_size_differences.append(rec_size_difference)

            # HandLandmark Detectionモデルの入力サイズ 224x224 へアスペクトレシオを維持しながらリサイズして正方形にパディング
            #   padded_hand_image: アスペクトレシオを維持しながら224x224へリサイズしたうえで224x224へパディングした画像
            #   resized_hand_image: アスペクトレシオを維持しながら224x224へリサイズしたうえで224x224へパディングしていない画像（元画像からクロップした手画像からのスケールレシオ算出用）
            padded_hand_image, resized_hand_image = \
                keep_aspect_resize_and_pad(
                    image=hand_image,
                    resize_width=int(self.input_shapes[0][3]),
                    resize_height=int(self.input_shapes[0][2]),
                )

            # 処理対象：アスペクトレシオを維持しながら224x224にリサイズとパディングを施した手画像
            # 非ノーマライゼーション
            unnormalized_hand_image = copy.deepcopy(padded_hand_image)
            # [N,224,224,3]
            hand_224x224_images.append(unnormalized_hand_image)

            # 処理対象：アスペクトレシオを維持しながら224x224にリサイズとパディングを施した手画像
            # ノーマライゼーション
            normalized_hand_image = np.divide(padded_hand_image, 255.0)
            normalized_hand_image = normalized_hand_image.transpose(swap)
            normalized_hand_image = \
                np.ascontiguousarray(
                    normalized_hand_image,
                    dtype=np.float32,
                )
            # [N,3,224,224]
            normalized_hand_224x224_images.append(normalized_hand_image)

            # 元画像[H,W,3]から224x224画像[224,224,3]に変換されたあとのパディング部分を除いた手のひら画像部分のスケールレシオと224x224に加工したときに付与したパディングサイズを計算する
            #   padding_width: 左右の合計パディングピクセル数
            #   padding_height: 上下の合計パディングピクセル数
            # paddings_and_scales: [padding_height, padding_width, scale_height, scale_width]
            padding_height: int = abs(padded_hand_image.shape[0] - resized_hand_image.shape[0]) \
                if padded_hand_image.shape[0] - resized_hand_image.shape[0] >= 0 else 0
            padding_width: int = abs(padded_hand_image.shape[1] - resized_hand_image.shape[1]) \
                if padded_hand_image.shape[1] - resized_hand_image.shape[1] >= 0 else 0
            scale_height: float = resized_hand_image.shape[0] / hand_image.shape[0]
            scale_width: float = resized_hand_image.shape[1] / hand_image.shape[1]
            paddings_and_scales.append(
                PaddingsAndScale(
                    padding_height=padding_height,
                    padding_width=padding_width,
                    scale_height=scale_height,
                    scale_width=scale_width,
                )
            )

        return \
            hand_images, \
            np.asarray(hand_224x224_images, dtype=np.uint8), \
            np.asarray(normalized_hand_224x224_images, dtype=np.float32), \
            paddings_and_scales, \
            rec_size_differences

    def __postprocess(
        self,
        *,
        image: np.ndarray,
        hand_224x224_images: List[np.ndarray],
        hand_infos: List[Hand],
        xyz_x21s: np.ndarray,
        hand_scores: np.ndarray,
        lefthand_0_or_righthand_1s: np.ndarray,
        paddings_and_scales: List[PaddingsAndScale],
        rec_size_difference_rotation_angles: List[List[int]]
    ) -> Tuple[List[Hand], np.ndarray]:
        """__postprocess
        検出した手のキーポイント21点を元画像のスケールに戻す

        Parameters
        ----------
        image: np.ndarray
            全体画像

        hand_224x224_images: np.ndarray
            アスペクトレシオを維持したまま224x224へ収まるようにリサイズしたうえで224x224へパディングした手の画像\n
            uint8 [N,224,224,3]

        hand_infos: List[Hand]
            元画像スケールでの手の位置情報のリスト\n
            すべてスケール値で値を保持している (ピクセル座標ではない)

        xyz_x21s: np.ndarray
            float32 [N, 63], xyz*21

        hand_scores: np.ndarray
            float32 [N, 1]

        lefthand_0_or_righthand_1s: np.ndarray
            float32 [N, 1], 左手=0, 右手=1, 精度が低い

        paddings_and_scales: List[PaddingsAndScale]
            パディングピクセル数と手画像のスケールレシオ ([H,W,3]から[224,224,3]へ変換したときの変換倍率)\n
            [[padding_height, padding_width, scale_height, scale_width],[...],[...]]

        rec_size_difference_rotation_angles: List[List[int]]
            矩形クロップをするときに生じた幅と高さのサイズの差分。左右合計差分サイズと上下合計差分サイズ。\n
            最終的に元の座標系に戻すときには各値を２分の１にして減算して使用する。\n
            [[w,h], [w,h], ...] -> [[6,3], [10,15], ...]

        Returns
        -------
        hand_infos: List[Hand]
            手のキーポイント21点をセットした手の位置情報\n
            すべてスケール値で保持
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        xyz_x21s = xyz_x21s.reshape([-1, 21, 3]) # [N, 65] -> [N, 21, XYZ]
        xy_x21s = xyz_x21s[..., 0:2] # 奥行き情報Zを捨てる, [N, 21, XYZ] -> [N, 21, XY]

        keep = hand_scores[:, 0] > self.class_score_th
        hand_224x224_images = [
            hand_224x224_image \
                for idx, hand_224x224_image in enumerate(hand_224x224_images) if keep[idx]
        ]
        xy_x21s = xy_x21s[keep, ...]
        hand_scores = hand_scores[keep, ...]
        lefthand_0_or_righthand_1s = lefthand_0_or_righthand_1s[keep, ...]
        rec_size_difference_rotation_angles = [
            rec_size_difference_rotation_angle \
                for idx, rec_size_difference_rotation_angle in enumerate(rec_size_difference_rotation_angles) if keep[idx]
            ]
        paddings_and_scales = [
            paddings_and_scale \
                for idx, paddings_and_scale in enumerate(paddings_and_scales) if keep[idx]
        ]
        keeps = [
            kidx \
                for kidx, kvalue in enumerate(keep) if kvalue
        ]

        for idx, (hand_224x224_image, kidx, xy_x21, lefthand_0_or_righthand_1, paddings_and_scale) \
            in enumerate(zip(hand_224x224_images, keeps, xy_x21s, lefthand_0_or_righthand_1s, paddings_and_scales)):

            # 入力画像のスケールに戻す
            #   1. 224x224画像を使用して推論したキーポイントの座標情報は、パディングとスケールが反映された値になっている
            #   2. キーポイントの座標情報からパディング幅の2分の1を引き算、パディング高の2分の1を引き算（注意点は、パディングサイズが奇数だったときに1ピクセルの調整が必要になること）
            #   3. キーポイントの座標情報を幅スケール率と高スケール率を使用して元画像のスケールに戻す
            #   4. hand_info の degree (回転角) を元にして座標値に回転戻しを適用する
            # xyz_x21: [21, XY]
            hand_info = hand_infos[kidx]

            # 手のひらの情報が取得できていないときは処理をスキップする
            if hand_info.palm is None:
                continue

            # 224x224の座標系に対して、224x224へ変更するときに適用したパディングとリサイズの影響を取り除く
            # 座標値は全体画像の座標系ではなく、パディングとリサイズを元に戻した手のひらの部分だけの座標系
            xy_x21[:, 0] = xy_x21[:, 0] - paddings_and_scale.padding_width // 2
            xy_x21[:, 0] = xy_x21[:, 0] / paddings_and_scale.scale_width
            xy_x21[:, 1] = xy_x21[:, 1] - paddings_and_scale.padding_height // 2
            xy_x21[:, 1] = xy_x21[:, 1] / paddings_and_scale.scale_height
            cx = 112 - paddings_and_scale.padding_width // 2
            cx = cx / paddings_and_scale.scale_width
            cy = 112 - paddings_and_scale.padding_height // 2
            cy = cy / paddings_and_scale.scale_height

            temp_image = copy.deepcopy(hand_224x224_image)
            temp_image = temp_image[
                paddings_and_scale.padding_height // 2:temp_image.shape[0]-paddings_and_scale.padding_height // 2,
                paddings_and_scale.padding_width // 2:temp_image.shape[1]-paddings_and_scale.padding_width // 2,
                :
            ]

            temp_image = cv2.resize(
                src=temp_image,
                dsize=(int(temp_image.shape[1]/paddings_and_scale.scale_width), int(temp_image.shape[0]/paddings_and_scale.scale_height))
            )
            temp_image_cx = int(temp_image.shape[1] / 2)
            temp_image_cy = int(temp_image.shape[0] / 2)
            temp_image = image_rotation_without_crop(images=[temp_image], angles=[-hand_info.palm.degree])

            # パディングとリサイズを元に戻した手のひら画像に対して回転角を元に戻す
            rotated_xy_x21 = \
                rotate_points_around_center(
                    points_xy=xy_x21,
                    degree=hand_info.palm.degree,
                    cx=temp_image_cx,
                    cy=temp_image_cy,
                )

            hand_info.keypoints = [
                Point3D(
                    x=(x + hand_info.x1 * image_width - rec_size_difference_rotation_angles[idx][0] / 2) / image_width,
                    y=(y + hand_info.y1 * image_height - rec_size_difference_rotation_angles[idx][1] / 2) / image_height,
                    z=None,
                    depth=None,
                ) for x, y in rotated_xy_x21
            ]

        # 元画像スケールの手のひら位置情報
        return hand_infos


def pad_image(
    *,
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> np.ndarray:
    """画像の周囲を、指定された外接矩形サイズにパディング

    Parameters
    ----------
    image: np.ndarray
        Image to be resize and pad.

    resize_width: int
        Width of outer rectangle.

    resize_width: int
        Height of outer rectangle

    Returns
    -------
    padded_image: np.ndarray
        Image after padding.
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    if resize_width < image_width:
        resize_width = image_width
    if resize_height < image_height:
        resize_height = image_height

    padded_image = np.zeros(
        (resize_height, resize_width, 3),
        np.uint8
    )
    start_h = int(resize_height / 2 - image_height / 2)
    end_h = int(resize_height / 2 + image_height / 2)
    start_w = int(resize_width / 2 - image_width / 2)
    end_w = int(resize_width / 2 + image_width / 2)
    padded_image[start_h:end_h, start_w:end_w, :] = image

    return padded_image

def keep_aspect_resize_and_pad(
    *,
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """アスペクト比を維持したまま、指定された外接矩形内に収まるように短辺を指定されたサイズにパディングしながら長辺を基準に画像のサイズを変更

    Parameters
    ----------
    image: np.ndarray
        Image to be resize and pad.

    resize_width: int
        Width of outer rectangle.

    resize_width: int
        Height of outer rectangle

    Returns
    -------
    padded_image: np.ndarray
        Image after padding.

    resized_image: np.ndarray
        Image after resize. (Before padding)
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    padded_image = np.zeros(
        (resize_height, resize_width, 3),
        np.uint8
    )
    ash = resize_height / image_height
    asw = resize_width / image_width
    if asw < ash:
        sizeas = (
            int(image_width * asw) if int(image_width * asw) > 0 else 1,
            int(image_height * asw) if int(image_height * asw) > 0 else 1
        )
    else:
        sizeas = (
            int(image_width * ash) if int(image_width * ash) > 0 else 1,
            int(image_height * ash) if int(image_height * ash) > 0 else 1
        )
    resized_image: np.ndarray = cv2.resize(image, dsize=sizeas)
    start_h = int(resize_height / 2 - sizeas[1] / 2)
    end_h = int(resize_height / 2 + sizeas[1] / 2)
    start_w = int(resize_width / 2 - sizeas[0] / 2)
    end_w = int(resize_width / 2 + sizeas[0] / 2)
    padded_image[start_h:end_h, start_w:end_w, :] = resized_image.copy()

    return padded_image, resized_image

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def normalize_radians(
    *,
    angle: float
) -> float:
    """ラジアンの正規化

    Parameters
    ----------
    angle(radians): float

    Returns
    -------
    normalized_angle(radians): float
    """
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))

def is_inside_rect(
    *,
    rects: np.ndarray,
    width_of_outer_rect: int,
    height_of_outer_rect: int,
) -> np.ndarray:
    """バウンディングボックス(rects)が指定された外側の矩形の範囲の内側か外側かを判定する

    Parameters
    ----------
    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle(degree)]\n
        Area to be verified.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle(degree): float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    width_of_outer_rect: int
        Width of outer rectangle.

    height_of_outer_rect: int
        Height of outer rectangle

    Returns
    -------
    result: np.ndarray
        True: if the rotated sub rectangle is side the up-right rectange, False: else
    """
    results: List[bool] = []

    for rect in rects:
        cx: float = rect[0]
        cy: float = rect[1]
        width: float = rect[2]
        height: float = rect[3]
        angle: float = rect[4]

        if (cx < 0) or (cx > width_of_outer_rect):
            # Center X coordinate is outside the range of the outer rectangle
            results.append(False)

        elif (cy < 0) or (cy > height_of_outer_rect):
            # Center Y coordinate is outside the range of the outer rectangle
            results.append(False)

        else:
            # Coordinate acquisition of bounding rectangle considering rotation
            # http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b
            rect_tuple = ((cx, cy), (width, height), angle)
            box = cv2.boxPoints(rect_tuple)

            x_max = int(np.max(box[:,0]))
            x_min = int(np.min(box[:,0]))
            y_max = int(np.max(box[:,1]))
            y_min = int(np.min(box[:,1]))

            if (x_min >= 0) and (x_max <= width_of_outer_rect) and \
                (y_min >= 0) and (y_max <= height_of_outer_rect):
                # All 4 vertices are within the perimeter rectangle
                results.append(True)
            else:
                # Any of the 4 vertices is outside the perimeter rectangle
                results.append(False)

    return np.asarray(results, dtype=np.bool_)

def bounding_box_from_rotated_rect(
    *,
    rects: np.ndarray,
) -> np.ndarray:
    """入力されたバウンディングボックスの回転角をゼロ度に調整したバウンディングボックスを返す

    Parameters
    ----------
    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Rotated rectangle.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    Returns
    -------
    result: np.ndarray
        e.g.:\n
        [input] rotated rectangle:\n
            [center:(10, 10), height:4, width:4, angle:45 degree]\n
        [output] bounding box for this rotated rectangle:\n
            [center:(10, 10), height:4*sqrt(2), width:4*sqrt(2), angle:0 degree]
    """
    results: List[List[int]] = []

    for rect in rects:
        cx = rect[0]
        cy = rect[1]
        width = rect[2]
        height = rect[3]
        angle = rect[4]
        rect_tuple = ((cx, cy), (width, height), angle)
        box = cv2.boxPoints(rect_tuple)
        x_max = int(np.max(box[:,0]))
        x_min = int(np.min(box[:,0]))
        y_max = int(np.max(box[:,1]))
        y_min = int(np.min(box[:,1]))
        cx = int((x_min + x_max) // 2)
        cy = int((y_min + y_max) // 2)
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        angle = 0
        results.append([cx, cy, width, height, angle])
    return np.asarray(results, dtype=np.float32)

def crop_rectangle(
    *,
    image: np.ndarray,
    rects: np.ndarray,
) -> List[np.ndarray]:
    """入力画像から指定領域を切り取った画像を返す（入力画像の範囲外が指定されているときは切り取らない）

    Parameters
    ----------
    image: np.ndarray
        Image to be rotate and crop.

    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Rotate and crop rectangle.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    Returns
    -------
    croped_images: List[np.ndarray]
        Image after cropping.
    """
    croped_images: List[np.ndarray] = []
    height = image.shape[0]
    width = image.shape[1]

    # Determine if rect is inside the entire image
    inside_or_outsides = \
        is_inside_rect(
            rects=rects,
            width_of_outer_rect=width,
            height_of_outer_rect=height,
        )

    rects = rects[inside_or_outsides, ...]

    for rect in rects:
        cx = int(rect[0])
        cy = int(rect[1])
        rect_width = int(rect[2])
        rect_height = int(rect[3])

        croped_image: np.ndarray = image[
            cy-rect_height//2:cy+rect_height-rect_height//2,
            cx-rect_width//2:cx+rect_width-rect_width//2,
        ]
        croped_images.append(croped_image)

    return croped_images

def image_rotation_without_crop(
    *,
    images: List[np.ndarray],
    angles: np.ndarray,
) -> List[np.ndarray]:
    """画像を切り取らずに元のサイズのまま回転する

    Parameters
    ----------
    images: List[np.ndarray]
        Image to be rotated.

    angles: np.ndarray
        Rotation degree.

    Returns
    -------
    rotated_images: List[np.ndarray]
        Image after rotation.
    """
    rotated_images: List[np.ndarray] = []
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    for image, angle in zip(images, angles):
        height, width = image.shape[:2]
        image_center = (width//2, height//2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, int(angle), 1)
        abs_cos = abs(rotation_matrix[0,0])
        abs_sin = abs(rotation_matrix[0,1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += bound_w/2 - image_center[0]
        rotation_matrix[1, 2] += bound_h/2 - image_center[1]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))
        rotated_images.append(rotated_image)
    return rotated_images

def rotate_and_crop_rectangle(
    *,
    image: np.ndarray,
    rects: np.ndarray,
    operation_when_cropping_out_of_range: str='padding',
    update_rects: bool=False,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """画像から回転した矩形範囲を切り抜く

    Parameters
    ----------
    image: np.ndarray
        Image to be rotate and crop.

    rects: np.ndarray
        [boxcount, 5] = [boxcount, cx, cy, width, height, angle]\n
        Rotat and crop rectangle.

        cx: float
            Rectangle center X coordinate.

        cy: float
            Rectangle center Y coordinate.

        width: float
            Width of the rectangle.

        height: float
            Height of the rectangle.

        angle: float
            The rotation angle in a clockwise direction.
            When the angle is 0, 90, 180, 270, 360 etc.,
            the rectangle becomes an up-right rectangle.

    operation_when_cropping_out_of_range: str
        'padding' or 'ignore'

    update_rects: bool
        クロップした領域の情報で引数のrectsを更新するかどうか、True:更新する、False:更新しない

    Returns
    -------
    rotated_croped_image: List[np.ndarray]
        Image after cropping and rotation.

    rec_size_difference_rotation_angle: List[List[int]]
        矩形クロップをするときに生じた幅と高さのサイズの差分。左右合計差分サイズと上下合計差分サイズ。\n
        最終的に元の座標系に戻すときには各値を２分の１にして減算して使用する。
        [[w,h], [w,h], ...] -> [[6,3], [10,15], ...]
    """
    temp_image = copy.deepcopy(image)
    rects_: np.ndarray = copy.deepcopy(rects)
    rotated_croped_images: List[np.ndarray] = []
    height = temp_image.shape[0]
    width = temp_image.shape[1]
    rec_size_difference_rotation_angle: List[List] = []

    # Determine if rect is inside the entire image
    if operation_when_cropping_out_of_range == 'padding':
        size = (int(math.sqrt(width ** 2 + height ** 2)) + 2) * 2
        temp_image = \
            pad_image(
                image=temp_image,
                resize_width=size,
                resize_height=size,
            )
        rects_[:, 0] = rects_[:, 0] + abs(size - width) / 2
        rects_[:, 1] = rects_[:, 1] + abs(size - height) / 2

    elif operation_when_cropping_out_of_range == 'ignore':
        inside_or_outsides = \
            is_inside_rect(
                rects=rects_,
                width_of_outer_rect=width,
                height_of_outer_rect=height,
            )
        rects_ = rects_[inside_or_outsides, ...]

    rect_bbx_upright = \
        bounding_box_from_rotated_rect(
            rects=rects_,
        )

    rect_bbx_upright_images = \
        crop_rectangle(
            image=temp_image,
            rects=rect_bbx_upright,
        )
    # 回転角を考慮してクロップした矩形と非考慮の矩形の幅と高さの差分を計算する
    # 最終的に全体画像へプロットし直すための座標補正に使用する
    if len(rect_bbx_upright_images) > 0:
        for rect_bbx_upright_image, rect in zip(rect_bbx_upright_images, rects):
            original_width = rect[2]
            original_height = rect[3]
            width_difference_size = (rect_bbx_upright_image.shape[1] - original_width + 1)
            height_difference_size = (rect_bbx_upright_image.shape[0] - original_height + 1)
            rec_size_difference_rotation_angle.append([width_difference_size, height_difference_size])

    rotated_rect_bbx_upright_images = \
        image_rotation_without_crop(
            images=rect_bbx_upright_images,
            angles=rects_[..., 4:5],
        )

    if update_rects:
        for rect_bbx_upright_image, rect in zip(rect_bbx_upright_images, rects_):
            rect[2] = rect_bbx_upright_image.shape[1] # width
            rect[3] = rect_bbx_upright_image.shape[0] # height

    for rotated_rect_bbx_upright_image, rect in zip(rotated_rect_bbx_upright_images, rects_):
        crop_cx = rotated_rect_bbx_upright_image.shape[1]//2
        crop_cy = rotated_rect_bbx_upright_image.shape[0]//2
        rect_width = int(rect[2])
        rect_height = int(rect[3])
        rotated_croped_images.append(
            rotated_rect_bbx_upright_image[
                crop_cy-rect_height//2:crop_cy+(rect_height-rect_height//2),
                crop_cx-rect_width//2:crop_cx+(rect_width-rect_width//2),
            ]
        )

    return rotated_croped_images, rec_size_difference_rotation_angle

def rotate_points(*, points_xy: np.ndarray, degree: float) -> np.ndarray:
    # 角度をラジアンに変換
    theta = np.radians(degree)
    # 回転行列を作成
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # 各点を回転させる
    rotated_points: np.ndarray = np.dot(points_xy, rotation_matrix.T)
    return rotated_points

def rotate_points_around_center(*, points_xy: np.ndarray, degree: float, cx: float, cy: float) -> np.ndarray:
    # 角度をラジアンに変換
    theta = np.radians(degree)
    # 回転行列を作成
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # 中心点を原点に移動
    xy_translated = points_xy - np.array([cx, cy])
    # 移動した点を回転
    rotated_translated = np.dot(xy_translated, rotation_matrix.T)
    # 回転後の点を元の位置に戻す
    rotated_points: np.ndarray = rotated_translated + np.array([cx, cy])
    return rotated_points


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-mh',
        '--model_hand',
        type=str,
        default='gold_yolo_n_hand_post_0333_0.4040_1x3x512x896.onnx',
    )
    parser.add_argument(
        '-mp',
        '--model_palm',
        type=str,
        default='palm_detection_full_Nx3x192x192_post.onnx',
    )
    parser.add_argument(
        '-ml',
        '--model_handlandmark',
        type=str,
        default='hand_landmark_sparse_Nx3x224x224.onnx',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
    )
    args = parser.parse_args()

    model_hand = GoldYOLOONNX(
        model_path=args.model_hand,
    )
    model_palm = PalmDetectionONNX(
        model_path=args.model_palm,
    )
    model_handlandmark = HandLandmarkDetectionONNX(
        model_path=args.model_handlandmark,
    )

    cap = cv2.VideoCapture(
        int(args.video) if is_parsable_to_int(args.video) else args.video
    )
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(w, h),
    )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        start_time = time.time()

        image = image[..., ::-1] # BGR -> RGB
        inference_image = copy.deepcopy(image)
        debug_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Hand Detection - 手の物体検出
        boxes, scores = model_hand(inference_image)

        hand_infos: List[Hand] = []
        image_width = image.shape[1]
        image_height = image.shape[0]
        for box, score in zip(boxes, scores):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # Hands
            hand_infos.append(
                Hand(
                    cx=(x1+x2)/2/image_width, # scale
                    cy=(y1+y2)/2/image_height, # scale
                    x1=x1/image_width, # scale
                    y1=y1/image_height, # scale
                    x2=x2/image_width, # scale
                    y2=y2/image_height, # scale
                    vector_x=None,
                    vector_y=None,
                    vector_z=None,
                    quaternion_x=None,
                    quaternion_y=None,
                    quaternion_z=None,
                    quaternion_w=None,
                    keypoints=None,
                    palm=None,
                )
            )
            # Debug ###############################################################################
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0,0,255), 1)
            # Debug ###############################################################################

        # Palm Detection - 手の回転角計算, hand.palm.degree を計算してセットする
        hand_infos = \
            model_palm(
                image=image,
                hand_infos=hand_infos,
            )

        # Hand Landmark Detection - 手のキーポイント検出
        hand_infos = \
            model_handlandmark(
                image=image,
                hand_infos=hand_infos,
            )

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        cv2.putText(debug_image, f'{fps:.1f} FPS (inferece + post-process)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{fps:.1f} FPS (inferece + post-process)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        # Debug ###############################################################################
        lines_hand = [
            [0,1],[1,2],[2,3],[3,4],
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17],
        ]

        thick_coef = debug_image.shape[1] / 400
        radius = int(1 + thick_coef * 2)

        debug_hands_keypoints = []
        for hand_info in hand_infos:
            if hand_info == []:
                continue
            debug_keypoints = []
            if hand_info.keypoints is not None:
                for keypoint in hand_info.keypoints:
                    debug_keypoints.append([keypoint.x * image_width, keypoint.y * image_height])
                debug_hands_keypoints.append(debug_keypoints)
        if len(debug_hands_keypoints) > 0:
            debug_hands_keypoints = np.asarray(debug_hands_keypoints, dtype=np.int32)

        if len(debug_hands_keypoints) > 0:
            for debug_hands_keypoint in debug_hands_keypoints:
                lines = np.asarray(
                    [
                        np.array([debug_hands_keypoint[point] for point in line]).astype(np.int32) for line in lines_hand
                    ]
                )
                cv2.polylines(debug_image, lines, False, (255, 0, 0), int(radius), cv2.LINE_AA)
                _ = [
                    cv2.circle(debug_image, (int(x), int(y)), 2, (0,128,255), -1) \
                        for x, y in zip(debug_hands_keypoint[..., 0::2], debug_hands_keypoint[..., 1::2])
                ]

        cv2.imshow(f'landmark', debug_image)
        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        video_writer.write(debug_image)
        # Debug ###############################################################################

    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
    if cap:
        cap.release()


if __name__ == "__main__":
    main()
