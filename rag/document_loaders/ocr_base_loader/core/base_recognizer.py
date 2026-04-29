# YourPDFLoader/core/base_recognizer.py

import os
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

try:
    from paddle.inference import Config, create_predictor
except ImportError as e:
    raise ImportError(
        "未安装 paddlepaddle，请先安装，例如：\n"
        "pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple"
    ) from e


class BaseRecognizer:
    """
    Paddle Inference 通用推理基类

    适用于：
    - OCR det 模型（文本检测）
    - OCR rec 模型（文本识别）
    - 未来 layout 模型（如果也是 Paddle 导出）

    兼容模型文件：
    - inference.pdmodel
    - inference.pdiparams
    - inference.pdiparams.info（可有可无）
    """

    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = False,
        gpu_mem: int = 1024,
        cpu_threads: int = 4,
        enable_mkldnn: bool = False,
    ):
        """
        Args:
            model_dir: 模型目录（里面应包含 pdmodel / pdiparams）
            use_gpu: 是否启用 GPU
            gpu_mem: GPU 显存限制（MB）
            cpu_threads: CPU 线程数
            enable_mkldnn: 是否启用 MKLDNN（CPU 推理加速）
        """
        self.model_dir = model_dir
        self.model_file, self.params_file = self._find_model_files(model_dir)

        self.predictor = self._create_predictor(
            model_file=self.model_file,
            params_file=self.params_file,
            use_gpu=use_gpu,
            gpu_mem=gpu_mem,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
        )

        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

    # =========================================================
    # 初始化 Paddle Predictor
    # =========================================================
    def _find_model_files(self, model_dir: str) -> Tuple[str, str]:
        """
        自动查找 Paddle 推理模型文件

        Args:
            model_dir: 模型目录

        Returns:
            (pdmodel_path, pdiparams_path)
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        pdmodel = None
        pdiparams = None

        for f in os.listdir(model_dir):
            if f.endswith(".pdmodel"):
                pdmodel = os.path.join(model_dir, f)
            elif f.endswith(".pdiparams"):
                pdiparams = os.path.join(model_dir, f)

        if pdmodel is None or pdiparams is None:
            raise FileNotFoundError(
                f"在目录 {model_dir} 中未找到 .pdmodel 或 .pdiparams 文件"
            )

        return pdmodel, pdiparams

    def _create_predictor(
        self,
        model_file: str,
        params_file: str,
        use_gpu: bool = False,
        gpu_mem: int = 1024,
        cpu_threads: int = 4,
        enable_mkldnn: bool = False,
    ):
        """
        创建 Paddle Predictor
        """
        config = Config(model_file, params_file)

        if use_gpu:
            config.enable_use_gpu(gpu_mem, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_threads)
            if enable_mkldnn:
                config.enable_mkldnn()

        # 关闭多余日志
        config.disable_glog_info()

        # 开启内存优化
        config.enable_memory_optim()

        # 关闭 feed/fetch 依赖，提高性能
        config.switch_use_feed_fetch_ops(False)

        predictor = create_predictor(config)
        return predictor

    # =========================================================
    # 基础推理接口
    # =========================================================
    def infer(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        通用推理函数

        Args:
            input_dict: 输入字典，key 必须和模型输入名一致

        Returns:
            output_dict: 输出字典
        """
        # 设置输入
        for name in self.input_names:
            if name not in input_dict:
                raise KeyError(f"缺少模型输入: {name}")

            input_handle = self.predictor.get_input_handle(name)
            input_data = input_dict[name]

            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data)

            input_handle.copy_from_cpu(input_data)

        # 执行推理
        self.predictor.run()

        # 读取输出
        output_dict = {}
        for name in self.output_names:
            output_handle = self.predictor.get_output_handle(name)
            output_dict[name] = output_handle.copy_to_cpu()

        return output_dict

    # =========================================================
    # 图像工具函数（det / rec 都会用到）
    # =========================================================
    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        scale=1.0 / 255.0,
    ) -> np.ndarray:
        """
        图像归一化
        """
        image = image.astype("float32") * scale
        mean = np.array(mean).reshape(1, 1, 3).astype("float32")
        std = np.array(std).reshape(1, 1, 3).astype("float32")
        image = (image - mean) / std
        return image

    @staticmethod
    def to_chw(image: np.ndarray) -> np.ndarray:
        """
        HWC -> CHW
        """
        return image.transpose(2, 0, 1)

    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
        keep_ratio: bool = False
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        图像 resize

        Args:
            image: 原图
            target_size: (W, H)
            keep_ratio: 是否保持比例

        Returns:
            resized_image
            scale_ratio: (ratio_w, ratio_h)
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        if keep_ratio:
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas[:new_h, :new_w, :] = resized
            return canvas, (new_w / w, new_h / h)

        resized = cv2.resize(image, (target_w, target_h))
        return resized, (target_w / w, target_h / h)

    @staticmethod
    def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
        """
        对四边形点按顺时针排序
        输入 shape=(4,2)
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    @staticmethod
    def clip_bbox(bbox: List[float], width: int, height: int) -> List[float]:
        """
        裁剪 bbox 到图像范围内
        """
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(x0, width - 1))
        y0 = max(0, min(y0, height - 1))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        return [float(x0), float(y0), float(x1), float(y1)]

    @staticmethod
    def sort_boxes_top_to_bottom_left_to_right(
        boxes: List[Dict[str, Any]],
        y_thresh: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        将文本框按阅读顺序排序：
        先按 y，再按 x
        """
        def sort_key(b):
            return (round(b["bbox"][1] / y_thresh), b["bbox"][0])

        return sorted(boxes, key=sort_key)

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        计算两个 bbox 的 IoU
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h

        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

        union = area1 + area2 - inter_area + 1e-6
        return inter_area / union

    @staticmethod
    def remove_duplicate_boxes(
        boxes: List[Dict[str, Any]],
        iou_thresh: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        简单去重：高重叠框只保留一个
        """
        if not boxes:
            return []

        kept = []
        for box in boxes:
            duplicated = False
            for kb in kept:
                if BaseRecognizer.compute_iou(box["bbox"], kb["bbox"]) > iou_thresh:
                    duplicated = True
                    break
            if not duplicated:
                kept.append(box)
        return kept

'''
if __name__ == "__main__":
    """
    简单测试：
    只测试 Paddle 模型能否成功加载
    """

    # 你的 det 模型目录（注意：这里传目录，不是 .info 文件）
    model_dir = r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\rag\document_loaders\ocr_base_loader\ocr_models\ch_PP-OCRv4_det_infer"

    if not os.path.exists(model_dir):
        print(f"[ERROR] 模型目录不存在: {model_dir}")
    else:
        try:
            recognizer = BaseRecognizer(
                model_dir=model_dir,
                use_gpu=False,
                enable_mkldnn=False,
            )

            print("[OK] Paddle 模型加载成功")
            print("输入名:", recognizer.input_names)
            print("输出名:", recognizer.output_names)

            # 伪造一个 dummy 输入，仅用于查看接口是否可用
            # 注意：这里只是演示，不保证所有模型都能直接吃这个尺寸
            dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)

            input_name = recognizer.input_names[0]
            outputs = recognizer.infer({input_name: dummy})

            print("[OK] 推理接口运行成功")
            for k, v in outputs.items():
                print(f"{k}: shape={v.shape}")

        except Exception as e:
            print("[ERROR] 测试失败：", e)
'''