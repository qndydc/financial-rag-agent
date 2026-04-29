# YourPDFLoader/core/ocr_engine.py

from __future__ import annotations

import cv2
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from .base_recognizer import BaseRecognizer
except ImportError:
    from base_recognizer import BaseRecognizer


# =========================================================
# Utils
# =========================================================
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(np.clip(x, -50, 50))
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    4点排序：tl, tr, br, bl
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def clip_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    return box


def box_to_bbox(box: np.ndarray) -> List[float]:
    x0 = float(np.min(box[:, 0]))
    y0 = float(np.min(box[:, 1]))
    x1 = float(np.max(box[:, 0]))
    y1 = float(np.max(box[:, 1]))
    return [x0, y0, x1, y1]


def polygon_score(pred: np.ndarray, box: np.ndarray) -> float:
    """
    计算 polygon 内平均置信度
    pred: [H, W], 0~1
    box : [4, 2]
    """
    h, w = pred.shape[:2]
    box_int = np.round(box).astype(np.int32)
    box_int[:, 0] = np.clip(box_int[:, 0], 0, w - 1)
    box_int[:, 1] = np.clip(box_int[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box_int], 1)

    values = pred[mask == 1]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def expand_box(box: np.ndarray, ratio: float = 1.6) -> np.ndarray:
    """
    轻量版 unclip：以中心点做放缩
    对 PDF 文本行通常足够，比直接照搬 DB unclip 更轻
    """
    center = np.mean(box, axis=0, keepdims=True)
    expanded = (box - center) * ratio + center
    return expanded.astype(np.float32)


def get_min_side(box: np.ndarray) -> float:
    edge1 = np.linalg.norm(box[0] - box[1])
    edge2 = np.linalg.norm(box[1] - box[2])
    return float(min(edge1, edge2))


def crop_quad(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    透视裁剪
    """
    pts = order_points_clockwise(box)
    tl, tr, br, bl = pts

    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width = max(width, 2)
    height = max(height, 2)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(
        img,
        M,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    crop = np.ascontiguousarray(crop)
    if crop.dtype != np.uint8:
        crop = crop.astype(np.uint8)

    # 对于“竖着的长条文字框”，旋转成横排再识别
    h, w = crop.shape[:2]
    if h > w * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop


def sort_boxes_reading_order(boxes: List[np.ndarray], y_thresh_ratio: float = 0.5) -> List[np.ndarray]:
    """
    比简单(y, x)更稳一点的阅读顺序排序：
    先按 top 排，再按“行”分组，再组内按 x 排
    """
    if not boxes:
        return []

    items = []
    for box in boxes:
        bbox = box_to_bbox(box)
        x0, y0, x1, y1 = bbox
        h = max(1.0, y1 - y0)
        items.append(
            {
                "box": box,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "h": h,
                "cy": (y0 + y1) / 2.0,
            }
        )

    items.sort(key=lambda z: (z["y0"], z["x0"]))

    lines = []
    for item in items:
        placed = False
        for line in lines:
            ref = line[0]
            thresh = min(ref["h"], item["h"]) * y_thresh_ratio
            if abs(item["cy"] - ref["cy"]) <= thresh:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda z: z["x0"])

    sorted_boxes = []
    for line in lines:
        sorted_boxes.extend([z["box"] for z in line])

    return sorted_boxes


# =========================================================
# 文本检测模块（det）
# =========================================================
class TextDetector:
    """
    轻量版 DB detector 后处理，适配中文 PDF 页图
    """

    def __init__(
        self,
        model_dir: str,
        limit_side_len: int = 1280,
        thresh: float = 0.20,
        box_thresh: float = 0.45,
        unclip_ratio: float = 1.6,
        min_size: int = 2,
        use_dilation: bool = True,
    ):
        self.recognizer = BaseRecognizer(model_dir)
        self.limit_side_len = limit_side_len
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.use_dilation = use_dilation

    def preprocess(self, img: np.ndarray):
        """
        det 预处理：
        - 限制长边
        - resize 到 32 倍数
        - 归一化
        """
        ori_h, ori_w = img.shape[:2]

        scale = min(self.limit_side_len / max(ori_h, ori_w), 1.0)
        resize_h = max(32, int(round(ori_h * scale / 32) * 32))
        resize_w = max(32, int(round(ori_w * scale / 32) * 32))

        resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        img_norm = resized.astype(np.float32) / 255.0
        # PP-OCR det 常见归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_norm - mean) / std

        img_norm = img_norm.transpose(2, 0, 1)
        img_norm = np.expand_dims(img_norm, axis=0)

        shape_info = {
            "ori_h": ori_h,
            "ori_w": ori_w,
            "resize_h": resize_h,
            "resize_w": resize_w,
            "ratio_h": ori_h / float(resize_h),
            "ratio_w": ori_w / float(resize_w),
        }
        return img_norm, shape_info

    def _extract_pred_map(self, outputs: Dict[str, np.ndarray]) -> np.ndarray:
        pred = list(outputs.values())[0]

        # 常见情况: [1, 1, H, W]
        if pred.ndim == 4:
            pred_map = pred[0, 0]
        elif pred.ndim == 3:
            pred_map = pred[0]
        else:
            raise ValueError(f"Unexpected det output shape: {pred.shape}")

        # 兼容 logits / prob
        if pred_map.min() < 0.0 or pred_map.max() > 1.0:
            pred_map = _sigmoid(pred_map)

        return pred_map.astype(np.float32)

    def postprocess(self, pred_map: np.ndarray, shape_info: Dict[str, float]) -> List[np.ndarray]:
        """
        轻量版 DB 后处理
        """
        bitmap = (pred_map > self.thresh).astype(np.uint8) * 255

        if self.use_dilation:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            bitmap = cv2.dilate(bitmap, kernel, iterations=1)

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        resize_h, resize_w = pred_map.shape[:2]
        ori_h, ori_w = int(shape_info["ori_h"]), int(shape_info["ori_w"])
        ratio_h, ratio_w = shape_info["ratio_h"], shape_info["ratio_w"]

        for cnt in contours:
            if cnt.shape[0] < 4:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.float32)

            if get_min_side(box) < self.min_size:
                continue

            score = polygon_score(pred_map, box)
            if score < self.box_thresh:
                continue

            box = expand_box(box, self.unclip_ratio)
            if get_min_side(box) < self.min_size + 1:
                continue

            # resize 坐标映射回原图
            box[:, 0] *= ratio_w
            box[:, 1] *= ratio_h
            box = clip_box(box, ori_w, ori_h)

            # 再次过滤过小框
            bbox = box_to_bbox(box)
            if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
                continue

            boxes.append(box)

        # 按面积去重/过滤极端噪声
        filtered = []
        for box in boxes:
            bbox = box_to_bbox(box)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 16:
                continue
            filtered.append(box)

        return filtered

    def __call__(self, img: np.ndarray) -> List[np.ndarray]:
        input_tensor, shape_info = self.preprocess(img)
        outputs = self.recognizer.infer({
            self.recognizer.input_names[0]: input_tensor
        })
        pred_map = self._extract_pred_map(outputs)
        return self.postprocess(pred_map, shape_info)


# =========================================================
# 文本识别模块（rec）
# =========================================================
class TextRecognizer:
    def __init__(
        self,
        model_dir: str,
        dict_path: str,
        img_h: int = 48,
        img_w: int = 320,
        batch_size: int = 16,
        use_space_char: bool = False,
    ):
        self.recognizer = BaseRecognizer(model_dir)
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size

        self.character = self._load_character(dict_path, use_space_char=use_space_char)

        # PaddleOCR CTC: blank 在 0 号位
        self.character = ["blank"] + self.character
        self.blank_idx = 0

    def _load_character(self, dict_path: str, use_space_char: bool = False):
        chars = []
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n").rstrip("\r")
                if line != "":
                    chars.append(line)
        if use_space_char:
            chars.append(" ")
        return chars

    def preprocess_single(self, img: np.ndarray):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        ratio = w / max(float(h), 1.0)
        resized_w = int(math.ceil(self.img_h * ratio))
        resized_w = max(1, min(resized_w, self.img_w))

        resized = cv2.resize(img, (resized_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype("float32") / 255.0
        resized = (resized - 0.5) / 0.5
        resized = resized.transpose(2, 0, 1)

        padding = np.zeros((3, self.img_h, self.img_w), dtype=np.float32)
        padding[:, :, :resized_w] = resized
        return padding

    def _normalize_output(self, preds: np.ndarray) -> np.ndarray:
        """
        自动兼容 [B, T, C] / [B, C, T]
        """
        if preds.ndim != 3:
            raise ValueError(f"Unexpected rec output shape: {preds.shape}")

        # 一般字符类别数会远大于时间步数
        # 如果第二维特别大、第三维较小，则可能是 [B, C, T]
        if preds.shape[1] > preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))  # -> [B, T, C]

        return preds

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(np.clip(x, -50, 50))
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)

    def decode(self, preds: np.ndarray):
        preds = self._normalize_output(preds)

        # logits -> probs
        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = self._softmax(preds, axis=2)

        pred_idxs = preds.argmax(axis=2)
        pred_probs = preds.max(axis=2)

        results = []
        for b in range(pred_idxs.shape[0]):
            char_list = []
            conf_list = []
            prev_idx = None

            for t, idx in enumerate(pred_idxs[b]):
                idx = int(idx)

                # blank
                if idx == self.blank_idx:
                    prev_idx = idx
                    continue

                # CTC 去重
                if idx == prev_idx:
                    continue

                if 0 <= idx < len(self.character):
                    char_list.append(self.character[idx])
                    conf_list.append(float(pred_probs[b, t]))

                prev_idx = idx

            text = "".join(char_list).strip()
            score = float(np.mean(conf_list)) if conf_list else 0.0
            results.append((text, score))

        return results

    def __call__(self, img_list):
        if not img_list:
            return []

        outputs_all = []
        for start in range(0, len(img_list), self.batch_size):
            batch_imgs = img_list[start:start + self.batch_size]
            batch = np.stack([self.preprocess_single(im) for im in batch_imgs], axis=0)

            outputs = self.recognizer.infer({
                self.recognizer.input_names[0]: batch.astype(np.float32)
            })

            preds = list(outputs.values())[0]
            print("REC output shape:", preds.shape)
            print("DICT size(without blank):", len(self.character) - 1)
            outputs_all.extend(self.decode(preds))

        return outputs_all


# =========================================================
# OCR 主流程（det + rec）
# =========================================================
class OCR:
    """
    统一 OCR 接口（给 PDFLoader 调用）

    输出结构：
    [
        {
            "text": "...",
            "bbox": [x0, y0, x1, y1],
            "score": 0.95,
            "type": "text"
        }
    ]
    """

    def __init__(
        self,
        det_model_dir: str,
        rec_model_dir: str,
        rec_dict_path: str,
        drop_score: float = 0.45,
        min_crop_size: int = 6,
    ):
        self.detector = TextDetector(det_model_dir)
        self.recognizer = TextRecognizer(
            model_dir=rec_model_dir,
            dict_path=rec_dict_path,
            img_h=48,
            img_w=320,
            batch_size=16,
        )
        self.drop_score = drop_score
        self.min_crop_size = min_crop_size

    def _valid_crop(self, crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0:
            return False
        if crop.ndim != 3:
            return False
        h, w = crop.shape[:2]
        if h < self.min_crop_size or w < self.min_crop_size:
            return False
        return True

    def __call__(self, img: np.ndarray) -> List[Dict[str, Any]]:
        boxes = self.detector(img)
        if not boxes:
            return []

        boxes = sort_boxes_reading_order(boxes)

        crops = []
        valid_boxes = []
        for box in boxes:
            crop = crop_quad(img, box)
            if not self._valid_crop(crop):
                continue

            # 过滤极端异常框
            h, w = crop.shape[:2]
            ratio = w / max(float(h), 1.0)
            if ratio > 40:   # 横线、表格线等
                continue
            if ratio < 0.08: # 极窄竖线噪声
                continue

            crops.append(crop)
            valid_boxes.append(box)

        if not crops:
            return []

        rec_res = self.recognizer(crops)

        results = []
        for box, (text, score) in zip(valid_boxes, rec_res):
            text = text.strip()
            if not text:
                continue
            if score < self.drop_score:
                continue

            bbox = box_to_bbox(box)
            results.append({
                "text": text,
                "bbox": bbox,
                "score": float(score),
                "type": "text",
            })

        return results


# =========================================================
# 独立测试
# =========================================================
if __name__ == "__main__":
    import traceback

    det_model_dir = r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\rag\document_loaders\ocr_base_loader\ocr_models\ch_PP-OCRv4_det_infer"
    rec_model_dir = r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\rag\document_loaders\ocr_base_loader\ocr_models\ch_PP-OCRv4_rec_infer"
    rec_dict_path = r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\rag\document_loaders\ocr_base_loader\ocr_models\ppocr_keys_v1.txt"

    pdf_path = r"D:\Aprojet\py\RAG_Agent\financial_rag_agent\dataset\raw_pdf\半导体\H2_AN202502281643614026_1.pdf"

    try:
        print("=" * 60)
        print("测试 OCR 引擎独立运行...")

        ocr = OCR(
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            rec_dict_path=rec_dict_path,
            drop_score=0.45,
        )
        print("OCR 初始化成功")
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from parser.pdf_processor import PDFProcessor

        processor = PDFProcessor(
            dpi=300,
            keep_image=True,
            save_images=False
        )
        print("PDFProcessor 初始化成功")

        doc = processor.process(pdf_path=pdf_path)
        print(f"PDF 读取成功，总页数：{len(doc.pages)}")

        page = doc.pages[9]
        img = page.image

        if img is None:
            raise ValueError("page.image is None")

        # 假设 processor 输出 RGB，这里转 BGR 给 OpenCV
        img_bgr = img[:, :, ::-1].copy()

        results = ocr(img_bgr)

        print("=" * 60)
        print(f"OCR 完成，识别到 {len(results)} 条文本")
        for i, r in enumerate(results[:80]):
            print(f"[{i+1:03d}] {r['text']} | score={r['score']:.3f} | bbox={r['bbox']}")
        print("=" * 60)

    except Exception as e:
        print(f"OCR 运行失败: {e}")
        traceback.print_exc()