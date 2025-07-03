import os
import sys
import cv2
import torch
import logging
import numpy as np
import pathlib
from pathlib import Path

from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from yolov5.models.experimental import attempt_load


# ---------------------- 平台兼容设置 ---------------------- #
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


# ---------------------- 日志配置 ---------------------- #
def setup_logger():
    logger = logging.getLogger('first_detect_and_crop')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(name)s.py: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    return logger


logger = setup_logger()


# ---------------------- 工具函数 ---------------------- #
def enlarge_box(xyxy, gain=1.5, img_shape=None):
    """放大检测框比例, 为了更好地裁剪图"""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h = w * gain, h * gain       # 经过验证，发现很多目标都是小目标,因此gain可以设置大一点
    x1_new = max(0, int(cx - new_w / 2))
    y1_new = max(0, int(cy - new_h / 2))
    x2_new = min(img_shape[1], int(cx + new_w / 2))
    y2_new = min(img_shape[0], int(cy + new_h / 2))
    return [x1_new, y1_new, x2_new, y2_new]


def load_model(weights_path=None, device='cpu', imgsz=640, half=False):
    """加载YOLOv5模型"""
    if weights_path is None:
        weights_path = "D:/Code/PythonDS/yolov5/weights/best.pt"
    weights_path = Path(weights_path)

    device = select_device(device)
    model = attempt_load(weights_path, device=device)
    model.to(device).eval()
    if half:
        model.half()

    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
    imgsz = check_img_size(imgsz, s=stride)
    logger.info(f"模型加载成功，使用设备: {device}")
    return model, device, imgsz, stride


@torch.no_grad()
def detect_crop_single_image(
    model, device, imgsz, stride, image_path, save_dir,
    conf_thres=0.25, iou_thres=0.45, enlarge_ratio=1.2, half=False):

    image_path = Path(image_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    im0 = cv2.imread(str(image_path))
    assert im0 is not None, f"Image not found: {image_path}"
    # logger.info(f"读取图像: {image_path.name}  尺寸: {im0.shape[1]}x{im0.shape[0]}")

    # 图像预处理
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()
    im /= 255.0
    if im.ndim == 3:
        im = im.unsqueeze(0)

    # 推理
    pred = model(im)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    for det in pred:
        if det is None or len(det) == 0:
            logger.info(f"未检测到目标: {image_path.name}")
            continue

        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        for j, (*xyxy, conf, cls) in enumerate(det):
            box = list(map(int, xyxy))
            cls_id = int(cls)

            # 标注矩形与类别
            cv2.rectangle(im0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(im0, str(cls_id), (box[0], max(box[1] - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 裁剪并保存
            enlarged_box = enlarge_box(box, gain=enlarge_ratio, img_shape=im0.shape)
            crop = im0[enlarged_box[1]:enlarged_box[3], enlarged_box[0]:enlarged_box[2]]
            crop_path = save_dir / f"{image_path.stem}_crop_{j}.jpg"
            cv2.imwrite(str(crop_path), crop)
            # 暂时不输出 logger.info(f"Save subplot to: {crop_path}")

    # 保存带标注的原图
    annotated_path = save_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), im0)










# -------------------------------------- 本地测试入口 ----------------------------------------- #
if __name__ == "__main__":
    image_path = "C:/Users/13557/Desktop/DSdatabase/PK11_X01_1.jpg"
    save_dir = "D:/Code/PythonDS/pictures/"
    device_id = '0'

    model, device, imgsz, stride = load_model(device=device_id, imgsz=640, half=False)
    detect_crop_single_image(model, device, imgsz, stride, image_path, save_dir)
