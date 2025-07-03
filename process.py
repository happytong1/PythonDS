import time
from pathlib import Path
import logging

from first_detect_and_crop import load_model, detect_crop_single_image
from second_ocr_detect import ocr_detection


# -------------------- 日志配置 -------------------- #
def setup_logger():
    logger = logging.getLogger('Thread-Process')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(name)s Module: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()


# -------------------------------------- 主检测流程 -------------------------------------- #
def run_detection(standalone, img1_path, img2_path, img3_path):
    logger.info(f"\n=============================================== 开始检测任务:{standalone} ==================================================")
    start_time = time.time()

    try:
        # === a. 创建工作目录 ===
        work_dir = Path("D:/Code/PythonDS/pictures/")
        dir_path = work_dir / standalone
        dir_path.mkdir(parents=True, exist_ok=True)

        # === b. 加载模型 YOLOv5 ===
        model, device, imgsz, stride = load_model(device='0', imgsz=1280, half=False)

        # === c. 图像检测与裁剪 ===
        image_paths = [img1_path, img2_path, img3_path]
        for idx, img_path in enumerate(image_paths, start=1):
            try:
                logger.info(f"程序正在处理第{idx}张图像: {img_path}")
                detect_crop_single_image(model, device, imgsz, stride, img_path, dir_path)
            except Exception as e:
                logger.error(f"图像处理失败(第{idx}张): {e}", exc_info=True)

        # === d. OCR识别 ===
        logger.info("开始OCR识别阶段......")
        try:
            result_flag, result = ocr_detection(dir_path=dir_path)
            logger.info("OCR识别完成!")
        except Exception as e:
            result_flag, result = -1, f"OCR识别失败: {str(e)}"
            logger.error(result, exc_info=True)

        # === e. 返回与统计 ===
        return result_flag, result

    except Exception as e:
        logger.error(f"检测流程整体失败: {str(e)}", exc_info=True)
        return -1, f"任务失败: {str(e)}"











# --------------------------------------- 本地测试 --------------------------------------------- #
if __name__ == "__main__":
    result_flag, result = run_detection(
        standalone="PK11",
        img1_path="C:/Users/13557/Desktop/DSdatabase/PK11_X01_1.jpg",
        img2_path="C:/Users/13557/Desktop/DSdatabase/PK11_X01_2.jpg",
        img3_path="C:/Users/13557/Desktop/DSdatabase/PK11_X01_3.jpg"
    )
    print(f"任务完成，结果状态: {result_flag}, 结果内容: {result}")
