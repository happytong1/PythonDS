import os
import re
import json
import sys
import logging
from pathlib import Path
from paddleocr import PaddleOCR
from json.decoder import JSONDecodeError

# 设置环境变量减少线程冲突
os.environ['OMP_NUM_THREADS'] = '1'

# ---------------------- 日志配置 ---------------------- #
def setup_logger():
    logger = logging.getLogger("second_ocr_detect")
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


# ---------------------- OCR检测主函数 ---------------------- #
def ocr_detection(dir_path):
    dir_path = Path(dir_path)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 初始化OCR模型
    ocr = PaddleOCR(
        text_det_unclip_ratio=2.0,
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )

    # 三个阶段的识别结果
    flag_1 = flag_2 = flag_3 = 0

    for name in os.listdir(dir_path):
        file_path = dir_path / name

        if not file_path.is_file() or name.endswith("_annotated.jpg"):
            # 可靠性检查
            continue

        filename, _ = os.path.splitext(name)
        pattern = r'^([A-Z0-9]+)_([A-Za-z0-9]+(?:bak)?)_([0-9]+)_crop_([0-9]+)$'
        # pattern = r'^([A-Z0-9]+)_([A-Z0-9]+)_([0-9]+)_crop_([0-9]+)$' 考虑上bak的情况
        match = re.match(pattern, filename)

        if not match:
            logger.warning(f"[WARN] 文件名不符合规则，跳过：{name}")
            continue

        Standalone, Socket, Stage, Subplot = match.groups()
        logger.info(f"线缆 <Standalone={Standalone}, Socket={Socket}, Stage={Stage}, Subplot={Subplot}>")

        # 1. OCR 识别并保存 JSON
        try:
            result = ocr.predict(str(file_path))
            for res in result:
                res.save_to_json(str(output_dir))
        except Exception as e:
            logger.error(f"OCR识别失败: {file_path}，错误: {e}", exc_info=True)
            continue

        # 2. 加载 JSON 文件
        json_path = output_dir / f"{filename}_res.json"
        if not json_path.exists():
            logger.warning(f"[WARN] JSON结果文件缺失: {json_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"[WARN] 无法打开JSON文件: {json_path}")
            continue
        except JSONDecodeError:
            logger.warning(f"[WARN] JSON格式错误: {json_path}")
            continue

        # 3. 分析识别内容
        recognized_texts = json_data.get("rec_texts", [])

        # 4. 匹配阶段内容
        match_texts = [t for t in recognized_texts if Standalone in t or Socket in t]

        if Stage == '1' and any(Standalone in t for t in recognized_texts):
            flag_1 = 1
        elif Stage == '2' and any(Socket in t for t in recognized_texts):
            flag_2 = 1
        elif Stage == '3' and match_texts:
            flag_3 = 1

        logger.info(f"{name} -> 匹配内容: {match_texts}")

    # 5. 计算最终flag
    final_flag = flag_1 + flag_2 + flag_3
    if final_flag < 1:
        result_text = "单机和线缆不匹配!"
    # elif final_flag < 1:
    #     result_text = "图片拍摄不标准,未能全部识别,建议重新拍摄上传"
    else:
        result_text = "单机和线缆匹配成功!"

    return final_flag, result_text





# -------------------------------------------- 测试入口 ----------------------------------------------- #
if __name__ == "__main__":
    dir_path = "D:/Code/PythonDS/pictures/PK11"
    flag, result = ocr_detection(dir_path)
    print(f"OCR检测状态: {flag}, 结果: {result}")
