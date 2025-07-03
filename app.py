import os
from flask import Flask, jsonify, request
from task_manager import read_json, TaskManager
from task import Task
from datetime import datetime
import shutil



app = Flask(__name__)



# ---------------------- 日志配置 ---------------------- #
import logging
def setup_logger():
    logger = logging.getLogger('FlaskApp')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('app.log', mode='w', encoding='utf-8')
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(name)s.py: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    # 添加第一行启动标志
    now = datetime.now()
    start_line = now.strftime("%Y年%m月%d日  %Hh%Mmin%Ss  开始写入日志 ========================================")
    logger.info(start_line)

    return logger

logger = setup_logger()



# ---------------------------- 删除output和pictures -------------------------------------
if os.path.exists("D:/Code/PythonDS/output"):
    shutil.rmtree("D:/Code/PythonDS/output")
if os.path.exists("D:/Code/PythonDS/pictures"):
    shutil.rmtree("D:/Code/PythonDS/pictures")




# ---------------------------------- 初始化任务管理器 -------------------------------- #
task_manager = TaskManager(max_task_num=100)


# ---------------------------------- 启动检测任务 -------------------------------------- #
@app.route("/start_detection", methods=['POST'])
def start_detection():
    try:
        json_data = request.get_json()
        if not json_data:
            logger.error("请求中没有JSON数据")
            return jsonify({"status": "error", "message": "Missing JSON data"}), 400

        # 解包并校验 json
        result = read_json(json_data)
        if not isinstance(result, tuple) or len(result) != 3:
            logger.error("read_json 返回值格式异常")
            return jsonify({"status": "error", "message": "Internal parsing error"}), 500
        
        error_flag, task_id, params_ls = result

        if error_flag or task_id is None:
            logger.error(f"JSON解析失败: task_id={task_id}")
            return jsonify({"status": "error", "message": "Invalid request format"}), 400

        task = task_manager
        task = task_manager.create_task(task_id, params_ls)
        
        if task is None:
            logger.warning(f"任务创建失败: task_id=<{task_id}>(可能是重复或任务池已满)")
            return jsonify({
                "status": "error",
                "message": "Task creation failed (ID exists or task pool full)"
            }), 400
        return jsonify({"status": "started", "task_id": task_id}), 202

    except Exception as e:
        logger.exception("启动检测过程中发生异常")
        return jsonify({"status": "error", "message": "Internal server error"}), 500


# ------------------ 查询任务状态 ------------------ #
@app.route('/get_status', methods=['GET'])
def get_status():
    try:
        task_id_raw = request.args.get("task_id")
        if not task_id_raw:
            return jsonify({"status": "error", "message": "Missing task_id parameter"}), 400

        # 这里不转换成int，直接用字符串判断是否存在
        if task_id_raw not in task_manager.get_all_task_id():
            logger.warning(f"无效任务ID: {task_id_raw}")
            return jsonify({"status": "error", "message": "Invalid task ID"}), 400

        info_dict = task_manager.get_task_info(task_id_raw)
        if not info_dict:
            logger.error(f"无法获取任务信息: {task_id_raw}")
            return jsonify({"status": "error", "message": "Failed to get task info"}), 500

        response = {
            "task_id": task_id_raw,
            "status": info_dict.get("status", "unknown"),
            "result_flag": info_dict.get("result_flag", -1),
            "result": info_dict.get("task_result", "")
        }
        return jsonify(response)

    except Exception as e:
        logger.exception("查询任务状态过程中发生异常")
        return jsonify({"status": "error", "message": "Internal server error"}), 500





# ------------------ 启动应用 ------------------ #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
