import os
import threading
import time
import logging
from datetime import datetime, timedelta

from task import Task


# ------------------------- 日志配置 ------------------------- #
def setup_logger():
    logger = logging.getLogger('TaskManager')
    logger.setLevel(logging.INFO)

    # 文件处理器  确保中文编码
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(name)s Module: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()


# ------------------------- JSON 解析 ------------------------- #
def read_json(json_data):
    """读取并验证 JSON 数据格式"""
    error_flag = False
    params_list = []
    task_id = None

    try:
        task_id = json_data["task_id"]
        standalone = json_data["standalone"]        # PK12-PK11_X01
        img_paths = json_data["img_paths"]
        img_1_path = img_paths.get("state_1")
        img_2_path = img_paths.get("state_3")
        img_3_path = img_paths.get("state_4")

        standalone_name = standalone.split('-')[1]  # standalone_name: PK11_X01
        # 这里就是为了匹配 PK11_X01_1.jpg 和 P12-PK11对应不对应
        for path in [img_1_path, img_2_path, img_3_path]:
            file_name = os.path.basename(path)
            if standalone_name not in file_name:
                socket_name = os.path.basename(path)
                logger.warning(f"图像 {socket_name} 与单机 {standalone} 不匹配")

        params_list = [standalone, img_1_path, img_2_path, img_3_path]
        logger.info(f"成功解析任务 <{task_id}>")

    except KeyError as e:
        error_flag = True
        logger.error(f"JSON 解析错误: 缺少字段 {str(e)}")
    except Exception as e:
        error_flag = True
        logger.error(f"JSON 解析异常: {str(e)}", exc_info=True)

    return error_flag, task_id, params_list



# ------------------------- 任务管理器 ------------------------- #
class TaskManager:
    def __init__(self, max_task_num=100, cleanup_interval=300, task_timeout=3600):
        self.max_task_num = max_task_num
        self.cleanup_interval = cleanup_interval
        self.task_timeout = task_timeout
        self.tasks_dict = {}  # {task_id: Task}
        self.lock = threading.Lock()
        self._start_cleanup_thread()
        logger.info(f"TaskManager初始化完成!!!  max_task_num:{max_task_num}")

    def _start_cleanup_thread(self):
        """后台线程定期清理任务"""
        def cleanup_loop():
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup_tasks()
                logger.debug("已完成定期任务清理")

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def cleanup_tasks(self):
        """清理超时或已完成的任务，仅当任务数超过最大限制时才执行"""
        with self.lock:
            # 如果当前任务数未超过阈值，跳过清理
            if len(self.tasks_dict) <= self.max_task_num:
                logger.debug(f"当前任务数({len(self.tasks_dict)})未超过最大限制({self.max_task_num})，跳过清理。")
                return 0

            now = datetime.now()
            to_remove = []

            # 先找出需要清理的任务
            for task_id, task in self.tasks_dict.items():
                if task.status == "completed":
                    to_remove.append(task_id)
                elif (now - task.last_accessed).total_seconds() > self.task_timeout:
                    logger.warning(f"任务 <{task_id}> 已超时，将被清理")
                    to_remove.append(task_id)

            # 删除找到的任务
            for task_id in to_remove:
                del self.tasks_dict[task_id]
                logger.info(f"已清理任务 <{task_id}>")

            # 清理后如果仍超过最大任务数，执行强制清理
            if len(self.tasks_dict) > self.max_task_num:
                self._force_cleanup()

            return len(to_remove)


    def _force_cleanup(self):
        """强制删除最旧的已完成任务"""
        completed = [(tid, task.last_accessed)
                     for tid, task in self.tasks_dict.items() if task.status == "completed"]
        completed.sort(key=lambda x: x[1])  # 早的排前面

        while len(self.tasks_dict) > self.max_task_num and completed:
            tid, _ = completed.pop(0)
            del self.tasks_dict[tid]
            logger.warning(f"强制清理任务 <{tid}>   任务池已满")

    def create_task(self, task_id, params_list):
        with self.lock:
            if task_id in self.tasks_dict:
                logger.warning(f"任务 <{task_id}> 已存在，创建失败")
                return None

            if len(self.tasks_dict) >= self.max_task_num:
                logger.info("任务池达到上限，尝试清理")
                self.cleanup_tasks()
                if len(self.tasks_dict) >= self.max_task_num:
                    logger.error(f"任务池仍满，创建任务 <{task_id}> 失败")
                    return None

            try:
                task = Task(params_list)
                if task.start():
                    self.tasks_dict[task_id] = task
                    logger.info(f"TaskManager创建任务<{task_id}>成功!")
                    return task
                else:
                    logger.error(f"任务 <{task_id}> 启动失败")
            except Exception as e:
                logger.error(f"创建任务异常: {str(e)}", exc_info=True)

            return None

    def get_all_task_id(self):
        """返回当前所有任务ID"""
        with self.lock:
            for task in self.tasks_dict.values():
                task.last_accessed = datetime.now()
            return list(self.tasks_dict.keys())

    def get_task_info(self, task_id):
        """获取某个任务的详细信息"""
        with self.lock:
            task = self.tasks_dict.get(task_id)
            if not task:
                return None

            task.last_accessed = datetime.now()

            def fmt(dt):
                return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None

            runtime = "0 min 0 s"
            if task.start_time:
                delta = datetime.now() - task.start_time
                runtime = f"{delta.seconds // 60} min {delta.seconds % 60} s"

            return {
                "task_id": task_id,
                "status": task.status,
                "result_flag": task.result_flag,
                "task_result": task.result,
                "start_time": fmt(task.start_time),
                "end_time": fmt(task.end_time),
                "runtime": runtime
            }
