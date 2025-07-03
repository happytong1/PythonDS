import threading
import time
from datetime import datetime
import logging
from process import run_detection


# -------------------- 日志配置 -------------------- #
def setup_logger():
    logger = logging.getLogger('Task')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s %(name)s Module: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():  # 防止重复添加
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()


# -------------------- 任务类定义 -------------------- #
class Task:
    def __init__(self, params_list):
        """
        参数说明:
            params_list: [standalone, img1_path, img2_path, img3_path]
        """
        self.params_list = params_list
        self.status = "pending"       # 状态: pending / running / completed
        self.result_flag = -1         # 检测结果标志位
        self.result = ""              # 检测返回内容
        self.thread = None            # 后台线程
        self.start_time = None
        self.end_time = None
        self.last_accessed = datetime.now()

    def start(self):
        """启动任务检测线程"""
        if self.status != "pending":
            logger.warning(f"尝试启动非待处理任务: 当前状态为 {self.status}")
            return False

        try:
            self.thread = threading.Thread(target=self._run_detection)
            self.thread.daemon = True
            self.thread.start()
            self.start_time = datetime.now()
            logger.info("任务线程启动成功")
            return True

        except Exception as e:
            logger.error(f"启动任务线程失败: {str(e)}", exc_info=True)
            return False

    def _run_detection(self):
        """线程实际执行的检测逻辑"""
        self.status = "running"
        self.last_accessed = datetime.now()

        try:
            # 解包参数
            standalone, img1_path, img2_path, img3_path = self.params_list

            # 调用检测主函数
            result_flag, result = run_detection(standalone, img1_path, img2_path, img3_path)

            self.result_flag = result_flag
            self.result = result

            logger.info(f"检测任务完成\nFlag: {result_flag}, Result: {result}")

        except Exception as e:
            self.result = f"检测异常: {str(e)}"
            logger.error(f"检测过程发生异常: {str(e)}", exc_info=True)

        finally:
            # 标记完成
            self.status = "completed"
            self.end_time = datetime.now()
            self.last_accessed = datetime.now()

            # 日志记录耗时
            duration = (self.end_time - self.start_time).total_seconds()

            # 清理资源
            try:
                self.params_list = None
                self.thread = None  # 断开引用，帮助 GC
            except Exception as e:
                logger.error(f"资源释放失败: {str(e)}", exc_info=True)
