# 🎯 卫星检测系统 PythonDS v1.0

一个基于 **Flask** 构建的后端服务，用于管理和执行卫星线缆连接情况的检测任务。

系统支持任务提交、状态查询等功能，可与 Java 后端对接。

---

## 🚀 版本说明

### v1.0

- ✅ 实现图片的识别裁剪与 OCR 识别功能  
- ✅ 支持多线程运行检测任务（未实现 GPU 资源调度）  
- ✅ 提供基础日志功能，便于问题追踪与分析  

---

## 📦 一.环境依赖

请确保安装以下依赖环境：

- Python 3.10+
- Flask
- PaddlePaddle
- YOLOv5 所需依赖（如 `torch`, `opencv-python`, `numpy` 等）

使用虚拟环境管理依赖：

```bash
python -m venv venv
source venv/bin/activate     # Windows 使用 venv\Scripts\activate
pip install -r requirements.txt


## 📁  二.项目结构

```bash
project/
├── venv/                     # 虚拟环境文件夹（建议本地创建）
├── yolov5/                   # YOLOv5 模型目录
├── app.log                   # 日志文件
├── task.py                   # 任务数据结构定义
├── task_manager.py           # 任务管理器，处理任务队列逻辑
├── first_detect_and_crop.py  # 第一步目标检测与裁剪模块
├── second_ocr_detect/        # OCR 检测模块目录
├── process.py                # 核心处理流程，图片分析与结果返回
├── test.py                   # 用于本地测试的脚本
└── README.md                 # 本说明文档
```


## 🧪 三.接口文档

### 3.1 POST请求  
🔹 app.route("/start_detection")  
🔹 URL：POST /start_detection     
🔹 描述：提交任务并启动检测流程    
🔹 请求体：需包含任务基本参数(例如 task_id、图片路径等) 

### 3.2 GET请求  
🔹 URL：GET /get_status  
🔹 描述：通过任务 ID 查询处理状态和结果



## 📒 四.日志说明

日志文件为 app.log

默认采用 UTF-8 编码，记录所有请求和系统异常

```bash
日志格式：
[INFO] 2025-07-02 15:30:00 FlaskApp.py: 任务 123456 已启动
[ERROR] 2025-07-02 15:31:00 FlaskApp.py: JSON解析失败: task_id=None
```  


## 📫 联系与支持
如需技术支持或提交建议，请通过以下方式联系作者：

👤 Contributor：Tongtong Shen,

📧 邮箱：1355718091@qq.com

🛠 本项目仍在持续迭代中，欢迎 Star、Fork 或参与贡献！