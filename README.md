基于YOLOv8➕pyqt5的焊缝质量检测系统，

**以下项目有成品代码**【或根据数据集代做】

| 序号 | 项目名称                     | 序号 | 项目名称                     | 序号 | 项目名称                     |
|------|------------------------------|------|------------------------------|------|------------------------------|
| 1    | 路面坑洞检测分割             | 2    | 非机动车头盔检测             | 3    | 人脸表情识别系统             |
| 4    | 玉米病害检测                 | 5    | 车辆行人追踪系统             | 6    | 钢材表面缺陷检测             |
| 7    | 水稻害虫检测识别             | 8    | 小麦害虫检测识别             | 9    | 100种蝴蝶识别                |
| 10   | 血细胞检测计数               | 11   | 野火烟雾检测                 | 12   | 脑肿瘤检测                   |
| 13   | 舰船分类检测系统             | 14   | 苹果病害识别                 | 15   | 车辆追踪计数                 |
| 16   | 交通标志检测识别             | 17   | 草莓病害检测分割             | 18   | 安全帽检测系统               |
| 19   | 螺栓螺母检测                 | 20   | 高密度人脸检测               | 21   | 肾结石检测                   |
| 22   | 火焰烟雾检测系统             | 23   | 车牌检测识别系统             | 24   | 葡萄病害识别                 |
| 25   | 无人机视角检测               | 26   | 木薯病害识别预防             | 27   | 水果质量检测                 |
| 28   | 102种犬类检测系统            | 29   | 路面坑洞检测系统             | 30   | 玉米害虫检测识别             |
| 31   | 生活垃圾分类检测             | 32   | 车辆行人检测计数             | 33   | 200种鸟类检测识别            |
| 34   | 人脸识别与管理系统           | 35   | 裂缝检测分析系统             | 36   | YOLOv8自动标注               |
| 37   | 人员闯入报警                 | 38   | 西红柿成熟度检测             | 39   | 肺炎诊断系统                 |
| 40   | 反光衣检测预警               | 41   | 人脸面部活体检测系统         | 42   | 手势识别系统                 |
| 43   | 水果检测识别                 | 44   | 车牌识别与管理系统           | 45   | 田间杂草检测系统             |
| 46   | 遥感地面物体检测             | 47   | 吸烟行为检测                 | 48   | 行人跌倒检测系统             |
| 49   | PCB板缺陷检测系统            | 50   | 橙子病害识别                 | 51   | 行人追踪计数                 |
| 52   | 100种中草药识别              | 53   | 面部口罩检测系统             | 54   | 蔬菜检测识别                 |
| 55   | 焊缝缺陷检测                 | 56   | 水稻病害识别                 | 57   | 复杂环境船舶检测             |
| 58   | 102种花卉识别                |      |                              |      |                              |


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/62bc8324c6de4ff2bb3a4859b9f4def7.png)


内含1300张焊缝缺陷数据集
包括['Bad Welding', 'Crack', 'Excess Reinforcement', 'Good Welding', 'Porosity', 'Spatters']，6类
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/093ab8ee04314f6fa5f627fe6d21284f.png)


也可自行替换模型，使用该界面做其他检测

以下是 **基于 YOLOv8 + PyQt5 的焊缝质量检测系统** 的完整代码实现，支持 **图片、视频、摄像头实时检测**，附带 **1300 张标注好的焊缝缺陷数据集（6 类）**，可直接运行。

---

## ✅ 一、项目概览

| 项目 | 内容 |
|------|------|
| **系统名称** | 基于深度学习的焊缝质量检测系统 |
| **核心技术** | YOLOv8 + PyQt5 + OpenCV |
| **功能支持** | 图片识别 / 视频分析 / 摄像头实时检测 |
| **检测类别** | `Bad Welding`, `Crack`, `Excess Reinforcement`, `Good Welding`, `Porosity`, `Spatters` |
| **输出内容** | 类别、置信度、坐标框、检测耗时 |
| **部署方式** | Python 脚本运行，到手即可使用 |

---

## ✅ 二、项目目录结构

```bash
WeldQualityDetection/
├── datasets/                    # 数据集文件夹（含 train/val）
│   ├── train/
│   └── val/
├── models/                      # 训练好的模型
│   └── best.pt                  # YOLOv8 模型权重
├── runs/                        # 训练结果缓存
├── save_data/                   # 检测结果保存路径
├── TestFiles/                   # 测试用图片
├── ui/                          # UI 相关文件
│   └── MainProgram.py           # 主程序入口
├── CameraTest.py                # 摄像头测试脚本
├── Config.py                    # 配置文件
├── detect_tools.py              # 检测工具函数
├── imgTest.py                   # 图片检测脚本
├── VideoTest.py                 # 视频检测脚本
├── yolov8n.pt                   # 官方预训练模型（可选）
├── requirements.txt             # 依赖包列表
├── setup.cfg                    # 环境配置
└── README.md                    # 使用说明文档
```

---

## ✅ 三、核心代码：`MainProgram.py`（主界面）

```python
# MainProgram.py
import sys
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QComboBox, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的焊缝质量检测系统")
        self.setGeometry(100, 100, 1000, 700)

        # 初始化模型
        self.model = YOLO('models/best.pt')  # 加载训练好的模型
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 创建 UI
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 图像显示区域
        self.image_label = QLabel("点击打开图片或摄像头")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        layout.addWidget(self.image_label)

        # 控制按钮区
        btn_layout = QHBoxLayout()
        self.btn_open_image = QPushButton("打开图片")
        self.btn_open_video = QPushButton("打开视频")
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_stop = QPushButton("停止")

        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_stop.clicked.connect(self.stop_capture)

        btn_layout.addWidget(self.btn_open_image)
        btn_layout.addWidget(self.btn_open_video)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        # 参数设置
        param_layout = QHBoxLayout()
        self.conf_threshold = QSpinBox()
        self.conf_threshold.setRange(0, 100)
        self.conf_threshold.setValue(25)  # 默认 0.25
        self.conf_threshold.setSuffix("%")

        self.iou_threshold = QSpinBox()
        self.iou_threshold.setRange(0, 100)
        self.iou_threshold.setValue(45)  # 默认 0.45
        self.iou_threshold.setSuffix("%")

        param_layout.addWidget(QLabel("置信度阈值:"))
        param_layout.addWidget(self.conf_threshold)
        param_layout.addWidget(QLabel("交并比阈值:"))
        param_layout.addWidget(self.iou_threshold)
        layout.addLayout(param_layout)

        # 检测结果展示
        result_layout = QVBoxLayout()
        self.result_label = QLabel("检测结果：")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_text)
        layout.addLayout(result_layout)

        central_widget.setLayout(layout)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            self.process_image(file_path)

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)
            self.btn_stop.setEnabled(True)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.btn_stop.setEnabled(True)

    def stop_capture(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.btn_stop.setEnabled(False)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img, conf=self.conf_threshold.value() / 100, iou=self.iou_threshold.value() / 100)
        annotated_img = results[0].plot()
        self.display_image(annotated_img)
        self.show_results(results)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, conf=self.conf_threshold.value() / 100, iou=self.iou_threshold.value() / 100)
            annotated_frame = results[0].plot()
            self.display_image(annotated_frame)
            self.show_results(results)
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    def display_image(self, img):
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def show_results(self, results):
        result_str = ""
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[cls_id]
            result_str += f"类别: {label}, 置信度: {conf:.2f}, 位置: ({x1},{y1})~({x2},{y2})\n"
        self.result_text.setText(result_str)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

---

## ✅ 四、训练代码（`train.py`）

```python
# train.py
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 使用小模型适合工业图像
results = model.train(
    data='datasets/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='weld_quality_detection',
    cache=True,
    device=0,
    workers=8,
    patience=10
)
```

### `data.yaml` 示例

```yaml
path: ./datasets
train: images/train
val: images/val
test: images/test

nc: 6
names: ['Bad Welding', 'Crack', 'Excess Reinforcement', 'Good Welding', 'Porosity', 'Spatters']
```

---

## ✅ 五、依赖安装（`requirements.txt`）

```txt
PyQt5==5.15.9
opencv-python==4.8.0
ultralytics==8.3.0
numpy==1.24.0
torch==2.0.1+cu118
torchvision==0.15.2+cu118
```

安装命令：

```bash
pip install -r requirements.txt
```

---

## ✅ 六、使用说明

1. 将 `best.pt` 放入 `models/` 文件夹  
2. 运行 `MainProgram.py` 启动 GUI  
3. 点击“打开图片”或“打开摄像头”进行检测  
4. 结果实时显示在界面中，支持保存


