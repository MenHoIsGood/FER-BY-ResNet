from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from statistics import mode
from PIL import ImageFont, ImageDraw, Image
import cv2
import time

matplotlib.use("Qt5Agg")
from recognition import *


class UI(object):

    def __init__(self, form, model):
        self.setup_ui(form)
        self.model = model

    def setup_ui(self, form):
        form.setObjectName("FER")
        form.resize(1200, 800)
        # 原图无图时显示的label
        self.label_raw_pic = QtWidgets.QLabel(form)
        self.label_raw_pic.setGeometry(QtCore.QRect(10, 30, 320, 240))
        self.label_raw_pic.setStyleSheet("background-color:#bbbbbb;")
        self.label_raw_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_raw_pic.setObjectName("label_raw_pic")
        # 原图下方分割线
        self.line1 = QtWidgets.QFrame(form)
        self.line1.setGeometry(QtCore.QRect(340, 30, 20, 431))
        self.line1.setFrameShape(QtWidgets.QFrame.VLine)
        self.line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line1.setObjectName("line1")
        # 作者说明label
        self.label_designer = QtWidgets.QLabel(form)
        self.label_designer.setGeometry(QtCore.QRect(20, 700, 180, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_designer.setFont(font)
        self.label_designer.setObjectName("label_designer")
        # 结果布局设置
        self.layout_widget = QtWidgets.QWidget(form)
        self.layout_widget.setGeometry(QtCore.QRect(10, 310, 320, 240))
        self.layout_widget.setObjectName("layoutWidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.layout_widget)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setObjectName("verticalLayout")
        # 右侧水平线
        self.line2 = QtWidgets.QFrame(self.layout_widget)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line2")
        self.vertical_layout.addWidget(self.line2)
        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.horizontal_layout.setObjectName("horizontalLayout")
        self.pushButton_select_img = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_img.setObjectName("pushButton_2")
        self.horizontal_layout.addWidget(self.pushButton_select_img)
        self.pushButton_select_video = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_video.setObjectName("pushButton_3")
        self.horizontal_layout.addWidget(self.pushButton_select_video)
        self.pushButton_select_camera = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_video.setObjectName("pushButton_4")
        self.horizontal_layout.addWidget(self.pushButton_select_camera)
        self.vertical_layout.addLayout(self.horizontal_layout)
        self.graphicsView = QtWidgets.QGraphicsView(form)
        self.graphicsView.setGeometry(QtCore.QRect(360, 210, 800, 500))
        self.graphicsView.setObjectName("graphicsView")
        self.label_result = QtWidgets.QLabel(form)
        self.label_result.setGeometry(QtCore.QRect(361, 21, 71, 16))
        self.label_result.setObjectName("label_result")
        self.label_emotion = QtWidgets.QLabel(form)
        self.label_emotion.setGeometry(QtCore.QRect(715, 21, 71, 16))
        self.label_emotion.setObjectName("label_emotion")
        self.label_emotion.setAlignment(QtCore.Qt.AlignCenter)
        self.label_bar = QtWidgets.QLabel(form)
        self.label_bar.setGeometry(QtCore.QRect(720, 170, 80, 180))
        self.label_bar.setObjectName("label_bar")
        self.line = QtWidgets.QFrame(form)
        self.line.setGeometry(QtCore.QRect(361, 150, 800, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_rst = QtWidgets.QLabel(form)
        self.label_rst.setGeometry(QtCore.QRect(700, 50, 100, 100))
        self.label_rst.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rst.setObjectName("label_rst")

        self.pushButton_select_img.clicked.connect(self.open_file_browser)
        self.pushButton_select_video.clicked.connect(self.open_video_browser)
        self.pushButton_select_camera.clicked.connect(self.open_camera)
        self.retranslate_ui(form)
        QtCore.QMetaObject.connectSlotsByName(form)

    def retranslate_ui(self, form):
        _translate = QtCore.QCoreApplication.translate
        form.setWindowTitle(_translate("FER", "FER"))
        self.label_raw_pic.setText(_translate("FER", "O(∩_∩)O"))
        self.pushButton_select_img.setText(_translate("FER", "选择图像"))
        self.pushButton_select_video.setText(_translate("FER", "选择视频"))
        self.pushButton_select_camera.setText(_translate("FER", "摄像头模式"))
        self.label_result.setText(_translate("FER", "识别结果"))
        self.label_emotion.setText(_translate("FER", "null"))
        self.label_bar.setText(_translate("FER", "各表情得分"))
        self.label_bar.adjustSize()
        self.label_rst.setText(_translate("FER", "Result"))

    def open_file_browser(self):
        # 加载模型
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片", directory="input",
                                                                     filter="All Files (*);;Text Files (*.txt)")
        # 显示原图
        if file_name is not None and file_name != "":
            emotion, possibility = predict_expression(file_name, self.model)
            self.show_raw_img(file_name)
            self.show_results(emotion, possibility)

    def show_raw_img(self, filename):
        img = cv2.imread('output/output1.jpg')
        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 240))
        self.label_raw_pic.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                         QtGui.QImage.Format_RGB888)))

    def show_results(self, emotion, possibility):
        # 显示表情名
        self.label_emotion.setText(QtCore.QCoreApplication.translate("FER", emotion))
        self.label_emotion.adjustSize()
        # 显示emoji
        if emotion != '':
            img = cv2.imread('assets/icons/' + str(emotion) + '.png')
            frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (100, 100))
            self.label_rst.setPixmap(QtGui.QPixmap.fromImage(
                QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                             QtGui.QImage.Format_RGB888)))
        # else:
        #     self.label_rst.setText(QtCore.QCoreApplication.translate("Form", "no result"))
        # 显示直方图
        self.show_bars(list(possibility))

    def show_bars(self, possbility):
        dr = MyFigureCanvas()
        dr.draw_(possbility)
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)
        self.graphicsView.setScene(graphicscene)
        self.graphicsView.show()

    # 选择视频
    def open_video_browser(self):

        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取视频", directory="video",
                                                                     filter="All Files (*);;Text Files (*.txt)")

        if file_name is not None and file_name != "":
            # face_detection = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

            frame_window = 10

            # 表情标签 对应fer2013
            emotion_labels = {0: '生气', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '中性', 5: '伤心', 6: '惊讶'}
            emotion_labels_ck = {0: '生气', 1: '蔑视', 2: '厌恶', 3: '恐惧', 4: '开心', 5: '伤心', 6: '惊讶'}
            emotion_window = []

            video_capture = cv2.VideoCapture(file_name)

            cv2.startWindowThread()
            cv2.namedWindow('window_frame(press q to exit)')

            # 使用Pillow库创建字体对象
            font_path = 'font/simsun.ttc'  # 替换为你的中文字体文件路径
            font_size = 24
            font = ImageFont.truetype(font_path, font_size)
            # frames = []

            while True:
                # 读取一帧
                _, frame = video_capture.read()
                # frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
                if frame is None:
                    break
                frame = frame.copy()
                # 获得灰度图，并且在内存中创建一个图像对象
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 获取当前帧中的全部人脸
                # faces = face_detection.detectMultiScale(gray, 1.3, 5)
                boxes_c = infer_image(frame)

                if boxes_c is not None:
                    # 对于所有发现的人脸
                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]
                        # score = boxes_c[i, 4]
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                        # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
                        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                                      (corpbbox[2], corpbbox[3]), (84, 255, 159), 2)
                        # 获取人脸图像
                        # face = gray[y:y + h, x:x + w]
                        face = gray[corpbbox[1]:corpbbox[3], corpbbox[0]:corpbbox[2]]

                        try:
                            # shape变为(48,48)
                            face = cv2.resize(face, (48, 48))
                        except:
                            continue

                        # 扩充维度，shape变为(1,48,48,1)
                        # 将（1，48，48，1）转换成为(1,1,48,48)
                        face = np.expand_dims(face, 0)
                        face = np.expand_dims(face, 0)

                        # 人脸数据归一化，将像素值从0-255映射到0-1之间
                        face = preprocess_input(face)
                        new_face = torch.from_numpy(face)
                        new_new_face = new_face.float().requires_grad_(False)

                        # 调用我们训练好的表情识别模型，预测分类
                        emotion_arg = np.argmax(self.model.forward(new_new_face).detach().numpy())
                        emotion = emotion_labels[emotion_arg]

                        emotion_window.append(emotion)

                        if len(emotion_window) >= frame_window:
                            emotion_window.pop(0)

                        try:
                            # 获得出现次数最多的分类
                            emotion_mode = mode(emotion_window)
                        except:
                            continue

                        # 转换图像格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame)

                        # 在图像上绘制中文
                        draw = ImageDraw.Draw(pil_image)
                        text = emotion_mode
                        text_position = (corpbbox[0], corpbbox[1] - 30)
                        text_color = (255, 0, 0)
                        draw.text(text_position, text, font=font, fill=text_color)

                        # 将图像转回OpenCV格式
                        frame = np.array(pil_image)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    # 将图片从内存中显示到屏幕上
                    cv2.imshow('window_frame(press q to exit)', frame)
                except:
                    continue

                # 按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()

    # 打开摄像头
    def open_camera(self):

        # face_detection = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

        frame_window = 10

        # 表情标签 对应fer2013
        emotion_labels = {0: '生气', 1: '厌恶', 2: '恐惧', 3: '开心', 4: '中性', 5: '伤心', 6: '惊讶'}
        emotion_labels_ck = {0: '生气', 1: '蔑视', 2: '厌恶', 3: '恐惧', 4: '开心', 5: '伤心', 6: '惊讶'}
        emotion_window = []

        video_capture = cv2.VideoCapture(0)

        cv2.startWindowThread()
        cv2.namedWindow('window_frame(press q to exit)')

        # 使用Pillow库创建字体对象
        font_path = 'font/simsun.ttc'  # 替换为你的中文字体文件路径
        font_size = 24
        font = ImageFont.truetype(font_path, font_size)
        # frames = []

        while True:
            # 读取一帧
            _, frame = video_capture.read()
            # frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
            frame = frame.copy()
            # 获得灰度图，并且在内存中创建一个图像对象
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 获取当前帧中的全部人脸
            # faces = face_detection.detectMultiScale(gray, 1.3, 5)
            boxes_c = infer_image(frame)

            if boxes_c is not None:
                # 对于所有发现的人脸
                for i in range(boxes_c.shape[0]):
                    bbox = boxes_c[i, :4]
                    # score = boxes_c[i, 4]
                    corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                    # 在脸周围画一个矩形框，(255,0,0)是颜色，2是线宽
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (84, 255, 159), 2)
                    cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                                  (corpbbox[2], corpbbox[3]), (84, 255, 159), 2)
                    # 获取人脸图像
                    # face = gray[y:y + h, x:x + w]
                    face = gray[corpbbox[1]:corpbbox[3], corpbbox[0]:corpbbox[2]]

                    try:
                        # shape变为(48,48)
                        face = cv2.resize(face, (48, 48))
                    except:
                        continue

                    # 扩充维度，shape变为(1,48,48,1)
                    # 将（1，48，48，1）转换成为(1,1,48,48)
                    face = np.expand_dims(face, 0)
                    face = np.expand_dims(face, 0)

                    # 人脸数据归一化，将像素值从0-255映射到0-1之间
                    face = preprocess_input(face)
                    new_face = torch.from_numpy(face)
                    new_new_face = new_face.float().requires_grad_(False)

                    # 调用我们训练好的表情识别模型，预测分类
                    emotion_arg = np.argmax(self.model.forward(new_new_face).detach().numpy())
                    emotion = emotion_labels[emotion_arg]

                    emotion_window.append(emotion)

                    if len(emotion_window) >= frame_window:
                        emotion_window.pop(0)

                    try:
                        # 获得出现次数最多的分类
                        emotion_mode = mode(emotion_window)
                    except:
                        continue

                    # 转换图像格式
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame)

                    # 在图像上绘制中文
                    draw = ImageDraw.Draw(pil_image)
                    text = emotion_mode
                    text_position = (corpbbox[0], corpbbox[1] - 30)
                    text_color = (255, 0, 0)
                    draw.text(text_position, text, font=font, fill=text_color)

                    # 将图像转回OpenCV格式
                    frame = np.array(pil_image)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                # 将图片从内存中显示到屏幕上
                cv2.imshow('window_frame(press q to exit)', frame)
            except:
                continue

            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


class MyFigureCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

    def draw_(self, possibility):
        x = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
        self.axes.bar(x, possibility, align='center')
