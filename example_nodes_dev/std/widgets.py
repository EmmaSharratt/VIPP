import re

from ryven.NWENV import *

from qtpy.QtGui import QFont, QImage, QPixmap
from qtpy.QtCore import Qt, Signal, QEvent, QTimer
from qtpy.QtWidgets import (QPushButton, 
                            QComboBox, 
                            QSlider, 
                            QTextEdit, 
                            QPlainTextEdit, 
                            QWidget, 
                            QVBoxLayout, 
                            QLineEdit,
                            QLabel,
                            QFileDialog,                            
                            )
import cv2
import os


#-----------------DEVELOPED-----------------------
class OpenCVNodeSliderDev_MainWidget(MWB, QLabel, QSlider):
    def __init__(self, params):
        MWB.__init__(self, params)
        QLabel.__init__(self)
        self.resize(200, 200)

        QSlider.__init__(self, Qt.Horizontal)
        self.setRange(0, 1000)
        self.valueChanged.connect(self.value_changed)

    #image  
    def show_image(self, img):
        self.resize(500, 500)

        try:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        img_w = qt_image.width()
        img_h = qt_image.height()
        proportion = img_w / img_h
        self.resize(self.width() * proportion, self.height())
        qt_image = qt_image.scaled(self.width(), self.height())
        self.setPixmap(QPixmap(qt_image))
        self.node.update_shape()
        
    def clear_image(self):
        self.img.clear()

    #slider
    def value_changed(self, v):
        self.node.val = v/1000
        self.update_node()

    def get_state(self) -> dict:
        return {
            'val': self.value(),
        }

    def set_state(self, data: dict):
        self.setValue(data['val'])
    
    

    
#------------------ORIGINAL------------------------


class OpenCVNode_MainWidget(MWB, QLabel):
    def __init__(self, params):
        MWB.__init__(self, params)
        QLabel.__init__(self)

        self.resize(200, 200)

    def show_image(self, img):
        self.resize(500, 500)

        try:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        img_w = qt_image.width()
        img_h = qt_image.height()
        proportion = img_w / img_h
        self.resize(self.width() * proportion, self.height())
        qt_image = qt_image.scaled(self.width(), self.height())
        self.setPixmap(QPixmap(qt_image))
        # self.node.update_shape()
        
    def clear_image(self):
        self.image_label.clear()

        # self.img.clear()
        # self.node.update_shape()

# class QVBox_MainWidget(MWB, QWidget):
#     def __init__(self, params):
#         MWB.__init__(self, params)
#         QWidget.__init__(self)

#         self.img = QLabel(self)
#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setRange(0, 1000)
#         self.slider.valueChanged.connect(self.value_changed)

#         self.resize(200, 200)

#         self.setLayout(QVBoxLayout())
#         self.layout().addWidget(self.slider)
#         self.layout().addWidget(self.img)

#     def show_image(self, img):
#         self.resize(500, 500)

#         try:
#             rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         except cv2.error:
#             return

#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         img_w = qt_image.width()
#         img_h = qt_image.height()
#         proportion = img_w / img_h
#         self.resize(self.width() * proportion, self.height())
#         qt_image = qt_image.scaled(self.width(), self.height())
#         self.img.setPixmap(QPixmap(qt_image))
#         self.node.update_shape()     

#     def value_changed(self, v):
#         self.node.val = v/1000
#         self.update_node()

#     def get_state(self) -> dict:
#         return {
#             'val': self.value(),
#         }

#     def set_state(self, data: dict):
#         self.setValue(data['val'])


# class QvBoxDev_MainWidget(MWB, QWidget):
#     def __init__(self, params):
#         MWB.__init__(self, params)
#         QWidget.__init__(self)

#         self.resize(200, 200)

#         self.image_label = QLabel()
#         # self.image_label.resize(800, 800)

#         self.slider_label = QSlider(Qt.Horizontal)
#         self.slider_label.setRange(0, 1000)
#         self.slider_label.valueChanged.connect(self.value_changed)

#         self.text_label = QLabel("Slider Value: ")

#         self.layout = QVBoxLayout()
#         self.layout.addWidget(self.text_label)
#         self.layout.addWidget(self.slider_label)
#         self.layout.addWidget(self.image_label)
#         self.layout.setSpacing(0) 
#         self.setLayout(self.layout)
               

#     def show_image(self, img):
#         # self.resize(800,800)

#         try:
#             rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         except cv2.error:
#             return
        
#         # qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         qt_image = qt_image.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
#         self.image_label.setPixmap(QPixmap(qt_image))
#         self.image_label.setFixedSize(qt_image.size())


#         # h, w, ch = rgb_image.shape
#         # bytes_per_line = ch * w
#         # qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         # img_w = qt_image.width()
#         # img_h = qt_image.height()
#         # proportion = img_w / img_h
#         # self.resize(self.width() * proportion, self.height())
#         # qt_image = qt_image.scaled(self.width(), self.height())
#         # self.image_label.setPixmap(QPixmap(qt_image))
#         # self.node.update_shape()     
       
#     def value_changed(self, v):
#         self.node.val = v/1000
#         print(self.node.val)
#         self.update_node()

#     def get_state(self) -> dict:
#         return {
#             'val': self.slider_label.value(),
#         }

#     def set_state(self, data: dict):
#         self.slider_label.setValue(data['val'])
        # print(self.slider_label)

class QvBoxDev_MainWidget(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(200, 200)

        self.image_label = QLabel()
        # self.image_label.resize(800, 800)

        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(0, 1000)
        self.slider_label.valueChanged.connect(self.value_changed)

        self.text_label = QLabel('current ksize:')
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.text_label)
        self.layout1.addWidget(self.slider_label)
        self.layout1.addWidget(self.image_label)
        self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)
               

    def show_image(self, img):
        # self.resize(800,800)

        try:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return
        
        # qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(300, 300, Qt.KeepAspectRatio) #AspectRatioMode
        self.image_label.setPixmap(QPixmap(qt_image))
        
        
    def clear_img(self):
        if self.layout.indexOf(self.image_label) != -1:
                self.layout1.removeWidget(self.image_label)
                self.layout1.setSpacing(0)
                self.layout1.addStretch()
                # self.layout.setStretch(2, 0)  # Reset stretch factor when the image_label is removed
        self.image_label.clear()

        # h, w, ch = rgb_image.shape
        # bytes_per_line = ch * w
        # qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # img_w = qt_image.width()
        # img_h = qt_image.height()
        # proportion = img_w / img_h
        # self.resize(self.width() * proportion, self.height())
        # qt_image = qt_image.scaled(self.width(), self.height())
        # self.image_label.setPixmap(QPixmap(qt_image))
        # self.node.update_shape()     
       
    def value_changed(self, v):
        self.node.val = v/1000
        print(self.node.val)
        self.update_node()

    def get_state(self) -> dict:
        return {
            'val': self.slider_label.value(),
        }

    def set_state(self, data: dict):
        self.slider_label.setValue(data['val'])
        # print(self.slider_label)

class V2QvBoxDev_MainWidget(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(400, 400)

        self.image_label = QLabel()
        # self.image_label.resize(800, 800)

        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(0, 1000)
        self.slider_label.valueChanged.connect(self.value_changed)

        self.text_label = QLabel('current ksize:')
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.text_label)
        self.layout1.addWidget(self.slider_label)
        self.layout1.addWidget(self.image_label)
        self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)
               

    def show_image(self, img):
        # self.resize(800,800)

        try:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return
        
        # qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(self.width(), self.height(), Qt.KeepAspectRatio) #AspectRatioMode
        self.image_label.setPixmap(QPixmap(qt_image))
        
        
    def clear_img(self):
        if self.layout.indexOf(self.image_label) != -1:
                self.layout1.removeWidget(self.image_label)
                self.layout1.setSpacing(0)
                self.layout1.addStretch()
                # self.layout.setStretch(2, 0)  # Reset stretch factor when the image_label is removed
        self.image_label.clear()

        # h, w, ch = rgb_image.shape
        # bytes_per_line = ch * w
        # qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # img_w = qt_image.width()
        # img_h = qt_image.height()
        # proportion = img_w / img_h
        # self.resize(self.width() * proportion, self.height())
        # qt_image = qt_image.scaled(self.width(), self.height())
        # self.image_label.setPixmap(QPixmap(qt_image))
        # self.node.update_shape()     
       
    def value_changed(self, v):
        self.node.val = v/1000
        print(self.node.val)
        self.update_node()

    def get_state(self) -> dict:
        return {
            'val': self.slider_label.value(),
        }

    def set_state(self, data: dict):
        self.slider_label.setValue(data['val'])
        # print(self.slider_label)
        

    

class ButtonNode_MainWidget(QPushButton, MWB):

    def __init__(self, params):
        MWB.__init__(self, params)
        QPushButton.__init__(self)

        self.clicked.connect(self.update_node)


class ClockNode_MainWidget(MWB, QPushButton):

    def __init__(self, params):
        MWB.__init__(self, params)
        QPushButton.__init__(self)

        self.clicked.connect(self.node.toggle)


class LogNode_MainWidget(MWB, QComboBox):
    def __init__(self, params):
        MWB.__init__(self, params)
        QComboBox.__init__(self)

        self.addItems(self.node.targets)

        self.currentTextChanged.connect(self.text_changed)
        self.set_target(self.node.target)

    def text_changed(self, t):
        self.node.target = t

    def set_target(self, t):
        self.setCurrentText(t)


class SliderNode_MainWidget(MWB, QSlider):
    def __init__(self, params):
        MWB.__init__(self, params)
        QSlider.__init__(self, Qt.Horizontal)

        self.setRange(0, 1000)
        self.valueChanged.connect(self.value_changed)

    def value_changed(self, v):
        self.node.val = v/1000
        self.update_node()

    def get_state(self) -> dict:
        return {
            'val': self.value(),
        }

    def set_state(self, data: dict):
        self.setValue(data['val'])


class CodeNode_MainWidget(MWB, QTextEdit):
    def __init__(self, params):
        MWB.__init__(self, params)
        QTextEdit.__init__(self)

        self.setFont(QFont('Consolas', 9))
        self.textChanged.connect(self.text_changed)
        self.setFixedHeight(150)
        self.setFixedWidth(300)

    def text_changed(self):
        self.node.code = self.toPlainText()
        self.update_node()

    def get_state(self) -> dict:
        return {
            'text': self.toPlainText(),
        }

    def set_state(self, data: dict):
        self.setPlainText(data['text'])


class EvalNode_MainWidget(MWB, QPlainTextEdit):
    def __init__(self, params):
        MWB.__init__(self, params)
        QPlainTextEdit.__init__(self)

        self.setFont(QFont('Consolas', 9))
        self.textChanged.connect(self.text_changed)
        self.setMaximumHeight(50)
        self.setMaximumWidth(200)

    def text_changed(self):
        self.node.expression_code = self.toPlainText()
        self.update_node()

    def get_state(self) -> dict:
        return {
            'text': self.toPlainText(),
        }

    def set_state(self, data: dict):
        self.setPlainText(data['text'])


class InterpreterConsole(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.inp_line_edit = ConsoleInpLineEdit()
        self.output_text_edit = ConsoleOutDisplay()

        self.inp_line_edit.returned.connect(self.node.process_input)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.output_text_edit)
        self.layout().addWidget(self.inp_line_edit)

        self.last_hist_len = 0

    def interp_updated(self):

        if self.last_hist_len < len(self.node.hist):
            self.output_text_edit.appendPlainText('\n'.join(self.node.hist[self.last_hist_len:]))
        else:
            self.output_text_edit.clear()
            self.output_text_edit.setPlainText('\n'.join(self.node.hist))

        self.last_hist_len = len(self.node.hist)


class ConsoleInpLineEdit(QLineEdit):

    returned = Signal(str)

    def __init__(self):
        super().__init__()

        self.hist_index = 0
        self.hist = []

    def event(self, ev: QEvent) -> bool:

        if ev.type() == QEvent.KeyPress:

            if ev.key() == Qt.Key_Tab:
                self.insert(' '*4)
                return True

            elif ev.key() == Qt.Key_Backtab:
                ccp = self.cursorPosition()
                text_left = self.text()[:ccp]
                text_right = self.text()[ccp:]
                ends_with_tab = re.match(r"(.*)\s\s\s\s$", text_left)
                if ends_with_tab:
                    self.setText(text_left[:-4]+text_right)
                    self.setCursorPosition(ccp-4)
                    return True

            elif ev.key() == Qt.Key_Up:
                self.recall(self.hist_index - 1)
                return True

            elif ev.key() == Qt.Key_Down:
                self.recall(self.hist_index + 1)
                return True

            elif ev.key() == Qt.Key_Return:
                self.return_key()
                return True

        return super().event(ev)

    def return_key(self) -> None:
        text = self.text()
        for line in text.splitlines():
            self.record(line)
        self.returned.emit(text)
        self.clear()

    def record(self, line: str) -> None:
        """store line in history buffer and update hist_index"""

        self.hist.append(line)

        if self.hist_index == len(self.hist)-1 or line != self.hist[self.hist_index]:
            self.hist_index = len(self.hist)

    def recall(self, index: int) -> None:
        """select a line from the history list"""

        if len(self.hist) > 0 and 0 <= index < len(self.hist):
            self.setText(self.hist[index])
            self.hist_index = index


class ConsoleOutDisplay(QPlainTextEdit):
    def __init__(self):
        super().__init__()

        self.setReadOnly(True)
        self.setFont(QFont('Source Code Pro', 9))


export_widgets(
    ButtonNode_MainWidget,
    ClockNode_MainWidget,
    LogNode_MainWidget,
    SliderNode_MainWidget,
    CodeNode_MainWidget,
    EvalNode_MainWidget,
    InterpreterConsole,
    OpenCVNode_MainWidget,
    #--------------------
    OpenCVNodeSliderDev_MainWidget,
    QvBoxDev_MainWidget,
    V2QvBoxDev_MainWidget,
)
