import re
import sys
import numpy as np 
from ryven.NWENV import *
from ryvencore_qt.src.flows.nodes import *

from ryvencore_qt.src.flows.nodes.PortItemInputWidgets import Integer_IW

# Failed attempts:  :)
# import sys
# # sys.path.append('C:/Users/dell/OneDrive/Documents/Em/2023/Skripsie/Development/venvs/sk_env2/Lib/site-packages/ryvencore_qt/src/flows/nodes')
# # # Now you can import the module from the other folder
# # from PortItemInputWidgets import IWB
# from ryvencore.NodePort import NodeInput, NodeOutput

# sys.path.append('C:/Users/dell/AppData/Local/Programs/Python/Python310/Lib/site-packages/ryvencore_qt/src/flows/nodes')
# # Now you can import the module from the other folder
# from PortItemInputWidgets import IWB

# import sys
# sys.path.append('/ryvencore_qt/src/flows/nodes')  
# from PortItemInputWidgets import *
# from .ryvencore_qt.src.PortItemInputWidget.py import IWB

# from ryvencore_qt.src.flows.nodes.PortItemInputWidget import IWB
from qtpy.QtGui import QFont, QImage, QPixmap, QColor
from qtpy.QtCore import Qt, Signal, QEvent, QTimer, QObject
from qtpy.QtWidgets import (QPushButton, 
                            QComboBox, 
                            QSlider, 
                            QTextEdit, 
                            QPlainTextEdit, 
                            QWidget, 
                            QVBoxLayout, 
                            QHBoxLayout,
                            QLineEdit,
                            QLabel,
                            QFileDialog,     
                            QCheckBox,     
                            QDoubleSpinBox,
                            QSpinBox,    
                            QAbstractSpinBox,  
                            QMainWindow, 
                            QTableWidgetItem,
                            QTableWidget,    
                            QApplication, 
                            QMainWindow,    
                            QDialog,
                            QScrollArea, 
                            QSizePolicy,
                            QAbstractItemView                           
                            )
import cv2
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# C:\Users\dell\OneDrive\Documents\Em\2023\Skripsie\Development\venvs\sk_env2\Lib\site-packages\ryven\example_nodes_dev\std\widgets.py
# C:\Users\dell\OneDrive\Documents\Em\2023\Skripsie\Development\venvs\sk_env2\Lib\site-packages\ryvencore_qt\src\flows\nodes\PortItemInputWidgets.py"
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return

        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
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

#adapted resolution
class OpenCVNode_MainWidget(MWB, QLabel):
    def __init__(self, params):
        MWB.__init__(self, params)
        QLabel.__init__(self)

    def show_image(self, img):

        try:
            print("slice",img.shape)
            if len(img.shape) == 2:
                # Grayscale image
                # img = (img*255).astype('uint8')
                qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*2, QImage.Format_Grayscale16)
                # print("came here")
            else:
                # RGB image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate the target size for scaling
            scale_factor = 0.8  # Increase the scaling factor for clarity
            target_width = int(qt_image.width() * scale_factor)
            target_height = int(qt_image.height() * scale_factor)
            
            # qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
            self.setPixmap(QPixmap.fromImage(qt_image))
            self.resize(target_width, target_height)
            self.node.update_shape()
        except Exception as e:
            print("Error:", e)

        # try:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # except cv2.error:
        #     return

        # h, w, ch = img.shape
        # aspect_ratio = w / h  # Calculate the aspect ratio of the image

        # # Calculate the new dimensions for the widget based on the aspect ratio
        # new_widget_width = 300  # You can set the width to a desired value
        # new_widget_height = int(new_widget_width / aspect_ratio)

        # self.resize(new_widget_width, new_widget_height)

        # qt_image = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        # qt_image = qt_image.scaled(new_widget_width, new_widget_height, Qt.KeepAspectRatio)
        # self.setPixmap(QPixmap.fromImage(qt_image))

        # self.node.update_shape()

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
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         except cv2.error:
#             return

#         h, w, ch = img.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         except cv2.error:
#             return
        
#         # qt_image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
#         h, w, ch = img.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         qt_image = qt_image.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
#         self.image_label.setPixmap(QPixmap(qt_image))
#         self.image_label.setFixedSize(qt_image.size())


#         # h, w, ch = img.shape
#         # bytes_per_line = ch * w
#         # qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return
        
        # qt_image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        qt_image = qt_image.scaled(300, 300, Qt.KeepAspectRatio) #AspectRatioMode
        self.image_label.setPixmap(QPixmap(qt_image))
        
        
        
    def clear_img(self):
        if self.layout.indexOf(self.image_label) != -1:
                self.layout1.removeWidget(self.image_label)
                self.layout1.setSpacing(0)
                self.layout1.addStretch()
                # self.layout.setStretch(2, 0)  # Reset stretch factor when the image_label is removed
        self.image_label.clear()
    
    

        # h, w, ch = img.shape
        # bytes_per_line = ch * w
        # qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return
        
        # qt_image = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Calculate the target size for scaling
        scale_factor = 0.4  # Adjust the scaling factor as needed
        target_width = int(w * scale_factor)
        target_height = int(h * scale_factor)
        qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)

        self.image_label.setPixmap(QPixmap(qt_image))
        self.node.update_shape()
        
        
    def clear_img(self):
        if self.layout.indexOf(self.image_label) != -1:
                self.layout1.removeWidget(self.image_label)
                self.layout1.setSpacing(0)
                self.layout1.addStretch()
                # self.layout.setStretch(2, 0)  # Reset stretch factor when the image_label is removed
        self.image_label.clear() 
       
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

#coupled with slider v6, preview works well
#but does not update shape immediately after unchecked
class V3QvBoxDev_MainWidget(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(200, 200)

        self.image_label = QLabel()
        # self.image_label.resize(800, 800)
        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(0, 1000)
        self.slider_label.valueChanged.connect(self.value_changed)
        #sigmaX slider
        self.sliderX_label = QSlider(Qt.Horizontal)
        self.sliderX_label.setRange(0, 1000)
        self.sliderX_label.valueChanged.connect(self.value_changed)

        self.ksize_label = QLabel('adjust ksize:')
        self.sigmaX_label = QLabel('adjust SigmaX:')
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.slider_label)
        self.layout1.addWidget(self.sigmaX_label)
        self.layout1.addWidget(self.sliderX_label)
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
        # Calculate the target size for scaling
        scale_factor = 0.4  # Adjust the scaling factor as needed
        target_width = int(w * scale_factor)
        target_height = int(h * scale_factor)
        qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
        print("H:", target_height, "W:", target_width)
        self.image_label.setPixmap(QPixmap(qt_image))
        self.resize(200, 200)
        self.node.update_shape()
        print('Update Shape:',  print(self.width(), self.height()))
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        # Convert the QImage to QPixmap and set it to a QLabel
        # pixmap = QPixmap.fromImage(clr_img)
        # self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        # self.update_node()
        # self.node.update()  #--Does not work well (recursion)
        # self.node.repaint()

    # def preview_input_changed(self):
    #     print("preview_input_changed")
    #     self.update_node()
       
    def value_changed(self, v):  #own method
        #val -> ksize
        self.node.val = v/1000
        print(self.node.val)
        #sigmaX 
        #if this is removed, the size of the node no longer changes
        self.update_node()  
        

    def get_state(self) -> dict:
        return {
            'val': self.slider_label.value(),
        }

    def set_state(self, data: dict):
        self.slider_label.setValue(data['val'])
        # print(self.slider_label)       

#--------------------------------------------------------------------------------
#Try get update automatically after uncheck preview
# sigmaX, SigmaY 
class V4QvBoxDev_MainWidget(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(200, 200)
        
        #Added Widget -----------------------------------------------
        self.image_label = QLabel()
        # self.image_label.resize(800, 800)
        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(0, 1000)
        #sigmaX slider
        self.sliderX_label = QSlider(Qt.Horizontal)
        self.sliderX_label.setRange(0, 1000)
        
        self.ksize_label = QLabel('adjust ksize:')
        self.sigmaX_label = QLabel('adjust SigmaX:')
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.slider_label)
        self.layout1.addWidget(self.sigmaX_label)
        self.layout1.addWidget(self.sliderX_label)
        self.layout1.addWidget(self.image_label)
        self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)
        #Access input text-----------------------------------------------
        # Instance of InputWidgetBase (IWB) for ksize input
        input_params = (self.slider_label, None, self.node, self.node_item)
        self.ksize_input_widget_base = IWB(input_params)

        #Instance of InputWidgetBase (IWB) for sigmaX input
        input_params = (self.sliderX_label, None, self.node, self.node_item)
        self.sigmaX_input_widget_base = IWB(input_params)

        #Slider value changes  connected to respective IWB instances
        self.slider_label.valueChanged.connect(self.ksize_value_changed)
        self.sliderX_label.valueChanged.connect(self.sigmaX_value_changed)
        self.sliderX_label.valueChanged.connect(self.sigmaX_value_changed)
       

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
        # Calculate the target size for scaling
        scale_factor = 0.4  # Adjust the scaling factor as needed
        target_width = int(w * scale_factor)
        target_height = int(h * scale_factor)
        qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
        print("H:", target_height, "W:", target_width)
        self.image_label.setPixmap(QPixmap(qt_image))
        self.resize(200, 200)
        self.node.update_shape()
        print('Update Shape:',  print(self.width(), self.height()))
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        # Convert the QImage to QPixmap and set it to a QLabel
        # pixmap = QPixmap.fromImage(clr_img)
        # self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        # self.update_node()
        # self.node.update()  #--Does not work well (recursion)
        # self.node.repaint()

    # def preview_input_changed(self):
    #     print("preview_input_changed")
    #     self.update_node()
       
    def ksize_value_changed(self, v):  #own method
        #val = ksize
        self.node.val = v/1000
        print(self.node.val)
        self.ksize_input_widget_base.val_update_event(self.node.val)
        # print('Updated ksize textbox !!!:', self.ksize_input_widget_base.get_state())
        # self.input_widget(1).setText(str(self.node.val))
        self.update_node()  
    
    def sigmaX_value_changed(self, v):  #own method
        #sigmaX
        self.node.sigX = v/1000 
        self.sigmaX_input_widget_base.val_update_event(v)
        #if this is removed, the size of the node no longer changes
        self.update_node()  
        

    def get_state(self) -> dict:
        return {
            'val': self.slider_label.value(),
        }

    def set_state(self, data: dict):
        self.slider_label.setValue(data['val'])
        # print(self.slider_label)       


class V5QvBoxDev_MainWidget(MWB, QWidget):
       #define Signal
    kValueChanged = Signal(int)
    previewState = Signal(bool)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default = 5
        default_range = default*2
                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('ksize:')
        self.k_size_input = QSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range)    
        self.slider_label.setValue(default)
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaX:')
        self.sigmaX_size_input = QDoubleSpinBox()
        #Xslider
        self.sliderX_label = QSlider(Qt.Horizontal)
        # self.sliderX_label.setRange(0, 1000)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.k_size_input)
        self.layout1.addWidget(self.slider_label)
        # self.layout1.addWidget(self.sigmaX_label)
        # self.layout1.addWidget(self.sigmaX_size_input)
        # self.layout1.addWidget(self.sliderX_label)
        self.layout1.addWidget(self.preview_label)
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        self.range1 = self.k_size_input.value()*2
        #Signals 
        # Spinbox triggers
        self.k_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
          
    #slot functions
    def the_spinbox_was_changed(self):    # v: value emitted by a signal.
        self.slider_label.setRange(0, self.k_size_input.value()*2)  
        self.slider_label.setValue(self.k_size_input.value())  
        self.k = self.k_size_input.value()
        self.kValueChanged.emit(self.k)
        #debug
        # print(f'slider:{self.slider_label.value()}')

    def the_slider_was_changed(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.k_size_input.setValue(v)
        self.k = int(v)
        self.kValueChanged.emit(self.k)
        #debug
        # print(self.slider_label.value())
        # print(self.slider_label.value())
        # print(f'y:{int(y)}')

    # def updateImage(self):
    #     v = int(self.k)
    #     return cv2.blur(
    #         src=self.node.input(0).img,
    #         ksize=(v,v),
    #             )
    
    # def val_update_event(self, val):
    #     """triggered when input is connected and received new data;
    #     displays the data in the widget (without updating)"""
    #     #If input is connected 
    #     self.block = True
    #     try:
    #         self.setValue(val)
    #     except Exception as e:
    #         pass
    #     finally:
    #         self.block = False

    # def set_state(self, data: dict):
    #     # just show value, DO NOT UPDATE
    #     self.setValue(data['val'])
    #     # self.val_update_event(data['text'])
        
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
        # Calculate the target size for scaling
        scale_factor = 0.4  # Adjust the scaling factor as needed
        target_width = int(w * scale_factor)
        target_height = int(h * scale_factor)
        qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
        print("H:", target_height, "W:", target_width)
        self.image_label.setPixmap(QPixmap(qt_image))
        self.resize(100, 100)
        self.node.update_shape()
        print('Update Shape:',  print(self.width(), self.height()))
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        # Convert the QImage to QPixmap and set it to a QLabel
        # pixmap = QPixmap.fromImage(clr_img)
        # self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        # self.update_node()
        # self.node.update()  #--Does not work well (recursion)
        # self.node.repaint()
        print(self.k_size_input.value())

    # def preview_input_changed(self):
    #     print("preview_input_changed")
    #     self.update_node()
       
    def ksize_value_changed(self, v):  #own method
        #val = ksize
        self.node.val = v/1000
        print(self.node.val)
        self.update_node()  
        
    
    def sigmaX_value_changed(self, v):  #own method
        #sigmaX
        self.node.sigX = v/1000 
        #if this is removed, the size of the node no longer changes
        self.update_node()  
    
    def get_state(self) -> dict:
        return {
            'val': self.value(),
        }

    def set_state(self, data: dict):
        self.k_size_input.setValue(data['val'])

    # def get_state(self) -> dict:
    #     return {
    #         'val': self.slider_label.value(),
    #     }

    # def set_state(self, data: dict):
    #     self.slider_label.setValue(data['val'])
    #     # print(self.slider_label)     

# Pipeline Widegets -----------------------------------------
# //////////////////////////////////////////////////////////
class ChooseFileInputWidget(IWB, QPushButton):

    path_chosen = Signal(str)

    def __init__(self, params):
        IWB.__init__(self, params)
        QPushButton.__init__(self, "Select")

        self.clicked.connect(self.button_clicked)

    def button_clicked(self):
        self.file_path = QFileDialog.getOpenFileName(self, 'Select image')[0]
        try:
            self.file_path = os.path.relpath(self.file_path)
        except ValueError:
            return

        self.path_chosen.emit(self.file_path)
    

class Widget_Base(QWidget):
    # update image ---------------------------------------------
    def show_image(self, img):
        # self.resize(800,800)

        try:
            if len(img.shape) == 2:
                # Grayscale image
                qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*2, QImage.Format_Grayscale16)
                # print("came here for Sliderwidget")
            else:
                # RGB image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate the target size for scaling
            scale_factor = 0.7  # Increase the scaling factor for clarity
            target_width = int(qt_image.width() * scale_factor)
            # Use scaledToWidth to reduce the size while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaledToWidth(target_width)
            
            # Set the scaled pixmap
            self.image_label.setPixmap(scaled_pixmap)
            
            # Resize the widget to match the pixmap size
            self.resize(scaled_pixmap.width(), scaled_pixmap.height())
            
            # Ensure that any necessary updates are performed
            self.node.update_shape()
            
        except Exception as e:
            print("Error:", e)

class Widget_Base8(QWidget):
    # update image ---------------------------------------------

    # the node emits the channel dictionary to the widget which will be used in the show_img / assign method
    def channels(self, channels_dict):
        self.stack_dict = channels_dict
        print("came to channels", self.stack_dict)

    def assign_channels_RGB(self, img):
        single_chan = img[:, :, 0]
        # Initialize RGB channels with zeros
        red_channel = np.zeros_like(single_chan)
        green_channel = np.zeros_like(single_chan)
        blue_channel = np.zeros_like(single_chan)

        for color, channel_value in self.stack_dict["colour"].items():
            # Check if the channel is part of the image
            if channel_value != 100:
                # Assign channels based on the color
                if color == "red":
                    red_channel += img[:, :, channel_value]
                elif color == "green":
                    green_channel += img[:, :, channel_value]
                elif color == "blue":
                    blue_channel += img[:, :, channel_value]
                elif color == "cyan":
                    cyan_channel = img[:, :, channel_value]
                    green_channel += cyan_channel
                    blue_channel += cyan_channel
                elif color == "magenta":
                    magenta_channel = img[:, :, channel_value]
                    red_channel += magenta_channel
                    blue_channel += magenta_channel
                elif color == "yellow":
                    yellow_channel = img[:, :, channel_value]
                    red_channel += yellow_channel
                    green_channel += yellow_channel

        # Clip values to ensure they remain within the valid range [0, 255]
        red_channel = np.clip(red_channel, 0, 255)
        green_channel = np.clip(green_channel, 0, 255)
        blue_channel = np.clip(blue_channel, 0, 255)

        # Combine channels back into image data
        rgb_image_stack = np.stack([red_channel, green_channel, blue_channel], axis=2)

        return rgb_image_stack
    
    def clr_img(self):
        # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        # print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape()
    
    # original image. If 3 channels, default RGB, may need to assign the custom channels 
    def show_image(self, orignial_img):
        # self.resize(800,800)
        # print(f"DIMENSION WIDGET {img.shape[-1]}")

        try:
            # Grayscale image
            if orignial_img.shape[-1] == 1:
                qt_image = QImage(orignial_img.data, orignial_img.shape[1], orignial_img.shape[0], orignial_img.shape[1], QImage.Format_Grayscale8)

            # 3 colour channels (default RGB, may need to assign the custom channels)
            elif orignial_img.shape[-1] == 3:
                # Assign the channels to the image 
                self.RGB_img = self.assign_channels_RGB(orignial_img)
                h, w, ch = self.RGB_img.shape
                bytes_per_line = ch * w
                qt_image = QImage(self.RGB_img.data, w, h, bytes_per_line, QImage.Format_RGB888) #Format_RGB888

            # not sure if this is necesssary
            # note rgb image is now assigned custom channels 
            # may need to adjust code for 4 channel image (take original_img and apply)   
            elif orignial_img.shape[-1] == 4:
                # self.clr_img()
                print("RGBA image")
                self.RGB_img = self.assign_channels_RGB(orignial_img)
                print("rgb shape", self.RGB_img.shape)
                # now treat as 3D image
                h, w, ch = self.RGB_img.shape
                bytes_per_line = ch * w
                qt_image = QImage(self.RGB_img.data, w, h, bytes_per_line, QImage.Format_RGB888) #Format_RGB888

                # h, w, ch = self.RGB_img.shape
                # #print(f"ch: {ch}")
                # bytes_per_line = ch * 4
                # qt_image = QImage(self.RGB_img.data, w, h, QImage.Format_RGBA8888) #Format_RGB888
            if qt_image is not None:
                # Calculate the target size for scaling
                scale_factor = 0.7  # Increase the scaling factor for clarity
                if qt_image.width() < 400:
                    scale_factor = 1
                if qt_image.width() > 900:
                    scale_factor = 0.5
                target_width = int(qt_image.width() * scale_factor)
                # Use scaledToWidth to reduce the size while maintaining aspect ratio
                scaled_pixmap = QPixmap.fromImage(qt_image).scaledToWidth(target_width)
                
                # Set the scaled pixmap
                self.image_label.setPixmap(scaled_pixmap)
                
                # Resize the widget to match the pixmap size
                self.resize(scaled_pixmap.width(), scaled_pixmap.height())
                
                # Ensure that any necessary updates are performed
                self.node.update_shape()
            
        except Exception as e:
            print("Error:", e)
        

'''
Move up later
Initially from special nodes
'''

#New Slider Guas design
class CVImage:
    """
    The OpenCV Mat(rix) data type seems to have overridden comparison operations to perform element-wise comparisons
    which breaks ryverncore-internal object comparisons.
    To avoid this, I'll simply use this wrapper class and recreate a new object every time for now, so ryvencore
    doesn't think two different images are the same.
    """

    def __init__(self, img):
        self.img = img


class Read_Image_MainWidget(MWB, QWidget):
    path_chosen = Signal(str)
    ValueChanged1 = Signal(int)  #time instance
    ValueChanged2 = Signal(int)  #z-slice (depth)
    released1 = Signal(int)
    # released2 = Signal(int)
    dict_widg = Signal(dict)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        default = 5
        default_range = 6
        self.val1 = default
        self.val2 = 3
        # update image when confirmed 
        self.update_img = 0

        self.old_choice_array = [None, None, None, None, None, None]
        self.color_name = [None, None, None, None, None, None] 
        self.dropdowns = []
        self.stack_dict = {
            "time_step": 0,
            "colour": {
                "red": 100,
                "green": 100,
                "blue": 100,
                "cyan": 100,
                "yellow": 100,
                "magenta": 100                
            }
        }
        self.temp_dict = {
            "time_step": 0,
            "colour": {
                "red": 100,
                "green": 100,
                "blue": 100,
                "cyan": 100,
                "yellow": 100,
                "magenta": 100                
            }
        }

        self.color_codes = {
        'red': 1,
        'green': 2,
        'blue': 3,
        #CHANGED
        'cyan': 4,
        'yellow': 5,
        'magenta': 6
        }
        
        self.select_path = QPushButton("Select single file")

        self.shape_label = QLabel()
        self.shape_label.setStyleSheet('background-color: #1E242A; color: white; font-size: 14px;')
        self.input_label1 = QLabel('frame (time instance):')
        self.input_label1.setStyleSheet('font-size: 14px;')
        self.input1 = QSpinBox()
        self.input1.setValue(default)
        self.input1.setKeyboardTracking(False)
        self.input1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #timeseries slider
        self.slider_label1 = QSlider(Qt.Horizontal)
        self.slider_label1.setRange(1, default_range)    
        self.slider_label1.setSingleStep(1)
        self.slider_label1.setValue(default)

        #z-stack------------
        self.input_label2 = QLabel('z-slice:')
        self.input_label2.setStyleSheet('font-size: 14px;')
        self.input2 = QSpinBox()
        self.input2.setValue(default)
        self.input2.setKeyboardTracking(False)
        self.input2.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #zstack slider
        self.slider_label2 = QSlider(Qt.Horizontal)   #MAKE VERTICLAL
        self.slider_label2.setRange(1, default_range)    
        # self.slider_label2.setSingleStep(2)
        self.slider_label2.setValue(default)

        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        longest_word_width = max(len(word) for word in self.color_codes.keys())
        col1_layout = QVBoxLayout()
        col2_layout = QVBoxLayout()

        # Add dropdowns to two columns
        for i, color in enumerate(self.color_codes.keys()):
            label = QLabel(f"Channel {i}:")
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label.setMinimumWidth(longest_word_width)
            self.dropdown_box = QComboBox()
            self.dropdown_box.addItem("Select")
            self.dropdown_box.setStyleSheet("QComboBox QAbstractItemView { background-color: #091C7F }")
            for color_code in self.color_codes.keys():
                self.dropdown_box.addItem(color_code)
            # Signal
            self.dropdown_box.currentIndexChanged.connect(lambda index, i=i, diction=self.temp_dict: self.handle_selection(index, i, diction))
            # self.dropdown_box is a single channel QComboBox()
            # self.dropdowns is a list of the QComboBox()'s
            self.dropdowns.append(self.dropdown_box)
            print(f"self dropdowns {self.dropdowns}")
            
            row_layout = QHBoxLayout()  # Create a QHBoxLayout for each row
            row_layout.addWidget(label)
            row_layout.addWidget(self.dropdown_box)
            
            # create two coloumns 
            if i < len(self.color_codes)/2:
                col1_layout.addLayout(row_layout)  # Add the row layout to the column layout
            else:
                col2_layout.addLayout(row_layout)  # Add the row layout to the column layout

        # Add column layouts to the main layout
        self.layout1.addWidget(self.select_path)
        col_layout = QHBoxLayout()
        col_layout.addLayout(col1_layout)
        col_layout.addLayout(col2_layout)
        self.layout1.addLayout(col_layout)

        self.button_layout = QHBoxLayout()
        # Confrim checkbox
        self.checkbox = QCheckBox("Confirm channel selection")
        self.checkbox.setStyleSheet("color: #FF2C2C;")
        self.checkbox.stateChanged.connect(lambda state,  diction = self.temp_dict : self.update_dict(state, diction))
        self.button_layout.addWidget(self.checkbox)


        self.clear_button = QPushButton("Clear Channel Choices")
        self.clear_button.clicked.connect(lambda: self.clear_choices(self.stack_dict, self.temp_dict))
        self.button_layout.addWidget(self.clear_button)
        self.layout1.addLayout(self.button_layout)
        
        #add widgets
            #shape message
        self.layout1.addWidget(self.shape_label)
            # z-stack
        self.layout1.addWidget(self.input_label2)
        self.layout1.addWidget(self.input2)
        self.layout1.addWidget(self.slider_label2)
            # time
        self.layout1.addWidget(self.input_label1)
        self.layout1.addWidget(self.input1)
        self.layout1.addWidget(self.slider_label1)
            # Image
        self.layout1.addWidget(self.image_label)        

        self.setLayout(self.layout1)
        # self.reset_widg(1)
        

        # Signals -------------------------------------------------
        # Select path
        self.select_path.clicked.connect(self.button_clicked)
        # Spinbox triggers
        self.input1.editingFinished.connect(self.the_spinbox1_was_changed)  #USER ONLY
        self.input2.editingFinished.connect(self.the_spinbox2_was_changed)
        # Slider triggers
        self.slider_label1.sliderMoved.connect(self.the_slider1_was_changed)
        self.slider_label2.sliderMoved.connect(self.the_slider2_was_changed)
        self.slider_label1.sliderReleased.connect(self.slider1_released)
        # self.slider_label2.sliderReleased.connect(self.slider2_released)

    def button_clicked(self):
        self.file_path = QFileDialog.getOpenFileName(self, 'Select image')[0]
        try:
            self.file_path = os.path.relpath(self.file_path)
        except ValueError:
            return

        self.path_chosen.emit(self.file_path)
        self.checkbox.setChecked(False)
        self.clear_choices(self.stack_dict, self.temp_dict)
    # Drop down channel selcetion ---------------------------------------

    def handle_selection(self, index, channel_index, dicttemp):
        print(f"current text {self.dropdowns[channel_index].currentText()}")
        if index != 0:
            # update the specific channel
            
            print(f"old choice array{self.old_choice_array}")
            old_colour = self.old_choice_array[channel_index]
            print(f"old colour {old_colour}")
            if old_colour is not None:
                dicttemp["colour"][old_colour] = 100
            
            # update dictionary 
            new_color = self.dropdowns[channel_index].currentText()
            self.color_name[channel_index] = new_color
            print(f"current choice array{self.color_name}")
            if new_color in self.old_choice_array:
                print(f"{new_color} is duplicated")
                self.clear_choices(self.stack_dict, self.temp_dict)
                # self.checkbox.setChecked(False)
                # self.clear_img()
                

            dicttemp["colour"][self.color_name[channel_index]] = channel_index
            print(f"{self.color_name[channel_index]}: Channel {channel_index}")
            print(f"self.stack_dict{self.stack_dict}")
            # Uncheck the checkbox when a dropdown is changed
            if self.checkbox.isChecked():
                self.checkbox.setChecked(False)
                self.clear_img()
            self.old_choice_array[channel_index] = self.color_name[channel_index]
            

    def update_dict(self, state, dicttemp):
        if state == 2:  # Checkbox checked state
            self.update_img = 1
            for color in dicttemp["colour"]:
                self.stack_dict["colour"][color] = dicttemp["colour"][color]
                # dicttemp["colour"][color] = 100
                #WILL NEED TO SEND TO SPECIAL NODES LAYER
            self.dict_widg.emit(self.stack_dict)
            print("Dictionary updated:", self.stack_dict)
            print("Dictionary temporary cleared:", dicttemp)

            # update image 
            # self.RGB_img = self.assign_channels_RGB(self.RGB_img)
            # self.show_image(CVImage(self.RGB_img))
            # self.ValueChanged1.emit(self.val1)
            # print(f"self.val1 {self.val1}")

        
    def clear_choices(self, dict, dicttemp):
        self.checkbox.setChecked(False)
        self.update_img = 0
        for dropdown in self.dropdowns:
            dropdown.setCurrentIndex(0)  # Reset dropdown menu to "None"
        
        for color in dict["colour"]:
            dict["colour"][color] = 100  # Reset color values to 100
            dicttemp["colour"][color] = 100

        self.old_choice_array = [None, None, None, None, None, None]
        self.color_name = [None, None, None, None, None, None]
        
        self.clear_img()
        
        print("Choices cleared and color values reset to 100:", self.stack_dict)    

    # --------------------------------------------------------------------
        
    #new image -> reset sliders and inputs
    def reset_widg(self, val):
        # self.input1.setValue(1) #val = midpoint
        # self.input2.setValue(1)

        # self.slider_label1.setValue(1)
        # # self.slider_label1.setRange(1, val[0])

        # self.slider_label2.setValue(1)
        # self.slider_label2.setRange(1, val[1])
        if val[0]==1 and val[1]==1:
            t = 1
            z = 1
            self.input1.setValue(t) #val = midpoint
            self.input2.setValue(z)

            self.slider_label1.setValue(t)

            self.slider_label2.setValue(z)
        else:
            t = 1
            z = round((val[1])/2)
            self.input1.setValue(t) #val = midpoint
            self.input2.setValue(z)

            self.slider_label1.setValue(t)
            self.slider_label1.setRange(1, val[0])

            self.slider_label2.setValue(z)
            self.slider_label2.setRange(1, val[1])
    
    def assign_channels_RGB(self, img):
        single_chan = img[:, :, 0]
        # Initialize RGB channels with zeros
        red_channel = np.zeros_like(single_chan)
        green_channel = np.zeros_like(single_chan)
        blue_channel = np.zeros_like(single_chan)

        for color, channel_value in self.stack_dict["colour"].items():
            # Check if the channel is part of the image
            if channel_value != 100:
                # Assign channels based on the color
                if color == "red":
                    red_channel += img[:, :, channel_value]
                elif color == "green":
                    green_channel += img[:, :, channel_value]
                elif color == "blue":
                    blue_channel += img[:, :, channel_value]
                elif color == "cyan":
                    cyan_channel = img[:, :, channel_value]
                    green_channel += cyan_channel
                    blue_channel += cyan_channel
                elif color == "magenta":
                    magenta_channel = img[:, :, channel_value]
                    red_channel += magenta_channel
                    blue_channel += magenta_channel
                elif color == "yellow":
                    yellow_channel = img[:, :, channel_value]
                    red_channel += yellow_channel
                    green_channel += yellow_channel

        # Clip values to ensure they remain within the valid range [0, 255]
        red_channel = np.clip(red_channel, 0, 255)
        green_channel = np.clip(green_channel, 0, 255)
        blue_channel = np.clip(blue_channel, 0, 255)

        # Combine channels back into image data
        rgb_image_stack = np.stack([red_channel, green_channel, blue_channel], axis=2)

        return rgb_image_stack    
    
    def clear_img(self):
        # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        # print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately

    def remove_widgets(self):
        #print("REMOVE")
        # Remove and delete all widgets from the layout
        for i in reversed(range(self.layout1.count())):
            widget = self.layout1.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                #print("removed")

            # Remove the widget from the layout
            self.layout1.takeAt(i)
            
    def update_widgets(self, dim):
        self.dimens = dim
        # self.dimens = dim
        num_z = dim[1]  # 10
        num_time = dim[0]  # 21
        width = dim[3]  # 512
        height = dim[2]  # 512
        chan = dim[-1]
        
        if (dim[0] != 1) & (dim[1] != 1):
            message = f"Z-Slices: {num_z}\n"
            message += f"Frames (time): {num_time}\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            # self.slider_label1.setRange(1, num_time)  
            # self.slider_label2.setRange(1, num_z) 
        elif dim[0] != 1:
            message = f"Z-Slices: {num_z}"
            message += f" (single slice)\n"
            message += f"Frames (time): {num_time}\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            # self.slider_label1.setRange(1, num_time)
            # self.slider_label2.setRange(1, 1) 
        elif dim[1] != 1:
            message = f"Z-Slices: {num_z}\n"
            message += f"Frames (time): {num_time}"
            message += f" (single frame)\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            #zslider
            # self.slider_label2.setRange(1, num_z) 
            #time slider (one frame)
            # self.slider_label1.setRange(1, 1) 
        elif (dim[0] == 1) & (dim[1] == 1):
            message = f"2D image\n"
            message += f"Z-Slices: {num_z}"
            message += f" (single slice)\n"
            message += f"Frames (time): {num_time}"
            message += f" (single frame)\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            #zslice
            # self.slider_label2.setRange(1, 1) 
            #time slider (one frame)
            # self.slider_label1.setRange(1, 1) 
        
        # Set the text in self.shape_label
        self.shape_label.setText(message)

    # value 1: TIME ----------------------------------------------------
    def the_spinbox1_was_changed(self):    
        # self.slider_label1.setRange(1, (self.input1.value()*2))  #Range should not change for time / z sliders
        self.val1 = self.input1.value()
             
        self.slider_label1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)
        self.released1.emit(self.val1)
    
    def the_slider1_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val1 = v
        self.input1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)
    
    def slider1_released(self):
        self.released1.emit(self.slider_label1.value())

    # value 2: z-stack --------------------------------------------------
    def the_spinbox2_was_changed(self): 
        # self.slider_label2.setRange(1, (self.input2.value()*2))  
        self.val2 = self.input2.value()
             
        self.slider_label2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
        # self.released2.emit(self.val2)
    
    def the_slider2_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val2 = v
        self.input2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
    
    # def slider2_released(self):
    #     self.released2.emit(self.slider_label2.value())

    # For batch process
    def channels(self, channels_dict):
        self.stack_dict = channels_dict
        print("came to channels", self.stack_dict)
    
    def show_image(self, old_img):
        # self.resize(800,800)

        self.RGB_img = self.assign_channels_RGB(old_img)
        print("rgb shape read_img", self.RGB_img.shape)
        # If confirm button has been pressed
        if self.update_img == 1:
            try:
                if self.RGB_img.shape[-1] == 1:
                    # Grayscale image
                    qt_image = QImage(self.RGB_img.data, self.RGB_img.shape[1], self.RGB_img.shape[0], self.RGB_img.shape[1], QImage.Format_Grayscale8)
                    # #print("came here for Sliderwidget")
                elif self.RGB_img.shape[-1] == 3:
                    h, w, ch = self.RGB_img.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(self.RGB_img.data, w, h, bytes_per_line, QImage.Format_RGB888) #Format_RGB888
                elif self.RGB_img.shape[-1] == 4:
                    print("rgb shape read_img", self.RGB_img.shape)
                    h, w, ch = self.RGB_img.shape
                    #print(f"ch: {ch}")
                    bytes_per_line = ch * 4
                    qt_image = QImage(self.RGB_img.data, w, h, QImage.Format_RGBA8888) #Format_RGB888
                if qt_image is not None:
                    # Calculate the target size for scaling
                    scale_factor = 0.7  # Increase the scaling factor for clarity
                    if qt_image.width() < 400:
                        scale_factor = 1
                    if qt_image.width() > 900:
                        scale_factor = 0.5
                    target_width = int(qt_image.width() * scale_factor)
                    # Use scaledToWidth to reduce the size while maintaining aspect ratio
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaledToWidth(target_width)
                    
                    # Set the scaled pixmap
                    self.image_label.setPixmap(scaled_pixmap)
                    
                    # Resize the widget to match the pixmap size
                    self.resize(scaled_pixmap.width(), scaled_pixmap.height())
                    
                    # Ensure that any necessary updates are performed
                    self.node.update_shape()
                
            except Exception as e:
                print("Error:", e)
        else:
            self.clear_img()
    
    def get_state(self) -> dict:
        return {
            'dimension': self.dimens,
            'val1': self.val1, #t
            'val2': self.val2,
            'channel_0': self.dropdowns[0].currentText(),
            'channel_1': self.dropdowns[1].currentText(),
            'channel_2': self.dropdowns[2].currentText(),
            'channel_3': self.dropdowns[3].currentText(),            
            'channel_4': self.dropdowns[4].currentText(),            
            'channel_5': self.dropdowns[5].currentText(),
            # 'checked': self.checkbox.isChecked(),
            # 'update_img': self.update_img,
            }

    def set_state(self, data: dict):
        # First clear:
        # self.update_img = data['update_img']
        for dropdown in self.dropdowns:
            dropdown.setCurrentIndex(0)  # Reset dropdown menu to "None"
        
        for color in self.stack_dict["colour"]:
            self.stack_dict["colour"][color] = 100  # Reset color values to 100
            self.temp_dict["colour"][color] = 100

        self.old_choice_array = [None, None, None, None, None, None]
        self.color_name = [None, None, None, None, None, None]
        
        # self.clear_img()
        self.checkbox.setChecked(False)

        for i in range(0,5):
            print(i)
            self.dropdowns[i].setCurrentText(data[f'channel_{i}'])
            if self.dropdowns[i].currentText() != "Select":
                self.color_name[i] = self.dropdowns[i].currentText()
                # Does this:
                # self.dropdowns[1].setCurrentText(data['channel_1'])
                # self.dropdowns[2].setCurrentText(data['channel_2'])
                # self.dropdowns[3].setCurrentText(data['channel_3'])
                # self.dropdowns[4].setCurrentText(data['channel_4'])
                # self.dropdowns[5].setCurrentText(data['channel_5'])

        print("self dropdown 5", self.dropdowns[5].currentText())
        print("colourname", self.color_name)
        self.update_widgets(data['dimension'])

        self.val1 = data['val1']
        self.slider_label1.setValue(data['val1'])
        self.slider_label1.setRange(1, (data['dimension'])[0])
        self.input1.setValue(data['val1'])
        self.input1.setMaximum((data['dimension'])[0])

        self.val2 = data['val2']
        self.slider_label2.setValue(data['val2'])
        self.slider_label2.setRange(1, (data['dimension'])[1])
        self.input2.setValue(data['val2'])
        self.input2.setMaximum((data['dimension'])[1])

        # self.checkbox.setChecked(data['checked'])
        self.node.update_shape()
    


    
    
    # def get_state(self) -> dict:
    #     return {
    #         'dimension': self.val1, 
    #         'val2': self.val2,
    #     }

    # def set_state(self, data: dict):
    #     #val1
    #     self.input1.setValue(data['val1'])
    #     self.slider_label1.setValue(data['val1'])
    #     self.slider_label1.setRange(1, data['val1']*2)
    #     self.val1 = data['val1']

    #     self.input2.setValue(data['val2'])
    #     self.slider_label2.setValue(data['val2'])
    #     self.slider_label2.setRange(1, data['val2']*2)
    #     self.val2 = data['val2']
        

    #     self.clicked.connect(self.button_clicked)

    # def button_clicked(self):
    #     file_path = QFileDialog.getOpenFileName(self, 'Select image')[0]
    #     try:
    #         file_path = os.path.relpath(file_path)
    #     except ValueError:
    #         return
        
    #     self.path_chosen.emit(file_path)


class PathInput(MWB, QWidget):
    path_chosen = Signal(str)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.path = ''
        # path_chosen = ''

        # setup UI
        l = QVBoxLayout()
        button = QPushButton('save processed image')
        button.clicked.connect(self.choose_button_clicked)
        l.addWidget(button)
        self.path_label = QLabel('(save output: select file path)') ##print(f"Image saved to {file_path}")
        self.path_label.setStyleSheet('font-size: 14px;')
        l.addWidget(self.path_label)
        self.setLayout(l)
    
    def choose_button_clicked(self): 
        self.abs_f_path = QFileDialog.getSaveFileName(self, 'Save stack',filter='TIFF Files (*.tif *.tiff)')[0] #filter='TIFF Files (*.tif *.tiff)
        self.path = os.path.relpath(self.abs_f_path)

        self.path_label.setText(f"output saved to\n {self.abs_f_path}")
        #print(f"setText {self.path}")
        #print(f"abs_f_path {self.abs_f_path}")
        self.adjustSize()  # important! otherwise the widget won't shrink

        self.path_chosen.emit(self.path)

        self.node.update_shape()
    
    def reset_w(self, int):
        # #print("reset received")
        self.path_label.setText('  (save output: select file path)')

    def get_state(self):
        return {'path': self.path,
                'abs': self.abs_f_path}
    
    def set_state(self, data):
        self.path = data['path']
        self.abs_f_path = data['abs']
        self.path_label.setText(self.abs_f_path)
        self.node.update_shape()

class BatchPaths(MWB, QWidget):
    path_chosen = Signal(tuple)
    morphproperties = Signal(tuple)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.path = ''
        # path_chosen = ''

        # setup UI
        l = QVBoxLayout()
        self.path_label0 = QLabel('0. Connect node to pipeline\n1. Select a folder to process\n2. Select directory for BatchProcessed Images\n3. Select directory for BatchProcessed CSVs\n\nNote: steps 2 & 3 can be done in any order\nPlease allow the batch processing to complete before using the pipeline\n(See chosen folders for progress)\n') ##print(f"Image saved to {file_path}")
        self.path_label0.setStyleSheet('font-size: 16px;')
        l.addWidget(self.path_label0)
        button = QPushButton('1. Select Input Folder')
        button.clicked.connect(self.choose_button_clicked)
        l.addWidget(button)
        self.path_label = QLabel('(Select a folder to batch process)\n') ##print(f"Image saved to {file_path}")
        self.path_label.setStyleSheet('font-size: 14px;')
        l.addWidget(self.path_label)


        button2 = QPushButton('2.Generate and Save Batch Processed Images')
        button2.clicked.connect(self.bacthp_button_clicked)
        l.addWidget(button2)
        self.path_label2 = QLabel('(Press the button to generate Batch Processed Images)\n') ##print(f"Image saved to {file_path}")
        self.path_label2.setStyleSheet('font-size: 14px;')
        l.addWidget(self.path_label2)

        # Morphological properties
        button3 = QPushButton('3. Generate and Save Morphological Properties CSVs')
        button3.clicked.connect(self.morph_button_clicked)
        l.addWidget(button3)
        self.path_label3 = QLabel('(Press the button to generate Morphalogical Properties)\n') ##print(f"Image saved to {file_path}")
        self.path_label3.setStyleSheet('font-size: 14px;')
        l.addWidget(self.path_label3)

        self.setLayout(l)
    
    def choose_button_clicked(self):
        self.abs_f_path = QFileDialog.getExistingDirectory(self, 'Select folder to batch process') #filter='TIFF Files (*.tif *.tiff)
        self.path = os.path.relpath(self.abs_f_path)

        # # self.path_label.setText(f"output saved to\n {self.abs_f_path}")
        #  # Create a new folder in the selected location
        # self.new_folder_name = "BatchProcessed"  # You can change this to your desired folder name
        # self.new_folder_path = os.path.join(self.abs_f_path, self.new_folder_name)
        # os.makedirs(self.new_folder_path)

        # Update the label text to show the selected folder 
        self.path_label.setText(f"Input folder selected:\n{self.abs_f_path}\n")
        #print(f"setText {self.path}")
        #print(f"abs_f_path {self.abs_f_path}")
        self.adjustSize()  # important! otherwise the widget won't shrink
        # self.path_chosen.emit(self.path)

        self.node.update_shape()
    
    def bacthp_button_clicked(self):
        self.output_path_abs = QFileDialog.getExistingDirectory(self, 'Select folder to save ouput') #filter='TIFF Files (*.tif *.tiff)
        self.output_path = os.path.relpath(self.output_path_abs)
        # Create a new folder in the selected location
        self.new_folder_name = "BatchProcessed_Imgs"  # You can change this to your desired folder name
        self.output_path_new_folder= os.path.join(self.output_path, self.new_folder_name)
        os.makedirs(self.output_path_new_folder)

        # Update the label text to show the selected folder 
        self.path_label2.setText(f"Output directory:\n{self.output_path_new_folder}\n")
        # emit tuple
        self.path_chosen.emit((self.path, self.output_path_new_folder))
        #print(f"setText {self.path}")
        #print(f"abs_f_path {self.abs_f_path}")
        self.adjustSize()  # important! otherwise the widget won't shrink
        # self.path_chosen.emit(self.path)
        self.node.update_shape()

    def morph_button_clicked(self):
        self.morph_props_path_abs = QFileDialog.getExistingDirectory(self, 'Select folder to save csv files') #filter='TIFF Files (*.tif *.tiff)
        self.morph_props_path = os.path.relpath(self.morph_props_path_abs)
        
         # Create a new folder in the selected location
        self.csv_folder_name = "BatchProcessed_CSVs"  # You can change this to your desired folder name
        self.folder_csv_output_path= os.path.join(self.morph_props_path, self.csv_folder_name)
        os.makedirs(self.folder_csv_output_path)
        # Update the label text to show the selected folder 
        self.path_label3.setText(f"Output directory:\n{self.folder_csv_output_path}\n")
        print(f'emit{self.folder_csv_output_path}')
        self.morphproperties.emit((self.path, self.folder_csv_output_path))
        #print(f"setText {self.path}")
        #print(f"abs_f_path {self.abs_f_path}")
        self.adjustSize()  # important! otherwise the widget won't shrink
        # self.path_chosen.emit(self.path)
        self.node.update_shape()


    def reset_w(self, int):
        # #print("reset received")
        self.path_label.setText('(select a folder to batch process)\n')
        self.path_label2.setText('(Press the button to perform batch processing)\n')
        self.path_label3.setText('(Press the button to generate Morphalogical Properties)\n')

    # def get_state(self):
    #     return {'path': self.path,
    #             'abs': self.abs_f_path}
    
    # def set_state(self, data):
    #     self.path = data['path']
    #     self.abs_f_path = data['abs']
    #     self.path_label.setText(self.abs_f_path)
    #     self.node.update_shape()

# Crop
class Crop_MainWidget(MWB, QWidget):
           #define Signal
    kValueChanged = Signal(int)
    bValueChanged = Signal(int)
    lValueChanged = Signal(int)
    rValueChanged = Signal(int)
    previewState = Signal(bool)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default = 1
        default_range = 10
        self.k = default
        self.b = default
        self.l = default
        self.r = default
        self.target_h = default
        self.target_w = default
                
        #Added Widget -----------------------------------------------
        # top------------
        self.top_label = QLabel('crop from the top:')
        self.top_label.setStyleSheet('font-size: 14px;')
        self.top_input = QSpinBox()
        self.top_input.setValue(default)
        self.top_input.setKeyboardTracking(False)
        self.top_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # top slider
        self.top_slider_label = QSlider(Qt.Horizontal)
        self.top_slider_label.setRange(1, default_range)    
        self.top_slider_label.setValue(default)

        # bottom
        self.bot_label = QLabel('crop from the bottom:')
        self.bot_label.setStyleSheet('font-size: 14px;')
        self.bot_input = QSpinBox()
        self.bot_input.setValue(default)
        self.bot_input.setKeyboardTracking(False)
        self.bot_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # slider
        self.bot_slider_label = QSlider(Qt.Horizontal)
        self.bot_slider_label.setRange(1, default_range)    
        self.bot_slider_label.setValue(default)

        # Left
        self.left_label = QLabel('crop from the left:')
        self.left_label.setStyleSheet('font-size: 14px;')
        self.left_input = QSpinBox()
        self.left_input.setValue(default)
        self.left_input.setKeyboardTracking(False)
        self.left_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # left slider
        self.left_slider_label = QSlider(Qt.Horizontal)
        self.left_slider_label.setRange(1, default_range)    
        self.left_slider_label.setValue(default)

        # Right
        self.right_label = QLabel('crop from the right:')
        self.right_label.setStyleSheet('font-size: 14px;')
        self.right_input = QSpinBox()
        self.right_input.setValue(default)
        self.right_input.setKeyboardTracking(False)
        self.right_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # right slider
        self.right_slider_label = QSlider(Qt.Horizontal)
        self.right_slider_label.setRange(1, default_range)    
        self.right_slider_label.setValue(default)

        
        #sigmaX ------------
        # self.sigmaX_label = QLabel('sigmaX:')
        # self.sigmaX_label.setStyleSheet('font-size: 14px;')
        # self.sigmaX_size_input = QDoubleSpinBox()
        # #Xslider
        # self.sliderX_label = QSlider(Qt.Horizontal)
        # self.sliderX_label.setRange(0, 1000)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setStyleSheet('font-size: 14px;')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.top_label)
        self.layout1.addWidget(self.top_input)
        self.layout1.addWidget(self.top_slider_label)

        self.layout1.addWidget(self.left_label)
        self.layout1.addWidget(self.left_input)
        self.layout1.addWidget(self.left_slider_label)
        
        self.layout1.addWidget(self.bot_label)
        self.layout1.addWidget(self.bot_input)
        self.layout1.addWidget(self.bot_slider_label)
        
        self.layout1.addWidget(self.right_label)
        self.layout1.addWidget(self.right_input)
        self.layout1.addWidget(self.right_slider_label)

        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)
        
        #Signals 
        # Spinbox triggers
        self.top_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        # Slider triggers
        self.top_slider_label.sliderMoved.connect(self.the_slider_was_changed)

        # bot
        # Spinbox triggers
        self.bot_input.editingFinished.connect(self.the_spinbox_was_changed2)  #USER ONLY
        # Slider triggers
        self.bot_slider_label.sliderMoved.connect(self.the_slider_was_changed2)

        # left
        # Spinbox triggers
        self.left_input.editingFinished.connect(self.the_spinbox_was_changed3)  #USER ONLY
        # Slider triggers
        self.left_slider_label.sliderMoved.connect(self.the_slider_was_changed3)

        # right
        # Spinbox triggers
        self.right_input.editingFinished.connect(self.the_spinbox_was_changed4)  #USER ONLY
        # Slider triggers
        self.right_slider_label.sliderMoved.connect(self.the_slider_was_changed4)

        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
          
    #slot functions
    def the_spinbox_was_changed(self):    # v: value emitted by a signal.
        self.top_slider_label.setRange(1, self.top_input.value()*2)  
        self.top_slider_label.setValue(self.top_input.value())  
        self.k = self.top_input.value()
        self.kValueChanged.emit(self.k)
        #debug
        # #print(f'slider:{self.top_slider_label.value()}')

    def the_slider_was_changed(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.top_input.setValue(v)
        self.k = int(v)
        self.kValueChanged.emit(self.k)
    
    # bot
    def the_spinbox_was_changed2(self):    # v: value emitted by a signal.
        self.bot_slider_label.setRange(1, self.bot_input.value()*2)  
        self.bot_slider_label.setValue(self.bot_input.value())  
        self.b = self.bot_input.value()
        self.bValueChanged.emit(self.b)
        #debug
        # #print(f'slider:{self.bot_slider_label.value()}')

    def the_slider_was_changed2(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.bot_input.setValue(v)
        self.b = int(v)
        self.bValueChanged.emit(self.b)

    # left
    def the_spinbox_was_changed3(self):    # v: value emitted by a signal.
        self.left_slider_label.setRange(1, self.left_input.value()*2)  
        self.left_slider_label.setValue(self.left_input.value())  
        self.l = self.left_input.value()
        self.lValueChanged.emit(self.l)
        #debug
        # #print(f'slider:{self.left_slider_label.value()}')

    def the_slider_was_changed3(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.left_input.setValue(v)
        self.l = int(v)
        self.lValueChanged.emit(self.l)
    
    # right
    def the_spinbox_was_changed4(self):    # v: value emitted by a signal.
        self.right_slider_label.setRange(1, self.right_input.value()*2)  
        self.right_slider_label.setValue(self.right_input.value())  
        self.r = self.right_input.value()
        self.rValueChanged.emit(self.r)
        #debug
        # #print(f'slider:{self.right_slider_label.value()}')

    def the_slider_was_changed4(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.right_input.setValue(v)
        self.r = int(v)
        self.rValueChanged.emit(self.r)
    
    def dimensions(self, dim):
        self.target_h = dim[0]
        self.target_w = dim[1]
        #print(f'dim.height{self.target_h}')
        self.top_input.setMaximum(self.target_w-1)
        self.top_slider_label.setRange(1,self.target_w-1)

        self.bot_input.setMaximum(self.target_w-1)
        self.bot_slider_label.setRange(1,self.target_w-1)

        self.left_input.setMaximum(self.target_h-1)
        self.left_slider_label.setRange(1,self.target_h-1)

        self.right_input.setMaximum(self.target_h-1)
        self.right_slider_label.setRange(1,self.target_h-1)  
    
    # the node emits the channel dictionary to the widget which will be used in the show_img / assign method
    def channels(self, channels_dict):
        self.stack_dict = channels_dict
        print("came to channels", self.stack_dict)

    def assign_channels_RGB(self, img):
        single_chan = img[:, :, 0]
        # Initialize RGB channels with zeros
        red_channel = np.zeros_like(single_chan)
        green_channel = np.zeros_like(single_chan)
        blue_channel = np.zeros_like(single_chan)

        for color, channel_value in self.stack_dict["colour"].items():
            # Check if the channel is part of the image
            if channel_value != 100:
                # Assign channels based on the color
                if color == "red":
                    red_channel += img[:, :, channel_value]
                elif color == "green":
                    green_channel += img[:, :, channel_value]
                elif color == "blue":
                    blue_channel += img[:, :, channel_value]
                elif color == "cyan":
                    cyan_channel = img[:, :, channel_value]
                    green_channel += cyan_channel
                    blue_channel += cyan_channel
                elif color == "magenta":
                    magenta_channel = img[:, :, channel_value]
                    red_channel += magenta_channel
                    blue_channel += magenta_channel
                elif color == "yellow":
                    yellow_channel = img[:, :, channel_value]
                    red_channel += yellow_channel
                    green_channel += yellow_channel

        # Clip values to ensure they remain within the valid range [0, 255]
        red_channel = np.clip(red_channel, 0, 255)
        green_channel = np.clip(green_channel, 0, 255)
        blue_channel = np.clip(blue_channel, 0, 255)

        # Combine channels back into image data
        rgb_image_stack = np.stack([red_channel, green_channel, blue_channel], axis=2)

        return rgb_image_stack
    
    # original image. If 3 channels, default RGB, may need to assign the custom channels 
    def show_image(self, orignial_img):
        # self.resize(800,800)
        # print(f"DIMENSION WIDGET {img.shape[-1]}")

        try:
            # Grayscale image
            if orignial_img.shape[-1] == 1:
                qt_image = QImage(orignial_img.data, orignial_img.shape[1], orignial_img.shape[0], orignial_img.shape[1], QImage.Format_Grayscale8)

            # 3 colour channels (default RGB, may need to assign the custom channels)
            elif orignial_img.shape[-1] == 3:
                # Assign the channels to the image 
                self.RGB_img = self.assign_channels_RGB(orignial_img)
                h, w, ch = self.RGB_img.shape
                bytes_per_line = ch * w
                qt_image = QImage(self.RGB_img.data, w, h, bytes_per_line, QImage.Format_RGB888) #Format_RGB888

            # not sure if this is necesssary
            # note rgb image is now assigned custom channels 
            # may need to adjust code for 4 channel image (take original_img and apply)   
            elif self.RGB_img.shape[-1] == 4:
                h, w, ch = self.RGB_img.shape
                #print(f"ch: {ch}")
                bytes_per_line = ch * 4
                qt_image = QImage(self.RGB_img.data, w, h, QImage.Format_RGBA8888) #Format_RGB888
            if qt_image is not None:
                # Calculate the target size for scaling
                scale_factor = 0.7  # Increase the scaling factor for clarity
                if qt_image.width() < 400:
                    scale_factor = 1
                if qt_image.width() > 900:
                    scale_factor = 0.5
                target_width = int(qt_image.width() * scale_factor)
                # Use scaledToWidth to reduce the size while maintaining aspect ratio
                scaled_pixmap = QPixmap.fromImage(qt_image).scaledToWidth(target_width)
                
                # Set the scaled pixmap
                self.image_label.setPixmap(scaled_pixmap)
                
                # Resize the widget to match the pixmap size
                self.resize(scaled_pixmap.width(), scaled_pixmap.height())
                
                # Ensure that any necessary updates are performed
                self.node.update_shape()
            
        except Exception as e:
            print("Error:", e)
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))
        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        # #print(self.top_input.value())
    
    def get_state(self) -> dict:
        return {
            'val1': self.k,
            'val2': self.b,
            'val3': self.l,
            'val4': self.r,
            'val5': self.target_h,
            'val6': self.target_w,
        }

    def set_state(self, data: dict):
        #top
        self.top_input.setValue(data['val1'])
        self.top_slider_label.setValue(data['val1'])
        self.top_input.setMaximum(data['val5'])
        self.top_slider_label.setValue(data['val5'])
        self.k = data['val1']

        #bot
        self.bot_input.setValue(data['val2'])
        self.bot_slider_label.setValue(data['val2'])
        self.bot_input.setMaximum(data['val5'])
        self.bot_slider_label.setValue(data['val5'])
        self.b = data['val2']

        #left
        self.left_input.setValue(data['val3'])
        self.left_slider_label.setValue(data['val3'])
        self.left_input.setMaximum(data['val6'])
        self.left_slider_label.setValue(data['val6'])
        self.l = data['val3']

        #right
        self.right_input.setValue(data['val4'])
        self.right_slider_label.setValue(data['val4'])
        self.right_input.setMaximum(data['val6'])
        self.right_slider_label.setValue(data['val6'])
        self.r = data['val4']


# Output Node
class OutputMetadataWidg(MWB, QWidget):
    new_data = Signal(bool)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)
        self.label = QLabel('When finished tuning parameters, press\nthe button to display the properties:')
        self.label.setStyleSheet('font-size: 14px;')
        self.pleaseWait = QLabel('Once pressed...please wait...')
        self.pleaseWait.setStyleSheet('font-size: 14px;')
        self.button = QPushButton('load properties')
        # self.table = QTableWidget()
        self.summaryProperties = QLabel(" ")
        self.summaryProperties.setFixedHeight(250)
        self.summaryProperties.setAlignment(Qt.AlignTop)
        self.summaryProperties.setStyleSheet('font-size: 14px;')
        self.summarylabel = QLabel(' ')
        self.summarylabel.setStyleSheet('font-size: 4px;')
        # self.summarylabel.setFixedHeight(150)
        self.fullTable = QTableWidget()
        self.label2 = QLabel('Once the the properties have been loaded,\n - view properties for each label\n - save properties as a .csv:')
        self.label2.setStyleSheet('font-size: 14px;')
        self.button2 = QPushButton('view all properties')
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        # self.scroll_area.viewport().setMinimumHeight(500)
        # self.scroll_area.viewport().setMaximumHeight(600)
        self.scroll_area.setWidget(self.summaryProperties)
        self.button3 = QPushButton('save properties')
        self.path_label=QLabel(' \n')
        self.summaryProperties.setStyleSheet('font-size: 14px;')

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.pleaseWait)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.summarylabel)
        self.layout.addWidget(self.scroll_area)
        # self.layout.addWidget(self.summaryProperties)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        self.layout.addWidget(self.path_label)
        

        self.setLayout(self.layout)

        self.button.clicked.connect(self.button_clicked)
        self.button2.clicked.connect(self.open_new_window)
        self.button3.clicked.connect(self.save_properties)
        # self.resize(400,400)
        
    def channels(self, channels_dict):
        self.stack_dict = channels_dict
        print("came to channels", self.stack_dict)

    def button_clicked(self):
        self.pleaseWait.setText('Please wait ...')
        # QCoreApplication.processEvents()
        self.new_data.emit(True)
    #     self.emitthis()

    # def emitthis(self):
               
    
    def show_data(self, df, summary):
        #print(df)
        print(summary)
        self.df = df
        self.summarylabel.setStyleSheet('font-size: 13px; color: #468DF5; font-weight: bold;')
        self.summarylabel.setText("PROPERTIES SUMMARY")

        self.summaryProperties.setText(summary)
        self.summaryProperties.adjustSize()
        # self.scroll_area.setWidget(self.summaryProperties)
        self.pleaseWait.setText('Finished!')
    
    def open_new_window(self):
        new_window = QDialog() #self, Qt.WindowStaysOnTopHint
        new_window.setWindowTitle("Properties of All Labels")
        new_window.resize(500,500)
        new_layout = QVBoxLayout()

        self.fullTable.setRowCount(self.df.shape[0])
        self.fullTable.setColumnCount(self.df.shape[1])
        self.fullTable.setHorizontalHeaderLabels(self.df.columns)

        for row in range(self.df.shape[0]):
            for col in range(self.df.shape[1]):
                item = QTableWidgetItem(str(self.df.iat[row, col])) #str
                self.fullTable.setItem(row, col, item)

        self.fullTable.verticalHeader().setVisible(False)

        new_layout.addWidget(self.fullTable)
        new_window.setLayout(new_layout)
        new_window.exec_()

    def save_properties(self):
        self.abs_f_path = QFileDialog.getSaveFileName(self, 'Save properties', filter = 'CSV Files (*.csv)')[0] #filter='TIFF Files (*.tif *.tiff)
        self.path = os.path.relpath(self.abs_f_path)

        self.path_label.setText(f"output saved to\n {self.abs_f_path}")
        self.path_label.setStyleSheet('font-size: 14px;')
        self.df.to_csv(self.path, index=False)
        #print(f"setText {self.path}")
        #print(f"abs_f_path {self.abs_f_path}")
        self.adjustSize()  # important! otherwise the widget won't shrink
        self.node.update_shape()
        



    # def get_state(self):
    #     return {'path': self.path,
    #             'abs': self.abs_f_path}
    
    # def set_state(self, data):
    #     self.path = data['path']
    #     self.abs_f_path = data['abs']
    #     self.path_label.setText(self.abs_f_path)
    #     self.node.update_shape()
# class TableDialog(QDialog):

class TestWidget1(QWidget):
    widgetResized = Signal(int, int)
    def __init__(self, parent=None):
        super(TestWidget1, self).__init__(parent)

        self.label = QLabel('This is Widget 1')

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def resizeEvent(self, event):
        # Emit the widget's new width and height when resized
        self.widgetResized.emit(self.width(), self.height())
        super(TestWidget1, self).resizeEvent(event)

class TestWidget2(QWidget):
    widgetResized = Signal(int, int)
    def __init__(self, parent=None):
        super(TestWidget2, self).__init__(parent)

        self.label = QLabel('This is Widget 2')

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def resizeEvent(self, event):
        # Emit the widget's new width and height when resized
        self.widgetResized.emit(self.width(), self.height())
        super(TestWidget2, self).resizeEvent(event)

class Slider_widget(MWB, Widget_Base):
           #define Signal
    
    previewState = Signal(bool)
    linkState = Signal(bool)
    ValueChanged1 = Signal(int)  #time instance
    ValueChanged2 = Signal(int)  #z-slice (depth)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        # self.resize(300,300) #width height 
        default = 5
        default_range = default*2
                
        #Added Widget -----------------------------------------------
        #time-series------------
        self.input_label1 = QLabel('time instance:')
        self.input1 = QSpinBox()
        self.input1.setValue(default)
        self.input1.setKeyboardTracking(False)
        self.input1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #timeseries slider
        self.slider_label1 = QSlider(Qt.Horizontal)
        self.slider_label1.setRange(1, default_range)    
        self.slider_label1.setSingleStep(2)
        self.slider_label1.setValue(default)

        #z-stack------------
        self.input_label2 = QLabel('z-slice:')
        self.input2 = QSpinBox()
        self.input2.setValue(default)
        self.input2.setKeyboardTracking(False)
        self.input2.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #zstack slider
        self.slider_label2 = QSlider(Qt.Horizontal)   #MAKE VERTICLAL
        self.slider_label2.setRange(1, default_range)    
        self.slider_label2.setSingleStep(2)
        self.slider_label2.setValue(default)
        
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # time
        self.layout1.addWidget(self.input_label1)
        self.layout1.addWidget(self.input1)
        self.layout1.addWidget(self.slider_label1)
        # z-stack
        self.layout1.addWidget(self.input_label2)
        self.layout1.addWidget(self.input2)
        self.layout1.addWidget(self.slider_label2)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        # Image
        self.layout1.addWidget(self.image_label)
        self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.input1.editingFinished.connect(self.the_spinbox1_was_changed)  #USER ONLY
        self.input2.editingFinished.connect(self.the_spinbox2_was_changed)
        # Slider triggers
        self.slider_label1.sliderMoved.connect(self.the_slider1_was_changed)
        self.slider_label2.sliderMoved.connect(self.the_slider2_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
              
    # value 1: TIME ----------------------------------------------------
    def the_spinbox1_was_changed(self):    
        self.slider_label1.setRange(1, (self.input1.value()*2))  

        self.val1 = self.input1.value()
             
        self.slider_label1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)
    
    def the_slider1_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val1 = v
        self.input1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)

    # value 2: z-stack --------------------------------------------------
    def the_spinbox2_was_changed(self): 
        self.slider_label2.setRange(1, (self.input2.value()*2))  

        self.val2 = self.input2.value()
             
        self.slider_label2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
    
    def the_slider2_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val2 = v
        self.input2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)

    # update image ---------------------------------------------
    # def show_image(self, img):
    #     # self.resize(800,800)

    #     try:
    #         if len(img.shape) == 2:
    #             # Grayscale image
    #             img = (img*255).astype('uint8')
    #             qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
    #             #print("came here for Sliderwidget")
    #         else:
    #             # RGB image
    #             #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             h, w, ch = rgb_image.shape
    #             bytes_per_line = ch * w
    #             qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
    #         # Calculate the target size for scaling
    #         scale_factor = 0.8  # Increase the scaling factor for clarity
    #         target_width = int(qt_image.width() * scale_factor)
    #         target_height = int(qt_image.height() * scale_factor)
            
    #         qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
    #         self.image_label.setPixmap(QPixmap.fromImage(qt_image))
    #         self.resize(target_width, target_height)
    #         self.node.update_shape()
    #     except Exception as e:
    #         #print("Error:", e)


    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.val1, 
            'val2': self.val2,
        }

    def set_state(self, data: dict):
        #val1
        self.input1.setValue(data['val1'])
        self.slider_label1.setValue(data['val1'])
        self.slider_label1.setRange(1, data['val1']*2)
        self.val1 = data['val1']

        self.input2.setValue(data['val2'])
        self.slider_label2.setValue(data['val2'])
        self.slider_label2.setRange(1, data['val2']*2)
        self.val2 = data['val2']

class Split_Img(MWB, Widget_Base8):

    #define Signal
    previewState = Signal(bool)
    # Value1Changed = Signal(float)  #kernel
    # Value2Changed = Signal(float)  #itt

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        # default1 = 3
        # default_range1 = default1*2
        # default2 = 1
        # default_range2 = 5
        # self.t = default1
        # self.t2 = default2
                
        #Added Widget -----------------------------------------------
        #alpha size------------
        # self.label_1 = QLabel('alpha:')
        # self.label_1.setStyleSheet('font-size: 14px;')
        # self.input_1 = QDoubleSpinBox()
        # # self.input_1.setMaximum(255)
        # self.input_1.setValue(default1)
        # self.input_1.setKeyboardTracking(False)
        # self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #a slider
        # self.slider_label_1 = QSlider(Qt.Horizontal)
        # self.slider_label_1.setRange(1, default_range1*100)    
        # self.slider_label_1.setValue(default1*100)

        # #beta size------------
        # self.label_2 = QLabel('beta:')
        # self.label_2.setStyleSheet('font-size: 14px;')
        # self.input_2 = QDoubleSpinBox()
        # # self.input_2.setMaximum(5)
        # self.input_2.setValue(default2)
        # self.input_2.setKeyboardTracking(False)
        # self.input_2.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # #b slider
        # self.slider_label_2 = QSlider(Qt.Horizontal)
        # self.slider_label_2.setRange(1, default_range2*100)    
        # self.slider_label_2.setValue(default2)
       
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # kernel
        # self.layout1.addWidget(self.label_1)
        # self.layout1.addWidget(self.input_1)
        # self.layout1.addWidget(self.slider_label_1)
        # # itt
        # self.layout1.addWidget(self.label_2)
        # self.layout1.addWidget(self.input_2)
        # self.layout1.addWidget(self.slider_label_2)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        # self.input_1.editingFinished.connect(self.spinbox_1_changed)  #USER ONLY
        # self.input_2.editingFinished.connect(self.spinbox_2_changed)
        # # Slider triggers
        # self.slider_label_1.sliderMoved.connect(self.slider_1_changed)
        # self.slider_label_2.sliderMoved.connect(self.slider_2_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.prev_checkbox_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def prev_checkbox_changed(self, state):
        self.previewState.emit(state)
    
    # def assign_channels_RGB(self, img):
    #     single_chan = img[:, :, 0]
    #     # Initialize RGB channels with zeros
    #     red_channel = np.zeros_like(single_chan)
    #     green_channel = np.zeros_like(single_chan)
    #     blue_channel = np.zeros_like(single_chan)

    #     for color, channel_value in self.stack_dict["colour"].items():
    #         # Check if the channel is part of the image
    #         if channel_value != 100:
    #             # Assign channels based on the color
    #             if color == "red":
    #                 red_channel += img[:, :, channel_value]
    #             elif color == "green":
    #                 green_channel += img[:, :, channel_value]
    #             elif color == "blue":
    #                 blue_channel += img[:, :, channel_value]
    #             elif color == "cyan":
    #                 cyan_channel = img[:, :, channel_value]
    #                 green_channel += cyan_channel
    #                 blue_channel += cyan_channel
    #             elif color == "magenta":
    #                 magenta_channel = img[:, :, channel_value]
    #                 red_channel += magenta_channel
    #                 blue_channel += magenta_channel
    #             elif color == "yellow":
    #                 yellow_channel = img[:, :, channel_value]
    #                 red_channel += yellow_channel
    #                 green_channel += yellow_channel

    #     # Clip values to ensure they remain within the valid range [0, 255]
    #     red_channel = np.clip(red_channel, 0, 255)
    #     green_channel = np.clip(green_channel, 0, 255)
    #     blue_channel = np.clip(blue_channel, 0, 255)

    #     # Combine channels back into image data
    #     rgb_image_stack = np.stack([red_channel, green_channel, blue_channel], axis=2)

    #     return rgb_image_stack    
    
    # # redefine show image
    # def show_image(self, old_img):
    #     # self.resize(800,800)

    #     self.RGB_img = self.assign_channels_RGB(old_img)
    #     # If confirm button has been pressed
    #     if self.update_img == 1:
    #         try:
    #             if self.RGB_img.shape[-1] == 1:
    #                 # Grayscale image
    #                 qt_image = QImage(self.RGB_img.data, self.RGB_img.shape[1], self.RGB_img.shape[0], img.shape[1], QImage.Format_Grayscale8)
    #                 # #print("came here for Sliderwidget")
    #             elif self.RGB_img.shape[-1] == 3:
    #                 h, w, ch = self.RGB_img.shape
    #                 bytes_per_line = ch * w
    #                 qt_image = QImage(self.RGB_img.data, w, h, bytes_per_line, QImage.Format_RGB888) #Format_RGB888
    #             elif self.RGB_img.shape[-1] == 4:
    #                 h, w, ch = self.RGB_img.shape
    #                 #print(f"ch: {ch}")
    #                 bytes_per_line = ch * 4
    #                 qt_image = QImage(self.RGB_img.data, w, h, QImage.Format_RGBA8888) #Format_RGB888
    #             if qt_image is not None:
    #                 # Calculate the target size for scaling
    #                 scale_factor = 0.7  # Increase the scaling factor for clarity
    #                 if qt_image.width() < 400:
    #                     scale_factor = 1
    #                 if qt_image.width() > 900:
    #                     scale_factor = 0.5
    #                 target_width = int(qt_image.width() * scale_factor)
    #                 # Use scaledToWidth to reduce the size while maintaining aspect ratio
    #                 scaled_pixmap = QPixmap.fromImage(qt_image).scaledToWidth(target_width)
                    
    #                 # Set the scaled pixmap
    #                 self.image_label.setPixmap(scaled_pixmap)
                    
    #                 # Resize the widget to match the pixmap size
    #                 self.resize(scaled_pixmap.width(), scaled_pixmap.height())
                    
    #                 # Ensure that any necessary updates are performed
    #                 self.node.update_shape()
                
    #         except Exception as e:
    #             print("Error:", e)
    #     else:
            # self.clear_img()
          
    # thresh ----------------------------------------------------
    # def spinbox_1_changed(self):    
    #     self.t = self.input_1.value()
    #     v = self.t*100
    #     self.slider_label_1.setRange(1, v*2) 
    #     self.slider_label_1.setValue(v)
    #     # self.input_1.setValue(self.t)
    #     self.Value1Changed.emit(self.t)
    
    # def slider_1_changed(self, v):    # v: value emitted by a slider signal
    #     self.t = v/100
    #         #  #print('odd')
    #     self.input_1.setValue(self.t)
    #     self.Value1Changed.emit(self.t)
    # #itt
    # def spinbox_2_changed(self):    
    #     self.t2 = self.input_2.value()
    #     v = self.input_2.value()*100
    #     self.slider_label_2.setRange(1, v*2)
    #     self.slider_label_2.setValue(v)
    #     # self.input_2.setValue(self.t2)
    #     self.Value2Changed.emit(self.t2)
    
    # def slider_2_changed(self, v):    # v: value emitted by a slider signal
    #     self.t2 = v/100
    #         #  #print('odd')
    #     self.input_2.setValue(self.t2)
    #     self.Value2Changed.emit(self.t2)    
       
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    # def get_state(self) -> dict:
    #     return {
    #         'val1': self.t,
    #         'val2': self.t2,
    #     }

    # def set_state(self, data: dict):
    #     #ksize
    #     self.input_1.setValue(data['val1'])
    #     self.slider_label_1.setValue(data['val1']*100)
    #     self.slider_label_1.setRange(1, data['val1']*200)
    #     self.t = data['val1']

    #     self.input_2.setValue(data['val2'])
    #     self.slider_label_2.setValue(data['val2']*100)
    #     self.slider_label_2.setRange(1, data['val2']*200)
    #     self.t2 = data['val2']
            
class Blur_Averaging_MainWidget(MWB, Widget_Base8):
           #define Signal
    kValueChanged = Signal(int)
    previewState = Signal(bool)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default = 5
        default_range = default*2
        self.k = default
                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('ksize:')
        self.ksize_label.setStyleSheet('font-size: 14px;')
        self.k_size_input = QSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range)    
        self.slider_label.setValue(default)
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaX:')
        self.sigmaX_label.setStyleSheet('font-size: 14px;')
        self.sigmaX_size_input = QDoubleSpinBox()
        #Xslider
        self.sliderX_label = QSlider(Qt.Horizontal)
        # self.sliderX_label.setRange(0, 1000)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setStyleSheet('font-size: 14px;')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.k_size_input)
        self.layout1.addWidget(self.slider_label)
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)
        
        #Signals 
        # Spinbox triggers
        self.k_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
          
    #slot functions
    def the_spinbox_was_changed(self):    # v: value emitted by a signal.
        self.slider_label.setRange(1, self.k_size_input.value()*2)  
        self.slider_label.setValue(self.k_size_input.value())  
        self.k = self.k_size_input.value()
        self.kValueChanged.emit(self.k)
        #debug
        # #print(f'slider:{self.slider_label.value()}')

    def the_slider_was_changed(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.k_size_input.setValue(v)
        self.k = int(v)
        self.kValueChanged.emit(self.k)
        
    # def show_image(self, img):
    #     # self.resize(800,800)

    #     try:
    #         #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     except cv2.error:
    #         return
        
    #     # qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
    #     h, w, ch = rgb_image.shape
    #     bytes_per_line = ch * w
    #     qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    #     # Calculate the target size for scaling
    #     scale_factor = 0.4  # Adjust the scaling factor as needed
    #     target_width = int(w * scale_factor)
    #     target_height = int(h * scale_factor)
    #     qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
    #     # #print("H:", target_height, "W:", target_width)
    #     self.image_label.setPixmap(QPixmap(qt_image))
    #     self.resize(100, 100)
    #     self.node.update_shape()
    #     # #print('Update Shape:',  #print(self.width(), self.height()))
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))
        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        #print(self.k_size_input.value())
    
    def get_state(self) -> dict:
        return {
            'val1': self.k,
        }

    def set_state(self, data: dict):
        #ksize
        self.k_size_input.setValue(data['val1'])
        self.slider_label.setValue(data['val1'])
        self.slider_label.setRange(1, data['val1']*2)
        self.k = data['val1']



class Blur_Median_MainWidget(MWB, Widget_Base8):
           #define Signal
    kValueChanged = Signal(int)
    kReleased = Signal(int)
    previewState = Signal(bool)
    

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default = 5
        default_range = default*2
        self.k = default

                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('ksize:')
        self.ksize_label.setStyleSheet('font-size: 14px;')
        self.k_size_input = QSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range)    
        self.slider_label.setValue(default)
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaX:')
        self.sigmaX_label.setStyleSheet('font-size: 14px;')
        self.sigmaX_size_input = QDoubleSpinBox()
        #Xslider
        self.sliderX_label = QSlider(Qt.Horizontal)
        # self.sliderX_label.setRange(0, 1000)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.k_size_input)
        self.layout1.addWidget(self.slider_label)
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)
        
        #Signals 
        # Spinbox triggers
        self.k_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        self.slider_label.sliderReleased.connect(self.releasedK)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    #slot functions
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
    
    def the_spinbox_was_changed(self):    
        self.slider_label.setRange(1, (self.k_size_input.value()*2 + 1))  
        if (self.k_size_input.value() % 2) == 0:
            self.k = self.k_size_input.value()+1 
        else:
             self.k = self.k_size_input.value()
             
        self.slider_label.setValue(self.k)
        self.k_size_input.setValue(self.k) #updates even number in spinbox to odd
        self.kValueChanged.emit(self.k)
          
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal 
        if (v % 2) == 0:
            self.k = v+1 
            # #print('even')
        else:
            self.k = v
            #  #print('odd')
        self.k_size_input.setValue(self.k)
        self.kValueChanged.emit(self.k)
    
    def releasedK(self):
        self.kReleased.emit(self.k)
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))
        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        #print(self.k_size_input.value())
    
    def get_state(self) -> dict:
        return {
            'val1': self.k,
        }

    def set_state(self, data: dict):
        #ksize
        self.k_size_input.setValue(data['val1'])
        self.slider_label.setValue(data['val1'])
        self.slider_label.setRange(1, data['val1']*2)
        self.k = data['val1']

class Gaus_Blur_MainWidget(MWB, Widget_Base8):
           #define Signal
    
    previewState = Signal(bool)
    linkState = Signal(bool)
    kValueChanged = Signal(int)
    XValueChanged = Signal(float)
    YValueChanged = Signal(float)
    sigValueChanged = Signal(float)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        # self.resize(300, 300)
        default = 5
        default_range = default*2
        self.k = default
        self.X = default
        self.Y = default
                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('ksize:')
        self.ksize_label.setStyleSheet('font-size: 14px;')
        self.k_size_input = QSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range)    
        self.slider_label.setSingleStep(2)
        self.slider_label.setValue(default)
        
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaX:')
        self.sigmaX_label.setStyleSheet('font-size: 14px;')
        self.sigmaX_size_input = QDoubleSpinBox()
        self.sigmaX_size_input.setValue(default)
        self.sigmaX_size_input.setKeyboardTracking(False)
        self.sigmaX_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #Xslider
        self.Xslider_label = QSlider(Qt.Horizontal)
        self.Xslider_label.setRange(1, default_range*100)
        self.Xslider_label.setValue(default*100)
        #sigmaY ------------
        self.sigmaY_label = QLabel('sigmaY:')
        self.sigmaY_label.setStyleSheet('font-size: 14px;')
        self.sigmaY_size_input = QDoubleSpinBox()
        self.sigmaY_size_input.setValue(default)
        self.sigmaY_size_input.setKeyboardTracking(False)
        self.sigmaY_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #Yslider
        self.Yslider_label = QSlider(Qt.Horizontal)
        self.Yslider_label.setRange(1, default_range*100)    
        self.Yslider_label.setValue(default*100)
        #link
        self.link_checkbox = QCheckBox()
        self.link_checkbox.setText('Link Sigma Y with Sigma X')
        self.link_checkbox.setStyleSheet('font-size: 14px;')
        self.sigma_label = QLabel('      (use sigma X slider)') #to adjust both sigmas
        self.sigma_label.setStyleSheet('font-size: 14px;')
        self.link_checkbox.setCheckState(Qt.Checked)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setStyleSheet('font-size: 14px;')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # ksize
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.k_size_input)
        self.layout1.addWidget(self.slider_label)
        # Link
        self.layout1.addWidget(self.link_checkbox)
        self.layout1.addWidget(self.sigma_label)
        # Sigma X
        self.layout1.addWidget(self.sigmaX_label)
        self.layout1.addWidget(self.sigmaX_size_input)
        self.layout1.addWidget(self.Xslider_label)
        # Sigma Y
        self.layout1.addWidget(self.sigmaY_label)
        self.layout1.addWidget(self.sigmaY_size_input)
        self.layout1.addWidget(self.Yslider_label)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        # Image
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.k_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        self.sigmaX_size_input.editingFinished.connect(self.the_Xspinbox_was_changed)
        self.sigmaY_size_input.editingFinished.connect(self.the_Yspinbox_was_changed)
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        self.Xslider_label.sliderMoved.connect(self.the_Xslider_was_changed)
        self.Yslider_label.sliderMoved.connect(self.the_Yslider_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)
        self.link_checkbox.stateChanged.connect(self.link_checkbox_checked)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
        
    # link
    def link_checkbox_checked(self, state):
        self.linkState.emit(state)
        if  state == Qt.Checked:
            self.sigValueChanged.emit(self.X)
            self.Y = self.X
            self.Yslider_label.setRange(1, self.Y*200)
            self.Xslider_label.setRange(1, self.X*200)
            self.Yslider_label.setValue(self.Y*100)
            self.sigmaY_size_input.setValue(self.Y)

        #self.linkState is a signal object, not a boolean value: if  self.linkState == True: will not work
              
    # kernell ----------------------------------------------------
    def the_spinbox_was_changed(self):    
        self.slider_label.setRange(1, (self.k_size_input.value()*2 + 1))  
        if (self.k_size_input.value() % 2) == 0:
            self.k = self.k_size_input.value()+1 
        else:
             self.k = self.k_size_input.value()
             
        self.slider_label.setValue(self.k)
        self.k_size_input.setValue(self.k) #updates even number in spinbox to odd
        self.kValueChanged.emit(self.k)
    
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal 
        if (v % 2) == 0:
            self.k = v+1 
            # #print('even')
        else:
            self.k = v
            #  #print('odd')
        self.k_size_input.setValue(self.k)
        self.kValueChanged.emit(self.k)
    # sigmaX --------------------------------------------------
    def the_Xspinbox_was_changed(self): 
        v = self.sigmaX_size_input.value()*100   
        self.Xslider_label.setRange(1, v*2)  
        self.Xslider_label.setValue(v)  
        self.X = self.sigmaX_size_input.value()
        
        if self.link_checkbox.isChecked():
            self.sigValueChanged.emit(self.X)
            # update Y spinbox
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)

            # update Y slider
            self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            self.Y = self.sigmaX_size_input.value()
        else: 
            self.XValueChanged.emit(self.X)
            
           
    def the_Xslider_was_changed(self, v):    
        self.X = v/100
        self.sigmaX_size_input.setValue(self.X)
        
        if self.link_checkbox.isChecked():
            self.sigValueChanged.emit(self.X)
            # update Y spinbox
            self.sigmaY_size_input.setValue(self.X)
            self.Yslider_label.setValue(v)                       
        else:
            self.XValueChanged.emit(self.X)

    # sigma Y -------------------------------------------------
    def the_Yspinbox_was_changed(self): 
        if self.link_checkbox.isChecked():
            self.sigmaY_size_input.setValue(self.X)
        else:
            v = self.sigmaY_size_input.value()*100   
            self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            self.Y = self.sigmaY_size_input.value()
            self.YValueChanged.emit(self.Y)

    def the_Yslider_was_changed(self, v):    
        if self.link_checkbox.isChecked():
            self.Yslider_label.setValue(self.X*100)  
        else:
            self.Y = v/100
            #print(f"Y emitted {self.Y}")
            self.sigmaY_size_input.setValue(self.Y)
            self.YValueChanged.emit(self.Y)

    # update image ---------------------------------------------
    #show image in widget8

    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        print("widget, get_state")
        data = {
            'ksize': self.k,
            'sigmaX': self.X,
            'sigmaY': self.Y,
            # self.link_checkbox is the pyqt checkbox (not a variable)
            'linked': self.link_checkbox.isChecked(),
        }
        print("widget didnt crash")
        return data

    def set_state(self, data: dict):

        self.link_checkbox.setChecked(data['linked'])

        #ksize
        self.k = data['ksize']
        self.k_size_input.setValue(self.k)
        self.slider_label.setValue(self.k)
        self.slider_label.setRange(1, self.k*2)
        
        self.X = data['sigmaX']
        self.sigmaX_size_input.setValue(self.X)
        self.Xslider_label.setValue(self.X*100)
        self.Xslider_label.setRange(1,self.X*200)
        
        self.Y = data['sigmaY']
        self.sigmaY_size_input.setValue(self.Y)
        self.Yslider_label.setValue(self.Y*100)
        self.Yslider_label.setRange(1, self.Y*200)

class Gaus_Blur_MainWidget3D(MWB, Widget_Base8):
           #define Signal
    
    previewState = Signal(bool)
    linkState = Signal(bool)
    kValueChanged = Signal(float) #change to "released" for better user experience (update when slider released)
    XValueChanged = Signal(float)
    YValueChanged = Signal(float)
    sigChanged = Signal(float)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        # self.resize(300, 300)
        default = 2
        default_range = default*2
        self.X = default
        self.Y = default
        self.k = default

                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('sigmaZ:')
        self.ksize_label.setStyleSheet('font-size: 14px;')
        self.k_size_input = QDoubleSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range*100)    
        self.slider_label.setSingleStep(2)
        self.slider_label.setValue(default*100)
        
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaX:')
        self.sigmaX_label.setStyleSheet('font-size: 14px;')
        self.sigmaX_size_input = QDoubleSpinBox()
        self.sigmaX_size_input.setValue(default)
        self.sigmaX_size_input.setKeyboardTracking(False)
        self.sigmaX_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #Xslider
        self.Xslider_label = QSlider(Qt.Horizontal)
        self.Xslider_label.setRange(1, default_range*100)
        self.Xslider_label.setValue(default*100)
        #sigmaY ------------
        self.sigmaY_label = QLabel('sigmaY:')
        self.sigmaY_label.setStyleSheet('font-size: 14px;')
        self.sigmaY_size_input = QDoubleSpinBox()
        self.sigmaY_size_input.setValue(default)
        self.sigmaY_size_input.setKeyboardTracking(False)
        self.sigmaY_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #Yslider
        self.Yslider_label = QSlider(Qt.Horizontal)
        self.Yslider_label.setRange(1, default_range*100)    
        self.Yslider_label.setValue(default*100)
        #link
        self.link_checkbox = QCheckBox()
        self.link_checkbox.setText('Link Sigma X and Y values')
        self.link_checkbox.setStyleSheet('font-size: 14px;')
        self.sigma_label = QLabel('      (use sigma X slider)') #to adjust both sigmas
        self.sigma_label.setStyleSheet('font-size: 14px;')
        self.link_checkbox.setCheckState(Qt.Checked)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setStyleSheet('font-size: 14px;')
        self.preview_checkbox.setCheckState(Qt.Checked)

        self.warning_2D = QLabel('')
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        #warning!
        self.layout1.addWidget(self.warning_2D)
        # Link
        self.layout1.addWidget(self.link_checkbox)
        self.layout1.addWidget(self.sigma_label)
        # Sigma X
        self.layout1.addWidget(self.sigmaX_label)
        self.layout1.addWidget(self.sigmaX_size_input)
        self.layout1.addWidget(self.Xslider_label)
        # Sigma Y
        self.layout1.addWidget(self.sigmaY_label)
        self.layout1.addWidget(self.sigmaY_size_input)
        self.layout1.addWidget(self.Yslider_label)
         # ksize
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.k_size_input)
        self.layout1.addWidget(self.slider_label)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        # Image
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.k_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        self.sigmaX_size_input.editingFinished.connect(self.the_Xspinbox_was_changed)
        self.sigmaY_size_input.editingFinished.connect(self.the_Yspinbox_was_changed)
        # Slider triggers
        # Note, the released signal does not return a value
        # Therfore within the connected method, the value is updated: value = self.slider.value()
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        self.Xslider_label.sliderMoved.connect(self.the_Xslider_was_changed)
        self.Yslider_label.sliderMoved.connect(self.the_Yslider_was_changed)

        self.slider_label.sliderReleased.connect(self.the_slider_was_released)
        self.Xslider_label.sliderReleased.connect(self.the_Xslider_was_released)
        self.Yslider_label.sliderReleased.connect(self.the_Yslider_was_released)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)
        self.link_checkbox.stateChanged.connect(self.link_checkbox_checked)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def warning(self, warn):
        if warn == 1:
            self.warning_2D.setText("WARNING!\n3D Gaussian Blur cannot be performed on 2D data.\nPreprocessing has not been performed on the data.\nTo use this node, please read 3D data into the pipeline.\nOtherwise, please delete this node from your pipeline.")
            style = """
                background-color: #BC0000;
                border-style: outset;
                border-width: 2px;
                border-radius: 10px;
                border-color: beige;
                font: 14px;
                min-width: 10em;
                padding: 6px;
            """
            self.warning_2D.setStyleSheet(style)
            # self.warning_2D.setStyleSheet('font-size: 14px; color: red;')
        elif warn == 0:
            self.warning_2D.setText(' ')
            self.warning_2D.setStyleSheet('font-size: 1px;')

    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
        
    # link
    def link_checkbox_checked(self, state):
        self.linkState.emit(state)
        if  state == Qt.Checked:
            self.sigChanged.emit(self.X)
            self.Y = self.X
            self.Yslider_label.setRange(1, self.Y*200)
            self.Xslider_label.setRange(1, self.X*200)
            self.Yslider_label.setValue(self.Y*100)
            self.sigmaY_size_input.setValue(self.Y)

            # self.k = self.X
            # self.slider_label.setRange(1, self.Y*200)
            # self.slider_label.setValue(self.Y*100)
            # self.k_size_input.setValue(self.Y)

        #self.linkState is a signal object, not a boolean value: if  self.linkState == True: will not work
              
    # sigmaZ (last parameter on node) ----------------------------------------------------
    def the_spinbox_was_changed(self):    
        # if self.link_checkbox.isChecked():
        #     self.k_size_input.setValue(self.X)
        # else:
        v = self.k_size_input.value()*100
        self.slider_label.setRange(1, v*2)
        self.slider_label.setValue(v)
        self.k = self.k_size_input.value()
        self.kValueChanged.emit(self.k)

    
    def the_slider_was_changed(self):    # v: value emitted by a slider signal 
        # if self.link_checkbox.isChecked():
        #     self.slider_label.setValue(self.X*100)
        v = self.slider_label.value()
        self.k = v/100
        self.k_size_input.setValue(self.k)
        
    
    def the_slider_was_released(self):    # v: value emitted by a slider signal 
        
        self.kValueChanged.emit(self.k)
        print(f"self.k emmitted when slider released {self.k}")

        
    # sigmaX --------------------------------------------------
    def the_Xspinbox_was_changed(self): 
        v = self.sigmaX_size_input.value()*100   
        self.Xslider_label.setRange(1, v*2)  
        self.Xslider_label.setValue(v)  
        self.X = self.sigmaX_size_input.value()

        if self.link_checkbox.isChecked():
            self.sigChanged.emit(self.X)
            # update Y spinbox
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)

            # update Y slider
            self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            self.Y = self.sigmaX_size_input.value()
        else:
            self.XValueChanged.emit(self.X) 
            
            

    def the_Xslider_was_changed(self, v):         
        self.X = v/100
        self.sigmaX_size_input.setValue(self.X)

        if self.link_checkbox.isChecked():
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)

            self.Yslider_label.setValue(v)  
    
    def the_Xslider_was_released(self):
        # emit to X value to node.py to perform 3D gaus blur 
        # (only when slider released to improve user experience)
        if self.link_checkbox.isChecked():
            # sigChanged updates both X and Y
            self.sigChanged.emit(self.X)
           
        else:
            # update X only 
            self.XValueChanged.emit(self.X)
            

    # sigma Y -------------------------------------------------
    def the_Yspinbox_was_changed(self): 
        if self.link_checkbox.isChecked():
            self.sigmaY_size_input.setValue(self.X)
        else:
            v = self.sigmaY_size_input.value()*100   
            self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            self.Y = self.sigmaY_size_input.value()
            self.YValueChanged.emit(self.Y)


    def the_Yslider_was_changed(self, v):  
        # v = self.Yslider_label.value()  
        if self.link_checkbox.isChecked():
            # prevent user from moving y (just set back to x value)
            self.Yslider_label.setValue(self.X*100) 
        else:
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)
            
    
    def the_Yslider_was_released(self):
        # emit to Y value to node.py to perform 3D gaus blur 
        # (only when slider released to improve user experience)
        if not self.link_checkbox.isChecked():
            self.YValueChanged.emit(self.Y)
            print(f"not linked update Y, emitted value: {self.Y}")


    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'sigmaZ': self.X,
            'sigmaX': self.X,
            'sigmaY': self.Y,
            'linked': self.link_checkbox.isChecked(),
        }

    def set_state(self, data: dict):
        self.link_checkbox.setChecked(data['linked'])
        #ksize
        self.k = data['sigmaZ']
        self.k_size_input.setValue(self.k)
        self.slider_label.setValue(self.k*100)
        self.slider_label.setRange(1, self.k*200)
        
        self.X = data['sigmaX']
        self.sigmaX_size_input.setValue(self.X)
        self.Xslider_label.setValue(self.X*100)
        self.Xslider_label.setRange(1, self.X*200)
    
        self.Y = data['sigmaY']
        self.sigmaY_size_input.setValue(self.Y)
        self.Yslider_label.setValue(self.Y*100)
        self.Yslider_label.setRange(1, self.Y*200)
        

class Bilateral_MainWidget(MWB, Widget_Base8):    
    previewState = Signal(bool)
    linkState = Signal(bool)
    kValueChanged = Signal(int)
    kReleased = Signal(int)
    XValueChanged = Signal(float)
    YValueChanged = Signal(float)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        # self.resize(300, 300)
        default = 5
        default_range = default*2
        self.X = default
        self.Y = default
        self.k = default
                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('diameter:')
        self.ksize_label.setStyleSheet('font-size: 14px;')
        self.k_size_input = QSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range)    
        self.slider_label.setSingleStep(2)
        self.slider_label.setValue(default)
        
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaColour:')
        self.sigmaX_label.setStyleSheet('font-size: 14px;')
        self.sigmaX_size_input = QDoubleSpinBox()
        self.sigmaX_size_input.setValue(default)
        self.sigmaX_size_input.setKeyboardTracking(False)
        self.sigmaX_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #Xslider
        self.Xslider_label = QSlider(Qt.Horizontal)
        self.Xslider_label.setRange(1, default_range*100)
        self.Xslider_label.setValue(default*100)
        #sigmaY ------------
        self.sigmaY_label = QLabel('sigmaSpace:')
        self.sigmaY_label.setStyleSheet('font-size: 14px;')
        self.sigmaY_size_input = QDoubleSpinBox()
        self.sigmaY_size_input.setValue(default)
        self.sigmaY_size_input.setKeyboardTracking(False)
        self.sigmaY_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #Yslider
        self.Yslider_label = QSlider(Qt.Horizontal)
        self.Yslider_label.setRange(1, default_range*100)    
        self.Yslider_label.setValue(default*100)
        #link
        self.link_checkbox = QCheckBox()
        self.link_checkbox.setText('Link Sigma values')
        self.link_checkbox.setStyleSheet('font-size: 14px;')
        self.sigma_label = QLabel('      (use sigmaColour slider)') #to adjust both sigmas
        self.sigma_label.setStyleSheet('font-size: 14px;')
        self.link_checkbox.setCheckState(Qt.Checked)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setStyleSheet('font-size: 14px;')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # ksize
        self.layout1.addWidget(self.ksize_label)
        self.layout1.addWidget(self.k_size_input)
        self.layout1.addWidget(self.slider_label)
        # Sigma X
        self.layout1.addWidget(self.sigmaX_label)
        self.layout1.addWidget(self.sigmaX_size_input)
        self.layout1.addWidget(self.Xslider_label)
        # Link
        self.layout1.addWidget(self.link_checkbox)
        self.layout1.addWidget(self.sigma_label)
        # Sigma Y
        self.layout1.addWidget(self.sigmaY_label)
        self.layout1.addWidget(self.sigmaY_size_input)
        self.layout1.addWidget(self.Yslider_label)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        # Image
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.k_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        self.sigmaX_size_input.editingFinished.connect(self.the_Xspinbox_was_changed)
        self.sigmaY_size_input.editingFinished.connect(self.the_Yspinbox_was_changed)
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        self.Xslider_label.sliderMoved.connect(self.the_Xslider_was_changed)
        self.Yslider_label.sliderMoved.connect(self.the_Yslider_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)
        self.link_checkbox.stateChanged.connect(self.link_checkbox_checked)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
        
    # link
    def link_checkbox_checked(self, state):
        self.linkState.emit(state)
        if  state == Qt.Checked:
            self.Y = self.X
            self.Yslider_label.setRange(1, self.Y*200)
            self.Xslider_label.setRange(1, self.X*200)
            self.Yslider_label.setValue(self.Y*100)
            self.sigmaY_size_input.setValue(self.Y)

        #self.linkState is a signal object, not a boolean value: if  self.linkState == True: will not work
              
    # kernell ----------------------------------------------------
    def the_spinbox_was_changed(self):    
        self.slider_label.setRange(1, (self.k_size_input.value()*2 + 1))  
        if (self.k_size_input.value() % 2) == 0:
            self.k = self.k_size_input.value()+1 
        else:
             self.k = self.k_size_input.value()
             
        self.slider_label.setValue(self.k)
        self.k_size_input.setValue(self.k) #updates even number in spinbox to odd
        self.kValueChanged.emit(self.k)
    
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal 
        # self.k = v
        # self.k_size_input.setValue(self.k)
        # self.kValueChanged.emit(self.k)
        if (v % 2) == 0:
            self.k = v+1 
            # #print('even')
        else:
            self.k = v
            #  #print('odd')
        self.k_size_input.setValue(self.k)
        self.kValueChanged.emit(self.k)
    # sigmaX --------------------------------------------------
    def the_Xspinbox_was_changed(self): 
        v = self.sigmaX_size_input.value()*100   
        self.Xslider_label.setRange(1, v*2)  
        self.Xslider_label.setValue(v)  
        self.X = self.sigmaX_size_input.value()
        self.XValueChanged.emit(self.X)
        # #print(self.link_checkbox.isChecked())

        if self.link_checkbox.isChecked():
            # update Y spinbox
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)

            # update Y slider
            self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            self.Y = self.sigmaX_size_input.value()
            
            self.YValueChanged.emit(self.Y)

    def the_Xslider_was_changed(self, v):    
        self.X = v/100
        self.sigmaX_size_input.setValue(self.X)
        self.XValueChanged.emit(self.X)
        
        if self.link_checkbox.isChecked():
            # update Y spinbox
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)

            # update Y slider
            #print(self.X, self.Y)
            # self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            
            self.YValueChanged.emit(self.Y)

    # sigma Y -------------------------------------------------
    def the_Yspinbox_was_changed(self): 
        if not self.link_checkbox.isChecked():
            v = self.sigmaY_size_input.value()*100   
            self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            self.Y = self.sigmaY_size_input.value()
            self.YValueChanged.emit(self.Y)

        if self.link_checkbox.isChecked():
            self.sigmaY_size_input.setValue(self.X)

    def the_Yslider_was_changed(self, v):    
        if not self.link_checkbox.isChecked():
            self.Y = v/100
            self.sigmaY_size_input.setValue(self.Y)
            self.YValueChanged.emit(self.Y)

        if self.link_checkbox.isChecked():
            self.Yslider_label.setValue(self.X*100)   
   
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'ksize': self.k,
            'sigmaX': self.X,
            'sigmaY': self.Y,
            'linked': self.link_checkbox.isChecked(),
        }

    def set_state(self, data: dict):
        self.link_checkbox.setChecked(data['linked'])
        #ksize
        self.k_size_input.setValue(data['ksize'])
        self.slider_label.setValue(data['ksize'])
        self.slider_label.setRange(1, data['ksize']*2)
        self.k = data['ksize']

        self.sigmaX_size_input.setValue(data['sigmaX'])
        self.Xslider_label.setValue(data['sigmaX']*100)
        self.Xslider_label.setRange(1, data['sigmaX']*200)
        self.X = data['sigmaX']

        self.sigmaY_size_input.setValue(data['sigmaY'])
        self.Yslider_label.setValue(data['sigmaY']*100)
        self.Yslider_label.setRange(1, data['sigmaY']*200)
        self.Y = data['sigmaY']

# Thresholding Widget Base
# thresh #maxval
class Threshold_Manual_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    threshValueChanged = Signal(int)  #threshold
    mvValueChanged = Signal(int)      #maxval

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 100
        default2 = 255
        self.t = default1
        self.mv = default2
        
                
        #Added Widget -----------------------------------------------
        #threshold------------
        self.tsize_label = QLabel('threshold value:')
        self.tsize_label.setStyleSheet('font-size: 14px;')
        self.t_size_input = QSpinBox()
        self.t_size_input.setMaximum(255)
        self.t_size_input.setValue(default1)
        self.t_size_input.setKeyboardTracking(False)
        self.t_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #thresh slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, 255)    
        self.slider_label.setValue(default1)
        #maxvalue ------------
        self.maxv_label = QLabel('maximum value:')
        self.maxv_label.setStyleSheet('font-size: 14px;')
        self.maxv_size_input = QSpinBox()
        self.maxv_size_input.setMaximum(255)
        self.maxv_size_input.setValue(default2)
        self.maxv_size_input.setKeyboardTracking(False)
        self.maxv_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #mv slider
        self.mvslider_label = QSlider(Qt.Horizontal)
        self.mvslider_label.setRange(1, 255)
        self.mvslider_label.setValue(default2)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # tsize
        self.layout1.addWidget(self.tsize_label)
        self.layout1.addWidget(self.t_size_input)
        self.layout1.addWidget(self.slider_label)
        # maxval
        self.layout1.addWidget(self.maxv_label)
        self.layout1.addWidget(self.maxv_size_input)
        self.layout1.addWidget(self.mvslider_label)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.t_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        self.maxv_size_input.editingFinished.connect(self.the_mvspinbox_was_changed)
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        self.mvslider_label.sliderMoved.connect(self.the_mvslider_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def the_spinbox_was_changed(self):    
        # self.slider_label.setRange(1, (self.t_size_input.value()*2))  
        self.t = self.t_size_input.value()
        self.slider_label.setValue(self.t)
        self.t_size_input.setValue(self.t)
        self.threshValueChanged.emit(self.t)
    
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal
        self.t = v
            #  #print('odd')
        self.t_size_input.setValue(self.t)
        self.threshValueChanged.emit(self.t)
    # maxval --------------------------------------------------
    def the_mvspinbox_was_changed(self): 
        # self.mvslider_label.setRange(1, (self.maxv_size_input.value()*2))  
        self.mv = self.maxv_size_input.value()
        self.mvslider_label.setValue(self.mv)
        self.maxv_size_input.setValue(self.mv)
        self.mvValueChanged.emit(self.mv)

    def the_mvslider_was_changed(self, v):    
        self.mv = v
        self.maxv_size_input.setValue(self.mv)
        self.mvValueChanged.emit(self.mv)
        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t,
            'val2': self.mv,
        }

    def set_state(self, data: dict):
        #ksize
        self.t_size_input.setValue(data['val1'])
        self.slider_label.setValue(data['val1'])
        self.slider_label.setRange(1,255)
        self.t = data['val1']

        self.maxv_size_input.setValue(data['val2'])
        self.mvslider_label.setValue(data['val2'])
        self.mvslider_label.setRange(1, 255)
        self.mv = data['val2']

# Local Thresholding Widget Base
# blocksize: odd integers 3, 5, 7, 11...
# C: constant subtracted from the mean or weighted mean: positive, zero, or negative. 
class Threshold_Local_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    threshValueChanged = Signal(int)  #threshold
    mvValueChanged = Signal(int)      #maxval

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        # Block size
        default1 = 11
        default_range1 = 255
        # C
        default2 = 2
        default_range2 = 10
        self.t = default1
        self.mv = default2

        
                
        #Added Widget -----------------------------------------------
        #threshold------------
        self.tsize_label = QLabel('block size:')
        self.tsize_label.setStyleSheet('font-size: 14px;')
        self.t_size_input = QSpinBox()
        self.t_size_input.setMaximum(255)
        self.t_size_input.setValue(default1)
        self.t_size_input.setKeyboardTracking(False)
        self.t_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #thresh slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(3, 255)    
        self.slider_label.setValue(default1)
        self.slider_label.setSingleStep(2)
        #maxvalue ------------
        self.maxv_label = QLabel('C:')
        self.maxv_label.setStyleSheet('font-size: 14px;')
        self.maxv_size_input = QSpinBox()
        # unlikely to enter a number this high (but doesn't matter)
        self.maxv_size_input.setMaximum(255)
        self.maxv_size_input.setValue(default2)
        self.maxv_size_input.setKeyboardTracking(False)
        self.maxv_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        #mv slider
        self.mvslider_label = QSlider(Qt.Horizontal)
        self.mvslider_label.setRange(1, default_range2)
        self.mvslider_label.setValue(default2)
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # tsize
        self.layout1.addWidget(self.tsize_label)
        self.layout1.addWidget(self.t_size_input)
        self.layout1.addWidget(self.slider_label)
        # maxval
        self.layout1.addWidget(self.maxv_label)
        self.layout1.addWidget(self.maxv_size_input)
        self.layout1.addWidget(self.mvslider_label)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.t_size_input.editingFinished.connect(self.the_spinbox_was_changed)  #USER ONLY
        self.maxv_size_input.editingFinished.connect(self.the_mvspinbox_was_changed)
        # Slider triggers
        self.slider_label.sliderMoved.connect(self.the_slider_was_changed)
        self.mvslider_label.sliderMoved.connect(self.the_mvslider_was_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def the_spinbox_was_changed(self):    
        self.slider_label.setRange(1, (self.t_size_input.value()*2 + 1))  
        if (self.t_size_input.value() % 2) == 0:
            self.t = self.t_size_input.value()+1 
        else:
             self.t = self.t_size_input.value()
             
        self.slider_label.setValue(self.t)
        self.t_size_input.setValue(self.t) #updates even number in spinbox to odd
        self.threshValueChanged.emit(self.t)
    
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal 
        if (v % 2) == 0:
            self.t = v+1 
            # #print('even')
        else:
            self.t = v
            #  #print('odd')
        self.t_size_input.setValue(self.t)
        self.threshValueChanged.emit(self.t)
    # def the_spinbox_was_changed(self):    
    #     # self.slider_label.setRange(1, (self.t_size_input.value()*2))  
    #     self.t = self.t_size_input.value()
    #     self.slider_label.setValue(self.t)
    #     self.t_size_input.setValue(self.t)
    #     self.threshValueChanged.emit(self.t)
    
    # def the_slider_was_changed(self, v):    # v: value emitted by a slider signal
    #     self.t = v
    #         #  #print('odd')
    #     self.t_size_input.setValue(self.t)
    #     self.threshValueChanged.emit(self.t)
    # maxval --------------------------------------------------
    def the_mvspinbox_was_changed(self): 
        # self.mvslider_label.setRange(1, (self.maxv_size_input.value()*2))  
        self.mv = self.maxv_size_input.value()
        self.mvslider_label.setValue(self.mv)
        self.maxv_size_input.setValue(self.mv)
        self.mvValueChanged.emit(self.mv)

    def the_mvslider_was_changed(self, v):    
        self.mv = v
        self.maxv_size_input.setValue(self.mv)
        self.mvValueChanged.emit(self.mv)
        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t,
            'val2': self.mv,
        }

    def set_state(self, data: dict):
        #ksize
        self.t_size_input.setValue(data['val1'])
        self.slider_label.setValue(data['val1'])
        # self.slider_label.setRange(1, data['val1']*2)
        self.t = data['val1']

        self.maxv_size_input.setValue(data['val2'])
        self.mvslider_label.setValue(data['val2'])
        # self.mvslider_label.setRange(1, data['val2']*2)
        self.mv = data['val2']

# Global Thresholding Widget Base
class Global_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
              
        #Added Widget -----------------------------------------------
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        # #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    

#thresh #maxval
class Morphological_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(int)  #kernel

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 2
        default_range1 = default1*2
        self.t = default1
                
        #Added Widget -----------------------------------------------
        #kernel size------------
        self.label_1 = QLabel('kernel/radius:')
        self.label_1.setStyleSheet('font-size: 14px;')
        self.input_1 = QSpinBox()
        self.input_1.setMaximum(255)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #kernal slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1)    
        self.slider_label_1.setValue(default1)

        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # kernel
        self.layout1.addWidget(self.label_1)
        self.layout1.addWidget(self.input_1)
        self.layout1.addWidget(self.slider_label_1)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.input_1.editingFinished.connect(self.spinbox_1_changed)  #USER ONLY
        # Slider triggers
        self.slider_label_1.sliderMoved.connect(self.slider_1_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.prev_checkbox_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def prev_checkbox_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def spinbox_1_changed(self):    
        self.slider_label_1.setRange(1, (self.input_1.value()*2))  
        self.t = self.input_1.value()
        self.slider_label_1.setValue(self.t)
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    
    def slider_1_changed(self, v):    # v: value emitted by a slider signal
        self.t = v
            #  #print('odd')
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t
        }

    def set_state(self, data: dict):
        #ksize
        self.input_1.setValue(data['val1'])
        self.slider_label_1.setValue(data['val1'])
        self.slider_label_1.setRange(1, data['val1']*2)
        self.t = data['val1']

class Dilate_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(int)  #kernel
    Value2Changed = Signal(int)  #itt

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 10
        default_range1 = 255
        default2 = 1
        default_range2 = 5
        # t2: itteration (max = 5 hard coded)
        # default_range2 = default2*2
        self.t = default1
        self.t2 = default2
                
        #Added Widget -----------------------------------------------
        #kernel size------------
        self.label_1 = QLabel('kernel:')
        self.label_1.setStyleSheet('font-size: 14px;')
        self.input_1 = QSpinBox()
        self.input_1.setMaximum(default_range1)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #kernal slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1)    
        self.slider_label_1.setValue(default1)

        #itteration size------------
        self.label_2 = QLabel('iteration:')
        self.label_2.setStyleSheet('font-size: 14px;')
        self.input_2 = QSpinBox()
        self.input_2.setMaximum(default_range2)
        self.input_2.setValue(1)
        self.input_2.setKeyboardTracking(False)
        self.input_2.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #itt slider
        self.slider_label_2 = QSlider(Qt.Horizontal)
        self.slider_label_2.setRange(1, default_range2)    
        self.slider_label_2.setValue(1)
       
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # kernel
        self.layout1.addWidget(self.label_1)
        self.layout1.addWidget(self.input_1)
        self.layout1.addWidget(self.slider_label_1)
        # itt
        self.layout1.addWidget(self.label_2)
        self.layout1.addWidget(self.input_2)
        self.layout1.addWidget(self.slider_label_2)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.input_1.editingFinished.connect(self.spinbox_1_changed)  #USER ONLY
        self.input_2.editingFinished.connect(self.spinbox_2_changed)
        # Slider triggers
        self.slider_label_1.sliderMoved.connect(self.slider_1_changed)
        self.slider_label_2.sliderMoved.connect(self.slider_2_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.prev_checkbox_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def prev_checkbox_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def spinbox_1_changed(self):    
        self.slider_label_1.setRange(1, (self.input_1.value()*2))  
        self.t = self.input_1.value()
        self.slider_label_1.setValue(self.t)
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    
    def slider_1_changed(self, v):    # v: value emitted by a slider signal
        self.t = v
            #  #print('odd')
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    #itt
    def spinbox_2_changed(self):    
        self.slider_label_2.setRange(1, (self.input_2.value()*2))  
        self.t2 = self.input_2.value()
        self.slider_label_2.setValue(self.t2)
        self.input_2.setValue(self.t2)
        self.Value2Changed.emit(self.t2)
    
    def slider_2_changed(self, v):    # v: value emitted by a slider signal
        self.t2 = v
            #  #print('odd')
        self.input_2.setValue(self.t2)
        self.Value2Changed.emit(self.t2)    
       
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t,
            'val2': self.t2,
        }

    def set_state(self, data: dict):
        #ksize
        self.input_1.setValue(data['val1'])
        self.slider_label_1.setValue(data['val1'])
        self.slider_label_1.setRange(1, data['val1']*2)
        self.t = data['val1']

        self.input_2.setValue(data['val2'])
        self.slider_label_2.setValue(data['val2'])
        self.slider_label_2.setRange(1, data['val2']*2)
        self.t2 = data['val2']

class Alpha_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(float)  #kernel
    Value2Changed = Signal(float)  #itt

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 3
        default_range1 = default1*2
        default2 = 1
        default_range2 = 5
        self.t = default1
        self.t2 = default2
                
        #Added Widget -----------------------------------------------
        #alpha size------------
        self.label_1 = QLabel('alpha:')
        self.label_1.setStyleSheet('font-size: 14px;')
        self.input_1 = QDoubleSpinBox()
        # self.input_1.setMaximum(255)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #a slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1*100)    
        self.slider_label_1.setValue(default1*100)

        #beta size------------
        self.label_2 = QLabel('beta:')
        self.label_2.setStyleSheet('font-size: 14px;')
        self.input_2 = QDoubleSpinBox()
        # self.input_2.setMaximum(5)
        self.input_2.setValue(default2)
        self.input_2.setKeyboardTracking(False)
        self.input_2.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #b slider
        self.slider_label_2 = QSlider(Qt.Horizontal)
        self.slider_label_2.setRange(1, default_range2*100)    
        self.slider_label_2.setValue(default2)
       
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # kernel
        self.layout1.addWidget(self.label_1)
        self.layout1.addWidget(self.input_1)
        self.layout1.addWidget(self.slider_label_1)
        # itt
        self.layout1.addWidget(self.label_2)
        self.layout1.addWidget(self.input_2)
        self.layout1.addWidget(self.slider_label_2)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.input_1.editingFinished.connect(self.spinbox_1_changed)  #USER ONLY
        self.input_2.editingFinished.connect(self.spinbox_2_changed)
        # Slider triggers
        self.slider_label_1.sliderMoved.connect(self.slider_1_changed)
        self.slider_label_2.sliderMoved.connect(self.slider_2_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.prev_checkbox_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def prev_checkbox_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def spinbox_1_changed(self):    
        self.t = self.input_1.value()
        v = self.t*100
        self.slider_label_1.setRange(1, v*2) 
        self.slider_label_1.setValue(v)
        # self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    
    def slider_1_changed(self, v):    # v: value emitted by a slider signal
        self.t = v/100
            #  #print('odd')
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    #itt
    def spinbox_2_changed(self):    
        self.t2 = self.input_2.value()
        v = self.input_2.value()*100
        self.slider_label_2.setRange(1, v*2)
        self.slider_label_2.setValue(v)
        # self.input_2.setValue(self.t2)
        self.Value2Changed.emit(self.t2)
    
    def slider_2_changed(self, v):    # v: value emitted by a slider signal
        self.t2 = v/100
            #  #print('odd')
        self.input_2.setValue(self.t2)
        self.Value2Changed.emit(self.t2)    
       
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t,
            'val2': self.t2,
        }

    def set_state(self, data: dict):
        #ksize
        self.input_1.setValue(data['val1'])
        self.slider_label_1.setValue(data['val1']*100)
        self.slider_label_1.setRange(1, data['val1']*200)
        self.t = data['val1']

        self.input_2.setValue(data['val2'])
        self.slider_label_2.setValue(data['val2']*100)
        self.slider_label_2.setRange(1, data['val2']*200)
        self.t2 = data['val2']


class Gamma_Corr_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(float)  #kernel
    Value2Changed = Signal(float)  #itt

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 0.5
        default_range1 = default1*3
        self.t = default1
                
        #Added Widget -----------------------------------------------
        #alpha size------------
        self.label_1 = QLabel('gamma:')
        self.label_1.setStyleSheet('font-size: 14px;')
        self.input_1 = QDoubleSpinBox()
        # self.input_1.setMaximum(255)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #a slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1*100)    
        self.slider_label_1.setValue(default1*100)
       
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # kernel
        self.layout1.addWidget(self.label_1)
        self.layout1.addWidget(self.input_1)
        self.layout1.addWidget(self.slider_label_1)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.input_1.editingFinished.connect(self.spinbox_1_changed)  #USER ONLY
        # Slider triggers
        self.slider_label_1.sliderMoved.connect(self.slider_1_changed)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.prev_checkbox_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def prev_checkbox_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def spinbox_1_changed(self):    
        self.t = self.input_1.value()
        v = self.t*100
        self.slider_label_1.setRange(1, v*3) 
        self.slider_label_1.setValue(v)
        self.Value1Changed.emit(self.t)
    
    def slider_1_changed(self, v):    # v: value emitted by a slider signal
        self.t = v/100
            #  #print('odd')
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)   
       
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t,
        }

    def set_state(self, data: dict):
        #ksize
        self.input_1.setValue(data['val1'])
        self.slider_label_1.setValue(data['val1']*100)
        self.slider_label_1.setRange(1, data['val1']*300)
        self.t = data['val1']

class HisogramWidg(MWB, QWidget):
    displayHist = Signal(bool)
    LogHist = Signal(bool)
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        layout = QVBoxLayout(self)
        self.image = QLabel()
        self.button = QPushButton('Show histogram')
        self.logCheckbox = QCheckBox()
        self.label = QLabel('Histogram for binarized images')
        self.logCheckbox.setText('log scale (click the checkbox)')
        self.logCheckbox.setStyleSheet('font-size: 14px;')
        self.logCheckbox.setChecked(False)
        self.figure = Figure(dpi=100) #figsize=(0.3, 0.2)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(420, 300)
        # self.figure.subplots_adjust(left=0.05, right=0.05, top=0.9, bottom=0.1)
        # self.figure.set_size_inches(800 / self.figure.dpi, 800/ self.figure.dpi)
        # Add a subplot and adjust its position and size within the figure
        self.sub = self.figure.add_subplot(111)
        self.sub.set_position([0.2, 0.2, 0.7, 0.7])  # Adjust the position and size as needed
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.logCheckbox)
        layout.addWidget(self.canvas)
        layout.addWidget(self.image)
        
        # self.ax = self.figure.add_subplot(111)
        self.button.clicked.connect(self.emitHist)
        # self.button.clicked.connect(self.emitHist)
        self.logCheckbox.stateChanged.connect(self.the_checkbox_was_changed)
        

        # self.plot_histogram(gray)
    def emitHist(self):
        self.displayHist.emit(True)

    def the_checkbox_was_changed(self, state):
        #print("statechanged")
        self.LogHist.emit(state)

    def clear_hist(self):
        self.logCheckbox.setChecked(False)
        self.figure.clear()
        self.sub = self.figure.add_subplot(111)
        self.sub.set_position([0.2, 0.2, 0.7, 0.7])  # Adjust the position and size as needed
        self.canvas.draw()

    def show_histogram(self, img):
        self.logCheckbox.setChecked(False)
        # if self.logCheckbox.isChecked:
        #     self.log_hist(img)
        self.figure.clear()  # Clear the previous plot, if any.
        self.sub = self.figure.add_subplot(111)
        self.sub.set_position([0.2, 0.2, 0.7, 0.7])  # Adjust the position and size as needed
        # ax = self.figure.add_subplot(111)
        # if img.shape==2:
        gray_hist = cv2.calcHist([img], [0], None, [255], [0, 255])

        self.sub.plot(gray_hist)
        self.sub.set_title('Grayscale Histogram')
        self.sub.set_xlabel('Pixel Value [0 - 255]')
        self.sub.set_ylabel('# pixels')
        self.sub.set_xlim([0, 255])
        #print(f'image.shape{img.shape}')
        # if img.shape==3:
        #     hist = cv2.calcHist([img], [1], None, [256], [0, 256])
        #     #print('hist!')
        #     self.sub.plot(hist, color='green')
        #     # colors = ('b', 'g', 'r')
        #     # for i, col in enumerate(colors):
        #     #     # self.sub = self.figure.add_subplot(111)
        #     #     # self.sub.set_position([0.2, 0.2, 0.7, 0.7]) 
        #     #     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        #     #     self.sub.plot(hist, color=col)

        #     self.sub.set_title('RGB Histogram')
        #     self.sub.set_xlabel('Pixel Value [0 - 255]')
        #     self.sub.set_ylabel('# pixels')
        #     self.sub.set_xlim([0, 255])

            # colors = ('b', 'g', 'r')
            # for i, col in enumerate(colors):
            #     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            #     self.plot.plot(hist, color=col)

            # self.plot.set_xlim([0, 255])

            # colors = ('b', 'g', 'r')
            # for i, col in enumerate(colors):
            #     hist = cv2.calcHist([img], [i], None, [256], [0,256])
            # self.sub.plot(hist, colors=col)
            # self.sub.set_title('RGB Histogram')
            # self.sub.set_xlabel('Pixel Value [0 - 255]')
            # self.sub.set_ylabel('# pixels')
            # self.sub.set_xlim([0, 255])


        # Update the canvas to display the new plot
        self.canvas.draw()
    
    def log_hist(self, img):
        self.figure.clear()  # Clear the previous plot, if any.
        self.sub = self.figure.add_subplot(111)
        self.sub.set_position([0.2, 0.2, 0.7, 0.7])  # Adjust the position and size as needed
        # ax = self.figure.add_subplot(111)
        
        gray_hist = cv2.calcHist([img], [0], None, [255], [0, 255])

        self.sub.plot(gray_hist)
        self.sub.set_title('Grayscale Histogram (Log Scale)')
        self.sub.set_xlabel('Pixel Value [0 - 255]')
        self.sub.set_ylabel('# pixels (log scale)')
        self.sub.set_xlim([0, 255])
        self.sub.set_yscale('log') 

        # Update the canvas to display the new plot
        self.canvas.draw()

class Volume_Filter(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(int)  #kernel

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 10
        default_range1 = default1*2
        self.t = default1
                
        #Added Widget -----------------------------------------------
        #kernel size------------
        self.label_1 = QLabel('volume (pixel size):')
        self.label_1.setStyleSheet('font-size: 14px;')
        self.input_1 = QSpinBox()
        self.input_1.setMaximum(2000)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #kernal slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1)    
        self.slider_label_1.setValue(default1)
        
        #preview
        self.preview_label = QLabel('Preview:')
        self.preview_label.setStyleSheet('font-size: 14px;')
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)
        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # kernel
        self.layout1.addWidget(self.label_1)
        self.layout1.addWidget(self.input_1)
        self.layout1.addWidget(self.slider_label_1)
        # Preview
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Spinbox triggers
        self.input_1.editingFinished.connect(self.spinbox_1_changed)  #USER ONLY
        # Slider triggers
        self.slider_label_1.sliderMoved.connect(self.slider_1_changed)
        self.slider_label_1.sliderReleased.connect(self.slider_1_released)
        # Check box
        self.preview_checkbox.stateChanged.connect(self.prev_checkbox_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def prev_checkbox_changed(self, state):
        self.previewState.emit(state)
          
    # thresh ----------------------------------------------------
    def spinbox_1_changed(self):    
        self.slider_label_1.setRange(1, (self.input_1.value()*2))  
        self.t = self.input_1.value()
        self.slider_label_1.setValue(self.t)
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    
    def slider_1_changed(self, v):    # v: value emitted by a slider signal
        self.t = v
            #  #print('odd')
        self.input_1.setValue(self.t)
        
    
    def slider_1_released(self):    # v: value emitted by a slider signal 
        
        self.Value1Changed.emit(self.t)
        print(f"self.k emmitted when slider released {self.t}")
        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'val1': self.t
        }

    def set_state(self, data: dict):
        #ksize
        self.input_1.setValue(data['val1'])
        self.slider_label_1.setValue(data['val1'])
        self.slider_label_1.setRange(1, data['val1']*2)
        self.t = data['val1']   

class Fill_Holes_MainWidget(MWB, Widget_Base8):
    #define Signal
    previewState = Signal(bool)
    fill_holes = Signal(bool)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
              
        #Added Widget -----------------------------------------------
        #preview
        self.preview_checkbox = QCheckBox()
        self.preview_checkbox.setText('Preview')
        self.preview_checkbox.setCheckState(Qt.Checked)

        #fill holes
        self.fill_checkbox = QCheckBox()
        self.fill_checkbox.setText('Fill holes')
        self.fill_checkbox.setCheckState(Qt.Unchecked)

        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        # Preview
        self.layout1.addWidget(self.fill_checkbox)
        self.layout1.addWidget(self.preview_checkbox)
        self.layout1.addWidget(self.image_label)
        #self.layout1.setSpacing(0) 
        self.setLayout(self.layout1)

        #Signals -------------------------------------------------
        # Check box
        self.preview_checkbox.stateChanged.connect(self.the_checkbox_was_changed)
        self.fill_checkbox.stateChanged.connect(self.the_fill_holes_checkbox_was_changed)

    # Slot Functions -------------------------------------------
    # checkbox -------------------------------------------------
    def the_checkbox_was_changed(self, state):
        self.previewState.emit(state)
    
    def the_fill_holes_checkbox_was_changed(self, state):
        self.fill_holes.emit(state)
    
    def clear_fill(self, state):
        if state == False:
            self.fill_checkbox.setChecked(False)
        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        # #print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
   

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
    V3QvBoxDev_MainWidget,
    V4QvBoxDev_MainWidget,
    V5QvBoxDev_MainWidget,

    #PipelineWidgets
    Slider_widget,
    ChooseFileInputWidget,
    PathInput,
    BatchPaths,
    Crop_MainWidget,
    OutputMetadataWidg,
    Split_Img,
    Blur_Averaging_MainWidget,
    Blur_Median_MainWidget,
    Gaus_Blur_MainWidget,
    Gaus_Blur_MainWidget3D,
    Bilateral_MainWidget,
    Threshold_Manual_MainWidget,
    Threshold_Local_MainWidget,
    Global_MainWidget,
    Morphological_MainWidget,
    Dilate_MainWidget,
    Alpha_MainWidget,
    Gamma_Corr_MainWidget,
    Read_Image_MainWidget,
    HisogramWidg,
    Volume_Filter,
    Fill_Holes_MainWidget,
)
