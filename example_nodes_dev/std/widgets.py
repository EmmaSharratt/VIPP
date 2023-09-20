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
                            QLineEdit,
                            QLabel,
                            QFileDialog,     
                            QCheckBox,     
                            QDoubleSpinBox,
                            QSpinBox,    
                            QAbstractSpinBox,  
                            QMainWindow,                                     
                            )
import cv2
import os
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
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
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
        #     rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # except cv2.error:
        #     return

        # h, w, ch = rgb_image.shape
        # aspect_ratio = w / h  # Calculate the aspect ratio of the image

        # # Calculate the new dimensions for the widget based on the aspect ratio
        # new_widget_width = 300  # You can set the width to a desired value
        # new_widget_height = int(new_widget_width / aspect_ratio)

        # self.resize(new_widget_width, new_widget_height)

        # qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
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
        file_path = QFileDialog.getOpenFileName(self, 'Select image')[0]
        try:
            file_path = os.path.relpath(file_path)
        except ValueError:
            return
        
        self.path_chosen.emit(file_path)

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

class ChooseFileInputWidgetBASE(MWB, Widget_Base):

    ValueChanged1 = Signal(int)  #time instance
    ValueChanged2 = Signal(int)  #z-slice (depth)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        default = 0
        default_range = 5

        self.shape_label = QLabel()
        self.shape_label.setStyleSheet('background-color: #1E242A; color: white; font-size: 14px;')
        self.input_label1 = QLabel('time instance:')
        self.input_label1.setStyleSheet('font-size: 14px;')
        self.input1 = QSpinBox()
        self.input1.setValue(default)
        self.input1.setKeyboardTracking(False)
        self.input1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #timeseries slider
        self.slider_label1 = QSlider(Qt.Horizontal)
        self.slider_label1.setRange(0, default_range)    
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
        self.slider_label2.setRange(0, default_range)    
        self.slider_label2.setSingleStep(2)
        self.slider_label2.setValue(default)

        #image
        self.image_label = QLabel()
               #self.image_label.resize(800, 800)
        # Layout ----------------------------------------------------
        self.layout1 = QVBoxLayout()
        self.setLayout(self.layout1)
        self.reset_widg(0)
        

        # Signals -------------------------------------------------
        # Spinbox triggers
        self.input1.editingFinished.connect(self.the_spinbox1_was_changed)  #USER ONLY
        self.input2.editingFinished.connect(self.the_spinbox2_was_changed)
        # Slider triggers
        self.slider_label1.sliderMoved.connect(self.the_slider1_was_changed)
        self.slider_label2.sliderMoved.connect(self.the_slider2_was_changed)

    #new image -> reset sliders and inputs
    def reset_widg(self, val):
        self.input1.setValue(val) #val = 0
        self.input2.setValue(val)
        self.slider_label1.setValue(val)
        self.slider_label2.setValue(val)
        

    def remove_widgets(self):
        print("REMOVE")
        # Remove and delete all widgets from the layout
        for i in reversed(range(self.layout1.count())):
            widget = self.layout1.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                print("removed")

            # Remove the widget from the layout
            self.layout1.takeAt(i)
            
    def add_widgets(self, dim):
        num_z = dim[1]  # 10
        num_time = dim[2]  # 21
        width = dim[3]  # 512
        height = dim[4]  # 512
        chan = dim[5]
        
        if dim[0] == 5:
            #shape message
            print("ADDED")
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

            message = f"Z-Slices: {num_z}\n"
            message += f"Frames (time): {num_time}\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            self.slider_label1.setRange(0, num_time-1)  
            self.slider_label2.setRange(0, num_z-1) 
        elif dim[0] == 3:
            self.layout1.addWidget(self.input_label1)
            self.layout1.addWidget(self.input1)
            self.layout1.addWidget(self.slider_label1)
            #shape message
            self.layout1.addWidget(self.shape_label)
            # z-stack
            self.layout1.addWidget(self.input_label2)
            self.layout1.addWidget(self.input2)
            self.layout1.addWidget(self.slider_label2)
            # Image
            self.layout1.addWidget(self.image_label)
            message = f"Z-Slices: {num_z}\n"
            message += f"Single Frame\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            self.slider_label2.setRange(0, num_z-1) 
        elif dim[0] == 2:
            #shape message
            self.layout1.addWidget(self.shape_label)
            # time
            self.layout1.addWidget(self.input_label1)
            self.layout1.addWidget(self.input1)
            self.layout1.addWidget(self.slider_label1)
            # Image
            self.layout1.addWidget(self.image_label)
            message = f"Single slice\n"
            message += f"Frames (time): {num_time}\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            self.slider_label2.setRange(0, num_time-1)
        elif dim[0] == 0:
            self.layout1.addWidget(self.shape_label)
            self.layout1.addWidget(self.image_label)
            message = f"2D image\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"

        
        # Set the text in self.shape_label
        self.shape_label.setText(message)

    # value 1: TIME ----------------------------------------------------
    def the_spinbox1_was_changed(self):    
        # self.slider_label1.setRange(1, (self.input1.value()*2))  #Range should not change for time / z sliders
        self.val1 = self.input1.value()
             
        self.slider_label1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)
    
    def the_slider1_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val1 = v
        self.input1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)

    # value 2: z-stack --------------------------------------------------
    def the_spinbox2_was_changed(self): 
        # self.slider_label2.setRange(1, (self.input2.value()*2))  
        self.val2 = self.input2.value()
             
        self.slider_label2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
    
    def the_slider2_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val2 = v
        self.input2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
    
    def shape_message(self, shape):
        
        if len(shape) == 4:
            num_time = shape[0]  # 21
            num_z = shape[1]  # 10
            height = shape[2]  # 512
            width = shape[3]  # 512
            # Create a formatted text message
            message = f"Number of Time Instances: {num_time}\n"
            message += f"Number of Z-Slices: {num_z}\n"
            message += f"Image Height: {height}\n"
            message += f"Image Width: {width}"
            self.slider_label1.setRange(0, num_time-1)  
            self.slider_label2.setRange(0, num_z-1) 
            print(f"numtime{num_time}")
            print(f"numz{num_z}")

        if len(shape) == 3:
            num_z = shape[0]  # 10
            height = shape[1]  # 512
            width = shape[2]  # 512
            # Create a formatted text message
            message = f"Number of Z-Slices: {num_z}\n"
            message += f"Image Height: {height}\n"
            message += f"Image Width: {width}"
            
            self.slider_label1.setRange(0, num_z-1) 
            print(num_z)
        
        # Set the text in self.shape_label
        self.shape_label.setText(message)

        

        #handel widgets
        # PERHAPS ONLY ADD THEM HERE doesnt work

        # if len(shape) >= 4:
        #     # time
        #     self.layout1.addWidget(self.input_label1)
        #     self.layout1.addWidget(self.input1)
        #     self.layout1.addWidget(self.slider_label1)

        #     self.layout1.addWidget(self.input_label2)
        #     self.layout1.addWidget(self.input2)
        #     self.layout1.addWidget(self.slider_label2)
        # #     # Remove the widget from its parent layout
        # #     self.layout1.removeWidget(self.input_label1)
        # #     self.layout1.removeWidget(self.input1)
        # #     self.layout1.removeWidget(self.slider_label1)
        # #     # Delete the widget from memory
        # #     self.input_label1.deleteLater()
        # #     self.input1.deleteLater()
        # #     self.slider_label1.deleteLater()
        # #     self.setLayout(self.layout)
        # if len(shape) >= 2:
        #     # z-stack
        #     self.layout1.addWidget(self.input_label2)
        #     self.layout1.addWidget(self.input2)
        #     self.layout1.addWidget(self.slider_label2)
        
        # # Image
        # self.layout1.addWidget(self.image_label)
        # # self.layout1.setSpacing(0) 
        # self.setLayout(self.layout1)
        # if len(shape) < 4:
        #     # Remove the widget from its parent layout if it exists
        #     self.layout1.removeWidget(self.input_label1)
        #     self.input_label1.deleteLater()
            
        #     self.layout1.removeWidget(self.input1)
        #     self.input1.deleteLater()
            
        #     self.layout1.removeWidget(self.slider_label1)
        #     self.slider_label1.deleteLater()
        # else:
        #     # Add the widgets if they are not already in the layout
            
        #     self.layout1.addWidget(self.input_label1)
            
        #     self.layout1.addWidget(self.input1)
            
        #     self.layout1.addWidget(self.slider_label1)
            
        # # Set the layout
        # self.setLayout(self.layout1)

    
    
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
        

    #     self.clicked.connect(self.button_clicked)

    # def button_clicked(self):
    #     file_path = QFileDialog.getOpenFileName(self, 'Select image')[0]
    #     try:
    #         file_path = os.path.relpath(file_path)
    #     except ValueError:
    #         return
        
    #     self.path_chosen.emit(file_path)

class ChooseFileInputWidgetBASE3(MWB, Widget_Base):

    ValueChanged1 = Signal(int)  #time instance
    ValueChanged2 = Signal(int)  #z-slice (depth)

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        default = 1
        default_range = 5

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
        self.reset_widg(1)
        

        # Signals -------------------------------------------------
        # Spinbox triggers
        self.input1.editingFinished.connect(self.the_spinbox1_was_changed)  #USER ONLY
        self.input2.editingFinished.connect(self.the_spinbox2_was_changed)
        # Slider triggers
        self.slider_label1.sliderMoved.connect(self.the_slider1_was_changed)
        self.slider_label2.sliderMoved.connect(self.the_slider2_was_changed)

    #new image -> reset sliders and inputs
    def reset_widg(self, val):
        self.input1.setValue(val) #val = 0
        self.input2.setValue(val)
        self.slider_label1.setValue(val)
        self.slider_label2.setValue(val)
        

    def remove_widgets(self):
        print("REMOVE")
        # Remove and delete all widgets from the layout
        for i in reversed(range(self.layout1.count())):
            widget = self.layout1.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                print("removed")

            # Remove the widget from the layout
            self.layout1.takeAt(i)
            
    def update_widgets(self, dim):
        num_z = dim[1]  # 10
        num_time = dim[2]  # 21
        width = dim[3]  # 512
        height = dim[4]  # 512
        chan = dim[5]
        
        if dim[0] == 5:
            message = f"Z-Slices: {num_z}\n"
            message += f"Frames (time): {num_time}\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            self.slider_label1.setRange(1, num_time)  
            self.slider_label2.setRange(1, num_z) 
        elif dim[0] == 3:
            message = f"Z-Slices: {num_z}\n"
            message += f"Frames (time): {num_time}"
            message += f" (single frame)\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            #zslider
            self.slider_label2.setRange(1, num_z) 
            #time slider (one frame)
            self.slider_label1.setRange(1, 1) 
        elif dim[0] == 2:
            #shape message
            self.layout1.addWidget(self.shape_label)
            # time
            self.layout1.addWidget(self.input_label1)
            self.layout1.addWidget(self.input1)
            self.layout1.addWidget(self.slider_label1)
            # Image
            self.layout1.addWidget(self.image_label)
            message = f"Z-Slices: {num_z}"
            message += f" (single slice)\n"
            message += f"Frames (time): {num_time}\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            self.slider_label2.setRange(0, num_time-1)
        elif dim[0] == 0:
            message = f"2D image\n"
            message += f"Z-Slices: {num_z}"
            message += f" (single slice)\n"
            message += f"Frames (time): {num_time}"
            message += f" (single frame)\n"
            message += f"Image Width: {width}\n"
            message += f"Image Height: {height}\n"
            message += f"Colour channels: {chan}"
            #zslice
            self.slider_label2.setRange(1, 1) 
            #time slider (one frame)
            self.slider_label1.setRange(1, 1) 
        
        # Set the text in self.shape_label
        self.shape_label.setText(message)

    # value 1: TIME ----------------------------------------------------
    def the_spinbox1_was_changed(self):    
        # self.slider_label1.setRange(1, (self.input1.value()*2))  #Range should not change for time / z sliders
        self.val1 = self.input1.value()
             
        self.slider_label1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)
    
    def the_slider1_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val1 = v
        self.input1.setValue(self.val1)
        self.ValueChanged1.emit(self.val1)

    # value 2: z-stack --------------------------------------------------
    def the_spinbox2_was_changed(self): 
        # self.slider_label2.setRange(1, (self.input2.value()*2))  
        self.val2 = self.input2.value()
             
        self.slider_label2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
    
    def the_slider2_was_changed(self, v):    # v: value emitted by a slider signal 
        self.val2 = v
        self.input2.setValue(self.val2)
        self.ValueChanged2.emit(self.val2)
       
    
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
        

    #     self.clicked.connect(self.button_clicked)

    # def button_clicked(self):
    #     file_path = QFileDialog.getOpenFileName(self, 'Select image')[0]
    #     try:
    #         file_path = os.path.relpath(file_path)
    #     except ValueError:
    #         return
        
    #     self.path_chosen.emit(file_path)

class ChooseFileInputWidgetBASE2(MWB, QMainWindow):
    def __init__(self, parent=None):
        MWB.__init__(self, parent)  # Call the MWB constructor
        QMainWindow.__init__(self)  # Call the QMainWindow constructor
        # Initialize any additional properties or widgets for your custom widget here

        # Button to add Widget1
        self.addButton1 = QPushButton('Add Widget 1')
        self.addButton1.clicked.connect(self.addWidget1)

        # Button to add Widget2
        self.addButton2 = QPushButton('Add Widget 2')
        self.addButton2.clicked.connect(self.addWidget2)

        # Button to remove all widgets
        self.removeAllButton = QPushButton('Remove All Widgets')
        self.removeAllButton.clicked.connect(self.removeAllWidgets)

        # Main layout
        self.mainLayout = QVBoxLayout()

        # Add buttons to the main layout
        self.mainLayout.addWidget(self.addButton1)
        self.mainLayout.addWidget(self.addButton2)
        self.mainLayout.addWidget(self.removeAllButton)

        # List to keep track of added widgets
        self.addedWidgets = []

        # Widget to contain added widgets
        self.widgetContainer = QWidget()
        self.widgetContainerLayout = QVBoxLayout(self.widgetContainer)
        self.mainLayout.addWidget(self.widgetContainer)

        # Set central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.centralWidget)

    def add_widgets(self):
        new_widget = TestWidget1()
        self.widgetContainerLayout.addWidget(new_widget)
        self.addedWidgets.append(new_widget)
        self.adjustSize()  # Resize the main window
        self.resize(self.width(), self.height()+15)
        self.node.update_shape()

    def addWidget2(self):
        new_widget = TestWidget2()
        self.widgetContainerLayout.addWidget(new_widget)
        self.addedWidgets.append(new_widget)
        self.adjustSize()  # Resize the main window
        self.resize(self.width(), self.height()+15)
        self.node.update_shape()

    def removeAllWidgets(self):
        for widget in self.addedWidgets:
            widget.deleteLater()
        self.addedWidgets = []
        self.adjustSize()  # Resize the main window
        self.resize(self.width(), self.height()+15)
        self.node.update_shape()

    def resizeNode(self, width, height):
        # Resize the Ryven node based on the widget's size (you should implement this)
        # For example:
        self.node.setGeometry(0, 0, width, height)

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
    #             print("came here for Sliderwidget")
    #         else:
    #             # RGB image
    #             rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    #         print("Error:", e)


    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
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


class Blur_Averaging_MainWidget(MWB, Widget_Base):
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
        # print(f'slider:{self.slider_label.value()}')

    def the_slider_was_changed(self, v):    # v: value emitted by a signal -> slider value (0-1000)
        self.k_size_input.setValue(v)
        self.k = int(v)
        self.kValueChanged.emit(self.k)
        
    # def show_image(self, img):
    #     # self.resize(800,800)

    #     try:
    #         rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    #     # print("H:", target_height, "W:", target_width)
    #     self.image_label.setPixmap(QPixmap(qt_image))
    #     self.resize(100, 100)
    #     self.node.update_shape()
    #     # print('Update Shape:',  print(self.width(), self.height()))
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))
        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        print(self.k_size_input.value())
    
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.value(),
    #     }

    # def set_state(self, data: dict):
    #     self.setValue(data['val'])



class Blur_Median_MainWidget(MWB, Widget_Base):
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
            # print('even')
        else:
            self.k = v
            #  print('odd')
        self.k_size_input.setValue(self.k)
        self.kValueChanged.emit(self.k)
        
    # def show_image(self, img):
    #     # self.resize(800,800)

    #     try:
    #         rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    #     # print("H:", target_height, "W:", target_width)
    #     self.image_label.setPixmap(QPixmap(qt_image))
    #     self.resize(100, 100)
    #     self.node.update_shape()
    #     # print('Update Shape:',  print(self.width(), self.height()))
        
        
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))
        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
        print(self.k_size_input.value())
    
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.value(),
    #     }

    # def set_state(self, data: dict):
    #     self.setValue(data['val'])



class Gaus_Blur_MainWidget(MWB, QWidget):
           #define Signal
    
    previewState = Signal(bool)
    linkState = Signal(bool)
    kValueChanged = Signal(int)
    XValueChanged = Signal(float)
    YValueChanged = Signal(float)

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
        self.slider_label.setSingleStep(2)
        self.slider_label.setValue(default)
        
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaX:')
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
        self.link_checkbox.setText('Link')
        self.link_checkbox.setCheckState(Qt.Checked)
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
            # print('even')
        else:
            self.k = v
            #  print('odd')
        self.k_size_input.setValue(self.k)
        self.kValueChanged.emit(self.k)
    # sigmaX --------------------------------------------------
    def the_Xspinbox_was_changed(self): 
        v = self.sigmaX_size_input.value()*100   
        self.Xslider_label.setRange(1, v*2)  
        self.Xslider_label.setValue(v)  
        self.X = self.sigmaX_size_input.value()
        self.XValueChanged.emit(self.X)
        # print(self.link_checkbox.isChecked())

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
            print(self.X, self.Y)
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
    # update image ---------------------------------------------
    def show_image(self, img):
        # self.resize(800,800)

        try:
            if len(img.shape) == 2:
                # Grayscale image
                qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            else:
                # RGB image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate the target size for scaling
            scale_factor = 0.4  # Increase the scaling factor for clarity
            target_width = int(qt_image.width() * scale_factor)
            target_height = int(qt_image.height() * scale_factor)
            
            qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.resize(target_width, target_height)
            self.node.update_shape()
        except Exception as e:
            print("Error:", e)


    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    def get_state(self) -> dict:
        return {
            'ksize': self.k,
            'sigmaX': self.X,
            'sigmaY': self.Y
        }

    def set_state(self, data: dict):
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



class Bilateral_MainWidget(MWB, Widget_Base):
           #define Signal
    
    previewState = Signal(bool)
    linkState = Signal(bool)
    kValueChanged = Signal(int)     #d
    XValueChanged = Signal(float)   #sigmaColour
    YValueChanged = Signal(float)   #sigmaSpace

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default = 5
        default_range = default*2
                
        #Added Widget -----------------------------------------------
        #ksize------------
        self.ksize_label = QLabel('diameter:')
        self.k_size_input = QSpinBox()
        self.k_size_input.setValue(default)
        self.k_size_input.setKeyboardTracking(False)
        self.k_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #ksize slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range)    
        self.slider_label.setValue(default)
        #sigmaX ------------
        self.sigmaX_label = QLabel('sigmaColour:')
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
        self.link_checkbox.setText('Link')
        self.link_checkbox.setCheckState(Qt.Checked)
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
              
    # kernel----------------------------------------------------
    def the_spinbox_was_changed(self):    
        self.slider_label.setRange(1, (self.k_size_input.value()*2))  
        self.k = self.k_size_input.value()
             
        self.slider_label.setValue(self.k)
        self.kValueChanged.emit(self.k)
    
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal 
        
        self.k = v
            #  print('odd')
        self.k_size_input.setValue(self.k)
        self.kValueChanged.emit(self.k)
    # sigmaX --------------------------------------------------
    def the_Xspinbox_was_changed(self): 
        v = self.sigmaX_size_input.value()*100   
        self.Xslider_label.setRange(1, v*2)  
        self.Xslider_label.setValue(v)  
        self.X = self.sigmaX_size_input.value()
        self.XValueChanged.emit(self.X)
        # print(self.link_checkbox.isChecked())

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
            print(self.X, self.Y)
            # self.Yslider_label.setRange(1, v*2)  
            self.Yslider_label.setValue(v)  
            
            self.YValueChanged.emit(self.Y)

    # sigma Y -------------------------------------------------
    def the_Yspinbox_was_changed(self): 
        v = self.sigmaY_size_input.value()*100   
        self.Yslider_label.setRange(1, v*2)  
        self.Yslider_label.setValue(v)  
        self.Y = self.sigmaY_size_input.value()
        self.YValueChanged.emit(self.Y)

    def the_Yslider_was_changed(self, v):    
        self.Y = v/100
        self.sigmaY_size_input.setValue(self.Y)
        self.XValueChanged.emit(self.Y)
        
    # update image ---------------------------------------------
    # def show_image(self, img):
    #     # self.resize(800,800)

    #     try:
    #         if len(img.shape) == 2:
    #             # Grayscale image
    #             qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
    #         else:
    #             # RGB image
    #             rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             h, w, ch = rgb_image.shape
    #             bytes_per_line = ch * w
    #             qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
    #         # Calculate the target size for scaling
    #         scale_factor = 0.4  # Increase the scaling factor for clarity
    #         target_width = int(qt_image.width() * scale_factor)
    #         target_height = int(qt_image.height() * scale_factor)
            
    #         qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
    #         self.image_label.setPixmap(QPixmap.fromImage(qt_image))
    #         self.resize(target_width, target_height)
    #         self.node.update_shape()
    #     except Exception as e:
    #         print("Error:", e)


    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.value(),
    #     }

    # def set_state(self, data: dict):
    #     self.setValue(data['val'])


#thresh #maxval
class Threshold_MainWidget(MWB, QWidget):
    #define Signal
    previewState = Signal(bool)
    threshValueChanged = Signal(int)  #threshold
    mvValueChanged = Signal(int)      #maxval

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 10
        default_range1 = default1*2
        default2 = 100
        default_range2 = default2*2

        
                
        #Added Widget -----------------------------------------------
        #threshold------------
        self.tsize_label = QLabel('thresh:')
        self.t_size_input = QSpinBox()
        self.t_size_input.setMaximum(255)
        self.t_size_input.setValue(default1)
        self.t_size_input.setKeyboardTracking(False)
        self.t_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #thresh slider
        self.slider_label = QSlider(Qt.Horizontal)
        self.slider_label.setRange(1, default_range1)    
        self.slider_label.setValue(default1)
        #maxvalue ------------
        self.maxv_label = QLabel('maxval:')
        self.maxv_size_input = QSpinBox()
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
        self.slider_label.setRange(1, (self.t_size_input.value()*2))  
        self.t = self.t_size_input.value()
        self.slider_label.setValue(self.t)
        self.t_size_input.setValue(self.t)
        self.threshValueChanged.emit(self.t)
    
    def the_slider_was_changed(self, v):    # v: value emitted by a slider signal
        self.t = v
            #  print('odd')
        self.t_size_input.setValue(self.t)
        self.threshValueChanged.emit(self.t)
    # maxval --------------------------------------------------
    def the_mvspinbox_was_changed(self): 
        self.mvslider_label.setRange(1, (self.maxv_size_input.value()*2))  
        self.mv = self.maxv_size_input.value()
        self.mvslider_label.setValue(self.mv)
        self.maxv_size_input.setValue(self.mv)
        self.mvValueChanged.emit(self.mv)

    def the_mvslider_was_changed(self, v):    
        self.mv = v
        self.maxv_size_input.setValue(self.mv)
        self.mvValueChanged.emit(self.mv)
    # update image ---------------------------------------------
    def show_image(self, img):
        # self.resize(800,800)

        try:
            if len(img.shape) == 2:
                # Grayscale image
                qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            else:
                # RGB image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate the target size for scaling
            scale_factor = 0.4  # Increase the scaling factor for clarity
            target_width = int(qt_image.width() * scale_factor)
            target_height = int(qt_image.height() * scale_factor)
            
            qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.resize(target_width, target_height)
            self.node.update_shape()
        except Exception as e:
            print("Error:", e)

        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    # def get_state(self) -> dict:
    #     return {
    #         'v': self.mv(),
    #     }

    # def set_state(self, data: dict):
    #     self.slider_label(data['v'])

#thresh #maxval
class Morphological_MainWidget(MWB, QWidget):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(int)  #kernel

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 10
        default_range1 = default1*2
        default2 = 100
        default_range2 = default2*2
                
        #Added Widget -----------------------------------------------
        #kernel size------------
        self.label_1 = QLabel('kernel/radius:')
        self.input_1 = QSpinBox()
        self.input_1.setMaximum(255)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #kernal slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1)    
        self.slider_label_1.setValue(default1)
        #maxvalue ------------
        # self.maxv_label = QLabel('maxval:')
        # self.maxv_size_input = QSpinBox()
        # self.maxv_size_input.setMaximum(255)
        # self.maxv_size_input.setValue(default2)
        # self.maxv_size_input.setKeyboardTracking(False)
        # self.maxv_size_input.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # #mv slider
        # self.mvslider_label = QSlider(Qt.Horizontal)
        # self.mvslider_label.setRange(1, default_range2)
        # self.mvslider_label.setValue(default2)
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
            #  print('odd')
        self.input_1.setValue(self.t)
        self.Value1Changed.emit(self.t)
    
    # update image ---------------------------------------------
    def show_image(self, img):
        # self.resize(800,800)

        try:
            if len(img.shape) == 2:
                # Grayscale image
                qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            else:
                # RGB image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate the target size for scaling
            scale_factor = 0.4  # Increase the scaling factor for clarity
            target_width = int(qt_image.width() * scale_factor)
            target_height = int(qt_image.height() * scale_factor)
            
            qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.resize(target_width, target_height)
            self.node.update_shape()
        except Exception as e:
            print("Error:", e)

        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    # def get_state(self) -> dict:
    #     return {
    #         'v': self.mv(),
    #     }

    # def set_state(self, data: dict):
    #     self.slider_label(data['v'])
# class MainWidget(QWidget):

#     def imageproc()

class Dilate_MainWidget(MWB, QWidget):
    #define Signal
    previewState = Signal(bool)
    Value1Changed = Signal(int)  #kernel
    Value2Changed = Signal(int)  #itt

    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.resize(300, 300)
        default1 = 10
        default_range1 = default1*2
        default2 = 100
        default_range2 = default2*2
                
        #Added Widget -----------------------------------------------
        #kernel size------------
        self.label_1 = QLabel('kernel:')
        self.input_1 = QSpinBox()
        self.input_1.setMaximum(255)
        self.input_1.setValue(default1)
        self.input_1.setKeyboardTracking(False)
        self.input_1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #kernal slider
        self.slider_label_1 = QSlider(Qt.Horizontal)
        self.slider_label_1.setRange(1, default_range1)    
        self.slider_label_1.setValue(default1)

        #itteration size------------
        self.label_2 = QLabel('iteration:')
        self.input_2 = QSpinBox()
        self.input_2.setMaximum(5)
        self.input_2.setValue(1)
        self.input_2.setKeyboardTracking(False)
        self.input_2.setButtonSymbols(QAbstractSpinBox.NoButtons)

        #itt slider
        self.slider_label_2 = QSlider(Qt.Horizontal)
        self.slider_label_2.setRange(1, 5)    
        self.slider_label_2.setValue(1)
       
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
            #  print('odd')
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
            #  print('odd')
        self.input_2.setValue(self.t2)
        self.Value2Changed.emit(self.t2)    
    # update image ---------------------------------------------
    def show_image(self, img):
        # self.resize(800,800)

        try:
            if len(img.shape) == 2:
                # Grayscale image
                qt_image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
            else:
                # RGB image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Calculate the target size for scaling
            scale_factor = 0.4  # Increase the scaling factor for clarity
            target_width = int(qt_image.width() * scale_factor)
            target_height = int(qt_image.height() * scale_factor)
            
            qt_image = qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)
            
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            self.resize(target_width, target_height)
            self.node.update_shape()
        except Exception as e:
            print("Error:", e)

        
    # hide image ---------------------------------------------   
    def clear_img(self):
         # Create a black image of size 1x1
        clr_img = QImage(1, 1, QImage.Format_RGB888)
        clr_img.setPixelColor(0, 0, QColor(Qt.black))

        self.image_label.setPixmap(QPixmap(clr_img))
        print(self.width(), self.height())
        self.resize(200,50)
        self.node.update_shape() #works the best. But doesnt minimize shape immediately
    
    # def get_state(self) -> dict:
    #     return {
    #         'v': self.mv(),
    #     }

    # def set_state(self, data: dict):
    #     self.slider_label(data['v'])

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
    ChooseFileInputWidgetBASE,
    Blur_Averaging_MainWidget,
    Blur_Median_MainWidget,
    Gaus_Blur_MainWidget,
    Bilateral_MainWidget,
    Threshold_MainWidget,
    Morphological_MainWidget,
    Dilate_MainWidget,
    ChooseFileInputWidgetBASE2,
    ChooseFileInputWidgetBASE3,
)
