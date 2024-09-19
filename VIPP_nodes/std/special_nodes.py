import code
from contextlib import redirect_stdout, redirect_stderr

from ryven.NENV import *
widgets = import_widgets(__file__)

import cv2
import numpy as np
import os
import tifffile as tiff 
from scipy.ndimage import gaussian_filter, label, sum, binary_fill_holes
import concurrent.futures
import json
import pandas as pd
import porespy  as ps
#THIS ALSO INCLUDES OPENCV CODE


class NodeBase0(Node):
    version = 'v0.1'
    color = '#00a6ff' # Blue - input / output nodes

    # def handle_stack(self):
    #     self.image_stack = self.input(0)[0]
    #     self.frame = self.input(0)[1]
    #     self.z_sclice = self.input(0)[2]
    #     # squeeze = np.squeeze(self.image_stack)
    #     if (self.image_stack.shape[0] != 1) & (self.image_stack.shape[1] != 1):
    #         self.sliced = squeeze[self.frame, self.z_sclice, :, :]
    #         self.frame_size = self.image_stack.shape[0]
    #         self.z_size = self.image_stack.shape[1]
    #     else:
    #         self.sliced = squeeze
    #     print(f"size frame {self.frame_size}, size z {self.z_size}")
    #     print(f"shape {self.image_stack.shape}, frame {self.frame}, z {self.z_sclice}, SCLICE {squeeze.shape}")
    def handle_stack(self):
        self.image_stack = self.input(0)[0]
        # self.frame = self.input(0)[1] #dont actually need this anymore, but keep incase. Good to know wich time step
        self.stack_dict = self.input(0)[1] #dictioary
        self.z_sclice = self.input(0)[2]
        # self.squeeze = np.squeeze(self.image_stack)
        self.z_size = self.image_stack.shape[0]
        self.sliced = self.image_stack[self.z_sclice, :, :, :] #Z, H, W, C

# class NodeBase(Node):
#     version = 'v0.1'
#     color = '#FFCA00' #yellow - Filtering 


class NodePipeline(Node):
    
    def handle_stack(self):
        self.image_stack = self.input(0)[0]
        self.stack_dict = self.input(0)[1] #dictioary
        self.z_sclice = self.input(0)[2]
        # self.squeeze = np.squeeze(self.image_stack)
        self.z_size = self.image_stack.shape[0]
        self.sliced = self.image_stack[self.z_sclice, :, :, :] #Z, H, W, C
        #Debug
        # print(f"size z {self.z_size}")
        # print(f"shape colour {self.image_stack.shape[-1]}, frame {self.frame}, z {self.z_sclice}, SCLICE {self.squeeze.shape}")
        # print(f"shape colour {self.sliced.shape[-1]}")

    
    def get_img(self, zslice):
        #PROCESS SLICE 
        #generate slice for dispay
        # reshaped = self.sliced.reshape(zslice.shape[:-1] + (-1,))
        # print(f"size shape {zslice.shape}")
    
        # Apply median blur to all channels simultaneously
        processed_data = self.proc_technique(zslice)
        
        # Reshape the processed data back to the original shape
        # Ensures [, , 1] one at the end stays 
        processed_slice = processed_data.reshape(zslice.shape)
        
        return processed_slice
    
    #signle time step
    def proc_stack_parallel(self):
        
        # Define the number of worker threads or processes
        num_workers = 6  # Adjust as needed

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            # print(f"z size {self.z_size}")
            for z in range(self.z_size):
                    img = self.image_stack[z]
                    # print(f"\nprocessed z slice {z}")
                    future = executor.submit(self.get_img,img)
                    futures.append((z, future))

            proc_data = np.empty_like(self.image_stack)

            for z, future in futures:
                processed_frame = future.result()
                proc_data[z] = processed_frame
        
        # print(f"proc_data shape: {proc_data.shape}")
        self.reshaped_proc_data = proc_data
        # print(f"reshaped_proc_data shape: {self.reshaped_proc_data.shape}")
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))

class NodeBase(NodePipeline):
    version = 'v0.1'
    color = '#FFCA00' #yellow - Filtering 


class NodeBase3(NodePipeline):
    version = 'v0.1'
    color = '#8064A2' #purple - contrast enh 

class NodeBase2(Node):
    version = 'v0.1'
    color = '#92D050' #green - Binarization: #Binaraization have to reshape to add a channel for standarzation

    def handle_stack(self):
        self.image_stack = self.input(0)[0]
        self.stack_dict = self.input(0)[1] #dictioary
        self.z_sclice = self.input(0)[2]
        # self.squeeze = np.squeeze(self.image_stack)
        self.z_size = self.image_stack.shape[0]
        self.sliced = self.image_stack[self.z_sclice, :, :, :] #Z, H, W, C
        #Debug
        #print(f"size shape {self.sliced.shape}")
        # #print(f"shape colour {self.image_stack.shape[-1]}, frame {self.frame}, z {self.z_sclice}, SCLICE {self.squeeze.shape}")
        #print(f"shape colour {self.sliced.shape[-1]}")

    
    def get_img(self, zslice):
        
        processed_data = self.proc_technique(zslice)
        #dont need to rshape because do at the end of binarization in proc_technique
        
        return processed_data
    
    #signle time step
    def proc_stack_parallel(self):
        
        # Define the number of worker threads or processes
        num_workers = 6  # Adjust as needed

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            #print(f"z size {self.z_size}")
            for z in range(self.z_size):
                    img = self.image_stack[z]
                    # #print(f"\nprocessed z slice {z}")
                    future = executor.submit(self.get_img,img)
                    futures.append((z, future))
            
            #THIS IS DIFFERENT***
            # Create proc_data from processed_frame (not sclice - this may be grayscale)
            proc_data = np.empty((self.image_stack.shape[0], self.image_stack.shape[1], self.image_stack.shape[2], 1), dtype=np.uint8)

            for z, future in futures:
                processed_frame = future.result()
                # #print(f"processed_frame* {processed_frame.shape}")
                # processed_frame = np.expand_dims(processed_frame, axis=-1)  #Binaraization have to reshape to add a channel for standarzation
                proc_data[z] = processed_frame
        
        # #print(f"proc_data shape: {proc_data.shape}")
        self.reshaped_proc_data = proc_data
        #print(f"reshaped_proc_data shape*: {self.reshaped_proc_data.shape}")
        #print(f"data : {self.reshaped_proc_data.dtype}")
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))

class NodeBase4(NodePipeline):
    version = 'v0.1'
    color = '#C55A11' #red - post binarization

    


class Checkpoint_Node(NodeBase):
    """Provides a simple checkpoint to reroute your connections"""

    title = 'checkpoint'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(type_='data'),
    ]
    init_outputs = [
        NodeOutputBP(type_='data'),
    ]
    style = 'small'

    def __init__(self, params):
        super().__init__(params)

        self.display_title = ''

        self.active = False

        # initial actions
        self.actions['add output'] = {
            'method': self.add_output
        }
        self.actions['remove output'] = {
            '0': {'method': self.remove_output, 'data': 0}
        }
        self.actions['make active'] = {
            'method': self.make_active
        }

    """State transitions"""

    def clear_ports(self):
        # remove all outputs
        for i in range(len(self.outputs)):
            self.delete_output(0)

        # remove all inputs
        for i in range(len(self.inputs)):
            self.delete_input(0)

    def make_active(self):
        self.active = True

        # rebuild inputs and outputs
        self.clear_ports()
        self.create_input(type_='exec')
        self.create_output(type_='exec')

        # update actions
        del self.actions['make active']
        self.actions['make passive'] = {
            'method': self.make_passive
        }
        self.actions['remove output'] = {
            '0': {'method': self.remove_output, 'data': 0}
        }

    def make_passive(self):
        self.active = False

        # rebuild inputs and outputs
        self.clear_ports()
        self.create_input(type_='data')
        self.create_output(type_='data')

        # update actions
        del self.actions['make passive']
        self.actions['make active'] = {
            'method': self.make_active
        }
        self.actions['remove output'] = {
            '0': {'method': self.remove_output, 'data': 0}
        }

    """Actions"""

    def add_output(self):
        index = len(self.outputs)

        if self.active:
            self.create_output(type_='exec')
        else:
            self.create_output(type_='data')

        self.actions['remove output'][str(index)] = {
            'method': self.remove_output,
            'data': index,
        }

    def remove_output(self, index):
        self.delete_output(index)

        del self.actions['remove output'][str(len(self.outputs))]

    """Behavior"""

    def update_event(self, inp=-1):
        if self.active and inp == 0:
            for i in range(len(self.outputs)):
                self.exec_output(i)

        elif not self.active:
            data = self.input(0)
            for i in range(len(self.outputs)):
                self.set_output_val(i, data)

    """State Reload"""

    # def get_state(self) -> dict:
    #     return {
    #         'active': self.active,
    #         'num outputs': len(self.outputs),
    #     }

    # def set_state(self, data: dict, version):
    #     self.actions['remove output'] = {
    #         {'method': self.remove_output, 'data': i}
    #         for i in range(data['num outputs'])
    #     }

    #     if data['active']:
    #         self.make_active()


class Slider_Node(NodeBase):
    title = 'slider'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Integer(default=1), label='scl'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='round'),
    ]
    init_outputs = [
        NodeOutputBP()
    ]
    main_widget_class = widgets.SliderNode_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        self.val = 0

    def place_event(self):  #??
        self.update()

    def view_place_event(self):
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):

        v = self.input(0) * self.val
        if self.input(1):
            v = round(v)

        self.set_output_val(0, v)

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }

    def set_state(self, data: dict, version):
        self.val = data['val']


class _DynamicPorts_Node(NodeBase):
    version = 'v0.1'
    init_inputs = []
    init_outputs = []

    def __init__(self, params):
        super().__init__(params)

        self.actions['add input'] = {'method': self.add_inp}
        self.actions['add output'] = {'method': self.add_out}

        self.num_inputs = 0
        self.num_outputs = 0

    def add_inp(self):
        self.create_input()

        index = self.num_inputs
        self.actions[f'remove input {index}'] = {
            'method': self.remove_inp,
            'data': index
        }

        self.num_inputs += 1

    def remove_inp(self, index):
        self.delete_input(index)
        self.num_inputs -= 1
        del self.actions[f'remove input {self.num_inputs}']

    def add_out(self):
        self.create_output()

        index = self.num_outputs
        self.actions[f'remove output {index}'] = {
            'method': self.remove_out,
            'data': index
        }

        self.num_outputs += 1

    def remove_out(self, index):
        self.delete_output(index)
        self.num_outputs -= 1
        del self.actions[f'remove output {self.num_outputs}']

    def get_state(self) -> dict:
        return {
            'num inputs': self.num_inputs,
            'num outputs': self.num_outputs,
        }

    def set_state(self, data: dict):
        self.num_inputs = data['num inputs']
        self.num_outputs = data['num outputs']


class Exec_Node(_DynamicPorts_Node):
    title = 'exec'
    version = 'v0.1'
    main_widget_class = widgets.CodeNode_MainWidget
    main_widget_pos = 'between ports'

    def __init__(self, params):
        super().__init__(params)

        self.code = None

    def place_event(self):
        pass

    def update_event(self, inp=-1):
        exec(self.code)

    def get_state(self) -> dict:
        return {
            **super().get_state(),
            'code': self.code,
        }

    def set_state(self, data: dict, version):
        super().set_state(data, version)
        self.code = data['code']


class Eval_Node(NodeBase):
    title = 'eval'
    version = 'v0.1'
    init_inputs = [
        # NodeInputBP(),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    main_widget_class = widgets.EvalNode_MainWidget
    main_widget_pos = 'between ports'

    def __init__(self, params):
        super().__init__(params)

        self.actions['add input'] = {'method': self.add_param_input}

        self.number_param_inputs = 0
        self.expression_code = None

    def place_event(self):
        if self.number_param_inputs == 0:
            self.add_param_input()

    def add_param_input(self):
        self.create_input()

        index = self.number_param_inputs
        self.actions[f'remove input {index}'] = {
            'method': self.remove_param_input,
            'data': index
        }

        self.number_param_inputs += 1

    def remove_param_input(self, index):
        self.delete_input(index)
        self.number_param_inputs -= 1
        del self.actions[f'remove input {self.number_param_inputs}']

    def update_event(self, inp=-1):
        inp = [self.input(i) for i in range(self.number_param_inputs)]
        self.set_output_val(0, eval(self.expression_code))

    def get_state(self) -> dict:
        return {
            'num param inputs': self.number_param_inputs,
            'expression code': self.expression_code,
        }

    def set_state(self, data: dict, version):
        self.number_param_inputs = data['num param inputs']
        self.expression_code = data['expression code']


class Interpreter_Node(NodeBase):
    """Provides a python interpreter via a basic console with access to the
    node's properties."""
    title = 'interpreter'
    version = 'v0.1'
    init_inputs = []
    init_outputs = []
    main_widget_class = widgets.InterpreterConsole

    # DEFAULT COMMANDS

    def clear(self):
        self.hist.clear()
        self._hist_updated()

    def reset(self):
        self.interp = code.InteractiveInterpreter(locals=locals())

    COMMANDS = {
        'clear': clear,
        'reset': reset,
    }

    def __init__(self, params):
        super().__init__(params)

        self.interp = None
        self.hist: [str] = []
        self.buffer: [str] = []

        self.reset()

    def _hist_updated(self):
        if self.session.gui:
            self.main_widget().interp_updated()

    def process_input(self, cmds: str):
        m = self.COMMANDS.get(cmds)
        if m is not None:
            m()
        else:
            for l in cmds.splitlines():
                self.write(l)  # print input
                self.buffer.append(l)
            src = '\n'.join(self.buffer)

            def run_src():
                more_inp_required = self.interp.runsource(src, '<console>')
                if not more_inp_required:
                    self.buffer.clear()

            if self.session.gui:
                with redirect_stdout(self), redirect_stderr(self):
                    run_src()
            else:
                run_src()

    def write(self, line: str):
        self.hist.append(line)
        self._hist_updated()


class Storage_Node(NodeBase):
    """Sequentially stores all the data provided at the input in an array.
    A COPY of the storage array is provided at the output"""

    title = 'store'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    color = '#aadd55'

    def __init__(self, params):
        super().__init__(params)

        self.storage = []

        self.actions['clear'] = {'method': self.clear}

    def clear(self):
        self.storage.clear()
        self.set_output_val(0, [])

    def update_event(self, inp=-1):
        self.storage.append(self.input(0))
        self.set_output_val(0, self.storage.copy())

    def get_state(self) -> dict:
        return {
            'data': self.storage,
        }

    def set_state(self, data: dict, version):
        self.storage = data['data']


import uuid


class LinkIN_Node(NodeBase):
    """You can use link OUT nodes to link them up to this node.
    Whenever a link IN node receives data (or an execution signal),
    if there is a linked OUT node, it will receive the data
    and propagate it further."""

    title = 'Link in'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(),
    ]
    init_outputs = []  # no outputs

    # instances registration
    INSTANCES = {}  # {UUID: node}

    def __init__(self, params):
        super().__init__(params)
        self.display_title = 'link'

        # register
        self.ID: uuid.UUID = uuid.uuid4()
        self.INSTANCES[str(self.ID)] = self

        self.actions['add input'] = {
            'method': self.add_inp
        }
        self.actions['remove inp'] = {}
        self.actions['copy ID'] = {
            'method': self.copy_ID
        }

        self.linked_node: LinkOUT_Node = None

    def copy_ID(self):
        from qtpy.QtWidgets import QApplication
        QApplication.clipboard().setText(str(self.ID))

    def add_inp(self):
        index = len(self.inputs)

        self.create_input()

        self.actions['remove inp'][str(index)] = {
            'method': self.rem_inp,
            'data': index,
        }
        if self.linked_node is not None:
            self.linked_node.add_out()

    def rem_inp(self, index):
        self.delete_input(index)
        del self.actions['remove inp'][str(len(self.inputs))]
        if self.linked_node is not None:
            self.linked_node.rem_out(index)

    def update_event(self, inp=-1):
        if self.linked_node is not None:
            self.linked_node.set_output_val(inp, self.input(inp))

    def get_state(self) -> dict:
        return {
            'ID': str(self.ID),
        }

    def set_state(self, data: dict, version):
        if data['ID'] in self.INSTANCES:
            # this happens when some existing node has been copied and pasted.
            # we only want to rebuild links when loading a project, considering
            # new links when copying nodes might get quite complex
            pass
        else:
            del self.INSTANCES[str(self.ID)]     # remove old ref
            self.ID = uuid.UUID(data['ID'])      # use original ID
            self.INSTANCES[str(self.ID)] = self  # set new ref

            # resolve possible pending link builds from OUT nodes that happened
            # to get initialized earlier
            LinkOUT_Node.new_link_in_loaded(self)

    def remove_event(self):
        # break existent link
        if self.linked_node:
            self.linked_node.linked_node = None
            self.linked_node = None


class LinkOUT_Node(NodeBase):
    """The complement to the link IN node"""

    title = 'Link out'
    version = 'v0.1'
    init_inputs = []  # no inputs
    init_outputs = []  # will be synchronized with linked IN node

    INSTANCES = []
    PENDING_LINK_BUILDS = {}
    # because a link OUT node might get initialized BEFORE it's corresponding
    # link IN, it then stores itself together with the ID of the link IN it's
    # waiting for in PENDING_LINK_BUILDS

    @classmethod
    def new_link_in_loaded(cls, n: LinkIN_Node):
        for out_node, in_ID in cls.PENDING_LINK_BUILDS.items():
            if in_ID == str(n.ID):
                out_node.link_to(n)

    def __init__(self, params):
        super().__init__(params)
        self.display_title = 'link'

        self.INSTANCES.append(self)
        self.linked_node: LinkIN_Node = None

        self.actions['link to ID'] = {
            'method': self.choose_link_ID
        }

    def choose_link_ID(self):
        """opens a small input dialog for providing a copied link IN ID"""

        from qtpy.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QLineEdit

        class IDInpDialog(QDialog):
            def __init__(self):
                super().__init__()
                self.id_str = None
                self.setLayout(QVBoxLayout())
                self.line_edit = QLineEdit()
                self.layout().addWidget(self.line_edit)
                self.line_edit.returnPressed.connect(self.return_pressed)

            def return_pressed(self):
                self.id_str = self.line_edit.text()
                self.accept()

        d = IDInpDialog()
        d.exec_()

        if d.id_str is not None:
            n = LinkIN_Node.INSTANCES.get(d.id_str)
            if n is None:
                QMessageBox.warning(title='link failed', text='couldn\'t find a valid link in node')
            else:
                self.link_to(n)

    def link_to(self, n: LinkIN_Node):
        self.linked_node = n
        n.linked_node = self

        o = len(self.outputs)
        i = len(self.linked_node.inputs)

        # remove outputs if there are too many
        for j in range(i, o):
            self.delete_output(0)

        # add outputs if there are too few
        for j in range(o, i):
            self.create_output()

        self.update()

    def add_out(self):
        # triggered by linked_node
        self.create_output()

    def rem_out(self, index):
        # triggered by linked_node
        self.delete_output(index)

    def update_event(self, inp=-1):
        if self.linked_node is None:
            return

        # update ALL ports
        for i in range(len(self.outputs)):
            self.set_output_val(i, self.linked_node.input(i))

    def get_state(self) -> dict:
        if self.linked_node is None:
            return {}
        else:
            return {
                'linked ID': str(self.linked_node.ID),
            }

    def set_state(self, data: dict, version):
        if len(data) > 0:
            n: LinkIN_Node = LinkIN_Node.INSTANCES.get(data['linked ID'])
            if n is None:
                # means that the OUT node gets initialized before it's link IN
                self.PENDING_LINK_BUILDS[self] = data['linked ID']
            elif n.linked_node is None:
                # pair up
                n.linked_node = self
                self.linked_node = n

    def remove_event(self):
        # break existent link
        if self.linked_node:
            self.linked_node.linked_node = None
            self.linked_node = None


# -------------------------------------------
#New

class OpenCVNodeBase(NodeBase0):   
    init_outputs = [
        NodeOutputBP()
    ]
    main_widget_class = widgets.OpenCVNode_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass

    def update_event(self, inp=-1):                                         #reset
        #extract from stack
        self.handle_stack()
        new_img_wrp = CVImage(self.get_img())

        if self.session.gui:
            self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(0, new_img_wrp)

    
    def get_img(self):
        return None



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
     
    
# NODES ------------------------------------------------------------------------------------------------------------------

#Single timestep
# new shape (10, 512, 512, 1)
class ReadImage(NodeBase0):
    """Reads an image from a file"""

    title = 'Read Image'
    # input_widget_classes = {
    #     'choose file IW': widgets.ChooseFileInputWidget
    # }
    init_inputs = [
        NodeInputBP('batch process') #add_data={'widget name': 'choose file IW', 'widget pos': 'besides'}
    ]
    init_outputs = [
        NodeOutputBP('img')
    ]
    main_widget_class = widgets.Read_Image_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
                image_shape = Signal(list)
                #reset sliders
                reset_widget = Signal(list)
                #remove widgets
                remove_widget = Signal()
                channels_dict = Signal(dict)
                stack_dictionery = Signal(dict)

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.image_filepath = ''
        self.ttval = 0
        self.zzval = 4
        self.stack_dict = {
            "time_step": self.ttval, #Note, +1 if want to display in biologists notation
            "total_time_frames": 1,
            "colour": {
                "red": 100,
                "green": 100,
                "blue": 100,
                "cyan": 100,
                "yellow": 100,
                "magenta": 100
            }
        }

    def view_place_event(self):
        self.main_widget().path_chosen.connect(self.path_chosen)
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.image_shape.connect(self.main_widget().update_widgets)
        self.SIGNALS.reset_widget.connect(self.main_widget().reset_widg)
        self.SIGNALS.remove_widget.connect(self.main_widget().remove_widgets)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.SIGNALS.stack_dictionery.connect(self.main_widget().update_time_frame_stack_dict)
        self.main_widget().ValueChanged1.connect(self.onValue1Changed)
        self.main_widget().ValueChanged2.connect(self.onValue2Changed) 
        self.main_widget().released1.connect(self.output_data)  
        self.main_widget().dict_widg.connect(self.dict_col)
         

        # self.main_widget().ValueChanged2.connect(self.output_data) 
        # try:
        #     self.SIGNALS.new_img.emit(self.get_img())
        # except:  # there might not be an image ready yet
        #     pass
        # self.main_widget_message.connect(self.main_widget().show_path)
   
   # When user presses confirm the dictionary is updated with the new colour choices
    def dict_col(self, dictt):
        for color in self.stack_dict["colour"]:
                self.stack_dict["colour"][color] = dictt["colour"][color]
        print(f'self.stack_dict["colour"] {self.stack_dict}')
        #update iimage
        new_img_wrp = CVImage(self.get_img()) 
        # if self.session.gui:
        # display immediately 
        self.SIGNALS.new_img.emit(new_img_wrp.img)

        # output dictionary to update next nodes
        self.output_data()
    
    
    def path_chosen(self, file_path):
        self.image_filepath = file_path

        if self.image_filepath == '':
            return
        # Check if the file has a .tiff extension   --> tif file capability Check tiff 
        if self.image_filepath.endswith('.tif'): #----------------------------------------------------------------------------------TIFF
            try:
                self.image_data = tiff.imread(self.image_filepath)

                


                # Normalize - generate dimension list (T,Z,H,W,C)
                self.dim = self.id_tiff_dim(self.image_filepath)
                self.stack_dict['total_time_frames'] = self.dim[0]
                self.SIGNALS.stack_dictionery.emit(self.stack_dict)
                print(f"total frames:{self.stack_dict['total_time_frames']}")
                # Reshape - STANDARDIZED 
                self.image_data = ((self.image_data / np.max(self.image_data)) * 255).astype(np.uint8)
                
                # IF shape: ZCXY ---------------------------------------------------------------
                # Change to TZXYC
                if self.image_data.shape[-3] <= 4:
                    
                    # dim = [1, 15, 1024, 1024, 3]
                    print("Strange dimension")
                    #if no time step data 
                    if len(self.image_data.shape) == 4: 
                        print(f'length {len(self.image_data.shape)}')
                        self.image_data = self.image_data[np.newaxis,:,:,:,:]

                    # Create an empty array with the specified dimensions
                    image_data_stacked = np.empty(self.dim, dtype=np.uint8)

                    num_time_frames = self.dim[0]  # Get the number of time frames
                    
                    for t in range(num_time_frames):  # Iterate over the time frames
                        for i in range(self.dim[1]):  # Iterate over the images in each time frame
                            # Get the red, green, and blue channels for the i-th image in the t-th time frame
                            print(i)
                            chan_0 = self.image_data[t, i, 0, :, :]  # Assuming the first dimension is time frame, then image index
                            chan_1 = self.image_data[t, i, 1, :, :]
                            chan_2 = self.image_data[t, i, 2, :, :]
                            
                            # Stack the channels along the last axis
                            image_data_stacked[t, i, :, :, 0] = chan_0
                            image_data_stacked[t, i, :, :, 1] = chan_1
                            image_data_stacked[t, i, :, :, 2] = chan_2   
                            # if have a an alpha channel
                            if self.image_data.shape[2] >= 4:
                                chan_3 = self.image_data[t, i, 3, :, :] 
                                image_data_stacked[t, i, :, :, 3] = chan_3 

                            # if have a an alpha channel
                            if self.image_data.shape[2] == 5:
                                chan_4 = self.image_data[t, i, 4, :, :] 
                                image_data_stacked[t, i, :, :, 4] = chan_4 

                    print(image_data_stacked.shape)   

                    self.image_data = image_data_stacked 
                # ---------------------------------------------------------------------------------           
                    
                self.reshaped_data = self.image_data.reshape(self.dim)
                # print(f"reshaped: {self.reshaped_data.shape}")
                # self.handle_stack()

                
                # Grayscale
                # if self.reshaped_data.shape[4] == 1: #self.dim[4]==1:
                # self.reshaped_data = ((self.reshaped_data/self.reshaped_data.max())*255).astype('uint8')
                    # print("NORMALIZED FOR GRAYSCALE")
                
                # Display image
                # 3D, 3D or 5D
                # if (self.reshaped_data.shape[0] != 1) | (self.reshaped_data.shape[1] != 1): #time and space (4D)
                    #squeeze standardized down to relevant dimension (remove 1s)
                    # self.squeezed = np.squeeze(self.reshaped_data)
                    #slice
                print("emit from path chosen ReadImage")
                new_img_wrp = CVImage(self.get_img())  
                    # print("shape", new_img_wrp.shape)
                if self.session.gui:
                        self.SIGNALS.new_img.emit(new_img_wrp.img)
                        # self.main_widget().update_shape()
                    # mulitple time steps
                # if self.reshaped_data.shape[0] != 1:
                self.set_output_val(0, (self.reshaped_data[self.stack_dict["time_step"], :, :, :, :], self.stack_dict, self.zzval))
                print(f"output: total frames: {self.stack_dict['total_time_frames']}, time step: {self.stack_dict['time_step']}")
                #     self.set_output_val(0, (self.reshaped_data[0, :, :, :, :], self.ttval, self.zzval))

                #2D images (tiff)
                # else:
                #     image2D = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
                #     self.reshaped_data = image2D.reshape(1, 1, *image2D.shape)
                #     if self.session.gui:
                #         new_img_wrp = CVImage(image2D)
                #         self.SIGNALS.new_img.emit(new_img_wrp.img)
                #     self.set_output_val(0, (self.reshaped_data[0, :, :, :, :], 0, 0)) # will be 0 and 0 
            
            except Exception as e:
                print(e)
                print("failed")

        else: #-------------------------------------------------------------------------------------------------------------------- Not TIFF
            # 2D not tiff
            try:
                image2D = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
                self.reshaped_data = image2D.reshape(1, 1, *image2D.shape) # (1, 1, *image2D.shape, 1) ??
                if self.session.gui:
                    new_img_wrp = CVImage(image2D)
                    self.SIGNALS.new_img.emit(new_img_wrp.img)
                self.set_output_val(0,(self.reshaped_data, 0, 0))
            except Exception as e:
                print(e)

    def update_event(self, inp=-1):   #called when the input is changed
        
        self.handle_stack()
        print("ReadImage update_event()")
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        print(f"dictionary{self.stack_dict}")
        print(f"zslice {self.z_sclice}")
        single_slice = self.image_stack[self.z_sclice, :,:,:]
        self.new_img_wrp = CVImage(single_slice)
        print("update gui")
        if self.session.gui:
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.set_output_val(0, (self.image_stack, self.stack_dict, self.z_sclice))
        print("set output")
        # 3D
        # if self.z_size > 1:
        #     self.SIGNALS.warning.emit(0)
        #     self.proc_technique()

        
        # if self.prev == True:
        #     if self.session.gui:
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # # 2D
        # if self.z_size == 1:
        #     self.SIGNALS.warning.emit(1)
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 
        # self.proc_stack_parallel()
        # 3D process stack
        
    

        # #therefore new image 
        # new_img_wrp = CVImage(self.get_img())  
        #         # print("shape", new_img_wrp.shape)
        # if self.session.gui:
        #         self.SIGNALS.new_img.emit(new_img_wrp.img)
        #         # self.main_widget().update_shape()
        #     # mulitple time steps
        # # if self.reshaped_data.shape[0] != 1:
        # self.set_output_val(0, (self.reshaped_data[self.stack_dict["time_step"], :, :, :, :], self.stack_dict, self.zzval))
        

    def id_tiff_dim(self,f_path):
        tif_file = tiff.TiffFile(f_path)
        # Check for TIFF metadata tags
        metadata = tif_file.pages[0].tags
        if metadata:
            # print("Metadata Tags:")
            # for tag_name, tag in metadata.items():
            #     print(f"{tag_name}: {tag.value}")
            print("ReadImage Metadata")

            #set dimension to 0 when a new tiff file is processed
            dimension = [1,1,1,1,1] #dim, slices , time
            
            
            #  T Z Y X C  (F, Z, H, W, C)
            #  0 1 2 3 4
            
            if 256 in metadata: #width
                            # Access the tag value directly
                            dimension[3] = metadata[256].value
            if 257 in metadata: #H
                            # Access the tag value directly
                            dimension[2] = metadata[257].value
            if 277 in metadata: #channels
                            # Access the tag value directly
                            dimension[4] = metadata[277].value
            if 259 in metadata:  # Tag for slices
                            print("meta",metadata[259].value)
                            dimension[1] = metadata[259].value
                        
            if 'ImageDescription' in metadata:
                    # Access 'ImageDescription' tag
                    image_description = metadata['ImageDescription']
            
                    # Split the 'ImageDescription' string into lines
                    description_lines = image_description.value.split('\n')
                    # Parse the lines to extract slices and frames information
                    for line in description_lines:
                        # if 262 in metadata:  # Tag for frames
                        #     dimension[4] = metadata[262].value
                        #     print("dim",dimension[4])
                        if line.startswith("slices="):
                            dimension[1] = int(line.split('=')[1]) #slice
                        if line.startswith("frames="):
                            dimension[0] = int(line.split('=')[1]) #frames
                            # print("frames", int(line.split('=')[1]))
                            # print("dim",dimension[4])
                        if line.startswith("channels="):
                            dimension[4] = int(line.split('=')[1]) #frames
                            # print("dim",dimension[4])
                        
        else:
                print("ImageDescription tag not found in metadata.")
                        
        # print(f'Width: {dimension[3]}')
        # print(f'Height: {dimension[2]}')
        # print(f'Channels: {dimension[4]}')
        # print(f"Slices: {dimension[1]}")
        # print(f"Frames: {dimension[0]}")
        # print(f'Dimension: {dimension[0]}')
        # self.dimension=dimension
        set_widg = [1,1]
        set_widg[0] = dimension[0] #t
        set_widg[1] = dimension[1] #z
        self.SIGNALS.reset_widget.emit(set_widg)
        self.SIGNALS.image_shape.emit(dimension)
        self.zzval= round(set_widg[1]/2)
        # if (dimension[0])==1 or (dimension[1])==1:
        #     set_widg = [1,1]
        #     self.SIGNALS.reset_widget.emit(set_widg)
        #     self.stack_dict["time_step"]=0
        #     self.zzval=0
        print(f'Image dim: {dimension}')
        return dimension
           
        
    def onValue1Changed(self, value):
        # print(f"timevalue{value}")
        self.ttval=value-1 # slider: 1-max for biologists
        self.stack_dict["time_step"] = value-1
        print(f'stack {self.stack_dict["time_step"]}')
        self.output_data()
        self.new_img_wrp = CVImage(self.get_img())
        
        if self.session.gui:
            #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)   
    
    def onValue2Changed(self, value):
        # print(f"zvalue{value}")
        self.zzval=value-1
        self.new_img_wrp = CVImage(self.get_img())
        self.output_data()
        
        if self.session.gui:
                #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    
    def output_data(self):
        if self.reshaped_data.shape[0] != 1:
            self.set_output_val(0, (self.reshaped_data[self.stack_dict["time_step"], :, :, :, :], self.stack_dict, self.zzval))
        else:
            self.set_output_val(0, (self.reshaped_data[0, :, :, :, :], self.stack_dict, self.zzval))


    def get_img(self):
        # 4D
        # if (self.dim[0] != 1) & (self.dim[1] != 1):
        self.sliced = self.reshaped_data[self.stack_dict["time_step"],self.zzval,:,:,:]
        # reshaped = self.sliced.reshape(self.sliced.shape[:-1] + (-1,))
        # print(f"THIS is the RESHAPE: {self.sliced.shape}")
        return self.sliced
        # 2D in time
        # elif self.dim[0] != 1:
        #     return self.reshaped_data[self.ttval,,:,:]  #wont have all these *** (SQUEEZED) 
        # # 3D (Z-stack)
        # elif self.dim[1] != 1:
        #     return self.reshaped_data[1,self.zzval,:,:]
    
    # def update_dict

    def get_state(self) -> dict:
        data = {
            'image file path': self.image_filepath,
            'val1': self.stack_dict["time_step"],
            'val2': self.zzval,
                # 'dimension': self.dim
            }
        # print(data)
        return data
        

    def set_state(self, data: dict, version):
        self.path_chosen(data['image file path'])
        self.stack_dict["time_step"] = data['val1']
        self.zzval = data['val2']
        
        

        # self.update()
        # self.dim = data['dimension']
        # self.id_tiff_dim(self.image_filepath)
        # self.image_filepath = data['image file path']

    

    

    # def path_chosen(self, file_path):
    #     self.image_filepath = file_path
    #     self.update()
    
    # def get_state(self) -> dict:
    #     return {
    #         'file': self.image_filepath
    #     }

    # def set_state(self, data: dict, version):
    #     self.image_filepath = data['file']


class SaveImg(NodeBase0):
    title = 'Save Pipeline Output'
    # input_widget_classes = {
    #     'choose file path': widgets.ChooseFileInputWidget
    # }
    init_inputs = [
        NodeInputBP('img'),
        # NodeInputBP('path', add_data={'widget name': 'path input', 'widget pos': 'below'}),
    ]

    main_widget_class = widgets.PathInput
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)
        
        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                reset_widget = Signal(int)
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        
        self.file_path = '' 
        # self.actions['make executable'] = {'method': self.action_make_executable}

    def view_place_event(self):
        # self.input_widget(1).path_chosen.connect(self.path_chosen)
        # self.file_path.path_chosen.connect(self.path_chosen)
        self.main_widget().path_chosen.connect(self.path_chosen)
        self.SIGNALS.reset_widget.connect(self.main_widget().reset_w)

    def path_chosen(self, new_path):
        self.file_path = new_path
        
        stack = self.input(0)[0]
        # print(f'self {self.input(0)[0].shape}')
        # print(f'stack shape {stack.shape}')
        # print(f'channel {self.input(0)[0].shape[3]}')
                
        if self.input(0)[0].shape[3] == 1:
            grayscale_stack = stack[:, :, :, 0]
            #NEED TO CHANGE 259 / slices=
            metadata = {
                'Description': 'Stack (RGB) preprocessed with Visual Processing Pipeline. Developed using Ryven at Stellenbosch University by Emma Sharratt and Dr Rensue Theart',
                'Author': 'Emma Sharratt and Dr Rensu Theart',
                'axes': 'ZYX'
                # 'Width': grayscale_stack.shape[2],
                # 'Height': grayscale_stack.shape[1],
                # 'BitsPerSample': 16,  # Adjust as needed
                # 'Photometric': 'minisblack',
                # Add more metadata fields as needed
                }
            # print("stack shape {grayscale_stack}")
            tiff.imwrite(self.file_path, grayscale_stack, photometric='minisblack', imagej=True, metadata=metadata)  # You can adjust options as needed
            
        
        if self.input(0)[0].shape[3] == 3:
            # print(f"RGB shape {RGB_stack.shape}")
            custom_metadata = {
                "Description": "Stack (RGB) preprocessed with Visual Processing Pipeline. Developed using Ryven by Emma Sharratt and Dr Rensue Theart",
                "Author": "Emma Sharratt and Dr Rensue Theart",
                "Date": "Pipeline created in 2023",
                'axes': 'ZCYX'
                # "256": RGB_stack.shape[2], #W
                # "257": RGB_stack.shape[1], #H
                # "slices=": RGB_stack.shape[0],
                # "frames=": 1,
                # "channels=": RGB_stack.shape[3],
            }

            tiff.imwrite(self.file_path, stack, photometric='rgb', imagej=True, metadata=custom_metadata)  # You can adjust options as needed

    def update_event(self, inp=-1):
        self.SIGNALS.reset_widget.emit(1)


    # def get_state(self):
    #     return {'path': self.file_path}

    # def set_state(self, data, version):
    #     self.file_path = data['path']
    
class Morphological_Props(NodeBase0):
    title = 'Morphological Properties'
    init_inputs = [
        NodeInputBP('img'),
        # NodeInputBP('path', add_data={'widget name': 'path input', 'widget pos': 'below'}),
    ]   
    init_outputs = [
        NodeOutputBP('output img'), #img
    ]
    main_widget_class = widgets.OutputMetadataWidg
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)
        
        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                propertiesDf = Signal(pd.DataFrame, str)
                channels_dict = Signal(dict) #store colour channels 
                # propertiesStr = Signal(str)
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
              
        
        # self.file_path = ''

    def view_place_event(self):
        self.SIGNALS.propertiesDf.connect(self.main_widget().show_data)
        self.main_widget().new_data.connect(self.properties)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        
        # try:
        #      self.new_img_wrp = CVImage(self.get_img(self.sliced))
        #      self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        #      self.set_output_val(0, self.new_img_wrp)
        # except:  # there might not be an image ready yet
        #     pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        # self.main_widget().new_data(self.image_stack)
        # self.properties(self.image_stack)
        # self.new_img_wrp = CVImage(self.get_img(self.sliced))
        # if self.prev == True:
        #     if self.session.gui:
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        # self.proc_stack_parallel()
      
        self.set_output_val(0, (self.image_stack, self.stack_dict, self.z_sclice))
    
    def properties(self): #,true
        squeeze = np.squeeze(self.image_stack)
        squeeze = squeeze.astype(int)  #solved -- NB
        # print(squeeze.shape)
        # print(squeeze.dtype)
        labeled, numfeatures = label(squeeze)
        properties = ps.metrics.regionprops_3D(labeled)

        # Create an empty list to store dictionaries of property values
        property_dicts = []

        for pr in properties:
            property_dict = {
                'Label': pr.label,
                'Area': pr.area,
                'Centroid': pr.centroid,
                'Equivalent Diameter': pr.equivalent_diameter,
                'Euler Number': pr.euler_number,
                'Extent': pr.extent,
                'Filled Area': pr.filled_area,
                'Inertia Tensor Eigvals': pr.inertia_tensor_eigvals,
                'Volume (Physical Space)': pr.volume,
                'Surface Area (Physical Space)': pr.surface_area,
                'Sphericity (Physical Space)': pr.sphericity,
                'Aspect Ratio (Major/Minor)': np.sqrt(pr.inertia_tensor_eigvals[0] / pr.inertia_tensor_eigvals[-1])
            }
            property_dicts.append(property_dict)

            # print('append')

        # Create a pandas DataFrame from the list of property dictionaries
        df = pd.DataFrame(property_dicts)
        # Calculate the statistics
        total_structures = numfeatures  # Assuming 'df' is your existing DataFrame #rows
        area_avg = df["Area"].mean()
        centroid_avg = np.mean(df['Centroid'].to_list(), axis=0)
        Equivalent_Diameter_avg = df["Equivalent Diameter"].mean()
        Euler_avg = df["Euler Number"].mean()
        Extent_avg = df["Extent"].mean()
        Filled_Area = df["Filled Area"].mean()
        Inertia_avg = np.mean(df['Inertia Tensor Eigvals'].to_list(), axis=0)
        volume_avg = df['Volume (Physical Space)'].mean()
        Surface_area_avg = df['Surface Area (Physical Space)'].mean()
        Sphericity_avg = df['Sphericity (Physical Space)'].mean()
        Aspect_ratio = df['Aspect Ratio (Major/Minor)'].mean()

        summary_metadata = (
            f"Total Number of Structures: {total_structures}\n"
            f"Volume Avg: {volume_avg:.4f}\n"
            f"Surface Area Avg: {Surface_area_avg:.4f}\n"
            f"Area Avg: {area_avg:.4f}\n"
            f"Centroid Avg: {tuple(f'{x:.4f}' for x in centroid_avg)}\n"
            f"Equivalent Diameter Avg: {Equivalent_Diameter_avg:.4f}\n"
            f"Euler Number Avg: {Euler_avg:.4f}\n"
            f"Extent Avg: {Extent_avg:.4f}\n"
            f"Filled Area Avg: {Filled_Area:.4f}\n"
            f"Inertia Tensor Eigenvalues Avg:\n{tuple(f'{x:.4f}' for x in Inertia_avg)}\n"
            f"Sphericity Avg: {Sphericity_avg:.4f}\n"
            f"Aspect Ratio Avg (Major/Minor): {Aspect_ratio}\n"
        )
        if self.image_stack.shape[0] > 1:
            # Formatting the 'Centroid' and 'Inertia Tensor Eigvals' columns
            df['Centroid'] = df['Centroid'].apply(lambda x: f"{x[0]}, {x[1]}, {x[2]}")
            df['Inertia Tensor Eigvals'] = df['Inertia Tensor Eigvals'].apply(lambda x: f"{x[0]}, {x[1]}, {x[2]}")
            # print(df)

        self.SIGNALS.propertiesDf.emit(df, summary_metadata)
        
        # Convert the DataFrame to a string representation
        
        # self.main_widget().show_data(df)

        # Display the DataFrame
        

    # def get_state(self) -> dict:
    #     return {
    #         'val1': self.kk
    #     }

    # def set_state(self, data: dict, version):
    #     self.kk = data['val1']
    
class BatchProcess(NodeBase0):
    title = 'Batch Process'
    # input_widget_classes = {
    #     'choose file path': widgets.ChooseFileInputWidget
    # }
    init_inputs = [
        NodeInputBP('connet to end'),
        # NodeInputBP('path', add_data={'widget name': 'path input', 'widget pos': 'below'}),
    ]

    init_outputs = [
        NodeOutputBP('connect to start')
    ]

    main_widget_class = widgets.BatchPaths
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)
        
        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                reset_widget = Signal(int)
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        
        self.file_path = ''
        self.ttval = 0
        self.TIME_SERIES = False
        self.time_series_z_stacks = []
        # self.zzval = 4
        self.stack_dict = {
            "time_step": self.ttval,
            "total_time_frames": 0,
            "colour": {
                "red": 100,
                "green": 100,
                "blue": 100,
                "magenta": 100,
                "cyan": 100
            }
        }
        
        # self.actions['make executable'] = {'method': self.action_make_executable}

    def view_place_event(self):
        # self.input_widget(1).path_chosen.connect(self.path_chosen)
        # self.file_path.path_chosen.connect(self.path_chosen)
        self.main_widget().path_chosen.connect(self.path_chosen)
        self.main_widget().morphproperties.connect(self.initiate_morph_properties)
        self.SIGNALS.reset_widget.connect(self.main_widget().reset_w)
        self.proceed_batchp = 0
        self.i = 0
        self.morph = 0
        self.m = 0
        self.summary_metadata = []
    
    def path_chosen(self, paths):
        # Allow batch processing to start
        self.proceed_batchp = 1
        # Move to second file 
        self.i = 1

        self.morph = 0
        self.m = 0
        
        self.input_path = paths[0]
        self.output_path = paths[1]
        
        # Path to new folder
        # self.new_folder_name = "BatchProcessed"
        # self.new_folder_path = os.path.join(self.file_path, self.new_folder_name)
        # Create new file path
        print(f'output folder: {self.output_path}')

        # List files in the selected folder
        # files = os.listdir(self.file_path)
        files = os.listdir(self.input_path)
        self.tiff_files = [file for file in files if file.lower().endswith(('.tif', '.tiff'))]
        self.num_tiff = len(self.tiff_files)
        print("Number of files:", self.num_tiff)
        firstfile = self.tiff_files[0]
        self.extract_imagedata(self.input_path, firstfile)

    def initiate_morph_properties(self, paths):
        # clear summary if new input
        self.summary_metadata = []
        self.input_path = paths[0]
        self.morph_output_path = paths[1]

        # Allow morph properties to start
        self.morph = 1
        # Move to second file (when in update event)
        self.m = 1

        self.proceed_batchp = 0
        self.i = 0
        

        # Path to new folder
        # self.new_folder_name = "BatchProcessed"
        # self.new_folder_path = os.path.join(self.file_path, self.new_folder_name)
        # Create new file path
        print(f'output folder: {self.morph_output_path}')
    

        # List files in the selected folder
        # files = os.listdir(self.file_path)
        files = os.listdir(self.input_path)
        self.tiff_files = [file for file in files if file.lower().endswith(('.tif', '.tiff'))]
        self.num_tiff = len(self.tiff_files)
        print("Number of files:", self.num_tiff)
        firstfile = self.tiff_files[0]
        self.extract_imagedata(self.input_path, firstfile)

        
        

    def update_event(self, inp=-1):
        print('')
        print(f"first line of update event batchproc {self.proceed_batchp} morph {self.morph}")
        # from previous settings (before user selects anything)
        if self.proceed_batchp == 0 and self.morph == 0:
            self.stack_dict = self.input(0)[1] #dictioary
            self.z_sclice = self.input(0)[2]
            print(f"Batch Processing update event (tot frames): {self.stack_dict}")
        
        # once user selected batch proc imgs or morph, "if statement" above not true,
        # therefore this is the first file coming in
        # check if is time_series data
        else: 
            if self.stack_dict['total_time_frames'] > 1:
                self.TIME_SERIES = True
                print("time series = True")
        
        if self.TIME_SERIES ==  True:
            stack = self.input(0)[0]
            dictionery = self.input(0)[1]
            self.time_series(stack, dictionery)
            # once finished self.TIME_SERIES == False
        
        elif self.TIME_SERIES ==  False:
            # save the previous file (z-stack) (input)
            stack = self.input(0)[0]
            self.reshaped_data = stack[np.newaxis,:,:,:,:]
        
        # Only enter if there is one time frame or time series is complete, therefore can save:
        if self.TIME_SERIES ==  False:

            #New batch process May need to include somewhere ***
            # self.SIGNALS.reset_widget.emit(1)
            # Allow batch processing to start if batch processing button pressed
            print("Time series = False (saving)")
            print(f"proceed_batchp {self.proceed_batchp}, i {self.i} morph {self.morph}, m{self.m}")
            print("tot num tiff", self.num_tiff)

            if self.proceed_batchp == 1:
                print("self.proceed_batchp == 1")
                print("update event self.i", self.i)

                # save TZYXC (T could be 1, or greater)
                self.save_stack(self.reshaped_data)
                # Generate the morph csv for this time series

                # once saved last file, won't enter this
                if self.i < self.num_tiff:
                    print('perform on next file')
                    
                    # Step through files in the folder
                    # From second file to the end
                    filename = self.tiff_files[self.i]
                    # update i for next update event 
                    self.i += 1

                    # Extract and output the stack to the pipeline
                    self.extract_imagedata(self.input_path, filename)

                 

            if self.morph == 1:            
                print("self.morph == 1")
                print("update event self.m", self.m)
                print(f"m {type(self.m)} num_tiff {type(self.num_tiff)}")
                print(f"m {self.m} num_tiff {self.num_tiff}")
                # once saved last file, won't enter this
                self.save_morph_csv(self.reshaped_data)

                if self.m < self.num_tiff:
                    print('perform on next file')
                    
                    # Step through files in the folder
                    # From second file to the end
                    filename_morph = self.tiff_files[self.m]
                    print("current morph_file", filename_morph)
                    # update m for next update event 
                    # move to next file - only effects line above
                    # filename_morph is on current file which goes to extract_imagedata
                    self.m += 1

                    # Extract and output the stack to the pipeline 
                    self.extract_imagedata(self.input_path, filename_morph)
                    
                
                elif self.m == self.num_tiff:
                    print("m == number of files, save summary")
                    self.save_morph_summary()
                    self.m += 1
                
    
    def extract_imagedata(self, file_path, filename):
        #create new file path for the stack - where it will be saved
        print("value of self.i is",self.i)
        print("value of m is", self.m)

        # create file name for when update even is triggered, to save this file
        if self.i > 0:
            self.new_file_path = os.path.join(self.output_path, "batched_processed_" + filename)
        
        elif self.m > 0:
            self.base_filename = os.path.splitext(filename)[0]
            csv_filename = f"{self.base_filename}.csv"
            print("current csv filename", csv_filename)
            self.new_morph_path_csv = os.path.join(self.morph_output_path, csv_filename)
            
    
        if os.path.isfile(os.path.join(file_path, filename)):
            currentfile = os.path.join(file_path, filename)
            print(f'filepath currently batch processing: {currentfile}')
            if currentfile == '':
                return
            # Check if the file has a .tiff extension   --> tif file capability Check tiff 
            if currentfile.endswith('.tif'): #----------------------------------------------------------------------------------TIFF
                try:
                    self.image_data = tiff.imread(currentfile)

                    # Normalize - generate dimension list (T,Z,H,W,C)
                    self.dim = self.id_tiff_dim(currentfile)
                    # find how many total frames in stack_n
                    self.stack_dict['total_time_frames'] = self.dim[0]
                    self.stack_dict['time_step'] = 0
                    # Reshape - STANDARDIZED 
                    self.image_data = ((self.image_data / np.max(self.image_data)) * 255).astype(np.uint8)
                    
                    # IF shape: ZCXY ---------------------------------------------------------------
                    # Change to TZXYC
                    if self.image_data.shape[-3] <= 4:
                        
                        # dim = [1, 15, 1024, 1024, 3]
                        print("Strange dimension")
                        #if no time step data 
                        if len(self.image_data.shape) == 4: 
                            print(f'length {len(self.image_data.shape)}')
                            self.image_data = self.image_data[np.newaxis,:,:,:,:]

                        # Create an empty array with the specified dimensions
                        image_data_stacked = np.empty(self.dim, dtype=np.uint8)

                        num_time_frames = self.dim[0]  # Get the number of time frames
                        
                        for t in range(num_time_frames):  # Iterate over the time frames
                            for i in range(self.dim[1]):  # Iterate over the images in each time frame
                                # Get the red, green, and blue channels for the i-th image in the t-th time frame
                                print(i)
                                chan_0 = self.image_data[t, i, 0, :, :]  # Assuming the first dimension is time frame, then image index
                                chan_1 = self.image_data[t, i, 1, :, :]
                                chan_2 = self.image_data[t, i, 2, :, :]
                                
                                # Stack the channels along the last axis
                                image_data_stacked[t, i, :, :, 0] = chan_0
                                image_data_stacked[t, i, :, :, 1] = chan_1
                                image_data_stacked[t, i, :, :, 2] = chan_2   
                                # if have a an alpha channel
                                if self.image_data.shape[2] >= 4:
                                    chan_3 = self.image_data[t, i, 3, :, :] 
                                    image_data_stacked[t, i, :, :, 3] = chan_3 

                                # if have a an alpha channel
                                if self.image_data.shape[2] == 5:
                                    chan_4 = self.image_data[t, i, 4, :, :] 
                                    image_data_stacked[t, i, :, :, 4] = chan_4 

                        print(image_data_stacked.shape)   

                        self.image_data = image_data_stacked 
                    # ---------------------------------------------------------------------------------           

                    # Full 5D (potentially) stack: TZYZC  
                    self.reshaped_data = self.image_data.reshape(self.dim)
                    # create empty stak for time series data
                    self.time_series_z_stacks = np.empty_like(self.reshaped_data)

                    print("set output, i =", self.i)

                    # emmit first z-stack 0ZYZC  
                    self.set_output_val(0, (self.reshaped_data[0, :, :, :, :], self.stack_dict, self.z_sclice))
                
                except Exception as e:
                    print(e)
                    print("failed")

    def time_series(self, stack, dictionery):
        if dictionery["time_step"] < (dictionery["total_time_frames"]-1):
            t_step = dictionery["time_step"]
            print(f"time series, timestep:{t_step}")
            self.time_series_z_stacks[int(t_step)] = stack
            dictionery["time_step"] += 1
            # output next time_step z-stack
            self.set_output_val(0, (self.reshaped_data[dictionery["time_step"], :, :, :, :], dictionery, self.z_sclice))
        
        elif dictionery["time_step"] == (dictionery["total_time_frames"]-1):
            final_step = dictionery["time_step"]
            print(f"FINAL series, timestep: {final_step}")
            # final stack
            self.time_series_z_stacks[int(final_step)] = stack
            self.TIME_SERIES = False


    def save_stack(self, stack):
        #perform a check to see if folder exists 
        # if it exisists rename + "n"
        # n += 1 (set to zero when create folder)
        if stack.shape[3] == 1:
            # SENDING IN 5D
            grayscale_stack = stack[:, :, :, :, 0]
            #NEED TO CHANGE 259 / slices=
            metadata = {
                'Description': 'Stack (RGB) preprocessed with Visual Processing Pipeline. Developed using Ryven at Stellenbosch University by Emma Sharratt and Dr Rensue Theart',
                'Author': 'Emma Sharratt and Dr Rensu Theart',
                'axes': 'TZYX'
                # 'Width': grayscale_stack.shape[2],
                # 'Height': grayscale_stack.shape[1],
                # 'BitsPerSample': 16,  # Adjust as needed
                # 'Photometric': 'minisblack',
                # Add more metadata fields as needed
                }
            # print("stack shape {grayscale_stack}")
            tiff.imwrite(self.new_file_path, grayscale_stack, photometric='minisblack', imagej=True, metadata=metadata)  # You can adjust options as needed
            print(f"Grayscale stack saved with shape {grayscale_stack.shape} to {self.new_file_path}")
        
        if stack.shape[3] == 3:
            # WHY ONLY SAVE FIRST??
            # RGB_stack = stack[:, :, :, :]
            # print(f"RGB shape {RGB_stack.shape}")
            custom_metadata = {
                "Description": "Stack (RGB) preprocessed with Visual Processing Pipeline. Developed using Ryven by Emma Sharratt and Dr Rensue Theart",
                "Author": "Emma Sharratt and Dr Rensue Theart",
                "Date": "Pipeline created in 2023",
                'axes': 'ZCYX'
                # "256": RGB_stack.shape[2], #W
                # "257": RGB_stack.shape[1], #H
                # "slices=": RGB_stack.shape[0],
                # "frames=": 1,
                # "channels=": RGB_stack.shape[3],
            }
            print(f"RGB save shape: {stack.shape}")
            tiff.imwrite(self.new_file_path, stack, photometric='rgb', imagej=True, metadata=custom_metadata)  # You can adjust options as needed
        # tiff.imwrite(self.new_file_path, stack, photometric='rgb', imagej=True, metadata=custom_metadata)  # You can adjust options as needed

        # Time series data
        if stack.shape[0]>1 :
            print("save time series stack img")
            grayscale_stack = stack[:, :, :, :, 0]
            # WHY ONLY SAVE FIRST??
            # RGB_stack = stack[:, :, :, :]
            # print(f"RGB shape {RGB_stack.shape}")
            custom_metadata = {
                "Description": "Stack (RGB) preprocessed with Visual Processing Pipeline. Developed using Ryven by Emma Sharratt and Dr Rensue Theart",
                "Author": "Emma Sharratt and Dr Rensue Theart",
                "Date": "Pipeline created in 2023",
                'axes': 'TZYX'
                # "256": RGB_stack.shape[2], #W
                # "257": RGB_stack.shape[1], #H
                # "slices=": RGB_stack.shape[0],
                # "frames=": 1,
                # "channels=": RGB_stack.shape[3],
            }
            tiff.imwrite(self.new_file_path, grayscale_stack, photometric='minisblack', imagej=True, metadata=custom_metadata)  # You can adjust options as needed
            
        # FOUR CHANNEL
        # TWO CHANNEL
        # Assuming stack is in TZCYX format with 2 channels
        # if stack.shape[3] == 2:  # Check if it's 2-channel data
        #     custom_metadata = {
        #         "Description": "Stack (2-channel) preprocessed with Visual Processing Pipeline. Developed using Ryven by Emma Sharratt and Dr Rensue Theart",
        #         "Author": "Emma Sharratt and Dr Rensue Theart",
        #         "Date": "Pipeline created in 2023",
        #         'axes': 'TZCYX'  # Reflects Time, Z-slices, Channels (2), Y, X
        #     }

        #     print(f"2-channel save shape: {stack.shape}")
            
        #     # Saving the 2-channel stack as a TIFF
        #     tiff.imwrite(self.new_file_path, stack, 
        #                 photometric='minisblack',  # Use 'minisblack' for grayscale-like data
        #                 imagej=True,  # ImageJ-compatible metadata
        #                 metadata=custom_metadata)

    def id_tiff_dim(self,f_path):
        tif_file = tiff.TiffFile(f_path)
        # Check for TIFF metadata tags
        metadata = tif_file.pages[0].tags
        if metadata:
            # print("Metadata Tags:")
            # for tag_name, tag in metadata.items():
                # print(f"{tag_name}: {tag.value}")

            #set dimension to 0 when a new tiff file is processed
            dimension = [1,1,1,1,1] #dim, slices , time
            
            
            #  T Z Y X C  (F, Z, H, W, C)
            #  0 1 2 3 4
            
            if 256 in metadata: #width
                            # Access the tag value directly
                            dimension[3] = metadata[256].value
            if 257 in metadata: #H
                            # Access the tag value directly
                            dimension[2] = metadata[257].value
            if 277 in metadata: #channels
                            # Access the tag value directly
                            dimension[4] = metadata[277].value
            if 259 in metadata:  # Tag for slices
                            print("meta",metadata[259].value)
                            dimension[1] = metadata[259].value
                        
            if 'ImageDescription' in metadata:
                    # Access 'ImageDescription' tag
                    image_description = metadata['ImageDescription']
            
                    # Split the 'ImageDescription' string into lines
                    description_lines = image_description.value.split('\n')
                    # Parse the lines to extract slices and frames information
                    for line in description_lines:
                        # if 262 in metadata:  # Tag for frames
                        #     dimension[4] = metadata[262].value
                        #     print("dim",dimension[4])
                        if line.startswith("slices="):
                            dimension[1] = int(line.split('=')[1]) #slice
                        if line.startswith("frames="):
                            dimension[0] = int(line.split('=')[1]) #frames
                            # print("frames", int(line.split('=')[1]))
                            # print("dim",dimension[4])
                        if line.startswith("channels="):
                            dimension[4] = int(line.split('=')[1]) #frames
                            # print("dim",dimension[4])
                        
        else:
                print("ImageDescription tag not found in metadata.")
                        
        # print(f'Width: {dimension[3]}')
        # print(f'Height: {dimension[2]}')
        # print(f'Channels: {dimension[4]}')
        # print(f"Slices: {dimension[1]}")
        # print(f"Frames: {dimension[0]}")
        # print(f'Dimension: {dimension[0]}')
        # self.dimension=dimension
        # set_widg = [1,1]
        # set_widg[0] = dimension[0] #t
        # set_widg[1] = dimension[1] #z
        # self.SIGNALS.reset_widget.emit(set_widg)
        # self.SIGNALS.image_shape.emit(dimension)
        # self.stack_dict["time_step"]= 1
        # self.zzval= round(set_widg[1]/2)
        # if (dimension[0])==1 or (dimension[1])==1:
        #     set_widg = [1,1]
        #     self.SIGNALS.reset_widget.emit(set_widg)
        #     self.stack_dict["time_step"]=0
        #     self.zzval=0
        print(f'Image dim: {dimension}')
        # Set the Z and T slice at the midpoint to display through the pipeline
        self.zzval = int(dimension[1] /2)
        self.stack_dict["time_step"] = int(dimension[0] / 2)
        return dimension

    
    def save_morph_csv(self, stack):
        print("SAVE MORPH CSV function")
        squeeze = np.squeeze(stack)
        squeeze = squeeze.astype(int)  #solved -- NB
        # print(squeeze.shape)
        # print(squeeze.dtype)
        labeled, numfeatures = label(squeeze)
        properties = ps.metrics.regionprops_3D(labeled)

        # Create an empty list to store dictionaries of property values
        property_dicts = []
        print("test batch process 1")
        #loop through regions
        for pr in properties:
            property_dict = {
                'Label': pr.label,
                'Area': pr.area,
                'Centroid': pr.centroid,
                'Equivalent Diameter': pr.equivalent_diameter,
                'Euler Number': pr.euler_number,
                'Extent': pr.extent,
                'Filled Area': pr.filled_area,
                'Inertia Tensor Eigvals': pr.inertia_tensor_eigvals,
                'Volume (Physical Space)': pr.volume,
                'Surface Area (Physical Space)': pr.surface_area,
                'Sphericity (Physical Space)': pr.sphericity,
                'Aspect Ratio (Major/Minor)': np.sqrt(pr.inertia_tensor_eigvals[0] / pr.inertia_tensor_eigvals[-1])
            }
            property_dicts.append(property_dict)
        print("test batch process 2")
        # Create a pandas DataFrame from the list of property dictionaries
        df = pd.DataFrame(property_dicts)
        # Calculate the statistics
        total_structures = numfeatures  # Assuming 'df' is your existing DataFrame #rows
        area_avg = df["Area"].mean()
        centroid_avg = np.mean(df['Centroid'].to_list(), axis=0)
        equivalent_diameter_avg = df["Equivalent Diameter"].mean()
        euler_avg = df["Euler Number"].mean()
        extent_avg = df["Extent"].mean()
        filled_area_avg = df["Filled Area"].mean()
        inertia_avg = np.mean(df['Inertia Tensor Eigvals'].to_list(), axis=0)
        volume_avg = df['Volume (Physical Space)'].mean()
        surface_area_avg = df['Surface Area (Physical Space)'].mean()
        sphericity_avg = df['Sphericity (Physical Space)'].mean()
        aspect_ratio_avg = df['Aspect Ratio (Major/Minor)'].mean()

        
        # SUMMARY -----------------
        # Append to summary_metadata list
        # This will be a row in the summary csv (one per timestep)
        summary_metadata_dict = {}
        summary_metadata_dict = {
            # CHECK THI S***
            'Filename': self.base_filename,
            'Timestep': self.stack_dict["time_step"],
            'Total Structures': total_structures,
            'Volume Avg (Physical Space)': volume_avg,
            'Surface Area Avg (Physical Space)': surface_area_avg,
            'Area Avg': area_avg,
            'Centroid Avg': tuple(centroid_avg),
            'Equivalent Diameter Avg': equivalent_diameter_avg,
            'Euler Number Avg': euler_avg,
            'Extent Avg': extent_avg,
            'Filled Area Avg': filled_area_avg,
            'Inertia Tensor Eigenvalues Avg': tuple(inertia_avg),
            'Sphericity Avg (Physical Space)': sphericity_avg,
            'Aspect Ratio Avg (Major/Minor)': aspect_ratio_avg
        }
    
        self.summary_metadata.append(summary_metadata_dict)
        
        print("summary:", self.summary_metadata)
        
        # Export csv per timestep
        # independent of the tuple length (therefore will work for 2D and 3D ++)
        df['Centroid'] = df['Centroid'].apply(lambda x: ', '.join(map(str, x)))
        df['Inertia Tensor Eigvals'] = df['Inertia Tensor Eigvals'].apply(lambda x: ', '.join(map(str, x)))
        
        print("path", self.new_morph_path_csv)
        print(df)
        df.to_csv(self.new_morph_path_csv, index =False)
        
    
    
    def save_morph_summary(self):
        # make summary
        print("SUMMARY function")
        summary_metadata_df = pd.DataFrame(self.summary_metadata)
        # independent of the tuple length (therefore will work for 2D and 3D ++)
        summary_metadata_df['Centroid Avg'] = summary_metadata_df['Centroid Avg'].apply(lambda x: ', '.join(map(str, x)))
        summary_metadata_df['Inertia Tensor Eigenvalues Avg'] = summary_metadata_df['Inertia Tensor Eigenvalues Avg'].apply(lambda x: ', '.join(map(str, x)))
        summary_path = self.morph_output_path + r"\time_series_summary.csv"
        print("path", summary_path)
        summary_metadata_df.to_csv(summary_path, index =False)


        

class DisplayImg(OpenCVNodeBase):
    title = 'Display Image'
    init_inputs = [
        NodeInputBP('img'),
    ]

    def get_img(self):
        return self.sliced #.img
        # return self.input(0).img

class Crop(NodeBase0):        #Nodebase just a different colour
          #Nodebase just a different colour
    # color = '#00a6ff'
    title = 'Crop Stack'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Crop_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                dimension = Signal(object) 
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict)
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.top = 1
        self.bot = 1
        self.left = 1
        self.right = 1

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.dimension.connect(self.main_widget().dimensions)
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)

        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        self.main_widget().kValueChanged.connect(self.crop_stack)
        self.main_widget().bValueChanged.connect(self.onBSliderValueChanged)
        self.main_widget().bValueChanged.connect(self.crop_stack)
        self.main_widget().lValueChanged.connect(self.onLSliderValueChanged)
        self.main_widget().lValueChanged.connect(self.crop_stack)
        self.main_widget().rValueChanged.connect(self.onRSliderValueChanged)
        self.main_widget().rValueChanged.connect(self.crop_stack)
        self.main_widget().previewState.connect(self.preview)
        
        try:
             self.new_img_wrp = CVImage(self.zoomed_cropped_image)
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        target_h = self.image_stack.shape[1]
        target_w =self.image_stack.shape[2] 
        self.SIGNALS.dimension.emit((target_h, target_w))
        # print(f"stack h {self.image_stack.shape[1]}")
        self.crop_stack()
        self.new_img_wrp = CVImage(self.cropped_slice)
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
      
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def onSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.top = value
        self.new_img_wrp = CVImage(self.cropped_slice)
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

    def onBSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.bot = value
        self.new_img_wrp = CVImage(self.cropped_slice)
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
        
    def onLSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.left = value
        self.new_img_wrp = CVImage(self.cropped_slice)
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
    
    def onRSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.right = value
        self.new_img_wrp = CVImage(self.cropped_slice)
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
    
    def crop_stack(self):
        #debug
        # print(f"bot value shape2 {self.bot}")
        # print(f'crop value k dtype {type(self.bot)}')
        
        # cropeped data
        self.reshaped_proc_data = np.ascontiguousarray(self.image_stack[:, self.top:-self.bot,self.left:-self.right, :])
        # zoomed_image_stack = cv2.resize(self.reshaped_proc_data, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        self.cropped_slice = self.reshaped_proc_data[self.z_sclice, :, :, :]
        # self.zoomed_cropped_image = cv2.resize(self.cropped_slice, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def get_state(self) -> dict:
        return {
            'val1': self.top,
            'val2': self.bot,
            'val3': self.left,
            'val4': self.right,
        }

    def set_state(self, data: dict, version):
        self.top = data['val1']
        self.bot = data['val2']
        self.left = data['val3']
        self.right = data['val4']

#Split Channels --------------------------------------------------------------------------------------------------------------
class Split_Img(NodeBase0):
    title = 'Split Channels'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('channel 0'), #img
        NodeOutputBP('channel 1'), #img
        NodeOutputBP('channel 2'), #img
        NodeOutputBP('channel 3'), #img
        NodeOutputBP('channel 4'), #img
        NodeOutputBP('channel 5'), #img
    ]
    main_widget_class = widgets.Split_Img
    main_widget_pos = 'below ports'

    # Please ensure the colour channels are selected in the correct order Eg RGB must be ch1:R ch2:G ch3:B ch4:None)
    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict)
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        # default1 = 3
        # default2 = 1
        # self.value_1 = default1  #threshold
        # self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        # self.main_widget().Value1Changed.connect(self.ValueChanged1)
        # self.main_widget().Value1Changed.connect(self.proc_stack_parallel)
        # self.main_widget().Value2Changed.connect(self.ValueChanged2)
        # self.main_widget().Value2Changed.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.sliced)
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        print(f"SPLIT stack dict:{self.stack_dict}")
        # send channel information to widget to display img in selected channel colours
        self.SIGNALS.channels_dict.emit(self.stack_dict) # stack dict defined in handel_stack
        
        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        print("SPLIT!")
        self.proc_technique()
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))

    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    # def ValueChanged1(self, value):
    #     # This method will be called whenever the widget's signal is emitted
    #     # #print(value)
    #     self.value_1 = value
    #     self.new_img_wrp = CVImage(self.get_img(self.sliced))
    #     if self.prev == True:
    #         if self.session.gui:
    #             #update continuously 
    #             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    #     else:
    #         self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    # def ValueChanged2(self, value):
    #     # This method will be called whenever the widget's signal is emitted
    #     # #print(value)
    #     self.value_2 = value
    #     self.new_img_wrp = CVImage(self.get_img(self.sliced))
    #     if self.prev == True:
    #         if self.session.gui:
    #             #update continuously 
    #             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    #     else:
    #         self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

    
    # def get_img(self, zslice):
    #     #PROCESS SLICE 
    #     #generate slice for dispay
    #     # reshaped = self.sliced.reshape(zslice.shape[:-1] + (-1,))
    #     # print(f"size shape {zslice.shape}")
    
    #     # Apply median blur to all channels simultaneously
    #     processed_data = self.proc_technique(zslice)
        
    #     # Reshape the processed data back to the original shape
    #     # Ensures [, , 1] one at the end stays 
    #     processed_slice = processed_data.reshape(zslice.shape)
        
    #     return processed_slice
              
              
    def proc_technique(self):
        # add check for RGB

        # if shape[-1] == 1:
        #     print("GRAYSCALE")

        # elif shape[-1] > 1: 

        # Initialize an empty image array to use for blank channels
        # self.image_stack: 4D stack ZXYC
        blank_image = np.zeros_like(self.image_stack[:, :, :, 0])

        # Initialize a list to store the output values
        output_values = [blank_image] * 6  # Initialize with blank images

        # Loop through the colors in the dictionary and assign channels to outputs
        for color, channel_idx in self.stack_dict["colour"].items():
            print(f'node split_img dict: {self.stack_dict}')
            if channel_idx != 100:
                # Extract the channel from the image
                channel = self.image_stack[:, :, :, channel_idx]
                channel = channel[:,:,:,np.newaxis]
                # Set the output value for the corresponding output index
                output_values[channel_idx] = channel

        # Set output values for each output idx: colour channel
        for output_idx, output_value in enumerate(output_values):
            print(f"channel {output_idx} shape: {output_value.shape}")
            self.set_output_val(output_idx, (output_value, self.stack_dict, self.z_sclice))
            # self.set_output_val(output_idx, output_value)



        # stack4D=self.image_stack
        # print(f"shape stack4D {stack4D.shape}")
        # # Split the RGB data into separate channels
        # shape = stack4D.shape

        # if shape[-1] == 1:
        #     print("GRAYSCALE")

        # elif shape[-1] > 1: 
        #     red_stack = stack4D[..., 0]  # Extract the red channel (index 0)
        #     red_stack = red_stack[:, :, :, np.newaxis]

        #     green_stack = stack4D[..., 1]  # Extract the green channel (index 1)
        #     green_stack = green_stack[:, :, :, np.newaxis]

        #     blue_stack = stack4D[..., 2]  # Extract the blue channel (index 2)
        #     blue_stack = blue_stack[:, :, :, np.newaxis]

        #     if shape[-1] == 4:
                

        #         magenta_stack = stack4D[..., 3]  # Extract the blue channel (index 2)
        #         magenta_stack = magenta_stack[:, :, :, np.newaxis]

        #         self.set_output_val(0, (red_stack, self.frame, self.z_sclice))
        #         print(f"shape split: {red_stack.shape}")
        #         self.set_output_val(1, (green_stack, self.frame, self.z_sclice))
        #         self.set_output_val(2, (blue_stack, self.frame, self.z_sclice))
        #         self.set_output_val(3, (magenta_stack, self.frame, self.z_sclice))

        #     # If the input is an RGB image
        #     elif shape[-1] == 3:
        #         blank_stack = np.zeros_like(stack4D[..., 0])
        #         self.set_output_val(0, (red_stack, self.frame, self.z_sclice))
        #         print(f"shape split: {red_stack.shape}")
        #         self.set_output_val(1, (green_stack, self.frame, self.z_sclice))
        #         self.set_output_val(2, (blue_stack, self.frame, self.z_sclice))    
        #         # send a blank stack to magenta channel 
        #         self.set_output_val(3, (blank_stack, self.frame, self.z_sclice))
        # #print(f"proc input stack: {stack4D.shape}")
         #Ensure only on 3D data
        # if stack4D.shape[0] > 1:
        #     # Apply the gaussian filter to the entire 4D array
        #     filtered_image = gaussian_filter(stack4D, sigma=(self.kk, self.yy,self.xx, 0))
            # #print(self.kk)
            # prevent the use of for loops, but no sigma applied to channels
            #print(f"filtered_image {filtered_image.shape}")
            # update displayed image
        # self.sliced = filtered_image[self.z_sclice, :, :, :]
        # self.reshaped_proc_data= filtered_image
            
            # DONT NEED TO UPDATE IMAGE 
        # self.new_img_wrp = CVImage(self.sliced)
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
        
                
    
    # def get_img(self, zslice):
        
    #     processed_data = self.proc_technique(zslice)
    #     #dont need to rshape because do at the end of binarization in proc_technique
        
    #     return processed_data

    # #signle time step
    # def proc_stack_parallel(self):
        
    #     # Define the number of worker threads or processes
    #     num_workers = 6  # Adjust as needed

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = []
    #         # print(f"z size {self.z_size}")
    #         for z in range(self.z_size):
    #                 img = self.image_stack[z]
    #                 # print(f"\nprocessed z slice {z}")
    #                 print("SPLIT")
    #                 future = executor.submit(self.get_img, img)
    #                 futures.append((z, future))

    #         #THIS IS DIFFERENT***
    #         # Create proc_data from processed_frame (not sclice - this may be grayscale)
    #         proc_data = np.empty((self.image_stack.shape[0], self.image_stack.shape[1], self.image_stack.shape[2], 3), dtype=np.uint8)
    #         # proc_data = [None] * 3
    #         # np.empty_like(self.image_stack)

    #         for z, future in futures:
    #             processed_frame = future.result()
    #             proc_data[z] = processed_frame
        
    #     # print(f"proc_data shape: {proc_data.shape}")
    #     self.reshaped_proc_data = proc_data
    #     # print(f"reshaped_proc_data shape: {self.reshaped_proc_data.shape}")
        
    #     self.set_output_val(0, (self.reshaped_proc_data[0], self.frame, self.z_sclice))
    #     print(f"shape split: {self.reshaped_proc_data[0].shape}")
    #     self.set_output_val(1, (self.reshaped_proc_data[1], self.frame, self.z_sclice))
    #     self.set_output_val(2, (self.reshaped_proc_data[2], self.frame, self.z_sclice))
      

    # def proc_technique(self,img):
    #     # b, g, r = cv2.split(img)
    #     # split = [b,g,r]
    #     # return split
    #     b, g, r = cv2.split(img)
    #     return np.stack([b, g, r], axis=-1)
    
    # def get_state(self) -> dict:
    #     return {
    #         'val1': self.value_1,
    #         'val2': self.value_2,
    #     }

    # def set_state(self, data: dict, version):
    #     self.value_1 = data['val1']
    #     self.value_2 = data['val2']

class Merge_Img(Node):
    title = 'Merge Channels (RGB_CMY)'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('channel 0'),
        NodeInputBP('channel 1'),
        NodeInputBP('channel 2'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img
    ]
    main_widget_class = widgets.Split_Img
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added     
                channels_dict = Signal(dict) 
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        # default1 = 3
        # default2 = 1
        # self.value_1 = default1  #threshold
        # self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        # self.main_widget().Value1Changed.connect(self.ValueChanged1)
        # self.main_widget().Value1Changed.connect(self.proc_stack_parallel)
        # self.main_widget().Value2Changed.connect(self.ValueChanged2)
        # self.main_widget().Value2Changed.connect(self.proc_stack_parallel)
        
        # try:
        #     #  self.new_img_wrp = CVImage(self.sliced)
        #     #  self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        #     #  self.set_output_val(0, self.new_img_wrp)
        # except:  # there might not be an image ready yet
        #     pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.proc_technique()
        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        print("SPLIT!")
        
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def handle_stack(self):
        self.chan_0 = self.input(0)[0]
        self.chan_1 = self.input(1)[0]
        self.chan_2 = self.input(2)[0]
        self.stack_dict = self.input(0)[1] #dictioary
        self.z_sclice = self.input(0)[2]
        # self.squeeze = np.squeeze(self.image_stack)
        self.z_size = self.chan_0.shape[0]
        print(f"z_size {self.z_size}")
        # self.sliced = self.image_stack[self.z_sclice, :, :, :] #Z, H, W, C
              
    def proc_technique(self):
        # if 3 inputs: 
        self.rgb_image = np.concatenate((self.chan_0, self.chan_1, self.chan_2), axis=-1)
        # if 4 inputs: 
        # still to implement ----------
        self.sliced = self.rgb_image[self.z_sclice, :, :, :] #Z, H, W, C
        self.set_output_val(0, (self.rgb_image, self.stack_dict, self.z_sclice))
        
        
        # print(f"shape split: {red_stack.shape}")
        # self.set_output_val(1, (green_stack, self.frame, self.z_sclice))
        # self.set_output_val(2, (blue_stack, self.frame, self.z_sclice))
    
    # def get_state(self) -> dict:
    #     return {
    #         'val1': self.value_1,
    #         'val2': self.value_2,
    #     }

    # def set_state(self, data: dict, version):
    #     self.value_1 = data['val1']
    #     self.value_2 = data['val2']

# Filttering ------------------------------------------------------------------------------------------------------------------

class Blur_Averaging(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Averaging (Blur)'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Blur_Averaging_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added 
                channels_dict = Signal(dict) #store colour channels     
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        self.main_widget().kValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().previewState.connect(self.preview)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        print("stack dict from handle dict: ", self.stack_dict)
        # send assigned channels to widegt to display / show image correctly
        self.SIGNALS.channels_dict.emit(self.stack_dict)

        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
      
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def onSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
    
    def proc_technique(self,img):
        #debug
        # print(f"getimageValue{value}")
        return cv2.blur(
            src=img,
            ksize=(self.kk,self.kk),
                )
    
    def get_state(self) -> dict:
        return {
            'val1': self.kk
        }

    def set_state(self, data: dict, version):
        self.kk = data['val1']
    


class Median_Blur(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Median Blur'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Blur_Median_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added
                channels_dict = Signal(dict) #store colour channels      
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        self.main_widget().kValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().previewState.connect(self.preview)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, (self.proc_stack(), self.frame, self.z_sclice))
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        # print(f"type {self.sliced.dtype}")
        # print(f"shape {self.sliced.shape}")
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
                
        self.proc_stack_parallel()
      
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def onSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        # print(f"new shape {self.input(0)[0].shape}")
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)        
    
    #get_img specific
    def proc_technique(self,img):
        # print(f"median Blur, ksize {self.kk}\n")
        return cv2.medianBlur(img, self.kk)
        
          # #use when save and close
    def get_state(self) -> dict:
        return {
            'val1': self.kk,

        }

    def set_state(self, data: dict, version):
        self.kk = data['val1']


class Gaussian_Blur(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Gaussian Blur2D'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Gaus_Blur_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added     
                channels_dict = Signal(dict) #store colour channels  
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 5
        self.kk = 1
        self.xx = 1
        self.yy = 1
        # self.kk = default
        # self.xx = default
        # self.yy = default

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().kValueChanged.connect(self.onkValueChanged)
        self.main_widget().XValueChanged.connect(self.onXvalueChanged)        
        self.main_widget().YValueChanged.connect(self.onYvalueChanged)
        self.main_widget().sigValueChanged.connect(self.onSigvalueChanged)

        self.main_widget().kValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().XValueChanged.connect(self.proc_stack_parallel)        
        self.main_widget().YValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().sigValueChanged.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def onkValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        #print(value)
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)


    def onXvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.xx = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
    
    def onYvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.yy = value
        # #print(self.yy)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
    
    def onSigvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.yy = value
        self.xx = value
        # #print(self.yy)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
    
    def proc_technique(self,img):
        # debug
        #print(self.xx)
        #print(self.yy)
        return cv2.GaussianBlur(
            src=img,
            ksize=(self.kk, self.kk),
            sigmaX=self.xx,
            sigmaY=self.yy,
        )
    
      # #use when save and close
    def get_state(self) -> dict:
        print("node, get_state")
        data = {
            'ksize': self.kk,
            'sigmaX': self.xx,
            # linked vs not linked?
            'sigmaY': self.yy,
        }
        print("node didnt crash")
        return data

    def set_state(self, data: dict, version):
        self.kk = data['ksize']
        self.xx = data['sigmaX']
        self.yy = data['sigmaY']

class Gaussian_Blur3D(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Gaussian Blur3D'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Gaus_Blur_MainWidget3D
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                warning = Signal(int)
                channels_dict = Signal(dict) #store colour channels  
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 2
        self.kk = default
        self.xx = default
        self.yy = default
        warning = 0

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.warning.connect(self.main_widget().warning)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().kValueChanged.connect(self.onkValueChanged)
        self.main_widget().XValueChanged.connect(self.onXvalueChanged)        
        self.main_widget().YValueChanged.connect(self.onYvalueChanged)
        # updates both X and Y when linked 
        self.main_widget().sigChanged.connect(self.onSigValueChanged) 
        

        self.main_widget().kValueChanged.connect(self.proc_technique)
        self.main_widget().XValueChanged.connect(self.proc_technique)        
        self.main_widget().YValueChanged.connect(self.proc_technique)
        self.main_widget().sigChanged.connect(self.proc_technique)
        
        try:
             self.new_img_wrp = CVImage(self.sliced)
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        # 3D
        if self.z_size > 1:
            self.SIGNALS.warning.emit(0)
            self.proc_technique()

        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # 2D
        if self.z_size == 1:
            self.SIGNALS.warning.emit(1)
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 
        # 3D process stack
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def onkValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        #print(value)
        self.kk = value


    def onXvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.xx = value
       
    def onYvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.yy = value
        
    def onSigValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(self.kk)
        # self.kk = value
        self.xx = value
        self.yy = value
       
    def proc_technique(self):
        stack4D=self.image_stack
        # #print(f"proc input stack: {stack4D.shape}")
         #Ensure only on 3D data
        if stack4D.shape[0] > 1:
            # Apply the gaussian filter to the entire 4D array
            print(f"Performing gaus blur 3D x: {self.xx}, y:{self.yy}, z:{self.kk}")
            print(f"stack4D shape: {stack4D.shape}")
            filtered_image = gaussian_filter(stack4D, sigma=(self.kk, self.yy,self.xx, 0))
          
            self.sliced = filtered_image[self.z_sclice, :, :, :]
            self.reshaped_proc_data= filtered_image
            
            self.new_img_wrp = CVImage(self.sliced)
            if self.prev == True:
                if self.session.gui:
                    #update continuously 
                    self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            else:
                self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

            self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
                
        
            #emit Warning! A 3D Gaussian Blur cannot be performed on 2D data
            # Preprocessing has not been performed on the data
            # To use this node please read 3D data into the pipeline 
    def get_state(self) -> dict:
        print("node, get_state")
        data = {
            'sigmaZ': self.kk,
            'sigmaX': self.xx,
            'sigmaY': self.yy,
        }
        print("node didnt crash")
        return data

    def set_state(self, data: dict, version):
        self.kk = data['sigmaZ']
        self.xx = data['sigmaX']
        self.yy = data['sigmaY']


class Bilateral_Filtering(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Bilateral Filtering'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Bilateral_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) #store colour channels     
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 5
        self.kk = default
        self.xx = default
        self.yy = default

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().kValueChanged.connect(self.onkValueChanged)
        self.main_widget().XValueChanged.connect(self.onXvalueChanged)        
        self.main_widget().YValueChanged.connect(self.onYvalueChanged)

        self.main_widget().kValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().XValueChanged.connect(self.proc_stack_parallel)        
        self.main_widget().YValueChanged.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        #extract slice
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def onkValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        #print(value)
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)


    def onXvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.xx = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
    
    def onYvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.yy = value
        # #print(self.yy)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
    
    def proc_technique(self,img):
        # debug
        # #print(self.xx)
        # #print(self.yy)
        return cv2.bilateralFilter(
            src=img,
            d=self.kk,
            sigmaColor=self.xx,
            sigmaSpace=self.yy,
        )
    
      # #use when save and close
    def get_state(self) -> dict:
        return {
            'ksize': self.kk,
            'sigmaX': self.xx,
            'sigmaY': self.yy,

        }

    def set_state(self, data: dict, version):
        self.kk = data['ksize']
        self.xx = data['sigmaX']
        self.yy = data['sigmaY']
    
    

#/////// Errosion, Dilation 

class Dilation(NodeBase4):        #Nodebase just a different colour
    title = 'Dilate'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Dilate_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added 
                channels_dict = Signal(dict) #store colour channels          
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        default2 = 1
        self.value_1 = default1  #threshold
        self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged1)
        self.main_widget().Value1Changed.connect(self.proc_stack_parallel)
        self.main_widget().Value2Changed.connect(self.ValueChanged2)
        self.main_widget().Value2Changed.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
             self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ValueChanged1(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    def ValueChanged2(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_2 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    def proc_technique(self,img):
        return cv2.dilate(
            src=img,
            kernel=np.ones((self.value_1,self.value_1),np.uint8),
            iterations=self.value_2 
        )
    
    def get_state(self) -> dict:
        return {
            'val1': self.value_1,
            'val2': self.value_2,
        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']
        self.value_2 = data['val2']


class Erosion(NodeBase4):        #Nodebase just a different colour
    title = 'Erosion'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Dilate_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) #store colour channels  
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        default2 = 1
        self.value_1 = default1  #threshold
        self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged1)
        self.main_widget().Value1Changed.connect(self.proc_stack_parallel)
        self.main_widget().Value2Changed.connect(self.ValueChanged2)
        self.main_widget().Value2Changed.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
             self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)

        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ValueChanged1(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    def ValueChanged2(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_2 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    def proc_technique(self,img):
        return cv2.erode(
            src=img,
            kernel=np.ones((self.value_1,self.value_1),np.uint8),
            iterations=self.value_2 
        )
    
    def get_state(self) -> dict:
        return {
            'val1': self.value_1,
            'val2': self.value_2,
        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']
        self.value_2 = data['val2']
#//////////////////////////////
# Morphological Transformations

class Morphological_Base(NodeBase4):        #Nodebase just a different colour
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Morphological_MainWidget
    main_widget_pos = 'below ports'

    morph_type = None

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) #store colour channels     
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 2
        self.value_1 = default1  #threshold

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged)
        self.main_widget().Value1Changed.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
             self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)

        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        
        self.proc_stack_parallel()

        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def proc_technique(self,img):
        return cv2.morphologyEx(
            src=img,
            op=self.morph_type,
            kernel=np.ones((self.value_1,self.value_1),np.uint8),
        )
    def get_state(self) -> dict:
        return {
            'val1': self.value_1

        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']
    
class Morph_Gradient(Morphological_Base):
    title = 'Morphological Gradient'
    morph_type = cv2.MORPH_GRADIENT 
    

class Opening(Morphological_Base):
    title = 'Opening (Morph)'
    morph_type = cv2.MORPH_OPEN

class Closing(Morphological_Base):
    title = 'Closing (Morph)'
    morph_type = cv2.MORPH_CLOSE 

class TopHat(Morphological_Base):
    title = 'Top Hat (Morph)'
    morph_type = cv2.MORPH_TOPHAT 

class BlackHat(Morphological_Base):
    title = 'Black Hat (Morph)'
    morph_type = cv2.MORPH_BLACKHAT 

#-----------------------------------------------------------------------------
# Thresholding
# Manual
class Threshold_Manual_Base(NodeBase2):        #Nodebase just a different colour
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Threshold_Manual_MainWidget
    main_widget_pos = 'below ports'

    thresh_type = None

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) #store colour channels     
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 100
        default2 = 255
        self.thr = default1  #threshold
        self.mv = default2   #maxvalue

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().threshValueChanged.connect(self.ontValueChanged)
        self.main_widget().threshValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().mvValueChanged.connect(self.onMvvalueChanged)   
        self.main_widget().mvValueChanged.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)

        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        print("current binary value", self.thr)
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ontValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.thr = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

    def onMvvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.mv = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

    def proc_technique(self,img):
        if img.shape[-1] == 1:
                # Grayscale image
                img_gray = img
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print(f"SHAPE OF PROC {img.shape}")    
        ret, result = cv2.threshold(
            src=img_gray,
            thresh=self.thr,
            maxval=self.mv,
            type=self.thresh_type,
        )
        img_rehsape = result[:, :, np.newaxis]
        return img_rehsape    
    
    def get_state(self) -> dict:
        return {
            'val1': self.thr,
            'val2': self.mv,
        }

    def set_state(self, data: dict, version):
        self.thr = data['val1']
        self.mv = data['val2']

# Local 
class Threshold_Local_Base(NodeBase2):        #Nodebase just a different colour
    # title = 'Threshold-Adaptive Mean'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Threshold_Local_MainWidget
    main_widget_pos = 'below ports'

    adaptive_type = None

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) 

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 11
        default2 = 2
        self.thr = default1  #threshold
        self.mv = default2   #maxvalue

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().threshValueChanged.connect(self.ontValueChanged)
        self.main_widget().threshValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().mvValueChanged.connect(self.onMvvalueChanged)   
        self.main_widget().mvValueChanged.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
        print("current threshold value local", self.thr)
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ontValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.thr = value
        #print(self.thr)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

    def onMvvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.mv = value
        #print(self.mv)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)


    def get_state(self) -> dict:
        return {
            'val1': self.thr,
            'val2': self.mv,
        }

    def set_state(self, data: dict, version):
        self.thr = data['val1']
        self.mv = data['val2']

class ThresholdAdaptiveMean(Threshold_Local_Base):
    title = 'Threshold-Adaptive Mean'
    adaptive_type=cv2.ADAPTIVE_THRESH_MEAN_C,

    def proc_technique(self,img):
            if img.shape[-1] == 1:
                    # Grayscale image
                    img_gray = img
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(f"SHAPE OF PROC {img.shape}")    
            
            # result = cv2.adaptiveThreshold(
            #     src=img_gray,
            #     maxValue=255,
            #     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            #     thresholdType=cv2.THRESH_BINARY_INV,  # Inverted threshold
            #     blockSize=self.thr,
            #     C=self.mv,
            # )
            # img_rehsape = result[:, :, np.newaxis]
            # return img_rehsape 

            result = cv2.adaptiveThreshold(
                src=img_gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,   #self.adaptive_type,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=self.thr,
                C=-self.mv, #self.mv
                
            )
            img_rehsape = result[:, :, np.newaxis]
            return img_rehsape 

class ThresholdAdaptiveGaussian(Threshold_Local_Base):
    title = 'Threshold-Adaptive Gaussian'
    adaptive_type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

    def proc_technique(self,img):
            if img.shape[-1] == 1:
                    # Grayscale image
                    img_gray = img
            else:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print(f"SHAPE OF PROC {img.shape}")    
            result = cv2.adaptiveThreshold(
                src=img_gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,   #self.adaptive_type,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=self.thr,
                C=-self.mv,
            )
            img_rehsape = result[:, :, np.newaxis]
            return img_rehsape 

#global
class Global_Thresholding(NodeBase2):
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Global_MainWidget
    main_widget_pos = 'below ports'

    thresh_type = None

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict)
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def proc_technique(self,img):
        if img.shape[-1] == 1:
                # Grayscale image
                img_gray = img
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # #print(f"SHAPE OF PROC {img.shape}")    
        ret, result = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + self.thresh_type)
        img_rehsape = result[:, :, np.newaxis]
        return img_rehsape    
    

class ThresholdBinary(Threshold_Manual_Base):
    title = 'Binary Threshold'
    thresh_type = cv2.THRESH_BINARY

class ThresholdOtsu(Global_Thresholding):
    title = 'Otsu Binarization'  
    thresh_type = cv2.THRESH_OTSU

class ThresholdTriangle(Global_Thresholding):
    title = 'Threshold Triangle'
    thresh_type = cv2.THRESH_TRIANGLE


# --------------------------------------------------
# Contrsat Enhancement

class AlphaNode(NodeBase3):        #Nodebase just a different colour
    title = 'Contrast Stretching'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Alpha_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added  
                channels_dict = Signal(dict)    
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 3
        default2 = 1
        self.value_1 = default1  #threshold
        self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged1)
        self.main_widget().Value1Changed.connect(self.proc_stack_parallel)
        self.main_widget().Value2Changed.connect(self.ValueChanged2)
        self.main_widget().Value2Changed.connect(self.proc_stack_parallel)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ValueChanged1(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    def ValueChanged2(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_2 = value
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      

    def proc_technique(self,img):
        return cv2.convertScaleAbs(
            src=img,
            alpha=self.value_1,
            beta=self.value_2 
        )
    
    def get_state(self) -> dict:
        return {
            'val1': self.value_1,
            'val2': self.value_2,
        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']
        self.value_2 = data['val2']

class GammaNode(NodeBase3): 
    title = 'Gamma Correction'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Gamma_Corr_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added
                channels_dict = Signal(dict)      
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 0.5
        # default2 = 1
        self.value_1 = default1  #threshold
        # self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged1)
        self.main_widget().Value1Changed.connect(self.gamma_correction)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.gamma_correction()
        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ValueChanged1(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
    
    def gamma_correction(self):
        gamma = self.value_1

        # Apply gamma correction
        gamma_corrected = np.power(self.image_stack/ 255.0, gamma) * 255.0
        self.reshaped_proc_data = np.uint8(gamma_corrected)
        self.sliced = self.reshaped_proc_data[self.z_sclice,:,:]
        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
    
  
    def get_state(self) -> dict:
        return {
            'val1': self.value_1,
        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']


class Histogram(NodeBase3):
    title = 'Histogram'
    init_inputs = [
        NodeInputBP('img'),
    ]

    init_outputs = [
        NodeOutputBP()
    ]
    main_widget_class = widgets.HisogramWidg
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
                logScale = Signal(object)
                clear_graph = Signal(bool)
                channels_dict = Signal(dict)
                channels_dict = Signal(dict)


            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_histogram)
        self.SIGNALS.logScale.connect(self.main_widget().log_hist)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.SIGNALS.clear_graph.connect(self.main_widget().clear_hist)
        self.main_widget().displayHist.connect(self.emitImage)
        self.main_widget().LogHist.connect(self.logScale)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)

        # try:
        #     self.SIGNALS.new_img.emit(CVImage(self.z_sclice).img)
        # except:  # there might not be an image ready yet
        #     pass

    def update_event(self, inp=-1):                                         #reset
        #extract from stack
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        # histo_s
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.new_img_wrp = CVImage(self.sliced)
        if self.session.gui:
            self.SIGNALS.clear_graph.emit(True)

        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
        
    def emitImage(self):
        if self.session.gui:
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    
    def logScale(self, state):
        #print(f'state: {state}')
        if self.session.gui:
            if state == True:
                self.SIGNALS.logScale.emit(self.new_img_wrp.img)
            elif state == False:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        

    
    # def get_img(self):
    #     return None
    
# Analysis Nodes -------------------------------------------------------
class Overlap_analysis(Node):
    title = 'Overlap analysis'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('binarized channel (1)'),
        NodeInputBP('binarized channel (2)'),
        # NodeInputBP(''),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img
    ]
    main_widget_class = widgets.Global_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added     
                channels_dict = Signal(dict) 
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        # default1 = 3
        # default2 = 1
        # self.value_1 = default1  #threshold
        # self.value_2 = default2

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.proc_technique()
        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        
        
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def handle_stack(self):
        self.chan_0 = self.input(0)[0] # eg autophasgasomes
        self.chan_1 = self.input(1)[0] # eg lysosomes

        self.stack_dict = self.input(0)[1] #dictioary
        self.z_sclice = self.input(0)[2]
        # self.squeeze = np.squeeze(self.image_stack)
        self.z_size = self.chan_0.shape[0]
        print(f"z_size {self.z_size}")
        # self.sliced = self.image_stack[self.z_sclice, :, :, :] #Z, H, W, C
              
    def proc_technique(self):
        normalized_chan_0 = self.chan_0 / 255
        normalized_chan_1 = self.chan_1 / 255
        print("Data type of self.chan_0:", self.chan_0.dtype)
        print("Data type of self.chan_1:", self.chan_1.dtype)
        overlap = np.multiply(normalized_chan_0, normalized_chan_1)
        overlap = (overlap * 255).astype(np.uint8)
        print("Data type of overlap:", overlap.dtype)
        print("shape overlap", overlap.shape)
        num_chan0 = np.sum(self.chan_0)
        num_chan1 = np.sum(self.chan_1)
        num_overlap_pixels = np.sum(overlap)
        print(f"number of pixels chan0 {num_chan0} \nchan1 {num_chan1} \noverlap {num_overlap_pixels}")
        # if 3 inputs: 
        # self.rgb_image = np.concatenate((self.chan_0, self.chan_1, self.chan_2), axis=-1)
        # if 4 inputs: 
        # still to implement ----------
        self.sliced = overlap[self.z_sclice, :, :, :] #Z, H, W, C
        print(f"z_slice number of pixels {np.sum(self.sliced)}")
        uniq = np.unique(self.chan_0)
        print(f"unique vlaues chan0 {uniq} \noverlap {np.unique(overlap)} ")
        print("channel 0 - slice", self.chan_0[self.z_sclice, :, :, :])
        self.set_output_val(0, (overlap, self.stack_dict, self.z_sclice))
        

# APPLY TO WHOLE STACK AT ONCE - dont use pipelin 
# Use 3d guaus
class Volume_filter(NodeBase):        #Nodebase just a different colour
    title = 'Volume Filter'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Volume_Filter
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) #store colour channels     
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        self.value_1 = default1  #threshold

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged)
        self.main_widget().Value1Changed.connect(self.proc_technique)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)

        self.proc_technique()

        # self.new_img_wrp = CVImage(self.get_img(self.sliced))
        # if self.prev == True:
        #     if self.session.gui:
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        
        
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def ValueChanged(self, value):

        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.value_1 = value
    
    def proc_technique(self):
        # single channel (single colour)
        # Z Y X C
        # z y x 1
        #so technically 3D
        binarisedImageStack=self.image_stack
        # #print(f"proc input stack: {stack4D.shape}")
         #Ensure only on 3D data
         # CHECK BINARIAZIED, Add warning if not
        if binarisedImageStack.shape[0] > 1:
            # Apply the volume filter to the entire 4D array
            # print(f"Performing volume filter across stack: {self.xx}, y:{self.yy}, z:{self.kk}")
            print(f"stack4D shape: {binarisedImageStack.shape}")

            labeled, numpatches = label(binarisedImageStack)
            # since labels start from 1 use this range
            sizes = sum(binarisedImageStack/np.max(binarisedImageStack), labeled, range(1, numpatches + 1))
            # print(np.sort(sizes.astype('uint32')))

            # to ensure "black background" is excluded add 1, and labels only start from 1
            filteredIndexes = np.where(sizes >= self.value_1)[0] + 1

            filteredBinaryIndexes = np.zeros(numpatches + 1, np.uint8)
            filteredBinaryIndexes[filteredIndexes] = 1
            filteredBinary = filteredBinaryIndexes[labeled]

            labeledStack, numLabels = label(filteredBinary)
            print("Initial num labels: {}, num lables after filter: {}".format(numpatches, numLabels))
            
            # Convert filteredBinary to 0 and 255 for displaying grayscale volume_filtered stack
            labeledStack = (labeledStack * 255).astype(np.uint8)
            print(self.value_1)


            # filtered_image = gaussian_filter(stack4D, sigma=(self.kk, self.yy,self.xx, 0))
            # # #print(self.kk)
            # # prevent the use of for loops, but no sigma applied to channels
            # #print(f"filtered_image {filtered_image.shape}")
            # # update displayed image
            self.sliced = labeledStack[self.z_sclice, :, :, :]
            self.reshaped_proc_data= labeledStack
            
            self.new_img_wrp = CVImage(self.sliced)
            if self.prev == True:
                if self.session.gui:
                    #update continuously 
                    self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            else:
                self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

            self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
                
    def get_state(self) -> dict:
        return {
            'val1': self.value_1

        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']
           
class Fill_holes(NodeBase):        #Nodebase just a different colour
    title = 'Fill Holes'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Fill_Holes_MainWidget
    main_widget_pos = 'below ports'


    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
                channels_dict = Signal(dict) #store colour channels   
                clr_fill_holes = Signal(bool) #clear fill holes checkbox  
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        self.value_1 = default1  #threshold

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.channels_dict.connect(self.main_widget().channels)
        self.SIGNALS.clr_fill_holes.connect(self.main_widget().clear_fill)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().fill_holes.connect(self.fill_holes_activate)
        
        try:
             self.new_img_wrp = CVImage(self.get_img(self.sliced))
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            #  self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.handle_stack()
        self.SIGNALS.channels_dict.emit(self.stack_dict)
        self.SIGNALS.clr_fill_holes.emit(False)
        unique_values = np.unique(self.image_stack)
        print(f"unique values: {unique_values}")
        
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp.img) 

    def fill_holes_activate(self, state):
        if state == True:
            self.proc_technique()

    
    def proc_technique(self):
        
        # single channel (single colour)
        # Z Y X C
        # z y x 1
        #so technically 3D
        unique_values = np.unique(self.image_stack)
        print(f"FILL HOLES unique values: {unique_values}")
        binarisedImageStack=self.image_stack
        # CHECK BINARIAZIED, Add warning if not

        # Will work for Z == 1 or Z > 1 (3D or 2D)
        print(f"stack4D shape: {binarisedImageStack.shape}")
        print("fill holes")
        # Need to send a #3D array through, therefore remove the redundant channel 
        # dimension and add back later to satisfy system standardization (ZYX -> ZYX -> ZYXC)
        # Same idea as "squeeze", performed else where
        filled_holes_stack = binary_fill_holes(binarisedImageStack[:,:,:,0]).astype(int)
        unique_values = np.unique(filled_holes_stack)
        print(f"binary_fill_holes unique values: {unique_values}")
        filled_holes_stack = (filled_holes_stack* 255).astype(np.uint8)
        unique_values = np.unique(filled_holes_stack)
        print(f"after *255 unique values: {unique_values}")
        filled_holes_stack = filled_holes_stack[:,:,:,np.newaxis]
    
        self.sliced = filled_holes_stack[self.z_sclice, :, :, :]
        self.reshaped_proc_data= filled_holes_stack
        
        self.new_img_wrp = CVImage(self.sliced)
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

        self.set_output_val(0, (self.reshaped_proc_data, self.stack_dict, self.z_sclice))
            
    
    def get_state(self) -> dict:
        return {
            'val1': self.value_1

        }

    def set_state(self, data: dict, version):
        self.value_1 = data['val1']


nodes = [
    Checkpoint_Node,
    # Button_Node,
    # Print_Node,
    # Log_Node,
    # Clock_Node,
    # Slider_Node,
    # Exec_Node,
    # Eval_Node,
    # Storage_Node,
    # LinkIN_Node,
    # LinkOUT_Node,
    # Interpreter_Node,

    # Pipeline Nodes
    ReadImage,
    SaveImg,
    Morphological_Props,
    BatchProcess,
    DisplayImg,
    Crop,
    # Contrast Enhancemnet 
    Histogram,
    AlphaNode,
    GammaNode,
    # Filtering 
    Split_Img,
    Merge_Img,
    Blur_Averaging,
    Median_Blur,
    Gaussian_Blur,
    Gaussian_Blur3D,
    Bilateral_Filtering,
    # Binatization 
    ThresholdBinary,
    ThresholdAdaptiveMean,
    ThresholdAdaptiveGaussian,
    ThresholdOtsu,
    ThresholdTriangle,
    # Post Binatization 
    Dilation,
    Erosion,
    Opening,
    Closing,
    TopHat,
    Morph_Gradient,
    BlackHat,
    Overlap_analysis,
    Volume_filter,    
    Fill_holes,
]
