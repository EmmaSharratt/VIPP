import code
from contextlib import redirect_stdout, redirect_stderr

from ryven.NENV import *
widgets = import_widgets(__file__)

import cv2
import numpy as np
import os
import tifffile as tiff 
from scipy.ndimage import gaussian_filter, label
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
        self.frame = self.input(0)[1] #dont actually need this anymore, but keep incase. Good to know wich time step
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
        self.frame = self.input(0)[1] #dont actually need this anymore, but keep incase. Good to know wich time step
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
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))

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
        self.frame = self.input(0)[1] #dont actually need this anymore, but keep incase. Good to know wich time step
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
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))

class NodeBase4(NodePipeline):
    version = 'v0.1'
    color = '#C55A11' #red - post binarization

    

class NodeBase3D(Node):
    color = '#FFCA00' #yellow - Filtering 

    def handle_stack(self):
        self.image_stack = self.input(0)[0]
        self.frame = self.input(0)[1] #dont actually need this anymore, but keep incase. Good to know wich time step
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
        processed_slice = processed_data.reshape(zslice.shape)
        
        return processed_slice
    
    #signle time step
    def proc_stack_parallel(self):
        
        # Define the number of worker threads or processes
        num_workers = 6  # Adjust as needed

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            #print(f"z size {self.z_size}")
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
      #print(f"reshaped_proc_data shape: {self.reshaped_proc_data.shape}")
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))

class DualNodeBase(NodeBase):
    """For nodes that can be active and passive"""

    version = 'v0.1'

    def __init__(self, params, active=True):
        super().__init__(params)

        self.active = active
        if active:
            self.actions['make passive'] = {'method': self.make_passive}
        else:
            self.actions['make active'] = {'method': self.make_active}

    def make_passive(self):
        del self.actions['make passive']

        self.delete_input(0)
        self.delete_output(0)
        self.active = False

        self.actions['make active'] = {'method': self.make_active}

    def make_active(self):
        del self.actions['make active']

        self.create_input(type_='exec', insert=0)
        self.create_output(type_='exec', insert=0)
        self.active = True

        self.actions['make passive'] = {'method': self.make_passive}

    def get_state(self) -> dict:
        return {
            'active': self.active
        }

    def set_state(self, data: dict, version):
        self.active = data['active']


# -------------------------------------------


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

    def get_state(self) -> dict:
        return {
            'active': self.active,
            'num outputs': len(self.outputs),
        }

    def set_state(self, data: dict, version):
        self.actions['remove output'] = {
            {'method': self.remove_output, 'data': i}
            for i in range(data['num outputs'])
        }

        if data['active']:
            self.make_active()


# class Button_Node(NodeBase):
#     title = 'Button'
#     version = 'v0.1'
#     main_widget_class = widgets.ButtonNode_MainWidget
#     main_widget_pos = 'between ports'
#     init_inputs = [

#     ]
#     init_outputs = [
#         NodeOutputBP(type_='exec')
#     ]
#     color = '#99dd55'

#     def update_event(self, inp=-1):
#         self.exec_output(0)


# class Print_Node(DualNodeBase):
#     title = 'Print'
#     version = 'v0.1'
#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP(dtype=dtypes.Data(size='m')),
#     ]
#     init_outputs = [
#         NodeOutputBP(type_='exec'),
#     ]
#     color = '#5d95de'

#     def __init__(self, params):
#         super().__init__(params, active=True)

    # def update_event(self, inp=-1):
        # if self.active and inp == 0:
          #print(self.input(1))
        # elif not self.active:
          #print(self.input(0))


# import logging


# class Log_Node(DualNodeBase):
#     title = 'Log'
#     version = 'v0.1'
#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('msg', type_='data'),
#     ]
#     init_outputs = [
#         NodeOutputBP(type_='exec'),
#     ]
#     main_widget_class = widgets.LogNode_MainWidget
#     main_widget_pos = 'below ports'
#     color = '#5d95de'

#     def __init__(self, params):
#         super().__init__(params, active=True)

#         self.logger = self.new_logger('Log Node')

#         self.targets = {
#             **self.script.logs_manager.default_loggers,
#             'own': self.logger,
#         }
#         self.target = 'global'

#     def update_event(self, inp=-1):
#         if self.active and inp == 0:
#             i = 1
#         elif not self.active:
#             i = 0
#         else:
#             return

#         msg = self.input(i)

#         self.targets[self.target].log(logging.INFO, msg=msg)

#     def get_state(self) -> dict:
#         return {
#             **super().get_state(),
#             'target': self.target,
#         }

#     def set_state(self, data: dict, version):
#         super().set_state(data, version)
#         self.target = data['target']
#         if self.session.gui and self.main_widget():
#             self.main_widget().set_target(self.target)


# class Clock_Node(NodeBase):
#     title = 'clock'
#     version = 'v0.1'
#     init_inputs = [
#         NodeInputBP(dtype=dtypes.Float(default=0.1), label='delay'),
#         NodeInputBP(dtype=dtypes.Integer(default=-1, bounds=(-1, 1000)), label='iterations'),
#     ]
#     init_outputs = [
#         NodeOutputBP(type_='exec')
#     ]
#     color = '#5d95de'
#     main_widget_class = widgets.ClockNode_MainWidget
#     main_widget_pos = 'below ports'

#     def __init__(self, params):
#         super().__init__(params)

#         self.actions['start'] = {'method': self.start}
#         self.actions['stop'] = {'method': self.stop}

#         if self.session.gui:

#             from qtpy.QtCore import QTimer
#             self.timer = QTimer(self)
#             self.timer.timeout.connect(self.timeouted)
#             self.iteration = 0


#     def timeouted(self):
#         self.exec_output(0)
#         self.iteration += 1
#         if -1 < self.input(1) <= self.iteration:
#             self.stop()

#     def start(self):
#         if self.session.gui:
#             self.timer.setInterval(self.input(0)*1000)
#             self.timer.start()
#         else:
#             import time
#             for i in range(self.input(1)):
#                 self.exec_output(0)
#                 time.sleep(self.input(0))

#     def stop(self):
#         self.iteration = 0
#         if self.session.gui:
#             self.timer.stop()

#     def toggle(self):
#         # triggered from main widget
#         if self.session.gui:
#             if self.timer.isActive():
#                 self.stop()
#             else:
#                 self.start()

#     def update_event(self, inp=-1):
#         if self.session.gui:
#             self.timer.setInterval(self.input(0)*1000)

#     def remove_event(self):
#         self.stop()


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

    title = 'link IN'
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

    title = 'link OUT'
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

class Slider_Gaus_Old(OpenCVNodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP(dtype=dtypes.Integer(default=1), label='min'),
        NodeInputBP(dtype=dtypes.Integer(default=1), label='max'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='round'),
        NodeInputBP('img'),
    ]
    init_outputs = [
        NodeOutputBP(),
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

        range = self.input(1)-self.input(0)
        self.v = self.input(0) + (range * self.val)
        if self.input(2):
            self.v = round(self.v)

        self.set_output_val(0, self.v)

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }
    
    # def get_img(self):
    #     return cv2.medianBlur(src=self.input(3).img, ksize=self.v)

    def set_state(self, data: dict, version):
        self.val = data['val']

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


class Slider_Gaus(NodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus'
    version = 'v0.2'
    init_inputs = [
        
        NodeInputBP(dtype=dtypes.Data(default=1), label='kSize'),              #0
        NodeInputBP(dtype=dtypes.Boolean(default=True), label='round'),        #1
        NodeInputBP('img'),                                                    #2
    ]
    init_outputs = [
        #NodeOutputBP('img'),
        NodeOutputBP('ksize')
    ]
    main_widget_class = widgets.OpenCVNode_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        self.val = 0

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        
    def place_event(self):  #??
        self.update()

    def view_place_event(self):
        #image
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()
        

    def update_event(self, inp=-1):
        new_img_wrp = CVImage(self.get_img())

        if self.session.gui:
            self.SIGNALS.new_img.emit(new_img_wrp.img)

        # self.set_output_val(0, new_img_wrp)
        #self.set_output_val(0, new_img_wrp)

        if self.input(0)<=10:
            range = self.input(0)*2
            self.v = 0.1 + (range * self.val)
        else:
            range = 20                                                          #add if statements to ensure positive
            self.v = (self.input(0)-(range/2)) + (range * self.val)
        if self.input(1):
            self.v = round(self.v)

        self.set_output_val(0, self.v)


    def get_state(self) -> dict:
        return {
            'val': self.val,
        }
    
   
    def set_state(self, data: dict, version):
        self.val = data['val']

    def get_img(self):
        return cv2.medianBlur(src=self.input(2), ksize=self.self.input(0)) #self.v
    
#Slider and image working -------------------------------------
   
class Slider_Gaus_Tick_v2(OpenCVNodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v2'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP(dtype=dtypes.Integer(default=1), label='min'),
        NodeInputBP(dtype=dtypes.Integer(default=1), label='max'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='round'),
        NodeInputBP('img'),
    ]
    init_outputs = [
        NodeOutputBP(),
        NodeOutputBP(),
    ]
    main_widget_class = widgets.QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()


        self.val = 0

    def place_event(self):  #??
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
        new_img_wrp = CVImage(self.get_img())

        if self.session.gui:
            self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(1, new_img_wrp)

        #slider
        range = self.input(1)-self.input(0)
        self.v = self.input(0) + (range * self.val)
        if self.input(2):
            self.v = round(self.v)

        self.set_output_val(0, self.v)

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }
    
    # def get_img(self):
    #     return cv2.medianBlur(src=self.input(3).img, ksize=self.v)

    def set_state(self, data: dict, version):
        self.val = data['val']

    def get_img(self):
        return self.input(3).img
    
#-------------Add blur functionality 
class Slider_Gaus_Tick_v3(OpenCVNodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v3'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP(dtype=dtypes.Integer(default=1), label='min'),
        NodeInputBP(dtype=dtypes.Integer(default=1), label='max'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='round'),
        NodeInputBP('img'),
    ]
    init_outputs = [
        NodeOutputBP(),
        NodeOutputBP(),
    ]
    main_widget_class = widgets.QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()


        self.val = 0

    def place_event(self):  #??
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
         #slider
        range1 = self.input(1)-self.input(0)
        self.v = self.input(0) + (range1 * self.val)
        if self.input(2):
            self.v = round(self.v)

        self.set_output_val(0, self.v)
        # print(self.v)

        #image
        new_img_wrp = CVImage(self.get_img())

        if self.session.gui:
            self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(1, new_img_wrp)

#good :))
    
#-------------change inputs and layout 
# - correctly updated code to only one input - ksize
# - further improvement: current ksize (self.v) printed to widget
#                        Preview still not working 
class Slider_Gaus_Tick_v4(NodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v4'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),      
    ]
    init_outputs = [
        NodeOutputBP(),
        NodeOutputBP(),
    ]
    main_widget_class = widgets.QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()


        self.val = 0

    def place_event(self):  #??
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
         #slider
        if self.input(1)<=10:
            range1 = self.input(1)*2
            self.v = 0.1 + (range1 * self.val)
        else:
            range1 = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
        self.v=int(self.v)
        self.set_output_val(0, self.v)

        #image
        new_img_wrp = CVImage(self.get_img())
        if self.input(2):   
            if self.session.gui:
                self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(1, new_img_wrp)
        self.main_widget().text_label.setText((f"current ksize: {self.v}"))
       

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }

    def set_state(self, data: dict, version):
        self.val = data['val']

    def get_img(self):
        # return self.input(0).img
        k = int(self.v)
        # print(k)
        return cv2.blur(
            src=self.input(0).img,
            ksize=(k,k),
                )
    
class Slider_Gaus_Tick_v5(NodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v5'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),      
    ]
    init_outputs = [
        NodeOutputBP(),
        NodeOutputBP(),
    ]
    main_widget_class = widgets.V2QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()


        self.val = 0

    def place_event(self):  #??
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
         #slider
        if self.input(1)<=10:
            range1 = self.input(1)*2
            self.v = 0.1 + (range1 * self.val)
        else:
            range1 = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
        self.v=int(self.v)
        self.set_output_val(0, self.v)

        #image
        new_img_wrp = CVImage(self.get_img())
        if self.input(2):   
            if self.session.gui:
                self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(1, new_img_wrp)
        self.main_widget().text_label.setText((f"current ksize: {self.v}"))
       

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }

    def set_state(self, data: dict, version):
        self.val = data['val']

    def get_img(self):
        # return self.input(0).img
        k = int(self.v)
        # print(k)
        return cv2.blur(
            src=self.input(0).img,
            ksize=(k,k),
                )
    
#Add preview from prototype
class Slider_Gaus_Tick_v6(NodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v6'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),      
    ]
    init_outputs = [
        NodeOutputBP(),
        NodeOutputBP(),
    ]
    main_widget_class = widgets.V3QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
                clr_img = Signal(object)
                # preview_input = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()


        self.val = 0
        

    def place_event(self):  
        self.update()
      #print("place")
        self.previous_checkbox = self.input(2)

    def view_place_event(self):
        #Signals
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        # self.SIGNALS.preview_input.connect(self.main_widget().preview_input_changed)
        #Preview
        

      #print("view_place and connections")
        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
        #debugging
      #print("update event")
        
        # if self.input(2) != self.previous_checkbox:
        #   #print("Checkbox not checked")
        

         #slider
        if self.input(1)<=10:
            range1 = self.input(1)*2
            self.v = 0.1 + (range1 * self.val)
        else:
            range1 = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
        self.v=int(self.v)
        self.set_output_val(0, self.v)
        

      #print(f"Old:{self.previous_checkbox }")
        
        #image
        new_img_wrp = CVImage(self.get_img())
        if self.input(2):   
            # if self.session.gui:
                self.SIGNALS.new_img.emit(new_img_wrp.img)
              #print("The input is checked", self.input(2))
        else:
            # if self.session.gui:
            # if self.input(2) != self.previous_checkbox:
              #print("Clear image now and reshape")
                self.SIGNALS.clr_img.emit(new_img_wrp.img)

        # Preview
        # if self.input(2) != self.previous_checkbox:
          #print("Checkbox just checked")
            # self.SIGNALS.preview_input.emit(new_img_wrp.img)
            # self.main_widget().update_shape()
        
        self.previous_checkbox = self.input(2)
      #print(f"New:{self.previous_checkbox }")

        self.set_output_val(1, new_img_wrp)
        self.main_widget().text_label.setText((f"current ksize: {self.v}"))
       

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }

    def set_state(self, data: dict, version):
        self.val = data['val']

    def get_img(self):
        # return self.input(0).img
        k = int(self.v)
        # print(k)
        return cv2.blur(
            src=self.input(0).img,
            ksize=(k,k),
                )
    

#Add preview from prototype
class Slider_Gaus_Tick_v7(NodeBase):        #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v7'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        NodeInputBP(dtype=dtypes.Integer(default=5), label='SigmaX'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),      
    ]
    init_outputs = [
        NodeOutputBP(),
        NodeOutputBP(),
        NodeOutputBP(), #sigX

    ]
    main_widget_class = widgets.V5QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
                clr_img = Signal(object)
                # preview_input = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()


        self.kval = 0
        self.sigX = 0
        # self.inputs[1] = 10
        

    def place_event(self):  
        self.update()
      #print("place")
        self.previous_checkbox = self.input(3)

    def view_place_event(self):
        #Signals
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        # self.SIGNALS.preview_input.connect(self.main_widget().preview_input_changed)
        #Preview
        

      #print("view_place and connections")
        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
        # # debugging
        # print("update event")
        
        # # if self.input(2) != self.previous_checkbox:
        # #     print("Checkbox not checked")
        
        # print("val VALUE:", self.val)
        #  #slider ksize
        # if self.input(1)<=10:
        #     range1 = self.input(1)*2
        #     self.v = 0.1 + (range1 * self.val)
        # else:
        #     range1 = 20                                                          #add if statements to ensure positive
        #     self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
        # self.v=int(self.v)
        # #Expression cannot be assignment target
        # self.set_output_val(0, self.v)
        # # self.inputs[1].dtype=dtypes.Integer(default=self.v)

        
        #  #slider sigmaX
        # if self.input(2)<=10:
        #     range1 = self.input(2)*2
        #     self.sX = 0.1 + (range1 * self.sigX)
        # else:
        #     range1 = 20                                                          #add if statements to ensure positive
        #     self.sX = (self.input(2)-(range1/2)) + (range1 * self.sigX)
        # # self.sX=int(self.sX)
        # #Expression cannot be assignment target
        # self.set_output_val(2, self.sX)
        
        

        # print(f"Old:{self.previous_checkbox }")
        
        #image
        new_img_wrp = CVImage(self.get_img())
        if self.input(3):   
            # if self.session.gui:
                self.SIGNALS.new_img.emit(new_img_wrp.img)
               
                
        else:
            # if self.session.gui:
            # if self.input(2) != self.previous_checkbox:
              #print("Clear image now and reshape")
                self.SIGNALS.clr_img.emit(new_img_wrp.img)

        # Preview
        # if self.input(3) != self.previous_checkbox:
          #print("Checkbox just checked")
            # self.SIGNALS.preview_input.emit(new_img_wrp.img)
            # self.main_widget().update_shape()
        
        # self.previous_checkbox = self.input(3)
      #print(f"New:{self.previous_checkbox }")

        self.set_output_val(1, new_img_wrp)
        self.main_widget().text_label.setText((f"current ksize: {self.v}"))
       

    def get_state(self) -> dict:
        return {
            'kval': self.kval,
        }

    def set_state(self, data: dict, version):
        self.kval = data['kval']

    def get_img(self):
        # return self.input(0).img
        k = int(self.kval)
        # print(k)
        return cv2.blur(
            src=self.input(0).img,
            ksize=(k,k),
                )

class Slider_Gaus_Tick_v8(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'slider_Gaus_Tick_v8'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('img'),
         
    ]
    init_outputs = [
        NodeOutputBP(), #img

    ]
    main_widget_class = widgets.V5QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added
                # moved = Signal(object)    #added
                # preview_input = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

            #instance of main widget
            
            #slider signals
            # self.main_widget().ksliderValueChanged.connect(self.onSliderValueChanged)   

        self.kval = 0
        self.sigX = 0
        self.prev = True
        # self.inputs[1] = 10

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        self.main_widget().previewState.connect(self.preview)
        
        try:
            # self.SIGNALS.new_img.emit(self.update_event())
             self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #shows image as soon as connected
    def update_event(self, inp=-1):  #called when an input is changed
        self.new_img_wrp = CVImage(self.input(0).img)

        if self.session.gui:
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.set_output_val(0, self.new_img_wrp)
    
    # def updateNode(self):
    #     self.new_img_wrp = CVImage(self.get_img())

    #     if self.session.gui:
    #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)

    #     self.set_output_val(0, self.new_img_wrp)
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
      #print(value)
        self.new_img_wrp = CVImage(self.get_img(value))
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)
        
    
    def get_state(self) -> dict:
        return {
            'val': self.val,
        }

    def set_state(self, data: dict, version):
        self.val = data['val']
        

    def get_img(self,value):
        # return self.input(0).img
        # return self.input(0).img
        # k = int(self.main_widget().kval)
      #print(f"getimageValue{value}")
        return cv2.blur(
            src=self.input(0).img,
            ksize=(value,value),
                )
    
# NODES ------------------------------------------------------------------------------------------------------------------
# ReadImage Original ///////////////

# class ReadImage(NodeBase0):
#     """Reads an image from a file"""

#     title = 'Read Image'
#     input_widget_classes = {
#         'choose file IW': widgets.ChooseFileInputWidget
#     }
#     init_inputs = [
#         NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
#     ]
#     init_outputs = [
#         NodeOutputBP('img')
#     ]
#     main_widget_class = widgets.ChooseFileInputWidgetBASE3
#     main_widget_pos = 'below ports'

#     def __init__(self, params):
#         super().__init__(params)

#         if self.session.gui:
#             from qtpy.QtCore import QObject, Signal
#             class Signals(QObject):
#                 new_img = Signal(object)
#                 image_shape = Signal(list)
#                 #reset sliders
#                 reset_widget = Signal(int)
#                 #remove widgets
#                 remove_widget = Signal()

#             # to send images to main_widget in gui mode
#             self.SIGNALS = Signals()

#         self.image_filepath = ''
#         self.ttval = 0
#         self.zzval = 0

#     def view_place_event(self):
#         self.input_widget(0).path_chosen.connect(self.path_chosen)
#         self.SIGNALS.new_img.connect(self.main_widget().show_image)
#         self.SIGNALS.image_shape.connect(self.main_widget().update_widgets)
#         self.SIGNALS.reset_widget.connect(self.main_widget().reset_widg)
#         self.SIGNALS.remove_widget.connect(self.main_widget().remove_widgets)
#         self.main_widget().ValueChanged1.connect(self.onValue1Changed)
#         self.main_widget().ValueChanged2.connect(self.onValue2Changed) 
#         # try:
#         #     self.SIGNALS.new_img.emit(self.get_img())
#         # except:  # there might not be an image ready yet
#         #     pass
#         # self.main_widget_message.connect(self.main_widget().show_path)

#     def update_event(self, inp=-1):   #called when the input is changed
#         #therefore new image 
#         self.ttval = 0
#         self.zzval = 0
#         self.SIGNALS.reset_widget.emit(1)
#         # self.SIGNALS.remove_widget.emit()

#         if self.image_filepath == '':
#             return
#         # Check if the file has a .tiff extension   --> tif file capability Check tiff 
#         if self.image_filepath.endswith('.tif'):
#             try:
#                 self.image_data = tiff.imread(self.image_filepath)
#                 #generate dimension list (dimension, slices, frames (time), width, height, channels)
#                 self.id_tiff_dim(self.image_filepath)
#                 # print(self.image_data)
                
#                 #4D images 
#                 if self.dimension[0] == 5: #time and space (4D)
#                     self.image_data = ((self.image_data/self.image_data.max())*65535).astype('uint16')
#                     new_img_wrp = CVImage(self.get_img())
#                     # print("shape", new_img_wrp.shape)
#                     if self.session.gui:
#                         self.SIGNALS.new_img.emit(new_img_wrp.img)

#                     self.set_output_val(0, new_img_wrp)
                    
#                     print("Image loaded successfully")
#                 #3D images - zstack only
#                 #2D images
#                 #dimension[0]==0
#                 else:
#                     image2D = CVImage(cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED))
#                     if self.session.gui:
#                         self.SIGNALS.new_img.emit(image2D.img)
#                     self.set_output_val(0,image2D)
            
#             except Exception as e:
#                 print(e)
#                 print("failed")

#         else: 
#             try:
#                 self.set_output_val(0, CVImage(cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)))
#             except Exception as e:
#                 print(e)

#     def id_tiff_dim(self,f_path):
#         tif_file = tiff.TiffFile(f_path)

#         # Check for TIFF metadata tags
#         metadata = tif_file.pages[0].tags
#         if metadata:
#             # print("Metadata Tags:")
#             for tag_name, tag in metadata.items():
#                 print(f"{tag_name}: {tag.value}")

#             #set dimension to 0 when a new tiff file is processed
#             dimension = [0,1,1,0,0,0] #dim, slices , time
            
#             if 256 in metadata: #width
#                             # Access the tag value directly
#                             dimension[3] = metadata[256].value
            
                
#             if 257 in metadata: #height
#                             # Access the tag value directly
#                             dimension[4] = metadata[257].value
            
#             if 277 in metadata: #channels
#                             # Access the tag value directly
#                             dimension[5] = int(metadata[277].value)
#             if 259 in metadata:  # Tag for slices
#                 dimension[1] = metadata[259].value

#             if 262 in metadata:  # Tag for frames
#                 frames = metadata[262].value
            
#             if 'ImageDescription' in metadata:
#                     # Access 'ImageDescription' tag
#                     image_description = metadata['ImageDescription']
            
#                     # Split the 'ImageDescription' string into lines
#                     description_lines = image_description.value.split('\n')
#                     # Parse the lines to extract slices and frames information
#                     for line in description_lines:
#                         if line.startswith("slices="):
#                             dimension[1] = int(line.split('=')[1]) #slices
#                             dimension[0] = 3
#                         if line.startswith("frames="):
#                             dimension[2] = int(line.split('=')[1]) #frames
#                             dimension[0] += 2
#                         if 256 in metadata: #width
#                             # Access the tag value directly
#                             dimension[3] = metadata[256].value
#                         if 257 in metadata: #H
#                             # Access the tag value directly
#                             dimension[4] = metadata[257].value
#                         if 277 in metadata: #channels
#                             # Access the tag value directly
#                             dimension[5] = metadata[277].value
#         else:
#                 print("ImageDescription tag not found in metadata.")
                        
#         print(f"Slices: {dimension[1]}")
#         print(f"Frames: {dimension[2]}")
#         print(f'Dimension: {dimension[0]}')
#         print(f'Width: {dimension[3]}')
#         print(f'Height: {dimension[4]}')
#         print(f'Channels: {dimension[5]}')
#         self.dimension=dimension
#         self.SIGNALS.image_shape.emit(dimension)


#     def get_state(self):
#         data = {'image file path': self.image_filepath}
#         return data

#     def set_state(self, data, version):
#         self.path_chosen(data['image file path'])
#         # self.image_filepath = data['image file path']

#     # def get_state(self) -> dict:
#     #     return {
#     #         # 'image file path': self.image_filepath,
#     #         'val1': self.ttval, 
#     #         'val2': self.zzval,
#     #     }

#     # def set_state(self, data: dict, version):
#     #     # self.path_chosen(data['image file path'])
#     #     self.ttval = data['val1']
#     #     self.zzval = data['val2']

#     def path_chosen(self, file_path):
#         self.image_filepath = file_path
#         self.update()
    
#     def onValue1Changed(self, value):
#         print(f"timevalue{value}")
#         self.ttval=value-1 #slider: 1-max for biologists
#         self.new_img_wrp = CVImage(self.get_img())
        
#         if self.session.gui:
#             #update continuously 
#             self.SIGNALS.new_img.emit(self.new_img_wrp.img)   

#         self.set_output_val(0, self.new_img_wrp)
    
#     def onValue2Changed(self, value):
#         print(f"zvalue{value}")
#         self.zzval=value-1
#         self.new_img_wrp = CVImage(self.get_img())
        
#         if self.session.gui:
#                 #update continuously 
#             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
      
#         self.set_output_val(0, self.new_img_wrp)

#     def get_img(self):
#         return self.image_data[self.ttval,self.zzval,:,:]

# ////////////////////////////////// Orginal above

#tuple implementation 
#all timesteps
class ReadImage0(NodeBase0):
    """Reads an image from a file"""

    title = 'Read Image'
    input_widget_classes = {
        'choose file IW': widgets.ChooseFileInputWidget
    }
    init_inputs = [
        NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
    ]
    init_outputs = [
        NodeOutputBP('img')
    ]
    main_widget_class = widgets.ChooseFileInputWidgetBASE3
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
                image_shape = Signal(list)
                #reset sliders
                reset_widget = Signal(int)
                #remove widgets
                remove_widget = Signal()

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.image_filepath = ''
        self.ttval = 0
        self.zzval = 0

    def view_place_event(self):
        self.input_widget(0).path_chosen.connect(self.path_chosen)
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.image_shape.connect(self.main_widget().update_widgets)
        self.SIGNALS.reset_widget.connect(self.main_widget().reset_widg)
        self.SIGNALS.remove_widget.connect(self.main_widget().remove_widgets)
        self.main_widget().ValueChanged1.connect(self.onValue1Changed)
        self.main_widget().ValueChanged2.connect(self.onValue2Changed) 
        self.main_widget().released1.connect(self.output_data1)  
        self.main_widget().released2.connect(self.output_data1) 
        # try:
        #     self.SIGNALS.new_img.emit(self.get_img())
        # except:  # there might not be an image ready yet
        #     pass
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):   #called when the input is changed
        #therefore new image 
        self.ttval = 1
        self.zzval = 1
        self.SIGNALS.reset_widget.emit(1)
        # self.SIGNALS.remove_widget.emit()

        if self.image_filepath == '':
            return
        # Check if the file has a .tiff extension   --> tif file capability Check tiff 
        if self.image_filepath.endswith('.tif'): #----------------------------------------------------------------------------------TIFF
            try:
                self.image_data = tiff.imread(self.image_filepath)
                # Normalize - generate dimension list (T,Z,H,W,C)
                self.dim = self.id_tiff_dim(self.image_filepath)
                # Reshape - STANDARDIZED 
                self.reshaped_data = self.image_data.reshape(self.dim)
                print(f"reshaped: {self.reshaped_data.shape}")
                # Grayscale
                if self.reshaped_data.shape[4] == 1: #self.dim[4]==1:
                    self.reshaped_data = ((self.reshaped_data/self.reshaped_data.max())*255).astype('uint8')
                    # print("NORMALIZED FOR GRAYSCALE")
                
                # Display image
                # 3D, 3D or 5D
                if (self.reshaped_data.shape[0] != 1) | (self.reshaped_data.shape[1] != 1): #time and space (4D)
                    #squeeze standardized down to relevant dimension (remove 1s)
                    self.squeezed = np.squeeze(self.reshaped_data)
                    #slice
                    new_img_wrp = CVImage(self.get_img())  
                    # print("shape", new_img_wrp.shape)
                    if self.session.gui:
                        self.SIGNALS.new_img.emit(new_img_wrp.img)
                    # if multiple time steps, output 
                    self.set_output_val(0, (self.reshaped_data, self.ttval, self.zzval))

                #2D images (tiff)
                else:
                    image2D = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
                    self.reshaped_data = image2D.reshape(1, 1, *image2D.shape)
                    if self.session.gui:
                        new_img_wrp = CVImage(image2D)
                        self.SIGNALS.new_img.emit(new_img_wrp.img)
                    self.set_output_val(0,(self.reshaped_data, 1, 1))
            
            except Exception as e:
                print(e)
                print("failed")

        else: #-------------------------------------------------------------------------------------------------------------------- Not TIFF
            # 2D not tiff
            try:
                image2D = cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)
                self.reshaped_data = image2D.reshape(1, 1, *image2D.shape)
                if self.session.gui:
                    new_img_wrp = CVImage(image2D)
                    self.SIGNALS.new_img.emit(new_img_wrp.img)
                self.set_output_val(0,(self.reshaped_data, 1, 1))
            except Exception as e:
                print(e)

    def id_tiff_dim(self,f_path):
        tif_file = tiff.TiffFile(f_path)
        # Check for TIFF metadata tags
        metadata = tif_file.pages[0].tags
        if metadata:
            # print("Metadata Tags:")
            for tag_name, tag in metadata.items():
                print(f"{tag_name}: {tag.value}")

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
                            print("frames", int(line.split('=')[1]))
                            print("dim",dimension[4])
                        
        else:
                print("ImageDescription tag not found in metadata.")
                        
        # print(f'Width: {dimension[3]}')
        # print(f'Height: {dimension[2]}')
        # print(f'Channels: {dimension[4]}')
        # print(f"Slices: {dimension[1]}")
        # print(f"Frames: {dimension[0]}")
        # print(f'Dimension: {dimension[0]}')
        # self.dimension=dimension
        self.SIGNALS.image_shape.emit(dimension)
        return dimension

    def get_state(self):
        data = {'image file path': self.image_filepath}
        return data

    def set_state(self, data, version):
        self.path_chosen(data['image file path'])
        # self.image_filepath = data['image file path']

    # def get_state(self) -> dict:
    #     return {
    #         # 'image file path': self.image_filepath,
    #         'val1': self.ttval, 
    #         'val2': self.zzval,
    #     }

    # def set_state(self, data: dict, version):
    #     # self.path_chosen(data['image file path'])
    #     self.ttval = data['val1']
    #     self.zzval = data['val2']

    def path_chosen(self, file_path):
        self.image_filepath = file_path
        self.update()
    
    def onValue1Changed(self, value):
        # print(f"timevalue{value}")
        self.ttval=value-1 #slider: 1-max for biologists
        self.new_img_wrp = CVImage(self.get_img())
        
        if self.session.gui:
            #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)   
    
    def onValue2Changed(self, value):
        # print(f"zvalue{value}")
        self.zzval=value-1
        self.new_img_wrp = CVImage(self.get_img())
        
        if self.session.gui:
                #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    
    def output_data1(self, value):
        self.set_output_val(0, (self.reshaped_data, self.ttval, self.zzval))
    
    def output_data2(self, value):
        self.set_output_val(0, (self.reshaped_data, self.ttval, self.zzval))

    def get_img(self):
        # 4D
        if (self.dim[0] != 1) & (self.dim[1] != 1):
            return self.squeezed[self.ttval,self.zzval,:,:]
        # 2D in time
        elif self.dim[0] != 1:
            return self.squeezed[self.ttval,1,:,:]  #wont have all these *** (SQUEEZED) 
        # 3D (Z-stack)
        elif self.dim[1] != 1:
            return self.squeezed[1,self.zzval,:,:]

#Single timestep
# new shape (10, 512, 512, 1)
class ReadImage(NodeBase0):
    """Reads an image from a file"""

    title = 'Read Image'
    input_widget_classes = {
        'choose file IW': widgets.ChooseFileInputWidget
    }
    init_inputs = [
        NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
    ]
    init_outputs = [
        NodeOutputBP('img')
    ]
    main_widget_class = widgets.ChooseFileInputWidgetBASE3
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

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.image_filepath = ''
        self.ttval = 6
        self.zzval = 4

    def view_place_event(self):
        self.input_widget(0).path_chosen.connect(self.path_chosen)
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.image_shape.connect(self.main_widget().update_widgets)
        self.SIGNALS.reset_widget.connect(self.main_widget().reset_widg)
        self.SIGNALS.remove_widget.connect(self.main_widget().remove_widgets)
        self.main_widget().ValueChanged1.connect(self.onValue1Changed)
        self.main_widget().ValueChanged2.connect(self.onValue2Changed) 
        self.main_widget().released1.connect(self.output_data)  
        # self.main_widget().ValueChanged2.connect(self.output_data) 
        # try:
        #     self.SIGNALS.new_img.emit(self.get_img())
        # except:  # there might not be an image ready yet
        #     pass
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):   #called when the input is changed
        #therefore new image 
        # self.ttval = 0
        # self.zzval = 0
        # self.SIGNALS.reset_widget.emit(1)
        # self.SIGNALS.remove_widget.emit()

        if self.image_filepath == '':
            return
        # Check if the file has a .tiff extension   --> tif file capability Check tiff 
        if self.image_filepath.endswith('.tif'): #----------------------------------------------------------------------------------TIFF
            try:
                self.image_data = tiff.imread(self.image_filepath)

                


                # Normalize - generate dimension list (T,Z,H,W,C)
                self.dim = self.id_tiff_dim(self.image_filepath)
                # Reshape - STANDARDIZED 
                self.image_data = ((self.image_data / np.max(self.image_data)) * 255).astype(np.uint8)
                
                # IF shape: ZCXY 
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
                            red_channel = self.image_data[t, i, 0, :, :]  # Assuming the first dimension is time frame, then image index
                            green_channel = self.image_data[t, i, 1, :, :]
                            blue_channel = self.image_data[t, i, 2, :, :]
                            
                            # Stack the channels along the last axis
                            image_data_stacked[t, i, :, :, 0] = red_channel
                            image_data_stacked[t, i, :, :, 1] = green_channel
                            image_data_stacked[t, i, :, :, 2] = blue_channel    

                    print(image_data_stacked.shape)   

                    self.image_data = image_data_stacked            

                    
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
                new_img_wrp = CVImage(self.get_img())  
                    # print("shape", new_img_wrp.shape)
                if self.session.gui:
                        self.SIGNALS.new_img.emit(new_img_wrp.img)
                    # mulitple time steps
                # if self.reshaped_data.shape[0] != 1:
                self.set_output_val(0, (self.reshaped_data[self.ttval, :, :, :, :], self.ttval, self.zzval))
                # else:
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

    def id_tiff_dim(self,f_path):
        tif_file = tiff.TiffFile(f_path)
        # Check for TIFF metadata tags
        metadata = tif_file.pages[0].tags
        if metadata:
            # print("Metadata Tags:")
            for tag_name, tag in metadata.items():
                print(f"{tag_name}: {tag.value}")

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
        self.ttval= 1
        self.zzval= round(set_widg[1]/2)
        if (dimension[0])==1 or (dimension[1])==1:
            set_widg = [1,1]
            self.SIGNALS.reset_widget.emit(set_widg)
            self.ttval=0
            self.zzval=0
        print(f'Image dim: {dimension}')
        return dimension
        
    def onValue1Changed(self, value):
        # print(f"timevalue{value}")
        self.ttval=value-1 #slider: 1-max for biologists
        self.new_img_wrp = CVImage(self.get_img())
        
        if self.session.gui:
            #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)   
    
    def onValue2Changed(self, value):
        # print(f"zvalue{value}")
        self.zzval=value-1
        self.new_img_wrp = CVImage(self.get_img())
        self.output_data(value)
        
        if self.session.gui:
                #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
    
    def output_data(self, value):
        if self.reshaped_data.shape[0] != 1:
            self.set_output_val(0, (self.reshaped_data[self.ttval, :, :, :, :], self.ttval, self.zzval))
        else:
            self.set_output_val(0, (self.reshaped_data[0, :, :, :, :], self.ttval, self.zzval))


    def get_img(self):
        # 4D
        # if (self.dim[0] != 1) & (self.dim[1] != 1):
        self.sliced = self.reshaped_data[self.ttval,self.zzval,:,:,:]
        # reshaped = self.sliced.reshape(self.sliced.shape[:-1] + (-1,))
        # print(f"THIS is the RESHAPE: {self.sliced.shape}")
        return self.sliced
        # 2D in time
        # elif self.dim[0] != 1:
        #     return self.reshaped_data[self.ttval,,:,:]  #wont have all these *** (SQUEEZED) 
        # # 3D (Z-stack)
        # elif self.dim[1] != 1:
        #     return self.reshaped_data[1,self.zzval,:,:]
    
    # def get_state(self):
    #     data = {'image file path': self.image_filepath}
    #     return data
    def get_state(self) -> dict:
        data = {
            'image file path': self.image_filepath,
            'val1': self.ttval,
            'val2': self.zzval,
                # 'dimension': self.dim
            }
        # print(data)
        return data
        

    def set_state(self, data: dict, version):
        self.path_chosen(data['image file path'])
        self.ttval = data['val1']
        self.zzval = data['val2']
        self.update()
        # self.dim = data['dimension']
        # self.id_tiff_dim(self.image_filepath)
        # self.image_filepath = data['image file path']

    def path_chosen(self, file_path):
        self.image_filepath = file_path
        self.update()

    

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
            RGB_stack = stack[0, :, :, :]
            # print(f"RGB shape {RGB_stack.shape}")
            custom_metadata = {
                "Description": "Stack (RGB) preprocessed with Visual Processing Pipeline. Developed using Ryven by Emma Sharratt and Dr Rensue Theart",
                "Author": "Emma Sharratt and Dr Rensue Theart",
                "Date": "Pipeline created in 2023",
                'axes': 'ZYX'
                # "256": RGB_stack.shape[2], #W
                # "257": RGB_stack.shape[1], #H
                # "slices=": RGB_stack.shape[0],
                # "frames=": 1,
                # "channels=": RGB_stack.shape[3],
            }

            tiff.imwrite(self.file_path, RGB_stack, photometric='rgb', imagej=True, metadata=custom_metadata)  # You can adjust options as needed

    def update_event(self, inp=-1):
        self.SIGNALS.reset_widget.emit(1)


    def get_state(self):
        return {'path': self.file_path}

    def set_state(self, data, version):
        self.file_path = data['path']
    
class OutputMetadata(NodeBase0):
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
                # propertiesStr = Signal(str)
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        
        # self.file_path = ''

    def view_place_event(self):
        self.SIGNALS.propertiesDf.connect(self.main_widget().show_data)
        self.main_widget().new_data.connect(self.properties)
        
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
        # self.main_widget().new_data(self.image_stack)
        # self.properties(self.image_stack)
        # self.new_img_wrp = CVImage(self.get_img(self.sliced))
        # if self.prev == True:
        #     if self.session.gui:
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        # self.proc_stack_parallel()
      
        self.set_output_val(0, (self.image_stack, self.frame, self.z_sclice))
    
    def properties(self, true):
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
            f"Average Area: {area_avg}\n"
            f"Average Centroid: {centroid_avg}\n"
            f"Average Equivalent Diameter: {Equivalent_Diameter_avg}\n"
            f"Average Euler Number: {Euler_avg}\n"
            f"Average Extent: {Extent_avg}\n"
            f"Average Space Filled: {Filled_Area}\n"
            f"Average Inertia Tensor Eigvals:\n{Inertia_avg}\n"
            f"Average Volume: {volume_avg}\n"
            f"Average Surface Area: {Surface_area_avg}\n"
            f"Average Sphericity: {Sphericity_avg}\n"
            f"Average Aspect Ratio: {Aspect_ratio}\n"
        )

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

class ReadImageTiff(NodeBase0):
    """Reads an image from a file"""

    title = 'Read Tiff Image'
    input_widget_classes = {
        'choose file IW': widgets.ChooseFileInputWidget
    }
    init_inputs = [
        NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
    ]
    init_outputs = [
        NodeOutputBP('img')
    ]

    def __init__(self, params):
        super().__init__(params)

        self.image_filepath = ''

    def view_place_event(self):
        self.input_widget(0).path_chosen.connect(self.path_chosen)
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):
        if self.image_filepath == '':
            return
        # Check if the file has a .tiff extension
        if self.image_filepath.endswith('.tif'):
            try:
                image_data = tiff.imread(self.image_filepath)
                slicetz = image_data[23,9,:,:]
                self.set_output_val(0, slicetz)
                # print("Image shape:", slicetz.shape)
                # print("Image loaded successfully")
            except Exception as e:
                print(e)
                print("failed")


        # _, file_extension = os.path.splitext(self.image_filepath)
        # os.
        # if file_extension.lower() == '.tif':
        #     image = CVImage(tiff.imread(self.image_filepath))
        #     self.set_output_val(0, image)
        #     print(image.shape)
        #     print("lower")

        # # else: 
        # try:
        #         image_data = tiff.imread(self.image_filepath)
        #         slicetz = image_data[23,9,:,:]
        #         self.set_output_val(0, slicetz)
        #         print("Image shape:", slicetz.shape)
        #         print("Image loaded successfully")
        # except Exception as e:
        #         print(e)
        #         print("failed")

    def get_state(self):
        data = {'image file path': self.image_filepath}
        return data

    def set_state(self, data, version):
        self.path_chosen(data['image file path'])
        # self.image_filepath = data['image file path']

    def path_chosen(self, file_path):
        self.image_filepath = file_path  #file_path defined in widget 
        print(self.image_filepath)
        self.update()

        

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
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
    title = 'Split Channels (RGB)'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('red channel'), #img
        NodeOutputBP('green channel'), #img
        NodeOutputBP('blue channel'), #img
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
        stack4D=self.image_stack
        print(f"shape stack4D {stack4D.shape}")
        # Split the RGB data into separate channels
        shape = stack4D.shape

        if shape[-1] == 1:
            print("GRAYSCALE")

        elif shape[-1] > 1: 
            red_stack = stack4D[..., 0]  # Extract the red channel (index 0)
            red_stack = red_stack[:, :, :, np.newaxis]

            green_stack = stack4D[..., 1]  # Extract the green channel (index 1)
            green_stack = green_stack[:, :, :, np.newaxis]

            blue_stack = stack4D[..., 2]  # Extract the blue channel (index 2)
            blue_stack = blue_stack[:, :, :, np.newaxis]

            if shape[-1] == 4:
                

                magenta_stack = stack4D[..., 3]  # Extract the blue channel (index 2)
                magenta_stack = magenta_stack[:, :, :, np.newaxis]

                self.set_output_val(0, (red_stack, self.frame, self.z_sclice))
                print(f"shape split: {red_stack.shape}")
                self.set_output_val(1, (green_stack, self.frame, self.z_sclice))
                self.set_output_val(2, (blue_stack, self.frame, self.z_sclice))
                self.set_output_val(3, (magenta_stack, self.frame, self.z_sclice))

            # If the input is an RGB image
            elif shape[-1] == 3:
                blank_stack = np.zeros_like(stack4D[..., 0])
                self.set_output_val(0, (red_stack, self.frame, self.z_sclice))
                print(f"shape split: {red_stack.shape}")
                self.set_output_val(1, (green_stack, self.frame, self.z_sclice))
                self.set_output_val(2, (blue_stack, self.frame, self.z_sclice))    
                # send a blank stack to magenta channel 
                self.set_output_val(3, (blank_stack, self.frame, self.z_sclice))
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
    title = 'Merge Channels (RGB)'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('red channel'),
        NodeInputBP('green channel'),
        NodeInputBP('blue channel'),
         
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
        self.red_stack = self.input(0)[0]
        self.gr_stack = self.input(1)[0]
        self.blue_stack = self.input(2)[0]
        self.frame = self.input(0)[1] #dont actually need this anymore, but keep incase. Good to know wich time step
        self.z_sclice = self.input(0)[2]
        # self.squeeze = np.squeeze(self.image_stack)
        self.z_size = self.red_stack.shape[0]
        print(f"z_size {self.z_size}")
        # self.sliced = self.image_stack[self.z_sclice, :, :, :] #Z, H, W, C
              
    def proc_technique(self):
        self.rgb_image = np.concatenate((self.red_stack, self.gr_stack, self.blue_stack), axis=-1)
        self.sliced = self.rgb_image[self.z_sclice, :, :, :] #Z, H, W, C
        self.set_output_val(0, (self.rgb_image, self.frame, self.z_sclice))
        
        
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        self.main_widget().kValueChanged.connect(self.proc_stack_parallel)
        self.main_widget().previewState.connect(self.preview)
        
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
        #extract slice
        self.handle_stack()
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
      
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        # print(f"type {self.sliced.dtype}")
        # print(f"shape {self.sliced.shape}")
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
                
        self.proc_stack_parallel()
      
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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

        #ALL time steps
    # def proc_stack_parallel(self, value):
    #     # Create an empty array to store the processed data
    #     proc_data = np.empty_like(self.squeeze)

    #     # Define the kernel size for median blur
    #     ksize = value

    #     # Define the number of worker threads or processes
    #     num_workers = 8  # Adjust as needed

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         futures = []
    #         for t in range(self.frame_size):
    #             for z in range(self.z_size):
    #                 img = self.squeeze[t, z, :, :]
    #                 future = executor.submit(self.process_frame,img, ksize)
    #                 futures.append((t, z, future))

    #         for t, z, future in futures:
    #             processed_image = future.result()
    #             proc_data[t, z, :, :] = processed_image

    #     print(f"proc_data shape {proc_data.shape}")
    #     self.reshaped_proc_data = proc_data[..., np.newaxis]
    #     print(f"reshaped_proc_data shape {self.reshaped_proc_data.shape}")
    #     self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))


class Guassian_Blur3D1(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Guassian_Blur3D'
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5
        # self.sigma = [5, 5, 5,]
    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        self.main_widget().kReleased.connect(self.proc_stack_parallel)
        self.main_widget().previewState.connect(self.preview)
        
        try:
             self.new_img_wrp = CVImage(self.get_img())
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
        self.proc_stack_parallel()
        # print(f"type {self.sliced.dtype}")
        # print(f"shape {self.sliced.shape}")
        self.new_img_wrp = CVImage(self.get_img())
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

    def onSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(value)
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
        #Only out put when slider released 
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    def get_img(self):
        #debug
        # #print(f"getimageValue{value}")
        # image_uint8 = ((self.sliced/self.sliced.max())*255).astype('uint8')
        # convert to 8, display as 8, but keep stack as original
        return self.proc[self.frame, self.z_sclice, :, :]
    
    def process_frame(self, img, sigma_z, sigma_x, sigma_y):
        return gaussian_filter(img, sigma=(sigma_x, sigma_y,sigma_z))

    def proc_stack_parallel(self, value):
        # Create an empty array to store the processed data
        proc_data = np.empty_like(self.squeeze)

        # Define the parameters for Gaussian blur
        sigma_z = value #make global, then call below # Assuming kk now represents sigma_z
        sigma_x = 2.0 #value[0]
        sigma_y = 2.0

        # Define the number of worker threads or processes
        num_workers = 8  # Adjust as needed

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for t in range(self.frame_size):
                img = self.squeeze[t, :, :, :]  # Get the 2D slice
                future = executor.submit(self.process_frame, img, sigma_z, sigma_x, sigma_y)
                futures.append((t, future))

            for t, future in futures:
                processed_image = future.result()
                proc_data[t, :, :, :] = processed_image


        #print(f"proc_data shape {proc_data.shape}")
        self.proc = proc_data
        self.reshaped_proc_data = proc_data[..., np.newaxis]
        #print(f"reshaped_proc_data shape {self.reshaped_proc_data.shape}")

        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    # def proc_stack(self):
    #     proc_data = np.empty_like(self.squeeze)
    #     #4D
    #     sigma_x = 2.0
    #     sigma_y = 2.0
    #     # if (self.image_stack.shape[0] != 1) & (self.image_stack.shape[1] != 1):
    #     for t in range(self.frame_size):
    #             blurred_stack = gaussian_filter(self.squeeze[t, :, :, :], sigma=(self.kk, sigma_x, sigma_y))
                
    #             proc_data[t, :, :, :] = blurred_stack
    #             # cv2.imshow(f"proced slice{z}", proc_data[2, z, :, :])
    #     #print(f"proc_data shape {proc_data.shape}")
    #     #reshape
    #     # Reshape 'proc_data' to add an extra dimension of size 1
    #     self.proc = proc_data
    #     self.reshaped_proc_data = proc_data[..., np.newaxis]
    #     #print(f"reshaped_proc_data shape {self.reshaped_proc_data.shape}")


     
        
	
    # #use when save and close
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']


class Dimension_Management(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'Dimension Management'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Slider_widget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 5
        self.vval1 = default
        self.vval2 = default

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().ValueChanged1.connect(self.onValue1Changed)
        self.main_widget().ValueChanged2.connect(self.onvalue2Changed)   
        
        try:
             self.new_img_wrp = CVImage(self.get_img())
             self.SIGNALS.new_img.emit(self.new_img_wrp)
             self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.dimensions = self.input(0).shape #[time, z, width, height]
        #print(self.dimensions)

        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.set_output_val(0, self.new_img_wrp)
    
    def preview(self, state):
        if state ==  True:
            self.prev = True
            #Bring image back immediately 
            self.SIGNALS.new_img.emit(self.new_img_wrp)
               
        else:
              self.prev = False 
              self.SIGNALS.clr_img.emit(self.new_img_wrp) 

    def onValue1Changed(self, value):
        # This method will be called whenever the widget's signal is emitted
        #print(value)
        self.vval1 = value
        self.update_new_img_wrp

    def onvalue2Changed(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.xx = value
        self.update_new_img_wrp

    def update_new_img_wrp(self):
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp)
      
        self.set_output_val(0, self.new_img_wrp)
    
    def get_img(self):
        # debug
        # #print(self.vval1)
        # #print(self.vval2)
        #print("sliceShapeComing")
        image = self.input(0)
        #print("sliceShape", image.shape)
        return self.input(0)
            
    
      # #use when save and close
    def get_state(self) -> dict:
        return {
            'val1': self.vval1,
            'val2': self.vval2,
        }

    def set_state(self, data: dict, version):
        self.vval1 = data['val1']
        self.vval2 = data['val2']

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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        return {
            'ksize': self.kk,
            'sigmaX': self.xx,
            # 'sigmaY': self.yy,

        }

    def set_state(self, data: dict, version):
        self.kk = data['ksize']
        self.xx = data['sigmaX']
        self.yy = data['sigmaX']

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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 5
        self.kk = 1
        self.xx = 1
        self.yy = 1
        warning = 0
        # self.kk = default
        # self.xx = default
        # self.yy = default

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.SIGNALS.warning.connect(self.main_widget().warning)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().kValueChanged.connect(self.onkValueChanged)
        self.main_widget().XValueChanged.connect(self.onXvalueChanged)        
        self.main_widget().YValueChanged.connect(self.onYvalueChanged)
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
        # self.proc_stack_parallel()
        # 3D process stack
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        # self.proc_technique(self.image_stack)
        # self.new_img_wrp = CVImage(self.sliced)
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))


    def onXvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.xx = value
        # self.proc_technique(self.image_stack)
        # self.new_img_wrp = CVImage(self.sliced)
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    def onYvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.yy = value
        # self.proc_technique(self.image_stack)
        # #print(self.yy)
        # self.new_img_wrp = CVImage(self.sliced)
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
        # self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
    def onSigValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # #print(self.kk)
        # self.kk = value
        self.xx = value
        self.yy = value
        # self.proc_technique()
        # self.new_img_wrp = CVImage(self.sliced)
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
    
    def proc_technique(self):
        stack4D=self.image_stack
        # #print(f"proc input stack: {stack4D.shape}")
         #Ensure only on 3D data
        if stack4D.shape[0] > 1:
            # Apply the gaussian filter to the entire 4D array
            filtered_image = gaussian_filter(stack4D, sigma=(self.kk, self.yy,self.xx, 0))
            # #print(self.kk)
            # prevent the use of for loops, but no sigma applied to channels
            #print(f"filtered_image {filtered_image.shape}")
            # update displayed image
            self.sliced = filtered_image[self.z_sclice, :, :, :]
            self.reshaped_proc_data= filtered_image
            
            self.new_img_wrp = CVImage(self.sliced)
            if self.prev == True:
                if self.session.gui:
                    #update continuously 
                    self.SIGNALS.new_img.emit(self.new_img_wrp.img)
            else:
                self.SIGNALS.clr_img.emit(self.new_img_wrp.img)

            self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
                
        
            #emit Warning! A 3D Gaussian Blur cannot be performed on 2D data
            # Preprocessing has not been performed on the data
            # To use this node please read 3D data into the pipeline 

      # #use when save and close
    def get_state(self) -> dict:
        return {
            # 'ksize': self.xx,
            'sigmaX': self.xx,
            # 'sigmaY': self.xx,

        }

    def set_state(self, data: dict, version):
        self.kk = data['sigmaX']
        self.xx = data['sigmaX']
        self.yy = data['sigmaX']

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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 5
        self.kk = default
        self.xx = default
        self.yy = default
        # self.kk = default
        # self.xx = default
        # self.yy = default

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
    
    
class Bilateral_Filtering1(NodeBase):        #Nodebase just a different colour
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default = 5
        self.kk = default  #d
        self.xx = default  #sigmaColour
        self.yy = default  #sigmaSpace

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().kValueChanged.connect(self.onkValueChanged)
        self.main_widget().XValueChanged.connect(self.onXvalueChanged)        
        self.main_widget().YValueChanged.connect(self.onYvalueChanged)
        
        try:
             self.new_img_wrp = CVImage(self.get_img())
             self.SIGNALS.new_img.emit(self.new_img_wrp.img)
             self.set_output_val(0, self.new_img_wrp)
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.set_output_val(0, self.new_img_wrp)
    
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
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def onXvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.xx = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)
    
    def onYvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.yy = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)
    
    def get_img(self):
        # debug
        # #print(self.xx)
        # #print(self.kk)
        return cv2.bilateralFilter(
            src=self.input(0).img,
            d=self.kk,
            sigmaColor=self.xx,
            sigmaSpace=self.yy,
        )
        
    # #use when save and close
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']

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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        default2 = 100
        self.value_1 = default1  #threshold
        self.value_2 = 1

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        default2 = 100
        self.value_1 = default1  #threshold
        self.value_2 = 1

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 2
        default2 = 100
        self.value_1 = default1  #threshold

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        
        self.proc_stack_parallel()

        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
    # def get_state(self) -> dict:
    #     return {
    #         'val1': self.value_1

    #     }

    # def set_state(self, data: dict, version):
    #     self.value_1 = data['val1']
    
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        default1 = 10
        default2 = 100
        self.thr = default1  #threshold
        self.mv = default2   #maxvalue

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        self.new_img_wrp = CVImage(self.get_img(self.sliced))
        if self.prev == True:
            if self.session.gui:
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        self.proc_stack_parallel()
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))
    
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
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged1)
        self.main_widget().Value1Changed.connect(self.gamma_correction)
        
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
        self.set_output_val(0, (self.reshaped_proc_data, self.frame, self.z_sclice))

  
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

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_histogram)
        self.SIGNALS.logScale.connect(self.main_widget().log_hist)
        # self.SIGNALS.show_img.connect(self.main_widget().show_image)
        self.SIGNALS.clear_graph.connect(self.main_widget().clear_hist)
        self.main_widget().displayHist.connect(self.emitImage)
        self.main_widget().LogHist.connect(self.logScale)

        # try:
        #     self.SIGNALS.new_img.emit(CVImage(self.z_sclice).img)
        # except:  # there might not be an image ready yet
        #     pass

    def update_event(self, inp=-1):                                         #reset
        #extract from stack
        self.handle_stack()
        # histo_s
        self.new_img_wrp = CVImage(self.sliced)
        if self.session.gui:
            self.SIGNALS.clear_graph.emit(True)

        self.set_output_val(0, (self.image_stack,self.frame, self.z_sclice))
    
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
        

    
    def get_img(self):
        return None
    

    
#Original GaussianBlur
class GaussianBlur(OpenCVNodeBase):
    title = 'Gaussian Blur New'
    init_inputs = [
        NodeInputBP('img'),
        #square matrix
        #NodeInputBP('ksize', dtype=dtypes.Tuple[int, int], value=(3, 3)), #default 3x3

        #NodeInputBP('ksize', dtype=dtypes.Int(3)),
        #original 
        NodeInputBP('ksize', dtype=dtypes.Data((3, 3))), #default 3x3
        NodeInputBP('sigmaX', dtype=dtypes.Float(1.0)),
        NodeInputBP('sigmaY', dtype=dtypes.Float(0.0)),
    ]

    def get_img(self):
        return cv2.GaussianBlur(
            src=self.input(0).img,
            ksize=self.input(1),
            sigmaX=self.input(2),
            sigmaY=self.input(3),
        )        
#NOTE -> changed OUTPUT ORDER -- img now first 
#Creating adapted gaussian with sliders & preview
#Gaussian blur - k can ONLY BE ODD!
class GaussianBlurNode(NodeBase):        
    title = 'GaussianBlurNode'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('img'),                                                    #0
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),           #1
        NodeInputBP('sigmaX', dtype=dtypes.Float(1.0)),                        #2
        NodeInputBP('sigmaY', dtype=dtypes.Float(0.0)),                        #3
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),     #4
    ]
    init_outputs = [
        NodeOutputBP(),  #image
        NodeOutputBP(),   #ksize debugging (only need image later)
        NodeOutputBP(),   #sigmaX debugging (only need image later)
    ]
    main_widget_class = widgets.V3QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)
                clr_img = Signal(object)
                # preview_input = Signal(object)
        
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.val = 0
      
    def place_event(self):  
        self.update()
        self.previous_checkbox = self.input(2)

    def view_place_event(self):
        #Signals
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        # self.SIGNALS.preview_input.connect(self.main_widget().preview_input_changed)
        #Preview
        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
        #debugging
        #print("update event")
        #print(f"Old:{self.previous_checkbox }")
        self.previous_checkbox = self.input(2)
        #print(f"New:{self.previous_checkbox }")

        # if self.input(2) != self.previous_checkbox:
        #     #print("Checkbox not checked")
        
        #sliderK
        self.v = self.input(1)
        if self.input(1)<=10:
            range1 = self.input(1)*2
            self.v = 0.1 + (range1 * self.val)
        else:
            range1 = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
        self.v=int(self.v)
        self.set_output_val(1, self.v)

        #sliderX
        self.x = self.input(1)
        if self.input(1)<=10:
            range1 = self.input(1)*2
            self.x = 0.1 + (range1 * self.val)
        else:
            range1 = 20                                                          #add if statements to ensure positive
            self.x = (self.input(1)-(range1/2)) + (range1 * self.val)
        # self.x=int(self.v)
        self.set_output_val(2, self.x)      

        #image
        new_img_wrp = CVImage(self.get_img())
        if self.input(2):   
            # if self.session.gui:
                self.SIGNALS.new_img.emit(new_img_wrp.img)
        else:
            # if self.session.gui:
            if self.input(2) != self.previous_checkbox:
                self.SIGNALS.clr_img.emit(new_img_wrp.img)
                

        # Preview
        # if self.input(2) != self.previous_checkbox:
            #print("Checkbox just checked")
            # self.SIGNALS.preview_input.emit(new_img_wrp.img)
            # self.main_widget().update_shape()

        self.set_output_val(0, new_img_wrp)
        self.main_widget().text_label.setText((f"adjust ksize: {self.v}"))
        self.main_widget().text_label.setText((f"adjust SigmaX: {self.x}")) #change self.v

    def get_state(self) -> dict:
        return {
            'val': self.val,
        }

    def set_state(self, data: dict, version):
        self.val = data['val']

    def get_img(self):
        # return self.input(0).img
        k = int(self.v)
        # #print(k)
        return cv2.GaussianBlur(
            src=self.input(0).img,
            # ksize=(k,k),
            ksize=(self.input(1),self.input(1)),
            sigmaX=self.input(2),
            sigmaY=self.input(3),
                )



    #v6
    # def __init__(self, params):
    #     super().__init__(params)

    #     if self.session.gui:
    #         from qtpy.QtCore import QObject, Signal
    #         class Signals(QObject):
    #             new_img = Signal(object)
        
        
    #         # to send images to main_widget in gui mode
    #         self.SIGNALS = Signals()

    #     self.val = 0
    #     self.preview_enabled = False 

    # def place_event(self): 
    #     self.update()

    # def view_place_event(self):
    #     if self.input(2):  # Assuming the input(2) represents the "Preview" checkbox
    #         if self.session.gui:
    #             self.SIGNALS.new_img.connect(self.main_widget().show_image)
            
    #     try:
    #         self.SIGNALS.new_img.emit(self.get_img())
    #     except:
    #         pass
        
    #     self.update()

    # def update_event(self, inp=-1):
    #      #slider
    #     if self.input(1)<=10:
    #         range1 = self.input(1)*2
    #         self.v = 0.1 + (range1 * self.val)
    #     else:
    #         range1 = 20                                                          #add if statements to ensure positive
    #         self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
    #     self.v=int(self.v)
    #     self.set_output_val(0, self.v)

    #     #image
    #     new_img_wrp = CVImage(self.get_img())
    #     if self.input(2):   
    #         if self.session.gui:
    #             self.SIGNALS.new_img.emit(new_img_wrp.img)

    #     self.set_output_val(1, new_img_wrp)

       

    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #         'preview_enabled': self.preview_enabled,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']
    #     self.preview_enabled = data['preview_enabled']

    # def get_img(self):
    #     # return self.input(0).img
    #     k = int(self.v)
    #     # #print(k)
    #     return cv2.blur(
    #         src=self.input(0).img,
    #         ksize=(k,k),
    #             )
# ADD SLIDER through widget
class BlurAdapted(NodeBase):
    """Performs a median blur on an img"""
    title = 'Blur Adapted & Slider'
    init_inputs = [
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        
        #Slider inputs
        #NodeInputBP(dtype=dtypes.Integer(default=1), label='scl'),
        NodeInputBP(dtype=dtypes.Boolean(default=True), label='round'), #2
                # NodeInputBP( dtype=dtypes.Boolean(default=False), label='Preview'),
    ]

    init_outputs = [
        NodeOutputBP('img'),
        NodeOutputBP()
    ]
       
    main_widget_class = widgets.QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        #slider
        self.val = 0
    
    def place_event(self):  #sldier
        self.update()

    def view_place_event(self):
        #slider
         # when running in gui mode, the value might come from the input widget
        self.update()
        #OpenCV
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass       

    def update_event(self, inp=-1):                                         #reset
        new_img_wrp = CVImage(self.get_img())

        if self.session.gui:
            self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(0, new_img_wrp)

        #slider
        if self.input(1)<=10:
            range = self.input(1)*2
            self.v = 0.1 + (range * self.val)
        else:
            range = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range/2)) + (range * self.val)
        
        if self.input(3):
            self.v = round(self.v)

        self.set_output_val(1, self.v)
        

    def get_img(self):
        #if self.input(2):
        blurred_img=cv2.blur(
            src=self.input(0).img,
            ksize=(self.v,self.v))
        
        return blurred_img 
        #else:
         #   return None
    def get_state(self) -> dict:
        return {
            'val': self.val,
        }
    def set_state(self, data: dict, version):
        self.val = data['val']



class GaussianBlur(OpenCVNodeBase):
    title = 'Gaussian Blur New'
    init_inputs = [
        NodeInputBP('img'),
        #square matrix
        #NodeInputBP('ksize', dtype=dtypes.Tuple[int, int], value=(3, 3)), #default 3x3

        #NodeInputBP('ksize', dtype=dtypes.Int(3)),
        #original 
        NodeInputBP('ksize', dtype=dtypes.Data((3, 3))), #default 3x3
        NodeInputBP('sigmaX', dtype=dtypes.Float(1.0)),
        NodeInputBP('sigmaY', dtype=dtypes.Float(0.0)),
    ]

    def get_img(self):
        return cv2.GaussianBlur(
            src=self.input(0).img,
            ksize=self.input(1),
            sigmaX=self.input(2),
            sigmaY=self.input(3),
        )        

class BlurMedian(OpenCVNodeBase):
    """Performs a median blur on an img"""
    title = 'Blur Median'
    init_inputs = [
        NodeInputBP('img'),
        NodeInputBP('ksize', dtype=dtypes.Data(default=3)),
    ]

    def get_img(self):
        return cv2.medianBlur(src=self.input(0).img, ksize=self.input(1))
    
# ADD SLIDER through widget
class BlurAdapted(NodeBase):
    """Performs a median blur on an img"""
    title = 'Blur Adapted & Slider'
    init_inputs = [
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        
        #Slider inputs
        #NodeInputBP(dtype=dtypes.Integer(default=1), label='scl'),
        NodeInputBP(dtype=dtypes.Boolean(default=True), label='round'), #2
                # NodeInputBP( dtype=dtypes.Boolean(default=False), label='Preview'),
    ]

    init_outputs = [
        NodeOutputBP('img'),
        NodeOutputBP()
    ]
       
    main_widget_class = widgets.QvBoxDev_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                new_img = Signal(object)

            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()
        #slider
        self.val = 0
    
    def place_event(self):  #sldier
        self.update()

    def view_place_event(self):
        #slider
         # when running in gui mode, the value might come from the input widget
        self.update()
        #OpenCV
        self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass       

    def update_event(self, inp=-1):                                         #reset
        new_img_wrp = CVImage(self.get_img())

        if self.session.gui:
            self.SIGNALS.new_img.emit(new_img_wrp.img)

        self.set_output_val(0, new_img_wrp)

        #slider
        if self.input(1)<=10:
            range = self.input(1)*2
            self.v = 0.1 + (range * self.val)
        else:
            range = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range/2)) + (range * self.val)
        
        if self.input(3):
            self.v = round(self.v)

        self.set_output_val(1, self.v)
        

    def get_img(self):
        #if self.input(2):
        blurred_img=cv2.blur(
            src=self.input(0).img,
            ksize=(self.v,self.v))
        
        return blurred_img 
        #else:
         #   return None
    def get_state(self) -> dict:
        return {
            'val': self.val,
        }
    def set_state(self, data: dict, version):
        self.val = data['val']
# title = 'slider'
    # version = 'v0.1'
    # init_inputs = [
    #     NodeInputBP(dtype=dtypes.Integer(default=1), label='scl'),
    #     NodeInputBP(dtype=dtypes.Boolean(default=False), label='round'),
    # ]
    # init_outputs = [
    #     NodeOutputBP(),
    # ]
    # main_widget_class = widgets.SliderNode_MainWidget
    # main_widget_pos = 'below ports'

    # def __init__(self, params):
    #     super().__init__(params)

    #     self.val = 0

    # def place_event(self):  #??
    #     self.update()

    # def view_place_event(self):
    #     # when running in gui mode, the value might come from the input widget
    #     self.update()

    # def update_event(self, inp=-1):

    #     v = self.input(0) * self.val
    #     if self.input(1):
    #         v = round(v)

    #     self.set_output_val(0, v)

    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']


    
class BlurAdaptedPreview(NodeBase):             #made base node NodeBase - 
    """Performs a median blur on an img"""      #adds text if hover over node
    title = 'Blur Adapted Preview'
    init_inputs = [
        NodeInputBP('img'),
        NodeInputBP(dtype=dtypes.Integer(default=3), label='ksize'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),        
    ]

    init_outputs = [
        NodeOutputBP('img')
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

    def view_place_event(self):
        #self.SIGNALS.new_img.connect(self.main_widget().show_image)

        try:
                self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass

    def update_event(self, inp=-1):                                         #reset
        new_img_wrp = CVImage(self.get_img())
        if self.input(2):    
            self.SIGNALS.new_img.connect(self.main_widget().show_image)
            if self.session.gui:                                #if this is taken out if the "if input" then the preview continualusy 
                self.SIGNALS.new_img.emit(new_img_wrp.img)      #updates wether preview is selected or not
        else:
            #self.SIGNALS.new_img.connect(self.main_widget().clear_image)
            self.main_widget().clear_image()

        self.set_output_val(0, new_img_wrp)

    def get_img(self):
        return cv2.blur(
            src=self.input(0).img,
            ksize=(self.input(1),self.input(1)),
                )

#12/07/23
#Simplified edit
class BlurSimplePrev(OpenCVNodeBase):
    """Performs a median blur on an img"""
    title = 'Blur Simple Prev'
    init_inputs = [
        NodeInputBP('img'),
        NodeInputBP('ksize', dtype=dtypes.Data(default=3)),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='Preview'),
    ]

    def get_img(self):
        if self.input(2):
            return cv2.blur(src=self.input(0).img, ksize=(self.input(1),self.input(1)))
#Experimented with the return funtionality
#This did not work well
#Always need to return the image. Need to deal with the preview boolean 
#This code just looks shorter because of OpenCVNodeBase


#Experimenting new Data transfer
class TransmitTuple(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'TransmitTuple'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Tuple
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5

    def place_event(self):  
        self.update()

    def view_place_event(self):
        # self.SIGNALS.new_img.connect(self.main_widget().show_image)
        # self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        # self.main_widget().previewState.connect(self.preview)
        
        try:
            #  self.new_img_wrp = CVImage(self.get_img())
            #  self.SIGNALS.new_img.emit(self.new_img_wrp.img)
             self.set_output_val(0, (self.kk, 5))
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        # self.new_img_wrp = CVImage(self.get_img())
        # if self.prev == True:
        #     if self.session.gui:
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)

        self.set_output_val(0, (self.kk, 5))
    
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
        # #print(value)
        self.kk = value
        # self.new_img_wrp = CVImage(self.get_img())
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, (self.kk, 5))
        # #print(self.outputs(0))
    
    def get_img(self):
        #debug
        # #print(f"getimageValue{value}")
        return cv2.blur(
            src=self.input(0).img,
            ksize=(self.kk,self.kk),
                )
        
    # #use when save and close
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']

class TransmitTuple2(NodeBase):        #Nodebase just a different colour
          #Nodebase just a different colour
    title = 'TransmitTuple2'
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Tuple
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        if self.session.gui:
            from qtpy.QtCore import QObject, Signal
            class Signals(QObject):
                #Signals used for preview
                new_img = Signal(object)    #original
                clr_img = Signal(object)    #added      
        
            # to send images to main_widget in gui mode
            self.SIGNALS = Signals()

        self.prev = True
        self.kk = 5

    def place_event(self):  
        self.update()

    def view_place_event(self):
        # self.SIGNALS.new_img.connect(self.main_widget().show_image)
        # self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().kValueChanged.connect(self.onSliderValueChanged)
        # self.main_widget().previewState.connect(self.preview)
        
        try:
            #  self.new_img_wrp = CVImage(self.get_img())
            #  self.SIGNALS.new_img.emit(self.new_img_wrp.img)
             self.set_output_val(0, (self.kk, 5))
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        # check
        self.update()

    #called when img connected - send output
    def update_event(self, inp=-1):  #called when an input is changed
        # self.new_img_wrp = CVImage(self.get_img())
        # if self.prev == True:
        #     if self.session.gui:
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        #print("INPUT", self.input(0)[0])
        self.set_output_val(0, (self.kk, 5))
    
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
        # #print(value)
        self.kk = value
        # self.new_img_wrp = CVImage(self.get_img())
        # if self.prev == True:
        #     if self.session.gui:
        #         #update continuously 
        #         self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        # else:
        #     self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
        new = self.kk*self.input(0)[0]
        new2 = 5*self.input(0)[1]
        self.set_output_val(0, (new, new2))
        #print(f"new{new}, new2{new2}")
        # #print(self.outputs(0))
    
    def get_img(self):
        #debug
        # #print(f"getimageValue{value}")
        return cv2.blur(
            src=self.input(0).img,
            ksize=(self.kk,self.kk),
                )
        
    # #use when save and close
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']


nodes = [
    # Checkpoint_Node,
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
    # Slider_Gaus,
    # GaussianBlur,
    # BlurMedian,
    # Slider_Gaus_Old,
    # Slider_Gaus_Tick_v2,
    # Slider_Gaus_Tick_v3,
    # Slider_Gaus_Tick_v4,
    # Slider_Gaus_Tick_v5,
    # Slider_Gaus_Tick_v6,
    # Slider_Gaus_Tick_v7,
    # Slider_Gaus_Tick_v8,
    # GaussianBlurNode,
    # BlurAdapted,
    # BlurAdaptedPreview,
    # BlurSimplePrev,

    # Pipeline Nodes
    ReadImage,
    SaveImg,
    OutputMetadata,
    # ReadImageTiff,
    DisplayImg,
    Crop,
    # Dimension_Management,
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
    # Contrast Enhancemnet 
    Histogram,
    AlphaNode,
    GammaNode,

    
    
    # TransmitTuple,
    # TransmitTuple2,
    
]
