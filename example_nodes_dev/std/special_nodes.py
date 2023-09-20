import code
from contextlib import redirect_stdout, redirect_stderr

from ryven.NENV import *
widgets = import_widgets(__file__)

import cv2
import numpy as np
import os
import tifffile as tiff 

#THIS ALSO INCLUDES OPENCV CODE

class NodeBase0(Node):
    version = 'v0.1'
    color = '#00a6ff' #yellow - Filtering 

class NodeBase(Node):
    version = 'v0.1'
    color = '#FFCA00' #yellow - Filtering 

class NodeBase3(Node):
    version = 'v0.1'
    color = '#C55A11' #red - contrast enh 

class NodeBase2(Node):
    version = 'v0.1'
    color = '#92D050' #green - Binarization

class NodeBase4(Node):
    version = 'v0.1'
    color = '#8064A2' #purple - post binarization


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


class Button_Node(NodeBase):
    title = 'Button'
    version = 'v0.1'
    main_widget_class = widgets.ButtonNode_MainWidget
    main_widget_pos = 'between ports'
    init_inputs = [

    ]
    init_outputs = [
        NodeOutputBP(type_='exec')
    ]
    color = '#99dd55'

    def update_event(self, inp=-1):
        self.exec_output(0)


class Print_Node(DualNodeBase):
    title = 'Print'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(type_='exec'),
        NodeInputBP(dtype=dtypes.Data(size='m')),
    ]
    init_outputs = [
        NodeOutputBP(type_='exec'),
    ]
    color = '#5d95de'

    def __init__(self, params):
        super().__init__(params, active=True)

    def update_event(self, inp=-1):
        if self.active and inp == 0:
            print(self.input(1))
        elif not self.active:
            print(self.input(0))


import logging


class Log_Node(DualNodeBase):
    title = 'Log'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(type_='exec'),
        NodeInputBP('msg', type_='data'),
    ]
    init_outputs = [
        NodeOutputBP(type_='exec'),
    ]
    main_widget_class = widgets.LogNode_MainWidget
    main_widget_pos = 'below ports'
    color = '#5d95de'

    def __init__(self, params):
        super().__init__(params, active=True)

        self.logger = self.new_logger('Log Node')

        self.targets = {
            **self.script.logs_manager.default_loggers,
            'own': self.logger,
        }
        self.target = 'global'

    def update_event(self, inp=-1):
        if self.active and inp == 0:
            i = 1
        elif not self.active:
            i = 0
        else:
            return

        msg = self.input(i)

        self.targets[self.target].log(logging.INFO, msg=msg)

    def get_state(self) -> dict:
        return {
            **super().get_state(),
            'target': self.target,
        }

    def set_state(self, data: dict, version):
        super().set_state(data, version)
        self.target = data['target']
        if self.session.gui and self.main_widget():
            self.main_widget().set_target(self.target)


class Clock_Node(NodeBase):
    title = 'clock'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Float(default=0.1), label='delay'),
        NodeInputBP(dtype=dtypes.Integer(default=-1, bounds=(-1, 1000)), label='iterations'),
    ]
    init_outputs = [
        NodeOutputBP(type_='exec')
    ]
    color = '#5d95de'
    main_widget_class = widgets.ClockNode_MainWidget
    main_widget_pos = 'below ports'

    def __init__(self, params):
        super().__init__(params)

        self.actions['start'] = {'method': self.start}
        self.actions['stop'] = {'method': self.stop}

        if self.session.gui:

            from qtpy.QtCore import QTimer
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.timeouted)
            self.iteration = 0


    def timeouted(self):
        self.exec_output(0)
        self.iteration += 1
        if -1 < self.input(1) <= self.iteration:
            self.stop()

    def start(self):
        if self.session.gui:
            self.timer.setInterval(self.input(0)*1000)
            self.timer.start()
        else:
            import time
            for i in range(self.input(1)):
                self.exec_output(0)
                time.sleep(self.input(0))

    def stop(self):
        self.iteration = 0
        if self.session.gui:
            self.timer.stop()

    def toggle(self):
        # triggered from main widget
        if self.session.gui:
            if self.timer.isActive():
                self.stop()
            else:
                self.start()

    def update_event(self, inp=-1):
        if self.session.gui:
            self.timer.setInterval(self.input(0)*1000)

    def remove_event(self):
        self.stop()


class Slider_Node(NodeBase):
    title = 'slider'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(dtype=dtypes.Integer(default=1), label='scl'),
        NodeInputBP(dtype=dtypes.Boolean(default=False), label='round'),
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
        print("place")
        self.previous_checkbox = self.input(2)

    def view_place_event(self):
        #Signals
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        # self.SIGNALS.preview_input.connect(self.main_widget().preview_input_changed)
        #Preview
        

        print("view_place and connections")
        try:
            self.SIGNALS.new_img.emit(self.get_img())
        except:  # there might not be an image ready yet
            pass
        # when running in gui mode, the value might come from the input widget
        self.update()

    def update_event(self, inp=-1):
        #debugging
        print("update event")
        
        # if self.input(2) != self.previous_checkbox:
        #     print("Checkbox not checked")
        

         #slider
        if self.input(1)<=10:
            range1 = self.input(1)*2
            self.v = 0.1 + (range1 * self.val)
        else:
            range1 = 20                                                          #add if statements to ensure positive
            self.v = (self.input(1)-(range1/2)) + (range1 * self.val)
        self.v=int(self.v)
        self.set_output_val(0, self.v)
        

        print(f"Old:{self.previous_checkbox }")
        
        #image
        new_img_wrp = CVImage(self.get_img())
        if self.input(2):   
            # if self.session.gui:
                self.SIGNALS.new_img.emit(new_img_wrp.img)
                print("The input is checked", self.input(2))
        else:
            # if self.session.gui:
            # if self.input(2) != self.previous_checkbox:
                print("Clear image now and reshape")
                self.SIGNALS.clr_img.emit(new_img_wrp.img)

        # Preview
        if self.input(2) != self.previous_checkbox:
            print("Checkbox just checked")
            # self.SIGNALS.preview_input.emit(new_img_wrp.img)
            # self.main_widget().update_shape()
        
        self.previous_checkbox = self.input(2)
        print(f"New:{self.previous_checkbox }")

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
        print("place")
        self.previous_checkbox = self.input(3)

    def view_place_event(self):
        #Signals
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        # self.SIGNALS.preview_input.connect(self.main_widget().preview_input_changed)
        #Preview
        

        print("view_place and connections")
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
                print("Clear image now and reshape")
                self.SIGNALS.clr_img.emit(new_img_wrp.img)

        # Preview
        if self.input(3) != self.previous_checkbox:
            print("Checkbox just checked")
            # self.SIGNALS.preview_input.emit(new_img_wrp.img)
            # self.main_widget().update_shape()
        
        self.previous_checkbox = self.input(3)
        print(f"New:{self.previous_checkbox }")

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
        print(value)
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
        print(f"getimageValue{value}")
        return cv2.blur(
            src=self.input(0).img,
            ksize=(value,value),
                )
    
# NODES ------------------------------------------------------------------------------------------------------------------

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
        # try:
        #     self.SIGNALS.new_img.emit(self.get_img())
        # except:  # there might not be an image ready yet
        #     pass
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):   #called when the input is changed
        #therefore new image 
        self.ttval = 0
        self.zzval = 0
        self.SIGNALS.reset_widget.emit(1)
        # self.SIGNALS.remove_widget.emit()

        if self.image_filepath == '':
            return
        # Check if the file has a .tiff extension   --> tif file capability Check tiff 
        if self.image_filepath.endswith('.tif'):
            try:
                self.image_data = tiff.imread(self.image_filepath)
                #generate dimension list (dimension, slices, frames (time), width, height, channels)
                self.id_tiff_dim(self.image_filepath)
                # print(self.image_data)
                
                #4D images 
                if self.dimension[0] == 5: #time and space (4D)
                    self.image_data = ((self.image_data/self.image_data.max())*65535).astype('uint16')
                    new_img_wrp = CVImage(self.get_img())
                    # print("shape", new_img_wrp.shape)
                    if self.session.gui:
                        self.SIGNALS.new_img.emit(new_img_wrp.img)

                    self.set_output_val(0, new_img_wrp)
                    
                    print("Image loaded successfully")
                #3D images - zstack only
                #2D images
                #dimension[0]==0
                else:
                    image2D = CVImage(cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED))
                    if self.session.gui:
                        self.SIGNALS.new_img.emit(image2D.img)
                    self.set_output_val(0,image2D)
            
            except Exception as e:
                print(e)
                print("failed")

        else: 
            try:
                self.set_output_val(0, CVImage(cv2.imread(self.image_filepath, cv2.IMREAD_UNCHANGED)))
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
            dimension = [0,1,1,0,0,0] #dim, slices , time
            
            if 256 in metadata: #width
                            # Access the tag value directly
                            dimension[3] = metadata[256].value
            
                
            if 257 in metadata: #height
                            # Access the tag value directly
                            dimension[4] = metadata[257].value
            
            if 277 in metadata: #channels
                            # Access the tag value directly
                            dimension[5] = int(metadata[277].value)
            if 259 in metadata:  # Tag for slices
                dimension[1] = metadata[259].value

            if 262 in metadata:  # Tag for frames
                frames = metadata[262].value
            
            if 'ImageDescription' in metadata:
                    # Access 'ImageDescription' tag
                    image_description = metadata['ImageDescription']
            
                    # Split the 'ImageDescription' string into lines
                    description_lines = image_description.value.split('\n')
                    # Parse the lines to extract slices and frames information
                    for line in description_lines:
                        if line.startswith("slices="):
                            dimension[1] = int(line.split('=')[1]) #slices
                            dimension[0] = 3
                        if line.startswith("frames="):
                            dimension[2] = int(line.split('=')[1]) #frames
                            dimension[0] += 2
                        if 256 in metadata: #width
                            # Access the tag value directly
                            dimension[3] = metadata[256].value
                        if 257 in metadata: #H
                            # Access the tag value directly
                            dimension[4] = metadata[257].value
                        if 277 in metadata: #channels
                            # Access the tag value directly
                            dimension[5] = metadata[277].value
        else:
                print("ImageDescription tag not found in metadata.")
                        
        print(f"Slices: {dimension[1]}")
        print(f"Frames: {dimension[2]}")
        print(f'Dimension: {dimension[0]}')
        print(f'Width: {dimension[3]}')
        print(f'Height: {dimension[4]}')
        print(f'Channels: {dimension[5]}')
        self.dimension=dimension
        self.SIGNALS.image_shape.emit(dimension)


    def get_state(self):
        data = {'image file path': self.image_filepath}
        return data

    def set_state(self, data, version):
        self.path_chosen(data['image file path'])
        # self.image_filepath = data['image file path']

    def path_chosen(self, file_path):
        self.image_filepath = file_path
        self.update()
    
    def onValue1Changed(self, value):
        print(f"timevalue{value}")
        self.ttval=value-1 #slider: 1-max for biologists
        self.new_img_wrp = CVImage(self.get_img())
        
        if self.session.gui:
            #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)   

        self.set_output_val(0, self.new_img_wrp)
    
    def onValue2Changed(self, value):
        print(f"zvalue{value}")
        self.zzval=value-1
        self.new_img_wrp = CVImage(self.get_img())
        
        if self.session.gui:
                #update continuously 
            self.SIGNALS.new_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def get_img(self):
        return self.image_data[self.ttval,self.zzval,:,:]



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
                print("Image shape:", slicetz.shape)
                print("Image loaded successfully")
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
        return self.input(0).img

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
        self.main_widget().previewState.connect(self.preview)
        
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

    def onSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)
    
    def get_img(self):
        #debug
        # print(f"getimageValue{value}")
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
        self.main_widget().previewState.connect(self.preview)
        
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

    def onSliderValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.kk = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)
    
    def get_img(self):
        #debug
        # print(f"getimageValue{value}")
        return cv2.medianBlur(
            src=self.input(0).img,
            ksize=self.kk,
                )
        
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
        print(self.dimensions)

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
        print(value)
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
        # print(self.vval1)
        # print(self.vval2)
        print("sliceShapeComing")
        image = self.input(0)
        print("sliceShape", image.shape)
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
    title = 'Gaussian Blur'
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
        self.kk = 0
        self.xx = 0
        self.yy = 0
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
        print(value)
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
        # print(self.yy)
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
        print(self.xx)
        print(self.yy)
        return cv2.GaussianBlur(
            src=self.input(0).img,
            ksize=(self.kk, self.kk),
            sigmaX=self.xx,
            sigmaY=self.yy,
        )
    
      # #use when save and close
    def get_state(self) -> dict:
        return {
            'ksize': self.kk,
            'sigmaX': self.xx,
            'sigmaY': self.yy

        }

    def set_state(self, data: dict, version):
        self.kk = data['ksize']
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
        print(value)
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
        # print(self.xx)
        # print(self.kk)
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

class Dilation(NodeBase):        #Nodebase just a different colour
    title = 'dilate'
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
        self.main_widget().Value2Changed.connect(self.ValueChanged2)
        
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

    def ValueChanged1(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def ValueChanged2(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.value_2 = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def get_img(self):
        return cv2.dilate(
            src=self.input(0).img,
            kernel=np.ones((self.value_1,self.value_1),np.uint8),
            iterations=self.value_2 
        )

#//////////////////////////////
# Morphological Transformations

class Morphological_Base(NodeBase):        #Nodebase just a different colour
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
        default1 = 10
        default2 = 100
        self.value_1 = default1  #threshold

    def place_event(self):  
        self.update()

    def view_place_event(self):
        self.SIGNALS.new_img.connect(self.main_widget().show_image)
        self.SIGNALS.clr_img.connect(self.main_widget().clear_img)
        self.main_widget().previewState.connect(self.preview)
        self.main_widget().Value1Changed.connect(self.ValueChanged)
        
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

    def ValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.value_1 = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def get_img(self):
        return cv2.morphologyEx(
            src=self.input(0).img,
            op=self.morph_type,
            kernel=np.ones((self.value_1,self.value_1),np.uint8),
        )
    
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

class Threshold_Base(NodeBase2):        #Nodebase just a different colour
    version = 'v0.1'
    init_inputs = [
        
        NodeInputBP('input img'),
         
    ]
    init_outputs = [
        NodeOutputBP('output img'), #img

    ]
    main_widget_class = widgets.Threshold_MainWidget
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
        self.main_widget().mvValueChanged.connect(self.onMvvalueChanged)   
        
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

    def ontValueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        # print(value)
        self.thr = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def onMvvalueChanged(self, value):
        # This method will be called whenever the widget's signal is emitted
        self.mv = value
        self.new_img_wrp = CVImage(self.get_img())
        if self.prev == True:
            if self.session.gui:
                #update continuously 
                self.SIGNALS.new_img.emit(self.new_img_wrp.img)
        else:
            self.SIGNALS.clr_img.emit(self.new_img_wrp.img)
      
        self.set_output_val(0, self.new_img_wrp)

    def get_img(self):
        if len(self.input(0).img.shape) == 2:
                # Grayscale image
                img_gray = self.input(0).img
        else:
            img_gray = cv2.cvtColor(self.input(0).img, cv2.COLOR_BGR2GRAY)
        ret, result = cv2.threshold(
            src=img_gray,
            thresh=self.thr,
            maxval=self.mv,
            type=self.thresh_type,
        )
        return result    
        
    # #use when save and close
    # def get_state(self) -> dict:
    #     return {
    #         'val': self.val,
    #     }

    # def set_state(self, data: dict, version):
    #     self.val = data['val']

class ThresholdBinary(Threshold_Base):
    title = 'Binary Threshold'
    thresh_type = cv2.THRESH_BINARY


class ThresholdOtsu(Threshold_Base):
    title = 'Otsu Binarization'
    thresh_type = cv2.THRESH_OTSU       


    
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
        print("update event")
        print(f"Old:{self.previous_checkbox }")
        self.previous_checkbox = self.input(2)
        print(f"New:{self.previous_checkbox }")

        # if self.input(2) != self.previous_checkbox:
        #     print("Checkbox not checked")
        
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
        if self.input(2) != self.previous_checkbox:
            print("Checkbox just checked")
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
        # print(k)
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
    #     # print(k)
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
    ReadImageTiff,
    DisplayImg,
    Dimension_Management,
    # Filtering 
    Blur_Averaging,
    Median_Blur,
    Gaussian_Blur,
    Bilateral_Filtering,
    Dilation,
    Opening,
    Closing,
    TopHat,
    Morph_Gradient,
    BlackHat,
    # Contrast Enhancemnet 
    # Binatization 
    ThresholdOtsu,
    ThresholdBinary,
    # Post Binatization 
    
]
