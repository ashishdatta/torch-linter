import astroid
import ast
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker


class PyTorchDataParallelChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = 'pytorch-data-parallel'

    priority = -1

    msgs = {
        'C9010': ('Consider using DistributedDataParallel with PyTorch for multi-GPU training instead of DataParallel.',
                  'distributed-data-parallel',
                  'Consider using DistributedDataParallel with PyTorch for multi-GPU training to speed up training time and improve efficiency.'
                  ),
        'C9011': ('Avoid using DataParallel with PyTorch for multi-GPU training.',
                    'avoid-data-parallel',
                    'Avoid using DataParallel with PyTorch for multi-GPU training to speed up training time and improve efficiency. Use DistributedDataParallel instead.',
                 ),
    }

    options = {}
    def visit_call(self, node: astroid.Call):
        if isinstance(node.func, astroid.Attribute) and node.func.attrname == 'DataParallel':# and node.func.expr.name == 'torch.nn':
            self.add_message('distributed-data-parallel', node=node)

    def visit_importfrom(self, node: astroid.ImportFrom):
        if node.modname == 'torch.nn.parallel' and 'DataParallel' in [n for n, _ in node.names]:
            self.add_message('avoid-data-parallel', node=node)

class PyTorchDDPConfigChecker(BaseChecker):
    __implements__ = IAstroidChecker
    name = 'pytorch-ddp-config'
    priority = -1
    msgs = {
        'C9015': ('Consider setting the device ID for each process when using PyTorch DistributedDataParallel.',
                  'ddp-device-id',
                  'Consider setting the device ID for each process when using PyTorch DistributedDataParallel to ensure that each process is using a different GPU.'),
        'C9012': ('Consider setting the number of worker processes for PyTorch DistributedDataParallel.',
                  'ddp-worker-count',
                  'Consider setting the number of worker processes for PyTorch DistributedDataParallel to maximize performance.'),
        'C9013': ('Consider setting the backend for PyTorch DistributedDataParallel.',
                  'ddp-backend',
                  'Consider setting the backend for PyTorch DistributedDataParallel to use the best available backend for your hardware.'),
        'C9014': ('Consider using PyTorch\'s native init process for DistributedDataParallel.',
                  'ddp-init-method',
                  'Consider using PyTorch\'s native init process for DistributedDataParallel to ensure that all processes are initialized correctly.'),
        'C9016': ('Consider setting the rank and world size for PyTorch DistributedDataParallel.',
                    'ddp-rank-world-size',
                    'Consider setting the rank and world size for PyTorch DistributedDataParallel to ensure that each process is using a different GPU.'),
        'W9017': ('Consider calling dist.destroy_process_group() after training.',
                    'ddp-destroy-process-group-not-called',
                    'Consider calling dist.destroy_process_group() after training to ensure that all processes are cleaned up properly.'),
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        self._init_process_group_lineno = 0

    def visit_call(self, node: astroid.Call):
        """
        Check for calls to torch.distributed.init_process_group() and torch.nn.parallel.DistributedDataParallel()
        world size and rank are required for DistributedDataParallel
        device_ids is required for DistributedDataParallel
        num_workers is required for DistributedDataParallel
        backend is required for DistributedDataParallel
        init_method is required for DistributedDataParallel
        """
        if isinstance(node.func, astroid.Attribute) and node.func.attrname == 'init_process_group':
            self._init_process_group_lineno = node.lineno
            if 'device_ids' not in [arg.arg for arg in node.keywords]:
                self.add_message('ddp-device-id', node=node)
            if 'backend' not in [arg.arg for arg in node.keywords]:
                self.add_message('ddp-backend', node=node)
            if 'init_method' not in [arg.arg for arg in node.keywords]:
                self.add_message('ddp-init-method', node=node)
            if 'rank' not in [arg.name for arg in node.args] and 'world_size' not in [arg.name for arg in node.args]:
                self.add_message('ddp-rank-world-size', node=node)

        if isinstance(node.func, astroid.Attribute) and node.func.attrname == 'DistributedDataParallel' and node.func.expr.name == 'torch.nn.parallel':
            if 'device_ids' not in [arg.arg for arg in node.keywords]:
                self.add_message('ddp-device-id', node=node)
            if 'num_workers' not in [arg.arg for arg in node.keywords]:
                self.add_message('ddp-worker-count', node=node)
    
    
    def leave_module(self, node):
        """After leaving a module, check if a corresponding destroy_process_group() call was made after init_process_group() call was made.
        """
        for child in node.body:#.get_children():
            found_destroy_process_group = False
            if isinstance(child, astroid.FunctionDef):
                for subchild in child.body:
                    if isinstance(subchild, astroid.Expr) and \
                        isinstance(subchild.value, astroid.Call) and \
                        isinstance(subchild.value.func, astroid.Attribute) and \
                        subchild.value.func.attrname == 'destroy_process_group' and \
                        isinstance(subchild.value.func.expr, astroid.Name) and \
                        subchild.value.func.expr.name == 'dist' and \
                        self._init_process_group_lineno != 0:
                        found_destroy_process_group = True
                        break
                if not found_destroy_process_group:
                    self.add_message('ddp-destroy-process-group-not-called', node=node)

def register(linter):
    linter.register_checker(PyTorchDataParallelChecker(linter))
    linter.register_checker(PyTorchDDPConfigChecker(linter))