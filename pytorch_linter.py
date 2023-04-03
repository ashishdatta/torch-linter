from __future__ import annotations

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class DDPImportChecker(BaseChecker):
    __implements__ = IAstroidChecker
    __torch_distributed_used__ = False

    name = "ddp-import"
    priority = -1
    msgs = {
            "C4401": (
                "torch DistributedDataParallel is commonly imported as DDP",
                "ddp-import-uncommon-name",
                "Import DistributedDataParrallel as DDP",
                ),
            "C4402": (
                "Torch DataParallel is not recommended for single machine multi-GPU anymore",
                "data-parallel-import",
                "It is recommended to use DataDistributedParallel rather than DataParallel",
                ),
            "C4403": (
                "torch.distributed is import and used but not checked if available",
                "torch-distributed-is-available",
                "It is recommended to use check if distributed is available",
                ),
            }
    options = {}
    def visit_import(self, module: astroid.Import):
        pass

    def visit_module(self, module: astroid.Module):
        _torch_distributed_used = False
        check_if_torch_dist_is_available = False
        for node in module.body:
            if isinstance(node, astroid.nodes.Import):
                for name, _ in node.names:
                    if name == "torch.distributed":
                        _torch_distributed_used = True
            if _torch_distributed_used == True:
                if isinstance(node, astroid.nodes.FunctionDef):
                    for n in node.body:
                        if isinstance(n, astroid.nodes.Expr):
                            if n.value.func.attrname != "is_available":
                                check_if_torch_dist_is_available = False
                            elif n.value.func.attrname == "is_available":
                                check_if_torch_dist_is_available = True
        if check_if_torch_dist_is_available == False:
            self.add_message("torch-distributed-is-available", node=node)


    def visit_importfrom(self, node:astroid.ImportFrom):
        if node.modname == "torch.nn.parallel":
            for name, alias in node.names:
                if name == "DistributedDataParallel":
                    if alias != "DDP":
                        self.add_message("ddp-import-uncommon-name", node=node)
                elif name == "DataParallel":
                    self.add_message("data-parallel-import", node=node)

def register(linter: PyLinter) -> None:
    """This required method auto registers the checker during initialization.
    :param linter: The linter to register the checker to.
    """
    linter.register_checker(DDPImportChecker(linter))
