from __future__ import annotations

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class DDPImportChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = "ddp-import"
    priority = -1
    msgs = {
            "C4401": (
                "torch DistributedDataParallel is commonly imported as DDP",
                "ddp-import-uncommon-name",
                "Import DistributedDataParrallel as DDP",
                ),
            }
    options = {}

    def visit_importfrom(self, node:astroid.ImportFrom):
        if node.modname == "torch.nn.parallel":
            for name, alias in node.names:
                if name == "DistributedDataParallel":
                    if alias != "DDP":
                        self.add_message("ddp-import-uncommon-name", node=node)

def register(linter: PyLinter) -> None:
    """This required method auto registers the checker during initialization.
    :param linter: The linter to register the checker to.
    """
    linter.register_checker(DDPImportChecker(linter))
