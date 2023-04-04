from pytorch_linter.checkers.pytorch_ddp_checker import PyTorchDDPConfigChecker

def register(linter):
    linter.register_checker(PyTorchDDPConfigChecker(linter))