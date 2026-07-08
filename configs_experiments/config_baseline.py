from yacs.config import CfgNode as CN

_C = CN()

# System settings
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cuda"

# Data settings
_C.DATA = CN()
_C.DATA.ROOT = "/home/zimame/data/imagewoof2"
_C.DATA.CLASS_INDICES = None  

# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = "resnet18"

# Training settings
_C.TRAIN = CN()
_C.TRAIN.SEED = 42
_C.TRAIN.EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_WORKERS = 4


# Optimizer settings
_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.MOMENTUM = 0.0
_C.OPTIMIZER.WEIGHT_DECAY = 0.0


# Scheduler settings
_C.SCHEDULER = CN()
_C.SCHEDULER.USE = True
_C.SCHEDULER.PATIENCE = 2 # in case i use ReduceLROnPlateau
_C.SCHEDULER.FACTOR = 0.5 # in case i use ReduceLROnPlateau
_C.SCHEDULER.MIN_LR = 1e-6

# Early stopping settings
_C.EARLY_STOPPING = CN()
_C.EARLY_STOPPING.USE = True
_C.EARLY_STOPPING.PATIENCE = 3
_C.EARLY_STOPPING.THRESHOLD = 0.0
_C.EARLY_STOPPING.MODE = "min"
_C.EARLY_STOPPING.BEST_OUTPUT_NAME = "best_model"

# Output settings
_C.OUTPUT = CN()
_C.OUTPUT.NAME = "baseline"
_C.OUTPUT.DIR = "results/baseline"

# Plotting
_C.OUTPUT.LOSS_PLOT_NAME = "loss_curve_baseline"

# confusion matrix
_C.OUTPUT.CONFUSION_MATRIX_NAME = "confusion_matrix_baseline.png"

def get_cfg_defaults():
    return _C.clone()
