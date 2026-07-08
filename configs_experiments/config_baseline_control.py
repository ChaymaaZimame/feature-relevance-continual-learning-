from yacs.config import CfgNode as CN

_C = CN()

# System settings
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = "cuda"

# Data settings
_C.DATA = CN()
_C.DATA.ROOT = "/home/zimame/data/imagewoof2"
_C.DATA.CLASS_INDICES_OLD = None
_C.DATA.CLASS_INDICES_NEW = None
_C.DATA.CLASS_INDICES_ALL = None  


# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = "resnet18"
_C.MODEL.OLD_CHECKPOINT= "results/baseline/best_model_baseline.pth"

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
_C.EARLY_STOPPING.BEST_OUTPUT_NAME = "best_model_baseline_control"

_C.LRP = CN()
_C.LRP.MODE = None
_C.LRP.BETA = None
_C.LRP.NORMALIZE = False
_C.LRP.BETA_QUANTILE = None
_C.LRP.BETA_SCOPE = None  # "global" oder "local"

_C.IMPORTANCE = CN()
_C.IMPORTANCE.MODE = None  # "once", "epoch" oder "batch"
_C.IMPORTANCE.DELTA = None
_C.IMPORTANCE.BETA = None
_C.IMPORTANCE.BETA_QUANTILE = None
_C.IMPORTANCE.BETA_SCOPE = None  # "global" oder "local"

# Output settings
_C.OUTPUT = CN()
_C.OUTPUT.NAME = "baseline_control"
_C.OUTPUT.DIR = "results/baseline_control"

# Neuron Importance
_C.OUTPUT.NEURON_IMPORTANCE = "importance_values_control.pt"

# LRP Relevance
_C.OUTPUT.NEURON_RELEVANCE = "relevance_values_control.pt"

# Plotting
_C.OUTPUT.LOSS_PLOT_NEW = "loss_curve_new_control"
_C.OUTPUT.LOSS_PLOT_ALL = "loss_curve_all_control"
_C.OUTPUT.LOSS_PLOT_OLD_NEW_ALL = "loss_curve_old_new_all_control"


# Confusion matrices
_C.OUTPUT.CM_NAME_OLD = "confusion_matrix_old_control"
_C.OUTPUT.CM_NAME_NEW = "confusion_matrix_new_control"
_C.OUTPUT.CM_NAME_ALL = "confusion_matrix_all_control"

def get_cfg_defaults():
    return _C.clone()
