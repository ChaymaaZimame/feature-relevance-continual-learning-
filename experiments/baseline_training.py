from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import random
import numpy as np
import os
from datetime import datetime

from configs_experiments.config_baseline import get_cfg_defaults
from model.resnet18 import ResNet18
from data.dataset import ImageWoofDataset
from experiments.early_stopping import EarlyStopper
from experiments.trainer import Trainer


####### Utils
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


####### Main
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Config
    cfg = get_cfg_defaults()
    cfg.merge_from_file("configs_experiments/config_baseline.yaml")
    cfg.freeze()
    print(cfg)

    os.makedirs(cfg.OUTPUT.DIR, exist_ok=True)

    # Device + Seed
    device = get_device()
    print(f"Using device: {device}")

    seed_everything(cfg.TRAIN.SEED)
    print(f"Set random seed to {cfg.TRAIN.SEED}")

    g = torch.Generator()
    g.manual_seed(cfg.TRAIN.SEED)

    # TensorBoard
    tb_dir = os.path.join(cfg.OUTPUT.DIR, "tensorboard", timestamp)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard logs: {tb_dir}")
    
    # Model
    model = ResNet18(num_classes=len(cfg.DATA.CLASS_INDICES))
    model = model.to(device)

    # Data only 7 classes
    train_data = ImageWoofDataset(
        img_path=cfg.DATA.ROOT,
        split="train",
        class_indices=cfg.DATA.CLASS_INDICES
    )

    val_data = ImageWoofDataset(
        img_path=cfg.DATA.ROOT,
        split="val",
        class_indices=cfg.DATA.CLASS_INDICES
    )

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Loss / Optimizer / Scheduler / EarlyStopping
    loss_fn = nn.CrossEntropyLoss()

    optimizer = SGD(
        model.parameters(),
        lr=cfg.OPTIMIZER.LR,
        momentum=cfg.OPTIMIZER.MOMENTUM,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
    )

    total_steps = cfg.TRAIN.EPOCHS * len(train_loader)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.SCHEDULER.MIN_LR
    )

    early_stopper = EarlyStopper(
        patience=cfg.EARLY_STOPPING.PATIENCE,
        threshold=cfg.EARLY_STOPPING.THRESHOLD,
        mode=cfg.EARLY_STOPPING.MODE
    )

    # Trainer for Baseline without importance/lrp
    trainer = Trainer(
        cfg=cfg,
        model=model,
        device=device,
        writer=writer,
        early_stopper=early_stopper,
    )

    # Training
    train_losses, val_losses = trainer.update_model(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        log_every=10,
    )

    # Loss plot
    loss_plot_path = os.path.join(
        cfg.OUTPUT.DIR,
        f"{cfg.OUTPUT.LOSS_PLOT_NAME}_{timestamp}.png"
    )
    trainer.plot_losses(train_losses, val_losses, loss_plot_path)

    # load best model
    trainer.load_best_model()

    # Evaluation of best model on the validation dataset
    trainer.evaluate_best_model(
        val_loader=val_loader,
        output_dir=cfg.OUTPUT.DIR,
        confusion_matrix_name=cfg.OUTPUT.CONFUSION_MATRIX_NAME,
        labels=cfg.DATA.CLASS_INDICES,
    )