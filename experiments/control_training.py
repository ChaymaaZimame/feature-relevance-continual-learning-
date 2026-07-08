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
import matplotlib.pyplot as plt

from configs_experiments.config_baseline_control import get_cfg_defaults
from model.resnet18 import ResNet18
from data.dataset import ImageWoofDataset
from experiments.early_stopping import EarlyStopper
from experiments.trainer_new import Trainer


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
    cfg.merge_from_file("configs_experiments/config_baseline_control.yaml")
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
    
    ####### 1. load old 7-heads model 
    model7 = ResNet18(num_classes=len(cfg.DATA.CLASS_INDICES_OLD)).to(device)
    model7.load_state_dict(
        torch.load(cfg.MODEL.OLD_CHECKPOINT, map_location=device)
    )
    model7.eval()

    ####### 2. Data
    train_data_new = ImageWoofDataset(
        img_path=cfg.DATA.ROOT,
        split="train",
        class_indices=cfg.DATA.CLASS_INDICES_NEW
    )

    val_data_old = ImageWoofDataset(
        img_path=cfg.DATA.ROOT,
        split="val",
        class_indices=cfg.DATA.CLASS_INDICES_OLD
    )

    val_data_new = ImageWoofDataset(
        img_path=cfg.DATA.ROOT,
        split="val",
        class_indices=cfg.DATA.CLASS_INDICES_NEW
    )

    val_data_all = ImageWoofDataset(
        img_path=cfg.DATA.ROOT,
        split="val",
        class_indices=cfg.DATA.CLASS_INDICES_ALL
    )

    train_loader_new = DataLoader(
        train_data_new,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader_old = DataLoader(
        val_data_old,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader_new = DataLoader(
        val_data_new,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader_all = DataLoader(
        val_data_all,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g
    )

    print("=== DataLoader Info ===")
    print(f"Train dataset NEW: {len(train_loader_new.dataset)}")
    print(f"Val dataset OLD:   {len(val_loader_old.dataset)}")
    print(f"Val dataset NEW:   {len(val_loader_new.dataset)}")
    print(f"Val dataset ALL:   {len(val_loader_all.dataset)}")

    ####### 3. Loss
    loss_fn = nn.CrossEntropyLoss()

    ####### 4. Accuracy of old 7-heads model on old 7 classes before control training
    trainer_old = Trainer(
        cfg=cfg,
        model=model7,
        device=device,
    )

    old_loss_before, old_acc_before = trainer_old.validate(val_loader_old, loss_fn)
    print(f"\nOLD accuracy before control training (7-heads model): {old_acc_before:.4f}")

    ####### 5. expand head to 10 classes 
    model = model7
    model.expand_head(new_num_classes=len(cfg.DATA.CLASS_INDICES_ALL))
    model = model.to(device)
    
    before_training_path = os.path.join(
        cfg.OUTPUT.DIR,
        "model_after_expand_before_training.pth"
    )
    
    torch.save(model.state_dict(), before_training_path)
    print(f"Saved model after expand before training to: {before_training_path}") 
      
    


    ####### 6. Accuracy of 10-heads model before control training
    trainer_before = Trainer(
        cfg=cfg,
        model=model,
        device=device,
    )

    old_loss_before, old_acc_before = trainer_before.validate(val_loader_old, loss_fn)
    new_loss_before, new_acc_before = trainer_before.validate(val_loader_new, loss_fn)
    all_loss_before, all_acc_before = trainer_before.validate(val_loader_all, loss_fn)

    print("\n=== BEFORE TRAINING ON NEW 3 CLASSES ===")
    print(f"\nOLD accuracy before control training (10-heads model): {old_acc_before:.4f}")
    print(f"\nNEW accuracy before control training (10-heads model): {new_acc_before:.4f}")
    print(f"\nALL accuracy before control training (10-heads model): {all_acc_before:.4f}")


    ####### 7. Optimizer / Scheduler / EarlyStopping
    optimizer = SGD(
        model.parameters(),
        lr=cfg.OPTIMIZER.LR,
        momentum=cfg.OPTIMIZER.MOMENTUM,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
    )

    total_steps = cfg.TRAIN.EPOCHS * len(train_loader_new)

    scheduler = None
    if cfg.SCHEDULER.USE:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=cfg.SCHEDULER.MIN_LR
        )

    early_stopper = None
    if cfg.EARLY_STOPPING.USE:
        early_stopper = EarlyStopper(
            patience=cfg.EARLY_STOPPING.PATIENCE,
            threshold=cfg.EARLY_STOPPING.THRESHOLD,
            mode=cfg.EARLY_STOPPING.MODE
        )

    ####### 8. Trainer
    # important: early stopping is based on val_loader_new, not val_loader_old, because we want to stop training when the new classes are learned well, since I nly train on new 3 classes
    trainer = Trainer(
        cfg=cfg,
        model=model,
        device=device,
        writer=writer,
        early_stopper=early_stopper,
    )

    ####### 9. Training on new 3 classes
    train_losses, val_losses_new = trainer.update_model(
        train_loader=train_loader_new,
        val_loader=val_loader_new,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        log_every=10,
        val_loader_old=val_loader_old,
        val_loader_all=val_loader_all,
        old_acc_before=old_acc_before,
    )

    ####### 10. Loss plot new
    loss_plot_new_path = os.path.join(
        cfg.OUTPUT.DIR,
        f"{cfg.OUTPUT.LOSS_PLOT_NEW}_{timestamp}.png"
    )
    trainer.plot_losses(train_losses, val_losses_new, loss_plot_new_path)

    ####### 11. load best model
    trainer.load_best_model()
    

    ####### 12. Evaluation of best model after training
    old_loss_after, old_acc_after = trainer.validate(val_loader_old, loss_fn)
    new_loss_after, new_acc_after = trainer.validate(val_loader_new, loss_fn)
    all_loss_after, all_acc_after = trainer.validate(val_loader_all, loss_fn)

    print("\n=== AFTER CONTROL TRAINING ===")
    print(f"\nOLD accuracy after control training (10-heads model): {old_acc_after:.4f}")
    print(f"\nNEW accuracy after control training (10-heads model): {new_acc_after:.4f}")
    print(f"\nALL accuracy after control training (10-heads model): {all_acc_after:.4f}")

    # Forgetting metrics
    knowledge_forgetting = old_acc_before - old_acc_after
    task_forgetting = old_acc_before - all_acc_after

    print("\n=== FORGETTING METRICS ===")
    print(f"Knowledge Forgetting (old acc drop): {knowledge_forgetting:.4f}")
    print(f"Task Forgetting (old_acc_before - all_acc_after): {task_forgetting:.4f}")

    ####### 13. Confusion Matrices / Reports
    trainer.evaluate_best_model(
        val_loader=val_loader_old,
        output_dir=cfg.OUTPUT.DIR,
        confusion_matrix_name=f"{cfg.OUTPUT.CM_NAME_OLD}_{timestamp}.png",
        labels=cfg.DATA.CLASS_INDICES_OLD,    
    )

    trainer.evaluate_best_model(
        val_loader=val_loader_new,
        output_dir=cfg.OUTPUT.DIR,
        confusion_matrix_name=f"{cfg.OUTPUT.CM_NAME_NEW}_{timestamp}.png",
        labels=cfg.DATA.CLASS_INDICES_NEW,    
    )

    trainer.evaluate_best_model(
        val_loader=val_loader_all,
        output_dir=cfg.OUTPUT.DIR,
        confusion_matrix_name=f"{cfg.OUTPUT.CM_NAME_ALL}_{timestamp}.png",
        labels=cfg.DATA.CLASS_INDICES_ALL,    
    )