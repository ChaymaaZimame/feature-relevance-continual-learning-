import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1) 
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

# method to plot the distribution of scales (importance or LRP) and their quantiles
""" THIS FUNCTION WAS IMPLEMENTED WITH AI ASSISTANCE """
def plot_scale_distribution(all_vals, title, xlabel, out_path):
    q50 = np.quantile(all_vals, 0.50)
    q75 = np.quantile(all_vals, 0.75)
    q90 = np.quantile(all_vals, 0.90)
    q95 = np.quantile(all_vals, 0.95)
    q99 = np.quantile(all_vals, 0.99)

    print(f"\nGLOBAL {title}:")
    print(f"min={all_vals.min():.6f}")
    print(f"max={all_vals.max():.6f}")
    print(f"mean={all_vals.mean():.6f}")
    print(f"std={all_vals.std():.6f}")

    print(f"\n{title} QUANTILE:")
    print(f"q50 (50%) = {q50:.6f}")
    print(f"q75 (75%) = {q75:.6f}")
    print(f"q90 (90%) = {q90:.6f}")
    print(f"q95 (95%) = {q95:.6f}")
    print(f"q99 (99%) = {q99:.6f}")

    plt.figure(figsize=(10, 6))
    plt.hist(all_vals, bins=300, edgecolor="orange", alpha=0.7)

    plt.axvline(q50, color="blue", linestyle="--", linewidth=2, label=f"q50 = {q50:.4f}")
    plt.axvline(q75, color="green", linestyle="--", linewidth=2, label=f"q75 = {q75:.4f}")
    plt.axvline(q90, color="red", linestyle="--", linewidth=2, label=f"q90 = {q90:.4f}")
    plt.axvline(q95, color="purple", linestyle="--", linewidth=2, label=f"q95 = {q95:.4f}")
    plt.axvline(q99, color="black", linestyle="--", linewidth=2, label=f"q99 = {q99:.4f}")

    plt.xlabel(xlabel)
    plt.ylabel("Frequenz")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved scale histogram with quantiles to: {out_path}")


class Trainer:
    def __init__(
        self,
        cfg,
        model,
        device,
        writer=None,
        early_stopper=None,
        neuron_imp_calc=None, 
        importance_scaler=None,
        lrp_runner=None,
        lrp_scaler=None,
    ):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.writer = writer
        self.early_stopper = early_stopper

        self.neuron_imp_calc = neuron_imp_calc
        self.importance_scaler = importance_scaler

        self.lrp_runner = lrp_runner
        self.lrp_scaler = lrp_scaler

        self.global_lrp_beta = None
        self.global_importance_beta = None

    # helper function to log scalars to TensorBoard
    def log_scalar(self, name, value, step):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
    
    # helper function to log histograms and imp/rel scales to TensorBoard
    """ THIS FUNCTION WAS IMPLEMENTED WITH AI ASSISTANCE """
    def log_scale_stats(self, scales_dict, step, prefix):
        if self.writer is None:
            return

        all_vals = []
        backbone_vals = []
        fc_vals = []

        for name, scale in scales_dict.items():
            vals = scale.detach().flatten().cpu().numpy()
            all_vals.append(vals)

            if "fc" in name:
                fc_vals.append(vals)
            else:
                backbone_vals.append(vals)

        def log_group(group_name, vals_list):
            if len(vals_list) == 0:
                return

            vals = np.concatenate(vals_list)

            self.writer.add_histogram(f"{prefix}/{group_name}_hist", vals, step)
            self.writer.add_scalar(f"{prefix}/{group_name}_min", vals.min(), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_max", vals.max(), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_mean", vals.mean(), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_std", vals.std(), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_q50", np.quantile(vals, 0.50), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_q90", np.quantile(vals, 0.90), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_q95", np.quantile(vals, 0.95), step)
            self.writer.add_scalar(f"{prefix}/{group_name}_q99", np.quantile(vals, 0.99), step)

        log_group("all", all_vals)
        log_group("backbone", backbone_vals)
        log_group("fc", fc_vals)

    # helper function to log factor statistics (zero/non-zero counts and ratios) to TensorBoard
    """ THIS FUNCTION WAS IMPLEMENTED WITH AI ASSISTANCE """
    def log_factor_stats(self, factor_dict, step, prefix):
        if self.writer is None or factor_dict is None:
            return

        groups = {"all": [], "backbone": [], "fc": []}

        for name, factor in factor_dict.items():
            vals = factor.detach().flatten().cpu().numpy()
            groups["all"].append(vals)

            if "fc" in name:
                groups["fc"].append(vals)
            else:
                groups["backbone"].append(vals)

        def log_group(group_name, vals_list):
            if len(vals_list) == 0:
                return

            vals = np.concatenate(vals_list)
            total = vals.size
            zero_count = np.sum(vals == 0)
            nonzero_count = np.sum(vals != 0)

            self.writer.add_scalar(f"{prefix}/{group_name}_zero_count", zero_count, step)
            self.writer.add_scalar(f"{prefix}/{group_name}_nonzero_count", nonzero_count, step)
            self.writer.add_scalar(f"{prefix}/{group_name}_zero_ratio", zero_count / total, step)
            self.writer.add_scalar(f"{prefix}/{group_name}_nonzero_ratio", nonzero_count / total, step)

        log_group("all", groups["all"])
        log_group("backbone", groups["backbone"])
        log_group("fc", groups["fc"])

    # returns the path where the best model will be saved
    def get_best_model_path(self):
        return os.path.join(
            self.cfg.OUTPUT.DIR,
            f"{self.cfg.EARLY_STOPPING.BEST_OUTPUT_NAME}.pth"
        )

    # saves the best model to the previously created path
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    # loads the best model after training for evaluation
    def load_best_model(self):
        best_model_path = self.get_best_model_path()
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded best model from: {best_model_path}")

    # prepares the hooks for importance calculation and scaling
    def prepare_importance(self):
        if self.neuron_imp_calc is None or self.importance_scaler is None:
            raise ValueError("Importance-Komponenten fehlen.")

        # for imp computation
        self.neuron_imp_calc.register_hooks(self.model)
        # for scaling with imp
        self.importance_scaler.register_all_params(self.model)

    # prepares the hooks for relevance scaling
    def prepare_lrp(self):
        if self.lrp_runner is None or self.lrp_scaler is None:
            raise ValueError("LRP-Komponenten fehlen.")
        
        self.lrp_scaler.register_all_params(self.model)
        
    
    # this function calculates the beta value based on the quantile of importance values
    # only for local beta scope (batchwise or epochwise)
    def set_importance_beta_from_quantile(self, scales_dict, q=0.99):
        if self.importance_scaler is None:
            return

        all_vals = []
        
        for scale in scales_dict.values():
            vals = scale.detach().flatten().cpu().numpy()
            all_vals.append(vals)

        if len(all_vals) == 0:
            return
       
        all_vals = np.concatenate(all_vals)
        beta = float(np.quantile(all_vals, q))
        self.importance_scaler.beta = beta

        print(f"Set importance beta dynamically to quantile {q} value: {beta:.6f}")

    # this function calculates the beta value based on the quantile of LRP values
    # only for local beta scope (batchwise or epochwise)
    def set_lrp_beta_from_quantile(self, scales_dict, q=0.99):
        if self.lrp_scaler is None:
            return

        all_vals = []
        for scale in scales_dict.values():
            vals = scale.detach().flatten().cpu().numpy()
            all_vals.append(vals)

        if len(all_vals) == 0:
            return

        all_vals = np.concatenate(all_vals)
        beta = float(np.quantile(all_vals, q))
        self.lrp_scaler.beta = beta

        print(f"Set beta dynamically to quantile {q} value: {beta:.6f}")
        
    # this function applies the importance scales to the model parameters and sets the beta value based on the configuration (local or global)
    def apply_importance_scales(self, scales_dict):
        if self.importance_scaler is None:
            raise ValueError("ImportanceScaler fehlt.")

        # if beta local (batch or epoch)
        if self.cfg.IMPORTANCE.BETA_SCOPE == "local":
            self.set_importance_beta_from_quantile(
                scales_dict,
                q=self.cfg.IMPORTANCE.BETA_QUANTILE
            )
        
        elif self.cfg.IMPORTANCE.BETA_SCOPE == "global":
            if self.global_importance_beta is None:
                raise ValueError("global_importance_beta wurde noch nicht gesetzt.")
            self.importance_scaler.beta = self.global_importance_beta
            # assert that the beta value of the importance scaler is equal to the global importance beta
            assert self.importance_scaler.beta == self.global_importance_beta

        else:
            raise ValueError(f"Unbekannter IMPORTANCE.BETA_SCOPE: {self.cfg.IMPORTANCE.BETA_SCOPE}")
        # apply the scales to the model parameters
        self.importance_scaler.update_scales(scales_dict)

    # this function applies the rel scales to the model parameters and sets the beta value based on the configuration (local or global)
    def apply_lrp_scales(self, scales_dict):
        if self.lrp_scaler is None:
            raise ValueError("LRPScaler fehlt.")

        # if beta local (batch or epoch)
        if self.cfg.LRP.BETA_SCOPE == "local":
            self.set_lrp_beta_from_quantile(
                scales_dict,
                q=self.cfg.LRP.BETA_QUANTILE
            )

        elif self.cfg.LRP.BETA_SCOPE == "global":
            if self.global_lrp_beta is None:
                raise ValueError("global_lrp_beta wurde noch nicht gesetzt.")
            self.lrp_scaler.beta = self.global_lrp_beta
            # assert that the beta value of the lrp scaler is equal to the global lrp beta
            assert self.lrp_scaler.beta == self.global_lrp_beta
            
            # DEBUGGING
            print(
                f"[DEBUG] scope={self.cfg.LRP.BETA_SCOPE} | "
                f"global_beta={self.global_lrp_beta:.6f} | "
                f"scaler_beta={self.lrp_scaler.beta:.6f}"
            )

        else:
            raise ValueError(f"Unbekannter LRP.BETA_SCOPE: {self.cfg.LRP.BETA_SCOPE}")

        print(
            "[LRP DEBUG]",
            "scope=", self.cfg.LRP.BETA_SCOPE,
            "beta=", self.lrp_scaler.beta
        )

        self.lrp_scaler.update_scales(scales_dict)

    # this function prints the statistics of the scales (importance or LRP) for each parameter layer
    def print_layer_scale_stats(self, scales_dict, title):
        print(f"\n=== {title} ===")

        all_vals = []

        for param_name, scale in scales_dict.items():
            vals = scale.detach().flatten().cpu().numpy()
            all_vals.append(vals)

            print(
                f"{param_name}: shape={tuple(scale.shape)}, "
                f"min={vals.min():.6f}, max={vals.max():.6f}, "
                f"mean={vals.mean():.6f}, std={vals.std():.6f}"
            )

            # all values for fc layers are printed for debugging
            if "fc" in param_name:
                print(f"\n--- ALLE WERTE für {param_name} ---")
                print(vals)

        return all_vals


    # this function computes the importance scales over the entire dataset and returns the mean scales for each parameter layer
    # it is only called in "epoch" or "once" importance mode to compute and analyze the importance scales over the entire dataset
    def compute_importance_scales(self, data_loader, loss_fn, epoch=None):
        if self.neuron_imp_calc is None or self.importance_scaler is None:
            raise ValueError("Importance-Komponenten fehlen.")

        self.model.eval()

        sum_scales = {}
        total_samples = 0

        for images, labels in data_loader:
            # forward and backward pass to compute gradients and activations
            # since they are needed for importance calculation
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            
            # zero the gradients before the backward pass 
            # to avoid accumulation of gradients from previous batches
            # because importance values are computed independently for each batch
            self.model.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = loss_fn(logits, labels)
            loss.backward()

            # it calls the update_importance method of the NeuronImportanceCalculator 
            # to compute the importance values for each layer based on the activations and their gradients
            self.neuron_imp_calc.update_importance()
            # it calls the get_param_scales method of the NeuronImportanceCalculator 
            # to get the importance scales for each parameter layer
            batch_scales = self.neuron_imp_calc.get_param_scales()

            # here the importance scales for each parameter layer are accumulated over all batches
            for param_name, scale in batch_scales.items():
                # if the parameter layer is not yet in the sum_scales dictionary, it is added with the current scale * by the batch size
                if param_name not in sum_scales:
                    sum_scales[param_name] = scale.detach().clone() * batch_size
                else:
                    sum_scales[param_name] += scale.detach() * batch_size

            total_samples += batch_size
        # here the mean importance scales for each parameter layer are computed 
        # by dividing the accumulated scales by the total number of samples
        mean_scales = {
            param_name: scale_sum / total_samples
            for param_name, scale_sum in sum_scales.items()
        }

        # print the statistics of the mean importance scales for each parameter layer
        all_vals = self.print_layer_scale_stats(
            mean_scales,
            "IMPORTANCE PARAM_SCALES STATISTIK"
        )

        # plot the distribution of the mean importance scales with quantiles
        if len(all_vals) > 0:
            all_vals = np.concatenate(all_vals)

            filename = "importance_scale_distribution_with_quantiles"
            if epoch is not None:
                filename += f"_epoch_{epoch + 1}"
            filename += ".png"

            out_path = os.path.join(self.cfg.OUTPUT.DIR, filename)

            plot_scale_distribution(
                all_vals=all_vals,
                title="IMPORTANCE PARAM_SCALES",
                xlabel="Wichtigkeitswert (Importance Scale)",
                out_path=out_path,
            )

        return mean_scales
    
    # this function computes the LRP scales over the entire dataset and returns the mean scales for each parameter layer
    # it is only called in "epoch" or "once" LRP mode to compute and analyze the LRP scales over the entire dataset
    def compute_lrp_scales(self, data_loader, topk=1, normalize=True, epoch=None):
        if self.lrp_runner is None or self.lrp_scaler is None:
            raise ValueError("LRP-Komponenten fehlen.")

        self.model.eval()

        sum_scales = {}
        total_samples = 0

        for images, _ in data_loader:
            # iterate over the dataset and compute the LRP scales for each batch
            images = images.to(self.device)
            batch_size = images.size(0)

            batch_scales = self.lrp_runner.get_param_scales(
                images,
                topk=topk,
                normalize=normalize
            )

            # here the relevance scales for each parameter layer are accumulated over all batches
            for param_name, scale in batch_scales.items():
                # if the parameter layer is not yet in the sum_scales dictionary, it is added with the current scale * by the batch size
                if param_name not in sum_scales:
                    sum_scales[param_name] = scale.detach().clone() * batch_size
                else:
                    sum_scales[param_name] += scale.detach() * batch_size

            total_samples += batch_size
        # here the mean relevance scales for each parameter layer are computed 
        # by dividing the accumulated scales by the total number of samples
        mean_scales = {
            param_name: scale_sum / total_samples
            for param_name, scale_sum in sum_scales.items()
        }
        
        # print the statistics of the mean relevance scales for each parameter layer
        all_vals = self.print_layer_scale_stats(
            mean_scales,
            "LRP RELEVANZEN STATISTIK"
        )

        # plot the distribution of the mean relevance scales with quantiles
        if len(all_vals) > 0:
            all_vals = np.concatenate(all_vals)
            
            # log the mean scales to TensorBoard for visualization
            step = 0 if epoch is None else epoch + 1
            self.log_scale_stats(
                mean_scales,
                step=step,
                prefix="LRP_Scales"
            )
                        
            filename = "lrp_scale_distribution_with_quantiles"
            if epoch is not None:
                filename += f"_epoch_{epoch + 1}"
            filename += ".png"

            out_path = os.path.join(self.cfg.OUTPUT.DIR, filename)

            plot_scale_distribution(
                all_vals=all_vals,
                title="LRP RELEVANZEN",
                xlabel="Relevanzwert (LRP Scale)",
                out_path=out_path,
            )

        return mean_scales

    # this function cleans up the hooks and resources used for importance and LRP calculations to avoid memory leaks
    def cleanup(self):
        
        if self.importance_scaler is not None:
            self.importance_scaler.remove()

        if self.lrp_scaler is not None:
            self.lrp_scaler.remove()

        if self.neuron_imp_calc is not None:
            self.neuron_imp_calc.remove()

    # this function collects the predictions and true labels over the entire validation dataset for later analysis of misclassifications
    def collect_predictions(self, val_loader):
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        return y_true, y_pred

    def collect_misclassifications(self, val_loader):
        y_true, y_pred = self.collect_predictions(val_loader)
        wrong = (y_pred != y_true).sum()
        total = len(y_true)
        return wrong, total

    # this function plots the training and validation losses over epochs
    def plot_losses(self, train_losses, val_losses, out_path):
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved loss plot to: {out_path}")

    # this function evaluates the best model on the validation dataset and 
    # prints the confusion matrix, per-class accuracy and classification report
    def evaluate_best_model(self, val_loader, output_dir, confusion_matrix_name, labels=None):
        y_true, y_pred = self.collect_predictions(val_loader)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()

        cm_path = os.path.join(output_dir, confusion_matrix_name)
        plt.savefig(cm_path, dpi=200)
        plt.close()

        print(f"Saved confusion matrix to: {cm_path}")

        print("\nPer-class accuracy / recall:")
        for label in labels:
            mask = y_true == label
            total = mask.sum()
            correct = ((y_true == label) & (y_pred == label)).sum()
            acc = correct / total if total > 0 else 0.0

            print(f"Class {label}: {acc:.3f} ({correct}/{total})")

        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=labels, digits=3))

    # validate is the function that evaluates the model on the validation dataset
    # and returns the average loss and accuracy
    def validate(self, val_loader, loss_fn):
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = loss_fn(logits, labels)

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                total_acc += accuracy_from_logits(logits, labels) * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        return avg_loss, avg_acc

    # train_one_epoch is the function that trains the model for one epoch on the training dataset
    # and returns the average loss and accuracy
    def train_one_epoch(
        self,
        train_loader,
        loss_fn,
        optimizer,
        scheduler=None,
        epoch=0,
        log_every=10,
        lrp_mode=None,
        importance_mode=None,
        lrp_topk=1,
        normalize=True,
        old_fc_weight=None,
        old_fc_bias=None,
        old_num_classes=None,
        ):
        
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # batch lrp
            if lrp_mode == "batch":
                print("[DEBUG] Vor backward: berechne batch_scales")
                batch_scales = self.lrp_runner.get_param_scales(
                    images,
                    topk=lrp_topk,
                    normalize=normalize
                )
                print("[DEBUG] Neue batch_scales berechnet")

                self.apply_lrp_scales(batch_scales)
                
                step = epoch * len(train_loader) + batch_idx
                
                if batch_idx % log_every == 0:
                    self.log_scale_stats(
                        batch_scales,
                        step=step,
                        prefix="LRP_Batch_Scales"
                    )
                    
                    self.log_factor_stats(
                        self.lrp_scaler.factor_dict,
                        step=step,
                        prefix="LRP_Batch_Factors"
                    )
                
                    if self.lrp_scaler.beta is not None:
                        self.log_scalar("LRP/current_beta", self.lrp_scaler.beta, step)
                        
                    if self.global_lrp_beta is not None:
                        self.log_scalar("LRP/beta_global", self.global_lrp_beta, step)
                
                print("[DEBUG] batch_scales angewendet, jetzt kommt zero_grad/forward/backward")

                if batch_idx == 0:
                    all_vals = self.print_layer_scale_stats(
                        batch_scales,
                        f"LRP RELEVANZEN STATISTIK Epoch {epoch + 1}, Batch {batch_idx}"
                    )

                    if len(all_vals) > 0:
                        all_vals = np.concatenate(all_vals)

                        out_path = os.path.join(
                            self.cfg.OUTPUT.DIR,
                            f"lrp_scale_distribution_batch_epoch_{epoch + 1}_batch_{batch_idx}.png"
                        )

                        plot_scale_distribution(
                            all_vals=all_vals,
                            title=f"LRP scales Batch-Modus - Epoch {epoch + 1}, Batch {batch_idx}",
                            xlabel="Relevanzwert (LRP Scale)",
                            out_path=out_path,
                        )

            optimizer.zero_grad()
            print("[DEBUG] zero_grad")

            logits = self.model(images)
            loss = loss_fn(logits, labels)
            print("[DEBUG] forward + loss")
            loss.backward()
            print("[DEBUG] backward fertig")

            #batch importance
            if (
                importance_mode == "batch"
                and self.neuron_imp_calc is not None
                and self.importance_scaler is not None
            ):
                self.neuron_imp_calc.update_importance()
                param_scales = self.neuron_imp_calc.get_param_scales()

                if batch_idx == 0:
                    all_vals = self.print_layer_scale_stats(
                        param_scales,
                        f"PARAM_SCALES STATISTIK Epoch {epoch + 1}, Batch {batch_idx}"
                    )

                    if len(all_vals) > 0:
                        all_vals = np.concatenate(all_vals)

                        out_path = os.path.join(
                            self.cfg.OUTPUT.DIR,
                            f"importance_param_scales_distribution_batch_epoch_{epoch + 1}_batch_{batch_idx}.png"
                        )

                        plot_scale_distribution(
                            all_vals=all_vals,
                            title=f"IMPORTANCE PARAM_SCALES Histogram Epoch {epoch + 1}, Batch {batch_idx}",
                            xlabel="Importance Parameter Scales",
                            out_path=out_path,
                        )

                self.apply_importance_scales(param_scales)
                
                step = epoch * len(train_loader) + batch_idx

                if batch_idx % log_every == 0:
                    self.log_scale_stats(
                        param_scales,
                        step=step,
                        prefix="Importance_Batch_Scales"
                    )

                    self.log_factor_stats(
                        self.importance_scaler.factor_dict,
                        step=step,
                        prefix="Importance_Batch_Factors"
                    )

            optimizer.step()

            # if train_one_epoch is initialized with old_fc_weight (ONLY control_training_frozen.py)
            # then the old weights and biases are copied to the new model's fc layer
            
            if old_fc_weight is not None:
                with torch.no_grad():
                    self.model.model.fc.weight[:old_num_classes].copy_(old_fc_weight)

                    if old_fc_bias is not None and self.model.model.fc.bias is not None:
                        self.model.model.fc.bias[:old_num_classes].copy_(old_fc_bias)

            if scheduler is not None:
                scheduler.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits, labels) * batch_size
            total_samples += batch_size

            if batch_idx % log_every == 0:
                step = epoch * len(train_loader) + batch_idx

                self.log_scalar("Train/Loss", loss.item(), step)
                self.log_scalar("Train/Accuracy", accuracy_from_logits(logits, labels), step)
                self.log_scalar("Train/LR", optimizer.param_groups[0]["lr"], step)

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        return avg_loss, avg_acc

    # update_model is the main function that trains the model over multiple epochs,
    # applies relevance and importance scaling
    # and evaluates the model on the validation dataset
    def update_model(
        self,
        train_loader, # train_loader_new
        val_loader, # val_loader_new
        loss_fn,
        optimizer,
        scheduler=None,
        log_every=10,
        lrp_mode=None,
        importance_mode=None,
        lrp_topk=1,
        normalize=True,
        val_loader_old=None,
        val_loader_all=None,
        old_acc_before=None,   
        old_fc_weight=None,    # only for control_training_frozen.py 
        old_fc_bias=None,      # only for control_training_frozen.py 
        old_num_classes=None,    
    ):
        train_losses = []
        val_losses = []

        best_model_path = self.get_best_model_path()
        
        # compute global beta values for LRP if the scope is set to "global" in the configuration
        if (
            lrp_mode is not None
            and self.lrp_runner is not None
            and self.lrp_scaler is not None
            and self.cfg.LRP.BETA_SCOPE == "global"
        ):
            lrp_scales = self.compute_lrp_scales(
                data_loader=train_loader,
                topk=lrp_topk,
                normalize=normalize,
            )

            all_vals = [
                s.detach().flatten().cpu().numpy()
                for s in lrp_scales.values()
            ]

            self.global_lrp_beta = float(
                np.quantile(
                    np.concatenate(all_vals),
                    self.cfg.LRP.BETA_QUANTILE
                )
            )
            
            #for DEBUGGING
            print(f"[DEBUG] Global beta wurde berechnet: {self.global_lrp_beta:.6f}")

            self.log_scalar("LRP/beta_global", self.global_lrp_beta, 0)
            # once lrp
            if lrp_mode == "once":
                self.apply_lrp_scales(lrp_scales)
            
                self.log_scalar("LRP/current_beta", self.lrp_scaler.beta, 0)
            
                self.log_factor_stats(
                    self.lrp_scaler.factor_dict,
                    step=0,
                    prefix="LRP_Initial_Factors"
                )

            print(f"Global LRP beta: {self.global_lrp_beta:.6f}")
            
        # compute global beta values for importance if the scope is set to "global" in the configuration
        if (
            importance_mode is not None
            and self.neuron_imp_calc is not None
            and self.importance_scaler is not None
            and self.cfg.IMPORTANCE.BETA_SCOPE == "global"
        ):
            imp_scales = self.compute_importance_scales(
                data_loader=train_loader,
                loss_fn=loss_fn,
            )

            all_vals = [
                s.detach().flatten().cpu().numpy()
                for s in imp_scales.values()
            ]

            self.global_importance_beta = float(
                np.quantile(
                    np.concatenate(all_vals),
                    self.cfg.IMPORTANCE.BETA_QUANTILE
                )
            )

            self.log_scale_stats(imp_scales, step=0, prefix="Importance_Initial_Scales")
            
            # once importance
            if importance_mode == "once":
                self.apply_importance_scales(imp_scales)

                self.log_factor_stats(
                    self.importance_scaler.factor_dict,
                    step=0,
                    prefix="Importance_Initial_Factors"
                )

            print(f"Global Importance beta: {self.global_importance_beta:.6f}")

        if lrp_mode == "once" and self.cfg.LRP.BETA_SCOPE != "global":
            print("Berechne LRP einmal vor dem Training ...")

            lrp_scales = self.compute_lrp_scales(
                data_loader=train_loader,
                topk=lrp_topk,
                normalize=normalize,
            )

            self.apply_lrp_scales(lrp_scales)

        if importance_mode == "once" and self.cfg.IMPORTANCE.BETA_SCOPE != "global":
            print("Berechne Importance einmal vor dem Training ...")

            imp_scales = self.compute_importance_scales(
                data_loader=train_loader,
                loss_fn=loss_fn,
            )

            self.apply_importance_scales(imp_scales)

        # main training loop over epochs
        for epoch in range(self.cfg.TRAIN.EPOCHS):
            print(f"Epoch {epoch + 1}/{self.cfg.TRAIN.EPOCHS}")

            # epoch lrp
            if lrp_mode == "epoch":
                print(f"[DEBUG] Epoche {epoch + 1}: berechne neue LRP-Scales")

                lrp_scales = self.compute_lrp_scales(
                    data_loader=train_loader,
                    topk=lrp_topk,
                    normalize=normalize,
                    epoch=epoch,
                )
                
                print(f"[DEBUG] Epoche {epoch + 1}: wende LRP-Scales an")

                self.apply_lrp_scales(lrp_scales)
                
                step = epoch + 1
                
                self.log_factor_stats(
                    self.lrp_scaler.factor_dict,
                    step=step,
                    prefix="LRP_Epoch_Factors"
                )
                
                if self.lrp_scaler is not None and self.lrp_scaler.beta is not None:
                    self.log_scalar("LRP/current_beta", self.lrp_scaler.beta, step)

                if self.global_lrp_beta is not None:
                    self.log_scalar("LRP/beta_global", self.global_lrp_beta, step)
             
            # epoch importance 
            if importance_mode == "epoch":
                print(f"Berechne Importance für Epoche {epoch + 1} ...")

                imp_scales = self.compute_importance_scales(
                    data_loader=train_loader,
                    loss_fn=loss_fn,
                    epoch=epoch,
                )

                self.apply_importance_scales(imp_scales)
                
                step = epoch + 1

                self.log_scale_stats(imp_scales, step=step, prefix="Importance_Epoch_Scales")

                self.log_factor_stats(
                    self.importance_scaler.factor_dict,
                    step=step,
                    prefix="Importance_Epoch_Factors"
                )

            # train model for one epoch and validate
            train_loss, train_acc = self.train_one_epoch(
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                log_every=log_every,
                lrp_mode=lrp_mode,
                importance_mode=importance_mode,
                lrp_topk=lrp_topk,
                normalize=normalize,
                old_fc_weight=old_fc_weight,
                old_fc_bias=old_fc_bias,
                old_num_classes=old_num_classes,
            )

            val_loss, val_acc = self.validate(val_loader, loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

            self.log_scalar("Train/Epoch_Loss", train_loss, epoch)
            self.log_scalar("Train/Epoch_Accuracy", train_acc, epoch)
            self.log_scalar("Val/Epoch_Loss", val_loss, epoch)
            self.log_scalar("Val/Epoch_Accuracy", val_acc, epoch)
            self.log_scalar("Train/Epoch_LR", optimizer.param_groups[0]["lr"], epoch)
            
            if val_loader_old is not None:
                old_loss_epoch, old_acc_epoch = self.validate(val_loader_old, loss_fn)
                self.log_scalar("Val/Old_Loss", old_loss_epoch, epoch)
                self.log_scalar("Val/Old_Accuracy", old_acc_epoch, epoch)

            if val_loader_all is not None:
                all_loss_epoch, all_acc_epoch = self.validate(val_loader_all, loss_fn)
                self.log_scalar("Val/All_Loss", all_loss_epoch, epoch)
                self.log_scalar("Val/All_Accuracy", all_acc_epoch, epoch)

            if (
                val_loader_old is not None
                and val_loader_all is not None
                and old_acc_before is not None
            ):
                knowledge_forgetting_epoch = old_acc_before - old_acc_epoch
                task_forgetting_epoch = old_acc_before - all_acc_epoch

                self.log_scalar("Forgetting/Knowledge", knowledge_forgetting_epoch, epoch)
                self.log_scalar("Forgetting/Task", task_forgetting_epoch, epoch)
            
            if self.early_stopper is not None:
                self.early_stopper.step(val_loss)

                if self.early_stopper.is_best():
                    self.save_model(best_model_path)
                    print(f"[BEST] saved model to {best_model_path}")

                if self.early_stopper.early_stop:
                    print("Early stopping triggered.")
                    break

            else:
                self.save_model(best_model_path)

        if self.writer is not None:
            self.writer.close()

        self.cleanup()

        return train_losses, val_losses