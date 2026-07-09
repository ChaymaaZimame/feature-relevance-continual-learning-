import sys
from pathlib import Path

import torch


# LRP repo path hinzufügen
# __file__ = aktueller Dateipfad (methods/lrp_runner_new_new.py)
# resolve() = absoluter Pfad (/home/zimame/ma-chaymaa-zimame/methods/lrp_runner_new_new.py)
# parents[1] = zwei Ebenen nach oben, also Projekt-Root (/home/zimame/ma-chaymaa-zimame)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pfad zu LRP-Repository
LRP_REPO_ROOT = PROJECT_ROOT / "lrp_repo"

# Damit ich Python Module aus dem lrp-repo importieren kann
sys.path.append(str(LRP_REPO_ROOT))

from models import OneWayResNet
from src.lrp import LRPModel

from methods.mapping_param_lrp import PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE


class LRPRunner:

    def __init__(self, model, device=None, skip_connection_prop="flows_skip"):
        # LRP Runner is initialized with the model, device, and skip_connection_prop (rule for handling skip connections in LRP)
        self.model = model
        self.device = device
        self.skip_connection_prop = skip_connection_prop

        self.model.to(self.device)
        self.model.eval()

        resnet_core = self.model.model
        self.oneway_model = OneWayResNet(resnet_core).to(self.device)
        self.oneway_model.eval()
        self.lrp_model = LRPModel(
            model=self.oneway_model,
            skip_connection_prop=self.skip_connection_prop,
        ).to(self.device)

    # this function gets one batch of images and computes the LRP relevances per layer
    # topk=1 means that only the most strongly predicted class is explained 
    # The output is a relevance_dict with rel_layer_name -> relevance tensor
    def get_layer_relevances(self, images, topk=1):        
        images = images.to(self.device)

        relevances, names = self.lrp_model.forward_per_layer(images, topk=topk)

        all_rel = {}
        for name, rel in zip(names, relevances):
            if isinstance(rel, torch.Tensor):
                all_rel[name] = rel

        allowed_rel_layers = []

        for param_layer, (rel_layer_name, layer_type) in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE.items():
            allowed_rel_layers.append(rel_layer_name)
        
        relevance_dict = {}
        for rel_layer_name in allowed_rel_layers:
            if rel_layer_name in all_rel:
                relevance_dict[rel_layer_name] = all_rel[rel_layer_name]

        return relevance_dict

    def reduce_to_per_output_neuron(self, R):
             
        # only the magnitude of the relevance matters, because both negative and positive relevance should be considered as "strong"
        R = R.abs()

        # conv/bn2d
        if R.dim() == 4:
            return R.sum(dim=(2, 3)).mean(dim=0)

        # linear/bn1d
        if R.dim() == 2:
            return R.mean(dim=0)

        raise ValueError(f"Unsupported relevance shape: {R.shape}")

    def normalize(self, s, eps=1e-6):
        return s / (s.mean() + eps)

    def scale_for_grad(self, relevance, layertype):
        
        if layertype in ("bn", "linear_b"):
            return relevance

        if layertype == "linear_w":
            return relevance.view(-1, 1)

        if layertype == "conv":
            return relevance.view(-1, 1, 1, 1)

        raise ValueError(f"Unknown layertype: {layertype}")

    # this function returns a dictionary mapping parameter layer names to their corresponding relevance scales, which were computed only for one batch of images
    def get_param_scales(self, images, topk=1, normalize=True):
        
        rel_dict = self.get_layer_relevances(images, topk=topk)

        scale_dict = {}

        for param_layer_name, (rel_layer_name, layertype) in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE.items():
            if rel_layer_name not in rel_dict:
                print(f"Relevanz-Layer '{rel_layer_name}' für Parameter '{param_layer_name}' wurde nicht berechnet")
                continue

            R = rel_dict[rel_layer_name]

            s = self.reduce_to_per_output_neuron(R)

            if normalize:
                s = self.normalize(s)   

            scale = self.scale_for_grad(s, layertype)

            scale_dict[param_layer_name] = scale

        return scale_dict