import sys
from pathlib import Path

import torch
import torch.nn as nn

# LRP repo path hinzufügen
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LRP_REPO_ROOT = PROJECT_ROOT / "lrp_repo"
sys.path.append(str(LRP_REPO_ROOT))

from models import OneWayResNet
from src.lrp import LRPModel

from methods.mapping_param_lrp import PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE


class LRPRunner:
    def __init__(self, model, device=None, skip_connection_prop="flows_skip"):
        self.model = model
        self.device = device
        self.skip_connection_prop = skip_connection_prop

        self.model.to(self.device)
        self.model.eval()

        # ResNet Kern von ResNet Klasse holen
        resnet_core = self.model.model
        
        # OneWay Modell bauen wie von LRPModel (Klasse von lrp_repo) erwartet
        self.oneway_model = OneWayResNet(resnet_core).to(self.device)
        self.oneway_model.eval()

        # Instanz von LRPModel Klasse bauen
        self.lrp_model = LRPModel(
            model=self.oneway_model,
            skip_connection_prop=self.skip_connection_prop,
        ).to(self.device)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # LRP-Modell nach dem Laden neu bauen
        resnet_core = self.model.model
        
        self.oneway_model = OneWayResNet(resnet_core)
        self.oneway_model.to(self.device)
        self.oneway_model.eval()

        self.lrp_model = LRPModel(
            model=self.oneway_model,
            skip_connection_prop=self.skip_connection_prop,
        )
        self.lrp_model.to(self.device)

    def get_layer_relevances(self, images, topk=1):
        images = images.to(self.device)

        relevances, names = self.lrp_model.forward_per_layer(images, topk=topk)

        # alle vom LRP-Code gelieferten Relevanzen sammeln
        all_rel = {}
        for name, rel in zip(names, relevances):
            if isinstance(rel, torch.Tensor):
                all_rel[name] = rel

        # nur die LRP-Layer aus dem Mapping behalten
        allowed_rel_layers = set()
        for _, (rel_layer_name, _) in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE.items():
            allowed_rel_layers.add(rel_layer_name)

        relevance_dict = {}
        for rel_layer_name in allowed_rel_layers:
            if rel_layer_name in all_rel:
                relevance_dict[rel_layer_name] = all_rel[rel_layer_name]

        return relevance_dict
    
    def reduce_to_per_output_neuron(self, R):
        # Conv / BN: (B, C, H, W) -> (C,)
        if R.dim() == 4: 
            return R.mean(dim=(0, 2, 3))

        # Linear: (B, C) -> (C,)
        if R.dim() == 2:
            return R.mean(dim=0)

        raise ValueError(f"Unsupported relevance shape: {tuple(R.shape)}")

    def scale_for_grad(self, relevance, layertype):
        
        if layertype == "bn" or layertype == "linear_b":
            scale = relevance
            
        elif layertype == "linear_w":
            scale = relevance.view(-1,1)
            
        elif layertype == "conv":
            scale = relevance.view(-1,1,1,1)
        
        else:
            raise ValueError(f"Unknown layertype: {layertype}")
        
        return scale
    
    # direkt Scales pro Parameter-Layer bauen
    # ----------------------------------------
    def get_param_scales(self, images, topk=1):
        rel_dict = self.get_layer_relevances(images, topk=topk)

        scale_dict = {}

        for param_layer_name in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE:
            rel_layer_name, layertype = PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE[param_layer_name]
           
            if rel_layer_name not in rel_dict:
                continue

            R = rel_dict[rel_layer_name]

            # wird gemittelt
            s = self.reduce_to_per_output_neuron(R)

            # wird gebroadcastet vorbereitet
            scale = self.scale_for_grad(s, layertype)

            scale_dict[param_layer_name] = scale

        return scale_dict
    """
    # printet Relevanzen vor mitteln und broadcasten per LRP-Layer
    def print_relevances(self, relevance_dict):
        print("Anzahl Layer:", len(relevance_dict))
        print("=" * 80)

        for name, rel in relevance_dict.items():
            print(name)
            print("Shape:", rel.shape)
            print("Min:", rel.min().item())
            print("Max:", rel.max().item())
            print("Mean:", rel.mean().item())
            print("-" * 80)
    """
            
    # printet Relevanzen inkl. mitteln und broadcasten per Parameter-Layer
    def print_param_scales(self, images, topk=1):
        rel_dict = self.get_layer_relevances(images, topk=topk)

        print("Anzahl Parameter-Layer:", len(PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE))
        print("=" * 80)

        for param_layer_name in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE:
            rel_layer_name, layertype = PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE[param_layer_name]
            
            if rel_layer_name not in rel_dict:
                continue

            R = rel_dict[rel_layer_name]
            s = self.reduce_to_per_output_neuron(R)
            scale = self.scale_for_grad(s, layertype)

            print("Param-Layer:", param_layer_name)
            print("LRP-Layer:", rel_layer_name)
            print("Layertype:", layertype)
            print("R shape:", R.shape)
            print("R Min:", R.min().item())
            print("R Max:", R.max().item())
            print("R Mean:", R.mean().item())
            #print("-" * 80)
            print("reduced shape (after mean):", s.shape)
            print("broadcasted shape:", scale.shape)
            #print("broadcasted relevance:")
            #print(scale)
            print("-" * 80)