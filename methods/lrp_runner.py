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

# Imports aus dem LRP-Repo
from models import OneWayResNet
# LRPModel (Klasse für die eigentliche Relevanzpropagation)
from src.lrp import LRPModel

# Mapping z.B. "model.layer1.0.conv1.weight" -> ("layer1.0.relu", "conv")
from methods.mapping_param_lrp import PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE


class LRPRunner:
    """
    Diese Klasse berechnet aus einem Modell LRP-Relevanzen
    und wandelt sie in Parameter-Scales um

    das Ziel ist:
    Für jeden trainierbaren Parameter-Layer einen Tensor erzeugen,
    der später benutzt wird, um Param-Gradienten zu skalieren
    """

    def __init__(self, model, device=None, skip_connection_prop="flows_skip"):
        """
        Initialisiert den LRPRunner mit Parametern:
        - model, device und
        - skip_connection_prop (Regel, wie Skip-Connections bei LRP behandelt werden)
        """
        self.model = model
        self.device = device
        self.skip_connection_prop = skip_connection_prop

        # Modell auf das Device verschieben
        self.model.to(self.device)
        # in Eval-Modus:
        self.model.eval()

        # Modell umbauen, damit es von LRP-Klasse (LRPModel) verwendet werden kann
        resnet_core = self.model.model
        self.oneway_model = OneWayResNet(resnet_core).to(self.device)
        self.oneway_model.eval()
        # LRP-Modell aufbauen (berechnet REL pro Layer)
        self.lrp_model = LRPModel(
            model=self.oneway_model,
            skip_connection_prop=self.skip_connection_prop,
        ).to(self.device)


    def get_layer_relevances(self, images, topk=1):
        """
        bekommt 1 Batch von Bildern und berechnet die LRP-Relevanzen pro Layer.
        topk=1 bedeutet, dass nur die am stärksten vorhergesagte Klasse erklärt wird (Relevanz dafür rückpropagiert).
        
        --> Output ist ein relevance_dict mit rel_layer_name -> Relevanztensor
        """
        
        images = images.to(self.device)

        # LRP pro Layer ausführen
        # relevances = Liste von Tensoren
        # names = Liste von dazugehörigen Layernamen
        relevances, names = self.lrp_model.forward_per_layer(images, topk=topk)

        # 1. alle Tensor-Relevanzen in dict all_rel speichern
        all_rel = {}
        for name, rel in zip(names, relevances):
            if isinstance(rel, torch.Tensor):
                all_rel[name] = rel

        # 2. aus Mapping nur die Relevanz-Layernamen sammeln, die für Parameterskalierung gebraucht sind
        """
        allowed_rel_layers = {
            rel_layer_name
            for _, (rel_layer_name, _) in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE.items()
        }
        """
        allowed_rel_layers = []

        for param_layer, (rel_layer_name, layer_type) in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE.items():
            allowed_rel_layers.append(rel_layer_name)
        
        # 3. nur die Relevanzen behalten, die im Mapping tatsächlich verwendet werden
        relevance_dict = {}
        for rel_layer_name in allowed_rel_layers:
            if rel_layer_name in all_rel:
                relevance_dict[rel_layer_name] = all_rel[rel_layer_name]

        return relevance_dict

    def reduce_to_per_output_neuron(self, R):
        """
        Reduziert einen Layer-Relevanztensor auf genau einen Wert
        pro Output-Kanal / Output-Neuron.
        - Betrag nehmen und
        - bei Conv und BN erstm Summer über H, W --> Gesamt-Relevanz eines Kanals
            dann über B mitteln
        - bei Linear Layer über B mitteln

        Warum?
        Für Gradient-Scaling brauchen wir später keinen kompletten
        räumlichen Relevanztensor, sondern nur einen Score pro Kanal.

        Output bei Conv, BN:
        1D-Tensor der Länge C, wobei jeder Wert
        die durchschnittliche Gesamt-Relevanz
        eines Output-Kanals über alle Samples im Batch darstellt.
        
        Output bei Linear:
        1D-Tensor (C,), der pro Output-Neuron 
        die über den Batch gemittelte Relevanz enthält.
        """
        
        # betrag, denn negative und positive Relevanz sollen beide als "stark" zählen
        R = R.abs()

        # Conv / BN
       
        if R.dim() == 4:
            # erst über H und W summieren -> (B, C)
            # dann über B mitteln -> (C,)
            return R.sum(dim=(2, 3)).mean(dim=0)

        # Linear
        if R.dim() == 2:
            # über B mitteln -> (C,)
            return R.mean(dim=0)

        # falls andere Layer-Relevanz-shape --> Fehlermeldung
        raise ValueError(f"Unsupported relevance shape: {R.shape}")

    def normalize(self, s, eps=1e-6):
        """
        Normalisiert die Relevanzen pro Layer.
        """
        return s / (s.mean() + eps)

    def scale_for_grad(self, relevance, layertype):
        """
        Relevanztensoren pro Layer werden umgeformt (broadcastet)
        sodass sie zur Form des jeweiligen Parameter-Tensors passen.
        Erwartet:
        - relevance: 1D-Tensor (C,)
        - layertype:
            "bn", "linear_b", "linear_w", "conv"
        """
        # BN (Gewichte/Bias) oder Linear (Bias):
        # passt schon zur Parameterform (C,)
        if layertype in ("bn", "linear_b"):
            return relevance

        # Linear (Gewichte):
        # Gewicht hat Form (out_features, in_features)
        # ich habe nur einen Wert pro Output-Neuron deswegen
        # -> (C,) zu (C, 1)
        if layertype == "linear_w":
            return relevance.view(-1, 1)

        # Conv (Gewichte):
        # Gewicht hat meist Form (out_channels, in_channels, kernel-H, kernel-W)
        # ich habe nur einen Wert pro Output-Neuron deswegen
        # -> (C,) zu (C, 1, 1, 1)
        if layertype == "conv":
            return relevance.view(-1, 1, 1, 1)

        raise ValueError(f"Unknown layertype: {layertype}")

    def get_param_scales(self, images, topk=1):
        """
        das ist die Hauptfunktion der Klasse.

        1. LRP-Relevanzen pro Layer berechnen
        2. passenden Relevanz-Layer zum Parameter-Layer finden
        3. Relevanz auf einen Wert pro Output-Kanal bzw. Out-Neuron reduzieren
        4. Relevanzwerte pro Layer normalisieren
        5. in Parameterform bringen

        Output: scale_dict mit
            param_layer_name -> Scale-Tensor
        """
        # für 1 batch!
        # Relevanzen der relevanten (d.h. die im Mapping drin sind) LRP-Layer holen
        rel_dict = self.get_layer_relevances(images, topk=topk)

        # dict für die Scales pro Parameter
        scale_dict = {}

        # Durch alle gemappten Parameter-Layer laufen
        for param_layer_name, (rel_layer_name, layertype) in PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE.items():
            # Falls dieser Relevanz-Layer nicht berechnet wurde, Meldung!
            if rel_layer_name not in rel_dict:
                print(f"Relevanz-Layer '{rel_layer_name}' für Parameter '{param_layer_name}' wurde nicht berechnet")
                continue

            # Relevanztensor des zugeordneten LRP-Layers holen
            R = rel_dict[rel_layer_name]

            # Relevanz auf einen Score pro Output-Kanal / Output-Neuron reduzieren
            s = self.reduce_to_per_output_neuron(R)

            # Scores mit Mean-Normalisierung stabilisieren
            s = self.normalize(s)

            # Broadcasting in Parameterform
            scale = self.scale_for_grad(s, layertype)

            # Parametername -> passender Scale-Tensor (brodcasteter Relevanzwert)
            scale_dict[param_layer_name] = scale

        return scale_dict