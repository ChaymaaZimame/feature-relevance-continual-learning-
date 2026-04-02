import torch

def relevance_to_factor(scale: torch.Tensor, beta: float, eps: float = 1e-6):
    # scale ist ein Tensor mit broadcasteten Relevanzwerten 
    # für genau einen Parameter-Layer
    # Standardmäßig ist der Faktor 0 
    # (d.h. alle Parameter bekommen Faktor 0 und werden nicht aktualisiert)
    # Parameter mit Relevanz < beta bekommen später Gradienten * 0
    # und werden dadurch nicht aktualisiert    #factor = torch.ones_like(scale)
    factor = torch.zeros_like(scale)

    # gleichzeitig über Tensoren scale und factor iterieren
    for s, f in zip(scale, factor):
        # nur wenn scale (broadcasteter Relevanzwert) >= beta 
        # (ab diesem Beta-Wert gilt ein Neuron bzw. Parameter (nach Broadcasting) als besonder relevant)
        
        if s >= beta:
            # berechne Faktorwert (wird dann mit Gradienten multipliziert)
            new_value = 1.0 / (s + eps)
            #new_value = 0
            #new_value= -0.2
            #new_value = -1
            #new_value = -0.1 * (s / (s + eps))
            #new_value = -s / (s + 1.0)
            #new_value = (beta/s) - 1.0
            #new_value = beta/s
            #new_value= 1.0 / (1 + s)
            #new_value = 1.0 - 1.2 * (s / (s + eps))
            #new_value = - 1 / (s + 1)
            #new_value = - 5 * beta / (s + eps)
            # Wert in Tensor schreiben
            f.copy_(new_value)  

    return factor
"""

def relevance_to_factor(scale: torch.Tensor, beta: float, eps: float = 1e-6):
    factor = torch.zeros_like(scale)
    mask = scale >= beta
    factor[mask] = 1.0 / (scale[mask] + eps)
    return factor
"""

class LRPScaler:
    '''
    diese Klasse skaliert Gradienten von Parametern basierend auf LRP-Relevanzen
    die Relevanzen werden in Faktoren umgewandelt, die dann mit den Gradienten multipliziert werden
    indem Hooks an den Parametern registriert werden die die Gradienten modifizieren
    '''
    def __init__(self, beta=None, eps=1e-6):
        self.beta = beta
        self.eps = eps

        self.handles = []
        # dict um Faktoren zu speichern die mit Parametergradienten im hook multipliziert werden
        # key: param_name, value: factor Tensor 
        # (Faktor heißt dass der brodcastete Relevanzwert in Faktor umgewandelt wird, d.h. durch 1/(Relevanzwert + eps))
        self.factor_dict = {}

    def update_scales(self, scale_dict):
        """
        bekommt: param_name -> scale Tensor
        macht: param_name -> factor Tensor
        """
        self.factor_dict = {}

        for pname, scale in scale_dict.items():
            factor = relevance_to_factor(scale, self.beta, self.eps)
            self.factor_dict[pname] = factor

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def param_hook(self, param_name):
        # für jeden Parameter seinene eigenen Hook
        # hook wird von pytorch automatsich im Backward-Pass aufgerufen, nachdem der Parametergradient berechnet wurde
        def hook(grad):
            # falls Parameter nicht in Faktoren-dict, Meldung! Gradienten bleibt unverändert
            if param_name not in self.factor_dict:
                print(f"Für {param_name} wurde kein Faktor im factor_dict gefunden, Gradienten bleibt unverändert")
                return grad
            
            # Passender Faktor zum Parameter holen
            factor = self.factor_dict[param_name]

            return grad * factor

        return hook

    def register_all_params(self, model):
        # hängt Hooks an alle trainierbaren Parameter des Modells
        self.remove()

        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue

            handle = p.register_hook(self.param_hook(pname))
            self.handles.append(handle)