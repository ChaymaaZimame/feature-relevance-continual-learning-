import torch
from methods.neuron_importance_calculator import NeuronImportanceCalculator


def test_neuron_importance_calculator_linear():
    # delta für EMA
    delta = 0.99
    imp_calculator = NeuronImportanceCalculator(delta=delta)

    # Fake-Daten
    # Shape = [B=2, C=3]
    activation = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], requires_grad=True)

    grad = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])

    activation.grad = grad

    # alter importance-Wert 
    old_importance = torch.tensor([2.1, 1.2, 0.5])
    
    # linear layer für Test
    layer_name = "test_layer"

    # activation speichern
    activations_dict = imp_calculator.activations
    activations_dict[layer_name] = activation

    # importance speichern
    importance_dict = imp_calculator.neuron_importance
    importance_dict[layer_name] = old_importance.clone()

    ########
    print("\n--- Multiplikation testen---")
    # Shape bleibt gleich (B,C)
    product = activation * grad
    print("product:")
    print(product)
    
    ########
    print("\n--- abs + mean testen ---")
    # Shape ändert sich da mean über B (C,)
    # also hier Shape (3,)
    raw_imp = product.abs().mean(dim=0)
    print("raw importance:", raw_imp)

   
    #########
    print("\n--- Normalisierung testen ---")
    # Shape bleibt gleich (C,) also (3,)

    norm = raw_imp / raw_imp.mean()
    print("normalized:", norm)

    #########
    print("\n--- EMA testen ---")
    # Shape bleibt gleich (C,) also (3,)

    expected = delta * old_importance + (1 - delta) * norm
    print("expected EMA:", expected)

    #########
    # NeuronImportanceCalculator Klasse laufen lassen
    print("\n--- NeuronImportanceCalculator Klasse ---")

    imp_calculator.update_importance()
    result = imp_calculator.neuron_importance["test_layer"]

    print("result:", result)

    #########
    # Vergleich
    print("\n--- Vergleich ---")

    if torch.allclose(result, expected, atol=1e-6):
        print("Test bestanden!")
    else:
        print("Test fehlgeschlagen!")
        print("Erwartet:", expected)
        print("Bekommen:", result)
        
    #########
    # Broadcasting testen
    print("\n--- Broadcasting Linear weight ---")
    # param_grad hat Shape (C_out, C_in)
    # importance hat Shape (C_out,)
    # mit view(-1,1) wird importance automatisch auf (C_out, C_in) gebroadcastet
    
    scale_w = imp_calculator.scale_importance_for_grad(importance=result, layertype="linear_w")
    # enthalten alle noch 1 weil
    # ich noch nicht die echte param.grad.shape einsetze
    # sondern nur einen vorbereitenden Skalierungstensor baue
    assert scale_w.shape == (3,1)
    print(f"linear_w:", scale_w)
    print("Linear_w-Shape korrekt!")

    
    print("\n--- Broadcasting Linear bias ---")
    # param_grad hat Shape (C_out,)
    # importance hat Shape (C_out,)
    # keine Änderung in NeuronImportanceCalculator Klasse

    scale_b = imp_calculator.scale_importance_for_grad(importance=result, layertype="linear_b")
    assert scale_b.shape == (3,)
    print("Linear_b-Shape korrekt!")
    
        
    



import torch
from methods.neuron_importance_calculator import NeuronImportanceCalculator


def test_neuron_importance_calculator_conv():
    # delta für EMA
    delta = 0.99
    imp_calculator = NeuronImportanceCalculator(delta=delta)

    # Fake-Daten:
    # Shape = [Batch=1, Channels=2, Height=2, Width=2]
    activation = torch.tensor([
        [
            [[1.0, 2.0],
             [3.0, 4.0]],

            [[5.0, 6.0],
             [7.0, 8.0]]
        ]
    ], dtype=torch.float32, requires_grad=True)

    grad = torch.tensor([
        [
            [[0.1, 0.2],
             [0.3, 0.4]],

            [[0.5, 0.6],
             [0.7, 0.8]]
        ]
    ], dtype=torch.float32)

    activation.grad = grad

    # alter importance-Wert
    old_importance = torch.tensor([2.1, 1.2], dtype=torch.float32)

    # conv layer für Test
    layer_name = "test_conv_layer"

    # activation speichern
    activations_dict = imp_calculator.activations
    activations_dict[layer_name] = activation

    # importance speichern
    importance_dict = imp_calculator.neuron_importance
    importance_dict[layer_name] = old_importance.clone()

    ########
    print("\n--- Multiplikation testen ---")
    # -> Shape bleibt (B, C, H, W)
    
    product = activation * grad
    print("product:")
    print(product)

    ########
    print("\n--- abs + mean testen ---")
    # -> mitteln über B, H und W, reduziert (B, C, H, W) -> (C,)
    raw_imp = product.abs().mean(dim=(0, 2, 3))
    print("raw importance:", raw_imp)

    #########
    print("\n--- Normalisierung testen ---")
    # bleibt (C,)
    norm = raw_imp / raw_imp.mean()
    print("normalized:", norm)

    #########
    print("\n--- EMA testen ---")

    expected = delta * old_importance + (1 - delta) * norm
    print("expected EMA:", expected)

    #########
    print("\n--- NeuronImportanceCalculator Klasse ---")

    imp_calculator.update_importance()
    result = imp_calculator.neuron_importance["test_conv_layer"]

    print("result:", result)

    #########
    print("\n--- Vergleich ---")

    if torch.allclose(result, expected, atol=1e-6):
        print("Test bestanden!")
    else:
        print("Test fehlgeschlagen!")
        print("Erwartet:", expected)
        print("Bekommen:", result)
        
    #########
    # Broadcasting testen
    print("\n--- Broadcasting Conv weight ---")
    # param_grad hat Shape (C_out, C_in, Kernel-Höhe, Kernel-Breite)
    # importance hat Shape (C_out,)
    # mit view(-1,1,1,1) wird importance automatisch auf (C_out, C_in, Kernel-Höhe, Kernel-Breite) gebroadcastet

    scale_w = imp_calculator.scale_importance_for_grad(importance=result, layertype="conv")
    # enthalten noch 1,1,1 weil
    # ich noch nicht die echte param.grad.shape einsetze
    # sondern nur einen vorbereitenden Skalierungstensor baue
    assert scale_w.shape == (2, 1, 1, 1)
    print("Conv_w-Shape korrekt!")
        
    print("\n--- Broadcasting BN ---")
    # param_grad hat Shape (C_out,)
    # importance hat Shape (C_out,)
    # keine Änderung in NeuronImportanceCalculator Klasse
    scale_bn = imp_calculator.scale_importance_for_grad(importance=result, layertype="bn")
    assert scale_bn.shape == (2,)
    print("BN-Shape korrekt!")


if __name__ == "__main__":
    print("\n########## Test für Linear ##########")
    test_neuron_importance_calculator_linear()
    print("\n########## Test für Conv ##########")
    test_neuron_importance_calculator_conv()
