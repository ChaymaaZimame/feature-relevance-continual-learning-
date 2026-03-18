
PARAMLAYER_TO_RELLAYER_AND_LAYERTYPE = {
    # in dem LRP-Repository wird bei BN-Layers die Relevanz unverändert durchgereicht, d.h. die Relevanz vor und nach BN ist gleich, da BN keinen Einfluss auf die Relevanz hat.
    # stem
    "model.conv1.weight": ("RelevancePropagationConv2d.out.14", "conv"),
    "model.bn1.weight":   ("RelevancePropagationBatchNorm2d.out.13", "bn"),
    "model.bn1.bias":     ("RelevancePropagationBatchNorm2d.out.13", "bn"),

    # layer1
    "model.layer1.0.conv1.weight": ("layer1.0.conv1_out", "conv"),
    "model.layer1.0.bn1.weight":   ("layer1.0.bn1_out",   "bn"),
    "model.layer1.0.bn1.bias":     ("layer1.0.bn1_out",   "bn"),
    "model.layer1.0.conv2.weight": ("layer1.0.conv2_out", "conv"),
    "model.layer1.0.bn2.weight":   ("layer1.0.bn2_out",   "bn"),
    "model.layer1.0.bn2.bias":     ("layer1.0.bn2_out",   "bn"),

    "model.layer1.1.conv1.weight": ("layer1.1.conv1_out", "conv"),
    "model.layer1.1.bn1.weight":   ("layer1.1.bn1_out",   "bn"),
    "model.layer1.1.bn1.bias":     ("layer1.1.bn1_out",   "bn"),
    "model.layer1.1.conv2.weight": ("layer1.1.conv2_out", "conv"),
    "model.layer1.1.bn2.weight":   ("layer1.1.bn2_out",   "bn"),
    "model.layer1.1.bn2.bias":     ("layer1.1.bn2_out",   "bn"),

    # layer2
    "model.layer2.0.conv1.weight": ("layer2.0.conv1_out", "conv"),
    "model.layer2.0.bn1.weight":   ("layer2.0.bn1_out",   "bn"),
    "model.layer2.0.bn1.bias":     ("layer2.0.bn1_out",   "bn"),
    "model.layer2.0.conv2.weight": ("layer2.0.conv2_out", "conv"),
    "model.layer2.0.bn2.weight":   ("layer2.0.bn2_out",   "bn"),
    "model.layer2.0.bn2.bias":     ("layer2.0.bn2_out",   "bn"),

    # downsample layer2.0
    # die Trennung zwischen conv und bn in LRP bringt nichts, da exakt dieselben Werte vor/nach BN.
    # denn in diesem repository wird bei BN-Layer die Relevanz unverändert durchgereicht, d.h. die Relevanz vor und nach BN ist gleich, da BN keinen Einfluss auf die Relevanz hat.
    "model.layer2.0.downsample.0.weight": ("layer2.0.shortcut_out", "conv"),
    "model.layer2.0.downsample.1.weight": ("layer2.0.shortcut_out", "bn"),
    "model.layer2.0.downsample.1.bias":   ("layer2.0.shortcut_out", "bn"),

    "model.layer2.1.conv1.weight": ("layer2.1.conv1_out", "conv"),
    "model.layer2.1.bn1.weight":   ("layer2.1.bn1_out",   "bn"),
    "model.layer2.1.bn1.bias":     ("layer2.1.bn1_out",   "bn"),
    "model.layer2.1.conv2.weight": ("layer2.1.conv2_out", "conv"),
    "model.layer2.1.bn2.weight":   ("layer2.1.bn2_out",   "bn"),
    "model.layer2.1.bn2.bias":     ("layer2.1.bn2_out",   "bn"),

    # layer3
    "model.layer3.0.conv1.weight": ("layer3.0.conv1_out", "conv"),
    "model.layer3.0.bn1.weight":   ("layer3.0.bn1_out",   "bn"),
    "model.layer3.0.bn1.bias":     ("layer3.0.bn1_out",   "bn"),
    "model.layer3.0.conv2.weight": ("layer3.0.conv2_out", "conv"),
    "model.layer3.0.bn2.weight":   ("layer3.0.bn2_out",   "bn"),
    "model.layer3.0.bn2.bias":     ("layer3.0.bn2_out",   "bn"),

    # downsample layer3.0
    "model.layer3.0.downsample.0.weight": ("layer3.0.shortcut_out", "conv"),
    "model.layer3.0.downsample.1.weight": ("layer3.0.shortcut_out", "bn"),
    "model.layer3.0.downsample.1.bias":   ("layer3.0.shortcut_out", "bn"),

    "model.layer3.1.conv1.weight": ("layer3.1.conv1_out", "conv"),
    "model.layer3.1.bn1.weight":   ("layer3.1.bn1_out",   "bn"),
    "model.layer3.1.bn1.bias":     ("layer3.1.bn1_out",   "bn"),
    "model.layer3.1.conv2.weight": ("layer3.1.conv2_out", "conv"),
    "model.layer3.1.bn2.weight":   ("layer3.1.bn2_out",   "bn"),
    "model.layer3.1.bn2.bias":     ("layer3.1.bn2_out",   "bn"),

    # layer4
    "model.layer4.0.conv1.weight": ("layer4.0.conv1_out", "conv"),
    "model.layer4.0.bn1.weight":   ("layer4.0.bn1_out",   "bn"),
    "model.layer4.0.bn1.bias":     ("layer4.0.bn1_out",   "bn"),
    "model.layer4.0.conv2.weight": ("layer4.0.conv2_out", "conv"),
    "model.layer4.0.bn2.weight":   ("layer4.0.bn2_out",   "bn"),
    "model.layer4.0.bn2.bias":     ("layer4.0.bn2_out",   "bn"),

    # downsample layer4.0
    "model.layer4.0.downsample.0.weight": ("layer4.0.shortcut_out", "conv"),
    "model.layer4.0.downsample.1.weight": ("layer4.0.shortcut_out", "bn"),
    "model.layer4.0.downsample.1.bias":   ("layer4.0.shortcut_out", "bn"),

    "model.layer4.1.conv1.weight": ("layer4.1.conv1_out", "conv"),
    "model.layer4.1.bn1.weight":   ("layer4.1.bn1_out",   "bn"),
    "model.layer4.1.bn1.bias":     ("layer4.1.bn1_out",   "bn"),
    "model.layer4.1.conv2.weight": ("layer4.1.conv2_out", "conv"),
    "model.layer4.1.bn2.weight":   ("layer4.1.bn2_out",   "bn"),
    "model.layer4.1.bn2.bias":     ("layer4.1.bn2_out",   "bn"),

    # classifier
    "model.fc.weight": ("RelevancePropagationLinear.out.0", "linear_w"),
    "model.fc.bias":   ("RelevancePropagationLinear.out.0", "linear_b"),
}
