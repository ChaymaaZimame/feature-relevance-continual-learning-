
PARAMLAYER_TO_IMPLAYER_AND_LAYERTYPE = {
    # stem
    "model.conv1.weight": ("model.conv1", "conv"),
    "model.bn1.weight":   ("model.bn1", "bn"),
    "model.bn1.bias":     ("model.bn1", "bn"),

    # layer1
    "model.layer1.0.conv1.weight": ("model.layer1.0.conv1", "conv"),
    "model.layer1.0.bn1.weight":   ("model.layer1.0.bn1",   "bn"),
    "model.layer1.0.bn1.bias":     ("model.layer1.0.bn1",   "bn"),
    "model.layer1.0.conv2.weight": ("model.layer1.0.conv2", "conv"),
    "model.layer1.0.bn2.weight":   ("model.layer1.0.bn2",   "bn"),
    "model.layer1.0.bn2.bias":     ("model.layer1.0.bn2",   "bn"),

    "model.layer1.1.conv1.weight": ("model.layer1.1.conv1", "conv"),
    "model.layer1.1.bn1.weight":   ("model.layer1.1.bn1",   "bn"),
    "model.layer1.1.bn1.bias":     ("model.layer1.1.bn1",   "bn"),
    "model.layer1.1.conv2.weight": ("model.layer1.1.conv2", "conv"),
    "model.layer1.1.bn2.weight":   ("model.layer1.1.bn2",   "bn"),
    "model.layer1.1.bn2.bias":     ("model.layer1.1.bn2",   "bn"),

    # layer2
    "model.layer2.0.conv1.weight": ("model.layer2.0.conv1", "conv"),
    "model.layer2.0.bn1.weight":   ("model.layer2.0.bn1",   "bn"),
    "model.layer2.0.bn1.bias":     ("model.layer2.0.bn1",   "bn"),
    "model.layer2.0.conv2.weight": ("model.layer2.0.conv2", "conv"),
    "model.layer2.0.bn2.weight":   ("model.layer2.0.bn2",   "bn"),
    "model.layer2.0.bn2.bias":     ("model.layer2.0.bn2",   "bn"),

    # downsample layer2.0
    # die Trennung zwischen conv und bn in LRP bringt nichts, da exakt dieselben Werte vor/nach BN.
    # denn in diesem repository wird bei BN-Layer die Relevanz unverändert durchgereicht, d.h. die Relevanz vor und nach BN ist gleich, da BN keinen Einfluss auf die Relevanz hat.
    "model.layer2.0.downsample.0.weight": ("model.layer2.0.downsample.0", "conv"),
    "model.layer2.0.downsample.1.weight": ("model.layer2.0.downsample.1", "bn"),
    "model.layer2.0.downsample.1.bias":   ("model.layer2.0.downsample.1", "bn"),

    "model.layer2.1.conv1.weight": ("model.layer2.1.conv1", "conv"),
    "model.layer2.1.bn1.weight":   ("model.layer2.1.bn1",   "bn"),
    "model.layer2.1.bn1.bias":     ("model.layer2.1.bn1",   "bn"),
    "model.layer2.1.conv2.weight": ("model.layer2.1.conv2", "conv"),
    "model.layer2.1.bn2.weight":   ("model.layer2.1.bn2",   "bn"),
    "model.layer2.1.bn2.bias":     ("model.layer2.1.bn2",   "bn"),

    # layer3
    "model.layer3.0.conv1.weight": ("model.layer3.0.conv1", "conv"),
    "model.layer3.0.bn1.weight":   ("model.layer3.0.bn1",   "bn"),
    "model.layer3.0.bn1.bias":     ("model.layer3.0.bn1",   "bn"),
    "model.layer3.0.conv2.weight": ("model.layer3.0.conv2", "conv"),
    "model.layer3.0.bn2.weight":   ("model.layer3.0.bn2",   "bn"),
    "model.layer3.0.bn2.bias":     ("model.layer3.0.bn2",   "bn"),

    # downsample layer3.0
    "model.layer3.0.downsample.0.weight": ("model.layer3.0.downsample.0", "conv"),
    "model.layer3.0.downsample.1.weight": ("model.layer3.0.downsample.1", "bn"),
    "model.layer3.0.downsample.1.bias":   ("model.layer3.0.downsample.1", "bn"),

    "model.layer3.1.conv1.weight": ("model.layer3.1.conv1", "conv"),
    "model.layer3.1.bn1.weight":   ("model.layer3.1.bn1",   "bn"),
    "model.layer3.1.bn1.bias":     ("model.layer3.1.bn1",   "bn"),
    "model.layer3.1.conv2.weight": ("model.layer3.1.conv2", "conv"),
    "model.layer3.1.bn2.weight":   ("model.layer3.1.bn2",   "bn"),
    "model.layer3.1.bn2.bias":     ("model.layer3.1.bn2",   "bn"),

    # layer4
    "model.layer4.0.conv1.weight": ("model.layer4.0.conv1", "conv"),
    "model.layer4.0.bn1.weight":   ("model.layer4.0.bn1",   "bn"),
    "model.layer4.0.bn1.bias":     ("model.layer4.0.bn1",   "bn"),
    "model.layer4.0.conv2.weight": ("model.layer4.0.conv2", "conv"),
    "model.layer4.0.bn2.weight":   ("model.layer4.0.bn2",   "bn"),
    "model.layer4.0.bn2.bias":     ("model.layer4.0.bn2",   "bn"),

    # downsample layer4.0
    "model.layer4.0.downsample.0.weight": ("model.layer4.0.downsample.0", "conv"),
    "model.layer4.0.downsample.1.weight": ("model.layer4.0.downsample.1", "bn"),
    "model.layer4.0.downsample.1.bias":   ("model.layer4.0.downsample.1", "bn"),

    "model.layer4.1.conv1.weight": ("model.layer4.1.conv1", "conv"),
    "model.layer4.1.bn1.weight":   ("model.layer4.1.bn1",   "bn"),
    "model.layer4.1.bn1.bias":     ("model.layer4.1.bn1",   "bn"),
    "model.layer4.1.conv2.weight": ("model.layer4.1.conv2", "conv"),
    "model.layer4.1.bn2.weight":   ("model.layer4.1.bn2",   "bn"),
    "model.layer4.1.bn2.bias":     ("model.layer4.1.bn2",   "bn"),

    # classifier
    "model.fc.weight": ("model.fc", "linear_w"),
    "model.fc.bias":   ("model.fc", "linear_b"),
}
