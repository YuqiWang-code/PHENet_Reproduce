from modeling.backbone import mobilenet_3f


def build_backbone(backbone, output_stride, BatchNorm, in_channels=3):
    if backbone == "mobilenet_3f":
        return mobilenet_3f.MobileNetV2(output_stride, BatchNorm, in_channels=in_channels)
    raise ValueError(f"PHENet only supports the paper backbone 'mobilenet_3f', got {backbone!r}")
