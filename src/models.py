"""Backbone for ForgeAD: ResNet-18 with a 3-way forge-classification head.

Classes: 0 = normal, 1 = cutpaste-forged, 2 = scar-forged.
`features()` exposes the 512-d pooled embedding (used by the Gaussian
density scorer) and `feature_map()` the pre-pool dense map (used for
localization).
"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ForgeNet(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True,
                 freeze_until: str = "layer2"):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=w)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = net.avgpool
        self.head = nn.Linear(net.fc.in_features, num_classes)

        # Documented CPU-speed deviation from the CutPaste paper (which
        # fine-tunes everything): early layers stay frozen, the network
        # learns forge discrimination in layer3/layer4 + head.
        _order = ["stem", "layer1", "layer2", "layer3", "layer4"]
        frozen = set(_order[: _order.index(freeze_until) + 1]) if freeze_until else set()
        for name, mod in [("stem", self.stem), ("layer1", self.layer1),
                          ("layer2", self.layer2), ("layer3", self.layer3),
                          ("layer4", self.layer4)]:
            if name in frozen:
                for p in mod.parameters():
                    p.requires_grad_(False)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Dense (B, 512, H', W') map before pooling."""
        return self._backbone(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled 512-d embedding."""
        fmap = self._backbone(x)
        return torch.flatten(self.pool(fmap), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
