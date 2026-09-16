"""CIFAR ResNet / plain nets (He et al. 2015, Section 4.2) re-implementation."""
import torch, torch.nn as nn, torch.nn.functional as F


class ShortcutA(nn.Module):
    """Option A: parameter-free shortcut; subsample + zero-pad channels."""
    def __init__(self, stride, cin, cout, impl="stride2_slice_then_zero_pad", placement="append_end"):
        super().__init__()
        self.stride, self.cin, self.cout = stride, cin, cout
        self.impl, self.placement = impl, placement
        if impl == "projection_1x1" and (stride != 1 or cin != cout):
            self.proj = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(cout)
        else:
            self.proj = None

    def forward(self, x):
        if self.proj is not None:
            return self.bn(self.proj(x))
        if self.stride != 1:
            if self.impl == "avgpool_then_zero_pad":
                x = F.avg_pool2d(x, self.stride, self.stride)
            else:
                x = x[:, :, ::self.stride, ::self.stride]
        pad = self.cout - self.cin
        if pad > 0:
            if self.placement == "split_front_back":
                a = pad // 2
                x = F.pad(x, (0, 0, 0, 0, a, pad - a))
            else:
                x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x


class Block(nn.Module):
    def __init__(self, cin, cout, stride, residual, cfg):
        super().__init__()
        bias = cfg["conv_bias"] == "bias"
        bn_kw = cfg["bn_kw"]
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(cout, **bn_kw)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(cout, **bn_kw)
        self.residual = residual
        self.short = ShortcutA(stride, cin, cout, cfg["shortcut_option_a_impl"],
                               cfg["shortcut_zero_pad_placement"]) if residual else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.short(x)
        return F.relu(out)  # second nonlinearity after the addition


class CifarNet(nn.Module):
    def __init__(self, depth, residual, cfg, num_classes=10, width=16):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        bias = cfg["conv_bias"] == "bias"
        self.conv1 = nn.Conv2d(3, width, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(width, **cfg["bn_kw"])
        layers = []
        cin = width
        for i, c in enumerate([width, width * 2, width * 4]):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(Block(cin, c, stride, residual, cfg))
                cin = c
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)
        self._init(cfg["init"])

    def _init(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == "glorot":
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out" if mode == "he_normal_fan_out" else "fan_in",
                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)
