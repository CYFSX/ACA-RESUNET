import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter


class Conv2d_r(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=not use_batchnorm
        )
        if kernel_size == 1:
            use_batchnorm = False
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, nn.ReLU(inplace=True))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DMS(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        num_scales = len(scales)
        self.scale_convs = nn.ModuleList()

        for scale in scales:
            if scale == 1:
                conv = Conv2d_r(in_channels, out_channels // num_scales, 3, padding=1)
            else:
                conv = Conv2d_r(in_channels, out_channels // num_scales, 3,
                                padding=scale, dilation=scale)
            self.scale_convs.append(conv)

        self.fusion_conv = Conv2d_r(out_channels, out_channels, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        feats = [conv(x) for conv in self.scale_convs]
        fused = torch.cat(feats, dim=1)
        fused = self.fusion_conv(fused)
        att = self.attention(fused)
        return self.dropout(fused * att)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp = nn.ModuleList([
            Conv2d_r(in_channels, out_channels, 1),
            *[Conv2d_r(in_channels, out_channels, 3, padding=d, dilation=d) for d in dilations[1:]]
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d_r(in_channels, out_channels, 1)
        )
        self.output = Conv2d_r(out_channels * 5, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        size = x.shape[2:]
        res = [m(x) for m in self.aspp]
        res.append(F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=False))
        out = torch.cat(res, dim=1)
        return self.dropout(self.output(out))


class GFF(nn.Module):
    def __init__(self, feature_channels, output_channels):
        super().__init__()
        compressed = max(32, output_channels // len(feature_channels))
        self.processors = nn.ModuleList([
            nn.Sequential(
                Conv2d_r(c, compressed, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ) for c in feature_channels
        ])

        total = compressed * len(feature_channels)
        self.fusion = nn.Sequential(
            nn.Linear(total, output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_channels, output_channels)
        )

        s_ch = max(8, output_channels // 16)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(output_channels, s_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s_ch, 1, 1),
            nn.Sigmoid()
        )

        c_red = max(8, output_channels // 32)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels, c_red, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_red, output_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        processed = [proc(f) for f, proc in zip(features, self.processors)]
        global_vec = self.fusion(torch.cat(processed, dim=1))

        _, _, h, w = features[-1].shape
        b = global_vec.shape[0]
        g_map = global_vec.view(b, -1, 1, 1).expand(-1, -1, h, w)

        g_map = g_map * self.spatial_att(g_map)
        g_map = g_map * self.channel_att(g_map)
        return g_map


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d_r(in_channels + skip_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2d_r(out_channels, out_channels, 3, padding=1)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.dropout(x)


class UNetResNet50(nn.Module):
    def __init__(self, n_channels=3, n_classes=16, pretrained=True):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)
        self.encoder = IntermediateLayerGetter(backbone, {
            'layer1': 'enc1',
            'layer2': 'enc2',
            'layer3': 'enc3',
            'layer4': 'enc4'
        })

        self.initial_conv = nn.Sequential(
            Conv2d_r(n_channels, 64, 7, padding=3),
            Conv2d_r(64, 64, 3, padding=1)
        )

        self.aspp = ASPP(2048, 512)

        self.dms1 = DMS(256, 256)
        self.dms2 = DMS(512, 512)
        self.dms3 = DMS(1024, 1024)

        self.gff = GFF([64, 256, 512, 1024, 2048], 256)

        self.decoder4 = DecoderBlock(512 + 256, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        self.final = nn.Sequential(
            Conv2d_r(64, 32, 3, padding=1),
            nn.Conv2d(32, n_classes, 1)
        )

        self.aux_head4 = nn.Conv2d(512, n_classes, 1)
        self.aux_head3 = nn.Conv2d(256, n_classes, 1)
        self.aux_head2 = nn.Conv2d(128, n_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = x.shape[2:]
        x0 = self.initial_conv(x)
        feats = self.encoder(x)
        enc1, enc2, enc3, enc4 = feats['enc1'], feats['enc2'], feats['enc3'], feats['enc4']

        gff_out = self.gff([x0, enc1, enc2, enc3, enc4])
        aspp_out = self.aspp(enc4)
        x = torch.cat([aspp_out, gff_out], dim=1)

        x = self.decoder4(x, self.dms3(enc3))
        aux4 = self.aux_head4(x)

        x = self.decoder3(x, self.dms2(enc2))
        aux3 = self.aux_head3(x)

        x = self.decoder2(x, self.dms1(enc1))
        aux2 = self.aux_head2(x)

        x = self.decoder1(x, x0)
        x = self.final(x)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        aux4 = F.interpolate(aux4, size=input_size, mode='bilinear', align_corners=False)
        aux3 = F.interpolate(aux3, size=input_size, mode='bilinear', align_corners=False)
        aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)

        return (x, aux4, aux3, aux2) if self.training else x

    def freeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.initial_conv.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True