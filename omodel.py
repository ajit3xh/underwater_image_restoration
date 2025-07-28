import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CNN_ViT_UnderwaterRestorer(nn.Module):
    def __init__(self):
        super(CNN_ViT_UnderwaterRestorer, self).__init__()

        # CNN for local structure
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        # ViT for global context
        self.vit = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        self.vit.head = nn.Identity()

        self.vit_decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224 * 224 * 3),
        )

        # Fusion of CNN + ViT
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, padding=1),
        )

        # Color Correction Module (CCM)
        self.color_correction = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),  # Learnable channel mixing
            nn.Tanh()  # Allow negative shifts in color space
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)  # [B, 3, 224, 224]

        vit_feat = self.vit(x)  # [B, 768]
        vit_out = self.vit_decoder(vit_feat).view(-1, 3, 224, 224)

        fused = torch.cat([cnn_feat, vit_out], dim=1)
        out = self.fusion(fused)  # [B, 3, 224, 224]

        out = self.color_correction(out)  # 💡 Apply CCM
        out = (out + 1) / 2  # [0,1] range for image output

        return out
