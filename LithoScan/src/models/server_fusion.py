import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class LithoScanServerFusion(nn.Module):
    """
    Server Edition: Texture-aware CNN + Transformer fusion model.
    Supports modes: baseline, cnn_transformer, cnn_only, transformer_ablation
    """
    def __init__(
        self,
        num_classes=7,
        architecture_mode="cnn_transformer",
        pretrained=True,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.architecture_mode = architecture_mode
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 1. Backbone
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_resnet = resnet50(weights=weights)
        
        # For baseline mode, we use the full resnet except the final fc
        if architecture_mode == "baseline":
            self.backbone = nn.Sequential(*list(base_resnet.children())[:-1]) # up to avgpool
            self.classifier = nn.Linear(base_resnet.fc.in_features, num_classes)
            return

        # For fusion modes, we drop both avgpool and fc to keep spatial maps
        # CNN output shape will be [B, 2048, H/32, W/32]
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-2])
        cnn_out_dim = 2048
        
        # 2. Projection
        self.projection = nn.Conv2d(cnn_out_dim, embed_dim, kernel_size=1)
        
        if architecture_mode == "cnn_only":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(embed_dim, num_classes)
            return
            
        # 3. Transformer modules
        # We expect a 512x512 image -> 16x16 spatial map (256 tokens)
        self.num_tokens = 256 # Note: this assumes a fixed 512 input size.
        
        # Only instantiate positional embedding if we aren't in transformer_ablation
        if architecture_mode == "cnn_transformer":
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        else:
            self.pos_embed = None
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if self.architecture_mode == "baseline":
            features = self.backbone(x)          # [B, 2048, 1, 1]
            features = torch.flatten(features, 1) # [B, 2048]
            return self.classifier(features)
            
        # Common forward path for the other modes
        spatial_features = self.backbone(x)      # [B, 2048, H', W']
        projected = self.projection(spatial_features) # [B, embed_dim, H', W']
        
        if self.architecture_mode == "cnn_only":
            pooled = self.global_pool(projected) # [B, embed_dim, 1, 1]
            pooled = torch.flatten(pooled, 1)    # [B, embed_dim]
            return self.classifier(pooled)
            
        # Transformer modes (cnn_transformer, transformer_ablation)
        B, C, H_prime, W_prime = projected.shape
        # Flatten spatial dims to tokens: [B, C, H'*W'] -> [B, H'*W', C]
        tokens = projected.flatten(2).transpose(1, 2)
        
        if self.pos_embed is not None:
            # Add positional embeddings (broadcasts over batch)
            tokens = tokens + self.pos_embed
            
        transformed_tokens = self.transformer(tokens) # [B, num_tokens, embed_dim]
        
        # Global Average Pooling over tokens
        pooled = transformed_tokens.mean(dim=1)       # [B, embed_dim]
        
        logits = self.classifier(pooled)
        return logits
