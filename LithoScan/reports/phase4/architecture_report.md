# Phase 4A Architecture Report: CNN-Transformer Server Edition

Date: 2026-09-14
Model: `LithoScanServerFusion`
File: `src/models/server_fusion.py`

## Architecture Summary
The LithoScan Server Edition is a hybrid CNN-Transformer designed to process spatial feature maps instead of globally pooled features. The architecture preserves spatial feature information as tokens and applies self-attention across those spatial representations.

- **Backbone**: ImageNet pretrained ResNet-50.
- **Fusion Logic**: The final average pooling layer and fully connected head are removed. The remaining spatial feature map (2048 channels) is passed to a 1x1 Convolution to project down to a configurable `embed_dim` (512). The spatial dimensions are flattened into tokens.
- **Transformer Encoder**: Tokenized features are added with learnable positional embeddings and passed through a PyTorch TransformerEncoder.
- **Classifier**: The output tokens are globally average-pooled and passed through a Linear classification head.

## Modes / Ablation Capabilities
The model supports an `architecture_mode` parameter to seamlessly support fair comparison ablations using identical preprocessing pipelines:
- `baseline`: Standard ResNet-50 behavior.
- `cnn_transformer`: Full hybrid pipeline with positional embeddings.
- `cnn_only`: Applies Global Average Pooling directly after the 512d projection, isolating the effect of the Transformer.
- `transformer_ablation`: Full hybrid pipeline but *without* learnable positional embeddings, testing reliance on strict spatial arrangements.

## Validation Metrics
- **Parameter Count**: 30,996,551
- **Trainable Parameter Count**: 30,996,551
- **Tensor Shapes (Batch Size = 2, Input = 512x512)**:
  - Input shape: `[2, 3, 512, 512]`
  - CNN feature-map shape: `[2, 2048, 16, 16]`
  - Token shape: `[2, 256, 512]`
  - Transformer output shape: `[2, 256, 512]`
  - Logits shape: `[2, 7]`

## Artifact Intervention Design
- **Module**: `src/evaluation/artifact_intervention.py`
- **Method**: `mask_image_border(img_tensor, border_width=40)`
- **Purpose**: Applies a strict mask to the borders of an image tensor during evaluation to test robustness against background/ruler artifacts, leaving the interior geological texture unchanged.
- **Coverage**: For a 512x512 image and a 40-px border, the exact spatial pixel percentage modified is **28.81%**.
