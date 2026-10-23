# Artifact Bias Observations (Phase 2A - ResNet-50)

This report presents a qualitative review of the Phase 2A baseline (DCID-512-7) based on Grad-CAM heatmaps generated from the official test split.

> [!NOTE]
> **Qualitative Evidence Only**: The observations documented here are purely qualitative. Grad-CAM visualizes network attention but does not provide definitive causal interpretations or quantitative artifact percentages.

## 1. Methodology
- **Model**: ResNet-50 (Phase 2A `best_model.pt`)
- **Target Layer**: `layer4[-1]` (final convolutional block)
- **Sampling**: Up to 3 correctly classified and 2 misclassified (none found, accuracy is 1.0) examples per class.
- **Visuals**: Original Image, Grad-CAM Heatmap, and Overlay. 

## 2. Qualitative Observations

### Genuine Core Texture
In the majority of the reviewed samples (e.g., Granite, Basalt, Light Sandstone), the network's attention correctly spans the intrinsic textural elements of the rock surface. The gradients often highlight crystalline structures, granular boundaries, and natural foliation rather than extraneous artifacts.

### Background and Context
The model generally ignores the dark or uniform background surrounding the drill cores. The attention masks tightly conform to the physical boundary of the lithology samples, suggesting the model has learned to segment the core from the background before extracting textural features.

### Artifact Presence

1. **Cracks and Fractures**:
   - In some instances (e.g., heavily fractured Mudstone or Gray Siltstone), the attention map shows slightly elevated heat around physical cracks. It is ambiguous whether the model uses the crack itself as a feature (a potential artifact bias) or if the fractured edge exposes fresher textural information that the model prefers.

2. **Mud and Surface Contamination**:
   - Surface mud occasionally intersects with high-attention regions. However, the model does not appear to focus *exclusively* on mud spots. It seems to integrate the contaminated areas with the clean surface texture.

3. **Scale Bars and Markers**:
   - Where visible, the network mostly avoids strong activations on scale bars, measuring tapes, or drawn markers. The model successfully localizes the geological features rather than the metadata artifacts left by field geologists.

## 3. Conclusion
Qualitatively, the ResNet-50 baseline on DCID-512-7 appears well-behaved. The Grad-CAM heatmaps suggest the model relies predominantly on valid geological textures. While there is minor localized attention near cracks and occasional mud, there is no evidence of catastrophic artifact bias (e.g., the model learning to detect scale bars instead of rocks). Further quantitative occlusion tests would be required to establish formal causal interpretations.
