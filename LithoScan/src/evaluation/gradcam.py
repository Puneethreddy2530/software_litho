import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        x.requires_grad_(True)
        self.model.zero_grad()
        
        preds = self.model(x)
        pred_idx = preds.argmax(dim=1).item()
        if class_idx is None:
            class_idx = pred_idx
            
        score = preds[0, class_idx]
        score.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy(), pred_idx

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False) -> np.ndarray:
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()

def main(args):
    import cv2 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ds = DCIDDataset(dcid_root=args.dcid_root, version=args.version, split="test", image_size=512, skip_corrupt=True)
    
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(ds.classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    
    target_layer = model.layer4[-1]
    cam_generator = GradCAM(model, target_layer)
    
    correct_counts = {c: 0 for c in range(len(ds.classes))}
    incorrect_counts = {c: 0 for c in range(len(ds.classes))}
    MAX_CORRECT = 10
    MAX_INCORRECT = 5
    
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    
    for idx, (img_tensor, label) in enumerate(loader):
        lbl = label.item()
        
        # Stop early if we have enough
        all_done = all(correct_counts[c] >= MAX_CORRECT for c in range(len(ds.classes))) and \
                   all(incorrect_counts[c] >= MAX_INCORRECT for c in range(len(ds.classes)))
        if all_done:
            break
            
        input_tensor = img_tensor.to(device)
        _, pred_idx = cam_generator(input_tensor, class_idx=lbl)
        
        is_correct = (lbl == pred_idx)
        
        if is_correct and correct_counts[lbl] < MAX_CORRECT:
            correct_counts[lbl] += 1
            cam, _ = cam_generator(input_tensor, class_idx=lbl)
            
            cam = cv2.resize(cam, (512, 512))
            rgb_img = denormalize(img_tensor[0])
            vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(rgb_img)
            axes[0].set_title(f"TRUE: {ds.classes[lbl]}")
            axes[0].axis('off')
            
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title(f"CAM TARGET: TRUE")
            axes[1].axis('off')
            
            axes[2].imshow(vis)
            axes[2].set_title(f"PREDICTED: {ds.classes[pred_idx]}")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(out_dir / f"gradcam_{ds.classes[lbl].replace(' ', '_')}_correct_{correct_counts[lbl]}.png")
            plt.close()
            
        elif not is_correct and incorrect_counts[lbl] < MAX_INCORRECT:
            incorrect_counts[lbl] += 1
            
            # True CAM
            cam_true, _ = cam_generator(input_tensor, class_idx=lbl)
            cam_true = cv2.resize(cam_true, (512, 512))
            rgb_img = denormalize(img_tensor[0])
            vis_true = show_cam_on_image(rgb_img, cam_true, use_rgb=True)
            
            # Pred CAM
            cam_pred, _ = cam_generator(input_tensor, class_idx=pred_idx)
            cam_pred = cv2.resize(cam_pred, (512, 512))
            vis_pred = show_cam_on_image(rgb_img, cam_pred, use_rgb=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: True
            axes[0, 0].imshow(rgb_img)
            axes[0, 0].set_title(f"TRUE: {ds.classes[lbl]}")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cam_true, cmap='jet')
            axes[0, 1].set_title(f"CAM TARGET: TRUE")
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(vis_true)
            axes[0, 2].set_title(f"PREDICTED: {ds.classes[pred_idx]}")
            axes[0, 2].axis('off')
            
            # Row 2: Pred
            axes[1, 0].imshow(rgb_img)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(cam_pred, cmap='jet')
            axes[1, 1].set_title(f"CAM TARGET: PREDICTED")
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(vis_pred)
            axes[1, 2].set_title(f"PREDICTED: {ds.classes[pred_idx]}")
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(out_dir / f"gradcam_{ds.classes[lbl].replace(' ', '_')}_incorrect_{incorrect_counts[lbl]}.png")
            plt.close()
            
    print(f"Grad-CAM generation complete.")
    print(f"Correct counts: {correct_counts}")
    print(f"Incorrect counts: {incorrect_counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--version", default="DCID-512-7")
    parser.add_argument("--dcid-root", default=r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")
    args = parser.parse_args()
    main(args)
