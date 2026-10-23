import os
import sys
import time
import torch
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.server_fusion import LithoScanServerFusion

def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  - CUDA allocated memory: {allocated:.2f} MB")
        print(f"  - CUDA reserved memory: {reserved:.2f} MB")
        print(f"  - CUDA max allocated memory (peak memory): {max_alloc:.2f} MB")

def run_synthetic_validation():
    print("==================================================")
    print("3. ARCHITECTURE VALIDATION (SYNTHETIC)")
    print("==================================================")
    
    model = LithoScanServerFusion(
        num_classes=7,
        architecture_mode="cnn_transformer",
        pretrained=False,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter count: {total_params:,}")
    print(f"Trainable parameter count: {trainable_params:,}")
    
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512)
    print(f"Input shape: {list(dummy_input.shape)}")
    
    # We hook into the forward pass manually just to print intermediate shapes
    # CNN output
    spatial_features = model.backbone(dummy_input)
    print(f"CNN feature-map shape: {list(spatial_features.shape)}")
    
    projected = model.projection(spatial_features)
    tokens = projected.flatten(2).transpose(1, 2)
    print(f"Token shape: {list(tokens.shape)}")
    
    transformed_tokens = model.transformer(tokens)
    print(f"Transformer output shape: {list(transformed_tokens.shape)}")
    
    logits = model(dummy_input)
    print(f"Logits shape: {list(logits.shape)}")
    
    assert not torch.isnan(logits).any(), "NaNs detected in logits!"
    assert not torch.isinf(logits).any(), "Infs detected in logits!"
    print("Verification: No NaNs, No Inf values.")
    
    loss = logits.sum()
    loss.backward()
    
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Verification: Gradients are non-zero after backward? {has_grads}")
    print()

def run_cuda_smoke_test():
    print("==================================================")
    print("4. RTX 3050 SMOKE TEST (CUDA FORWARD + BACKWARD)")
    print("==================================================")
    
    if not torch.cuda.is_available():
        print("CUDA is not available! Cannot run RTX 3050 Smoke Test.")
        return
        
    device = torch.device("cuda:0")
    model = LithoScanServerFusion(architecture_mode="cnn_transformer").to(device)
    
    batch_size = 4
    gradient_accumulation = 4
    
    print(f"model device: {next(model.parameters()).device}")
    
    # Simple loop to catch OOM
    success = False
    while not success and batch_size > 0:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda')
            
            dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
            dummy_target = torch.randint(0, 7, (batch_size,)).to(device)
            
            print(f"input device: {dummy_input.device}")
            print(f"target device: {dummy_target.device}")
            print(f"batch size: {batch_size}")
            print(f"gradient accumulation: {gradient_accumulation}")
            
            start_time = time.time()
            
            model.train()
            optimizer.zero_grad()
            
            # Forward + Backward step (with AMP and dummy loss)
            with torch.amp.autocast('cuda'):
                logits = model(dummy_input)
                loss = torch.nn.functional.cross_entropy(logits, dummy_target)
                loss = loss / gradient_accumulation
                
            scaler.scale(loss).backward()
            
            # Pretend we hit the accumulation step
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()
            exec_time = time.time() - start_time
            print(f"execution time: {exec_time:.4f} seconds")
            
            print_memory_stats()
            success = True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM caught with batch_size={batch_size}. Reducing...")
                batch_size -= 2
                gradient_accumulation *= 2
            else:
                raise e

if __name__ == "__main__":
    run_synthetic_validation()
    run_cuda_smoke_test()
