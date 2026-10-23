import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import traceback

def verify_cuda():
    print("==================================================")
    print("CUDA VERIFICATION")
    print("==================================================")

    cuda_avail = torch.cuda.is_available()
    print(f"1. torch.cuda.is_available(): {cuda_avail}")
    
    cuda_ver = torch.version.cuda if hasattr(torch.version, 'cuda') else None
    print(f"2. torch.version.cuda: {cuda_ver}")
    
    dev_count = torch.cuda.device_count() if cuda_avail else 0
    print(f"3. torch.cuda.device_count(): {dev_count}")
    
    if cuda_avail and dev_count > 0:
        dev_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        dev_name = "N/A"
        vram = 0.0
        
    print(f"4. torch.cuda.get_device_name(0): {dev_name}")
    print(f"5. total GPU VRAM: {vram:.2f} GB")
    
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"6. selected training device: {device}")
    
    print("\n--- Initializing Model ---")
    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)
        param_dev = next(model.parameters()).device
        print(f"7. model parameter device: {param_dev}")
        
        # Create dummy batch
        images = torch.randn(4, 3, 512, 512).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)
        
        print("8. first batch tensor devices:")
        print(f"   images.device: {images.device}")
        print(f"   labels.device: {labels.device}")
        
        print("\n9. Running ONE actual forward + backward + optimizer step...")
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        torch.cuda.reset_peak_memory_stats() if cuda_avail else None
        
        t0 = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_time = time.perf_counter() - t0
        
        print("   Step completed successfully.")
        
        print("\n10. Memory and Timing Stats:")
        if cuda_avail:
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mem_res = torch.cuda.memory_reserved() / (1024**2)
            mem_max = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"    torch.cuda.memory_allocated(): {mem_alloc:.2f} MB")
            print(f"    torch.cuda.memory_reserved(): {mem_res:.2f} MB")
            print(f"    max GPU memory allocated: {mem_max:.2f} MB")
        else:
            print("    torch.cuda.memory_allocated(): N/A (CPU)")
            print("    torch.cuda.memory_reserved(): N/A (CPU)")
            print("    max GPU memory allocated: N/A (CPU)")
            
        print(f"    batch execution time: {batch_time:.4f} seconds")
        
        if cuda_avail and "cuda" in str(device) and "cuda" in str(param_dev) and "cuda" in str(images.device):
            print("\nCUDA TRAINING VERIFIED = YES")
        else:
            print("\nCUDA TRAINING VERIFIED = NO")
            
    except Exception as e:
        print("\nERROR during execution:")
        traceback.print_exc()
        print("\nCUDA TRAINING VERIFIED = NO")

if __name__ == "__main__":
    verify_cuda()
