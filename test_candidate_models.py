import os
import sys
import numpy as np
np.trapz = np.trapezoid
import torch
import safetensors.torch
import mlstac

model_dir = r".\SEN2SRLite_RGBN_x4\SEN2SRLite\NonReference_RGBN_x4"
print("Loading Candidate A...")
cand_a = mlstac.load(model_dir).compiled_model(device="cpu")

print("Loading Candidate B...")
from sen2sr.models.opensr_baseline.cnn import CNNSR
trainable_f = os.path.join(model_dir, "model.safetensor")
sr_model_weights = safetensors.torch.load_file(trainable_f)
cand_b_base = CNNSR(4, 4, 24, 4, True, False, 6)
cand_b_base.load_state_dict(sr_model_weights)
cand_b_base.eval()
for param in cand_b_base.parameters():
    param.requires_grad = False
cand_b_base.to("cpu")

class UnconstrainedCNN(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return torch.clamp(out, min=0.0)

cand_b = UnconstrainedCNN(cand_b_base)

# Test with 128x128 patch
x_patch = torch.rand(1, 4, 128, 128)
with torch.no_grad():
    out_a = cand_a(x_patch)
    out_b = cand_b(x_patch)

print(f"Patch shape: {x_patch.shape}")
print(f"Cand A output shape: {out_a.shape}, range: [{out_a.min().item():.4f}, {out_a.max().item():.4f}]")
print(f"Cand B output shape: {out_b.shape}, range: [{out_b.min().item():.4f}, {out_b.max().item():.4f}]")
diff = torch.abs(out_a - out_b).mean().item()
max_diff = torch.abs(out_a - out_b).max().item()
print(f"Mean absolute difference: {diff:.6f}, Max diff: {max_diff:.6f}")
