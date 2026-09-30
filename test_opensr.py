import os
import sys
import numpy as np
np.trapz = np.trapezoid
import torch
import opensr_model

print("Inspecting opensr_model.srmodel...")
try:
    import inspect
    print(inspect.getsource(opensr_model.srmodel))
except Exception as e:
    print(f"Error inspecting srmodel: {e}")
