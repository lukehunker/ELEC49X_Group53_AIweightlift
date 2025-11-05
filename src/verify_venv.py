import sys
import os

# 1. Check Environment Isolation
in_venv = (sys.prefix != sys.base_prefix)
print(f"✅ Running inside venv: {in_venv}")
print(f"📂 Python executable: {sys.executable}")

# 2. Check for dangerous global leakage
# If paths include '/home/lukehunker/.local/', your venv is not isolated.
local_leak = any(".local/lib" in path for path in sys.path)
if local_leak:
    print("❌ WARNING: Environment is leaking global packages from ~/.local")
    print("   Fix: Run 'export PYTHONNOUSERSITE=1' before activating venv.")
else:
    print("✅ Environment is properly isolated.")

# 3. Test Critical Imports
print("\n--- Testing Imports ---")
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    import mmcv
    print(f"✅ MMCV: {mmcv.__version__}")
    import mmdet
    print(f"✅ MMDet: {mmdet.__version__}")
    import mmpose
    print(f"✅ MMPose: {mmpose.__version__}")
    import mediapipe
    print("✅ MediaPipe: loaded")
except ImportError as e:
    print(f"\n❌ FAILED IMPORT: {e}")
    sys.exit(1)

print("\n🎉 Environment verified! Ready for development.")