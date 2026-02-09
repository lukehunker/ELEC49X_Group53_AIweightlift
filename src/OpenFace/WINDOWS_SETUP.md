# OpenFace Windows Setup Guide

## Prerequisites

### 1. Visual Studio 2019 or 2022 (Community Edition is free)
- Download: https://visualstudio.microsoft.com/downloads/
- During installation, select **"Desktop development with C++"**
- Make sure to include:
  - MSVC C++ build tools
  - Windows 10/11 SDK
  - CMake tools for Windows

### 2. CMake (3.10 or higher)
- Download: https://cmake.org/download/
- Get the Windows installer (.msi)
- During install: Check **"Add CMake to system PATH for all users"**

### 3. Git for Windows
- Download: https://git-scm.com/download/win
- Use default settings during installation

---

## Step-by-Step Installation

### 1. Open PowerShell as Administrator
Right-click Start → "Windows Terminal (Admin)" or "PowerShell (Admin)"

### 2. Navigate to Your Project
```powershell
cd C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift
```

### 3. Download OpenFace Models
```powershell
cd OpenFace
.\download_models.ps1
```

If you get "execution policy" error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\download_models.ps1
```

### 4. Download Required Libraries
```powershell
.\download_libraries.ps1
```

### 5. Build OpenFace

#### **Option A: Using CMake GUI (Recommended for Windows)**

1. Open **CMake GUI** application
2. **Where is the source code**: `C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace`
3. **Where to build binaries**: `C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace\build`
4. Click **Configure**
5. Select your Visual Studio version (e.g., "Visual Studio 17 2022")
6. Select platform: **x64**
7. Click **Finish**
8. Wait for configuration to complete
9. If it finds errors, check these common issues:
   - **OpenCV not found**: 
     - Download OpenCV from https://opencv.org/releases/
     - Extract to `C:\opencv`
     - In CMake, click "Add Entry":
       - Name: `OpenCV_DIR`
       - Type: `PATH`
       - Value: `C:\opencv\build`
     - Click Configure again
10. Click **Generate**
11. Click **Open Project** (opens Visual Studio)
12. In Visual Studio:
    - At the top, change "Debug" to **Release**
    - Right-click on **ALL_BUILD** → Build
    - Wait 5-10 minutes for compilation

#### **Option B: Using Command Line**

```powershell
cd OpenFace
mkdir build -Force
cd build

# Configure with CMake
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release ..

# Build (this takes several minutes)
cmake --build . --config Release
```

### 6. Verify Installation

```powershell
cd build\bin\Release
.\FeatureExtraction.exe -h
```

You should see usage information. If you see "missing DLL" errors, see troubleshooting below.

### 7. Test with a Video

```powershell
.\FeatureExtraction.exe -f "C:\path\to\test_video.mp4" -out_dir "C:\output"
```

---

## Common Issues & Solutions

### Issue 1: "OpenCV not found"
**Solution:**
```powershell
# Download OpenCV pre-built binaries
# Go to https://opencv.org/releases/
# Download Windows version (e.g., opencv-4.9.0-windows.exe)
# Extract to C:\opencv

# In CMake GUI, add entry:
# Name: OpenCV_DIR
# Type: PATH  
# Value: C:\opencv\build
```

### Issue 2: "MSVCP140.dll is missing"
**Solution:**
- Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue 3: "dlib or boost not found"
**Solution:**
```powershell
cd C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace
.\download_libraries.ps1
```

### Issue 4: Exit Status 1 (Command returns non-zero exit status)
This usually means models are missing or incorrect paths. Try:

```powershell
# 1. Verify models exist
ls C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace\build\bin\Release\model\

# 2. Run with explicit model path
cd C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace\build\bin\Release
.\FeatureExtraction.exe -f "C:\full\path\to\video.mov" -out_dir "C:\output" -mloc ".\model"

# 3. Check if video file is accessible
# Try with a simple MP4 file first, not MOV
```

### Issue 5: "PowerShell scripts won't run"
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 6: MOV Format Not Supported
Windows OpenFace might not support certain MOV codecs. Convert to MP4:

```powershell
# Install ffmpeg via chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html

# Convert MOV to MP4
ffmpeg -i "input.mov" -c:v libx264 -c:a aac "output.mp4"
```

---

## Update Python Code for Windows

The Python code needs to use the Windows executable path. Update `src/OpenFace/openface_utils.py`:

```python
# Around line 36 in openface_utils.py
import platform

if platform.system() == 'Windows':
    OPENFACE_BIN = os.path.join(REPO_ROOT, "OpenFace", "build", "bin", "Release", "FeatureExtraction.exe")
else:
    OPENFACE_BIN = os.path.join(REPO_ROOT, "OpenFace", "build", "bin", "FeatureExtraction")
```

---

## Troubleshooting Exit Code 1

If you're getting exit status 1 with no error message:

### 1. Remove the `-q` (quiet) flag to see actual errors

```powershell
cd C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace\build\bin\Release

# Run WITHOUT -q to see error messages
.\FeatureExtraction.exe -f "C:\path\to\video.mov" -out_dir "C:\output" -aus
```

### 2. Check model files exist

```powershell
# Should see files like main_ceclm_general.txt, main_clnf_general.txt, etc.
ls .\model\
```

If models are missing:
```powershell
cd ..\..\..\  # Back to OpenFace root
.\download_models.ps1
```

### 3. Test with a simple command

```powershell
# Minimal command - just landmarks
.\FeatureExtraction.exe -f "C:\path\to\video.mp4" -out_dir "C:\output"
```

### 4. Check video file path

- Use **full absolute paths** (not relative)
- Avoid spaces in paths, or use quotes
- Try MP4 instead of MOV
- Verify file isn't corrupted: open it in VLC or Windows Media Player

---

## Final Checklist

- [ ] Visual Studio 2019/2022 installed with C++ tools
- [ ] CMake installed and in PATH
- [ ] OpenFace models downloaded (`download_models.ps1`)
- [ ] OpenFace libraries downloaded (`download_libraries.ps1`)
- [ ] OpenFace built in **Release** mode (not Debug)
- [ ] `FeatureExtraction.exe` exists in `build\bin\Release\`
- [ ] Model folder exists in `build\bin\Release\model\` with `.txt` files
- [ ] Test command works with verbose output (without `-q`)
- [ ] Python code updated to use correct Windows path

---

## Quick Reference: Build Commands

```powershell
# Complete rebuild from scratch
cd C:\Users\YourUsername\Documents\ELEC49X_Group53_AIweightlift\OpenFace
Remove-Item -Recurse -Force build
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# Verify
cd bin\Release
.\FeatureExtraction.exe -h
```

---

## Getting Help

If you're still having issues:

1. **Run without `-q` flag** to see the actual error message
2. **Check the error message** and search for it online
3. **Verify all prerequisites** are installed correctly
4. **Try a different video file** to rule out file-specific issues
5. **Check OpenFace GitHub issues**: https://github.com/TadasBaltrusaitis/OpenFace/issues

Common error keywords to search for:
- "OpenFace Windows exit code 1"
- "FeatureExtraction.exe model not found"
- "OpenFace CMake Windows configuration error"
