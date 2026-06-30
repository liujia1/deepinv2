---
name: "python-runner"
description: "Execute Python scripts with pre-configured Python path (D:\Programs\Python\Python310). Invoke when user needs to run Python scripts, check Python version, or execute Python commands without specifying the full path."
---

# Python Runner

## Overview

This skill provides convenient Python execution commands with a pre-configured Python installation path, eliminating the need to search for the Python executable each time.

## Configuration

- **Python Executable Path**: `D:\Programs\Python\Python310\python.exe`
- **Python Version**: 3.10.11 (verified)
- **pip Path**: `D:\Programs\Python\Python310\Scripts\pip.exe`

## Usage

### 1. Execute a Python Script

```powershell
& "D:\Programs\Python\Python310\python.exe" "path\to\script.py"
```

**Example**:
```powershell
& "D:\Programs\Python\Python310\python.exe" "d:\works\deepinv\实验9.2-1\9.2-1.py"
```

### 2. Check Python Version

```powershell
& "D:\Programs\Python\Python310\python.exe" --version
```

### 3. Install Python Packages

```powershell
& "D:\Programs\Python\Python310\Scripts\pip.exe" install <package-name>
```

**Example with Chinese mirror**:
```powershell
& "D:\Programs\Python\Python310\Scripts\pip.exe" install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Run Python One-liners

```powershell
& "D:\Programs\Python\Python310\python.exe" -c "print('Hello World')"
```

### 5. Run Python Module

```powershell
& "D:\Programs\Python\Python310\python.exe" -m <module-name>
```

**Example**:
```powershell
& "D:\Programs\Python\Python310\python.exe" -m pip list
```

## Quick Reference

| Task | Command |
|------|---------|
| Run script | `& "D:\Programs\Python\Python310\python.exe" "script.py"` |
| Check version | `& "D:\Programs\Python\Python310\python.exe" --version` |
| Install package | `& "D:\Programs\Python\Python310\Scripts\pip.exe" install <pkg>` |
| List packages | `& "D:\Programs\Python\Python310\python.exe" -m pip list` |
| Run module | `& "D:\Programs\Python\Python310\python.exe" -m <module>` |

## Notes

- Use `&` operator in PowerShell to execute paths with spaces
- Always use absolute paths for scripts to avoid working directory issues
- For package installation, consider using Chinese mirrors for faster downloads:
  - Tsinghua: `https://pypi.tuna.tsinghua.edu.cn/simple`
  - Aliyun: `https://mirrors.aliyun.com/pypi/simple/`
