---
name: "pip-install"
description: "Installs Python packages using pip with Chinese mirror for faster downloads. Invoke when user needs to install Python packages via pip or when pip install command is mentioned."
---

# pip-install

使用 pip 安装 Python 包时，自动使用国内镜像源加速下载。

## 可用镜像源

以下镜像源按推荐顺序排列：

1. **清华大学**：`https://pypi.tuna.tsinghua.edu.cn/simple`
2. **阿里云**：`https://mirrors.aliyun.com/pypi/simple/`
3. **腾讯云**：`https://mirrors.cloud.tencent.com/pypi/simple/`
4. **华为云**：`https://repo.huaweicloud.com/repository/pypi/simple/`

## 使用方法

### 临时使用镜像（推荐）

```bash
pip install <package-name> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 永久配置（可选）

**Windows (PowerShell):**
```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**Linux/macOS:**
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

配置文件位置：
- Windows: `%APPDATA%\pip\pip.ini`
- Linux: `~/.config/pip/pip.conf`
- macOS: `~/Library/Application Support/pip/pip.conf`

## 常用安装命令示例

```bash
# 安装单个包
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装多个包
pip install numpy pandas matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安静模式安装（不显示详细输出）
pip install statsmodels arviz -q -i https://pypi.tuna.tsinghua.edu.cn/simple

# 升级包
pip install --upgrade package-name -i https://pypi.tuna.tsinghua.edu.cn/simple

# 从 requirements.txt 安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 镜像源选择策略

1. **首选清华镜像**：速度最快，覆盖最全
2. **备用阿里云**：清华不可用时切换
3. **其他镜像**：根据网络环境选择

## 注意事项

- 镜像源可能偶尔同步延迟，如遇包版本过旧可尝试切换到其他镜像
- 某些特殊包（如企业内部包）可能需要使用官方源
- 使用 `-q` 参数可以减少输出噪音
