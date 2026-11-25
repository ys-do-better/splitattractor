# 外部基础设施配置

工程本身不保存密钥。默认模型配置是 `D:\.env`，至少包含：

```dotenv
key=...
base_url=https://.../v1
default_model=DeepSeek-V4.1-Flash-1017
```

可通过环境变量覆盖：

```powershell
$env:METACORSI_PYTHON = "D:\path\python.exe"
$env:METACORSI_DOTENV = "D:\path\.env"
$env:METACORSI_MODEL = "DeepSeek-V4.1-Flash-1017"
```

运行还需要 Docker Desktop 和 Harbor/DSH。`SETUP_LOCAL.ps1` 会建立本地 Python venv；`CHECK_ENV.ps1` 会检查 Python import、Docker 和 active-brain 检索隔离。
