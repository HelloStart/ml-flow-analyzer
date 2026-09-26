# ML Flow Analyzer

面向开发者和学习者的机器学习流程可视化学习工具。
示图：

![ML Flow Analyzer 操作演示](https://github.com/HelloStart/ml-flow-analyzer/raw/refs/heads/main/demo.gif)

当前版本：`v0.5`，增加统一“问 AI”。后续规划为候选方向，不代表固定发布日期或必须按顺序完成。

## Windows 可执行文件

Windows 可执行文件可从 [GitHub Releases 下载 ML Flow Analyzer v0.5](https://github.com/HelloStart/ml-flow-analyzer/releases/latest/download/MLFlowAnalyzer_v0.5.exe)。
下载后双击启动，无需按下方步骤安装 Python 和项目依赖。


## 源码环境与安装

当前版本需要 Windows 和 Python 3.11 或更高版本。建议在项目根目录执行可编辑安装，`pip` 会根据 `pyproject.toml` 自动安装全部依赖：

```powershell
py -3.11 -m pip install -e .
```


主要依赖包括：

- `PySide6`：桌面界面
- `scikit-learn`：数据集、数据划分、预处理、模型训练和评估
- `matplotlib`：散点图、混淆矩阵和评估图表
- `pandas`：表格数据处理与后续任务扩展
- `PyYAML`：加载可配置的任务和知识内容
- `joblib`：保存和加载模型与 StandardScaler

## 运行

推荐使用 Python 3.11 启动：

```powershell
py -3.11 app.py
```

如果 `python` 命令已经指向安装了上述依赖的解释器，也可以运行：

```powershell
python app.py
```

完成可编辑安装后，也可以使用安装生成的命令：

```powershell
ml-flow-analyzer
```



## 问 AI

课程门户、Iris、California Housing 和 Wine 页面的顶栏都提供“问 AI”入口。对话框会附带当前路线或任务、当前步骤、页面说明，以及当前可用的模型参数和实际运行指标；点击“上下文详情”可在发送前查看并编辑完整内容。



默认是本地 Ollama，应用会从 `http://localhost:11434/api/tags` 读取已安装模型。可先执行：

```powershell
ollama ls
ollama run qwen2.5:1.5b
```

也可选择通义千问或 DeepSeek。API Key 优先从环境变量读取，也可只在当前对话框输入，不会写入项目文件：

```powershell
$env:DASHSCOPE_API_KEY = "your-qwen-key"
$env:DEEPSEEK_API_KEY = "your-deepseek-key"
```

选择通义千问时使用 `DASHSCOPE_API_KEY`；选择 DeepSeek 时使用 `DEEPSEEK_API_KEY`。对话框只显示“已从环境变量读取”或“未设置”，不会显示 API Key 内容。

当前内置云端模型预设：

- 通义千问：`qwen3.8-flash`（推荐）、`qwen3.7-plus`、`qwen-plus`、`qwen-turbo`
- DeepSeek：`deepseek-v4-flash`（推荐）、`deepseek-chat`、`deepseek-reasoner`



## 提交问题与反馈

如遇到问题，请提交 Issue，或发送邮件至 [fanqiefox@foxmail.com](mailto:fanqiefox@foxmail.com)


## License

本项目使用 [MIT License](LICENSE)。


## ⚠️ 免责声明 / Disclaimer

> 本项目仅供**个人学习、技术整理和原型探索**，按 **MIT 协议**以“现状”提供，不作任何担保。代码中部分由 AI 辅助生成、经人工复核；如涉及任何权利问题，请提交 issue，我将尽快处理。涉及 TinyML、语音、设备控制和本地/云端 LLM 的部分内容为流程概览或 PC 模拟，不等同于目标硬件验证。
>
> This project is intended **only for personal learning, technical research, and prototype exploration**. Provided under the **MIT License** "AS IS", without warranty. Part of the code was **AI-assisted and human-reviewed**. Some TinyML, speech, device-control, and local/cloud LLM content is an architectural overview or PC simulation, not target-hardware validation.

完整中英免责声明见 [DISCLAIMER.md](DISCLAIMER.md)。