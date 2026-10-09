# 番茄叶片病害识别系统

这是一个面向人工智能课程大作业的十分类项目：使用固定的数据清单完成番茄叶片图像分类，对比 SimpleCNN 与 ResNet18，并提供独立评估、Top-3 预测、Grad-CAM 解释和 Streamlit 中文界面。

项目不会在网页启动时训练模型。README 也不预填任何准确率；只有运行 `evaluate.py` 后生成的指标才是本机、本数据版本和本检查点对应的实际结果。

## 功能范围

- 十个固定类别：九种病害或虫害状态，以及健康叶片。
- 数据准备时校验图片、计算 SHA-256、隔离重复内容，并按类别固定随机种子分层切分。
- SimpleCNN 基线与 ImageNet 预训练 ResNet18 两条训练流程。
- 测试集 Accuracy、Macro-Precision、Macro-Recall、Macro-F1、分类报告、混淆矩阵、错误样本和推理延迟。
- 单图 Top-3 预测、低置信度提醒，以及 ResNet18 的 Grad-CAM 可视化。
- 首页、叶片识别、模型分析三个 Streamlit 页面。

## 运行设备

- Python：3.11。
- CPU：可以完成数据准备、测试、界面运行和小规模训练；完整 ResNet18 训练会较慢。
- GPU：不是必需项。若要训练完整 ResNet18，建议使用支持 CUDA 的 NVIDIA 显卡；显存不足时显式减小 `--batch-size`，程序不会悄悄改变已记录的批量大小。
- 内存与磁盘：建议至少 8 GB 内存，并为原始数据、检查点和评估图表预留数 GB 空间。

设备选择参数支持 `auto`、`cpu`、`cuda`。`auto` 在 CUDA 可用时使用 GPU，否则自动回退到 CPU；显式传入 `cuda` 但 CUDA 不可用时也会回退到 CPU，并在运行记录或评估指标中写入实际设备。

## 1. 创建环境

在 Windows PowerShell 中进入项目目录：

```powershell
Set-Location 'D:\人工智能\番茄叶片病害识别系统'
py -3.11 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

安装 PyTorch 时，如需匹配特定 CUDA 版本，可先按 PyTorch 官方安装器给出的命令安装 `torch` 和 `torchvision`，再安装其余依赖。

## 2. 准备 PlantVillage 数据

本项目不随代码分发数据。原始数据来自 [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)，相关论文为 Mohanty、Hughes 与 Salathé（2016）[Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419)。使用数据前请自行核对源仓库当前的许可证与引用要求。

PlantVillage 图片主要在受控背景下采集，与手机拍摄、田间光照、遮挡和非叶片图片存在分布差异。项目的结果不能直接代表真实田间诊断能力。

从数据源取得图片后，只复制下面十个目录到 `data\raw`，目录名必须完全一致：

```text
data\raw\
├─ Tomato___Bacterial_spot\
├─ Tomato___Early_blight\
├─ Tomato___Late_blight\
├─ Tomato___Leaf_Mold\
├─ Tomato___Septoria_leaf_spot\
├─ Tomato___Spider_mites Two-spotted_spider_mite\
├─ Tomato___Target_Spot\
├─ Tomato___Tomato_Yellow_Leaf_Curl_Virus\
├─ Tomato___Tomato_mosaic_virus\
└─ Tomato___healthy\
```

生成可复现的数据清单：

```powershell
& '.\.venv\Scripts\python.exe' prepare_data.py `
  --source data\raw `
  --manifest data\manifest.csv
```

每次更换、增删或修改原始图片后都要重新生成清单。加载数据时会再次核对清单中的文件哈希，防止训练数据在实验过程中无记录地变化。

## 3. 训练两个模型

SimpleCNN 基线：

```powershell
& '.\.venv\Scripts\python.exe' train.py `
  --model simple_cnn `
  --manifest data\manifest.csv `
  --data-root data\raw `
  --simple-epochs 20 `
  --batch-size 32 `
  --device auto `
  --run-name simplecnn-main
```

ResNet18 迁移学习：

```powershell
& '.\.venv\Scripts\python.exe' train.py `
  --model resnet18 `
  --manifest data\manifest.csv `
  --data-root data\raw `
  --head-epochs 5 `
  --finetune-epochs 15 `
  --batch-size 32 `
  --device auto `
  --run-name resnet18-main
```

首次训练 ResNet18 可能需要下载 torchvision 的 ImageNet 预训练权重。若发生 CUDA 显存不足，请用新的运行名称重试并减小批量，例如 `--batch-size 16` 或 `8`；不要覆盖原运行目录。

每个训练目录位于 `outputs\<run-name>`，包含：

- `best_model.pth`：验证集 Macro-F1 最优的检查点及完整元数据；
- `config.json`、`runtime.json`：数据、训练和运行环境记录；
- `history.csv`：逐轮训练与验证指标；
- `training_curves.png`：损失、准确率和 Macro-F1 曲线。

如果目录已包含训练历史，程序会拒绝覆盖。

## 4. 独立测试集评估与模型对比

分别评估两个检查点：

```powershell
& '.\.venv\Scripts\python.exe' evaluate.py `
  --checkpoint outputs\simplecnn-main\best_model.pth `
  --manifest data\manifest.csv `
  --data-root data\raw `
  --output-dir outputs\evaluation\simplecnn-main `
  --device auto

& '.\.venv\Scripts\python.exe' evaluate.py `
  --checkpoint outputs\resnet18-main\best_model.pth `
  --manifest data\manifest.csv `
  --data-root data\raw `
  --output-dir outputs\evaluation\resnet18-main `
  --device auto
```

评估只读取 `test` 切分，且拒绝写入非空输出目录。它会生成 `metrics.json`、JSON/CSV 分类报告、原始与归一化混淆矩阵、`errors.csv` 和 `confusion_pairs.csv`。

生成两模型对比表：

```powershell
& '.\.venv\Scripts\python.exe' -c "from pathlib import Path; from evaluate import compare_runs; compare_runs([Path(r'outputs\evaluation\simplecnn-main\metrics.json'), Path(r'outputs\evaluation\resnet18-main\metrics.json')], Path(r'outputs\evaluation\comparison.csv'))"
```

## 5. 单图命令行预测

```powershell
& '.\.venv\Scripts\python.exe' predict.py data\raw\Tomato___healthy\example.jpg `
  --checkpoint outputs\resnet18-main\best_model.pth `
  --device auto
```

输出包含按概率降序排列的 Top-3 类别。最高概率低于阈值时，只给出“可能不属于训练分布”的提醒，不将未知图片强行解释为确定病害。

## 6. 启动中文界面

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

在“叶片识别”页填写 `outputs\resnet18-main\best_model.pth`，上传 JPG、JPEG 或 PNG 图片。在“模型分析”页填写某个独立评估目录，例如 `outputs\evaluation\resnet18-main`。训练曲线位于训练运行目录；评估指标与混淆矩阵位于评估目录，页面会对缺失的可选产物逐项提示。

## 7. 测试与代码检查

运行全部测试（包含一次离线、CPU、合成数据的最小训练闭环）：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -v
```

只运行端到端慢测试：

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\test_smoke_pipeline.py -v -m slow
```

执行源码编译检查：

```powershell
& '.\.venv\Scripts\python.exe' -m compileall -q -x '.*[\\/]\.venv[\\/].*' .
```

端到端测试使用程序生成的合成图片，只验证接口衔接和产物完整性，不能作为真实模型精度证据。

## 目录与版本控制

`data`、`outputs`、`checkpoints`、虚拟环境、模型权重、日志和 Streamlit 密钥均已加入 `.gitignore`。提交代码前请运行 `git status --short`，不要将原始数据、训练检查点或生成的报告图片提交到仓库。

## 免责声明

本系统仅用于课程实验和模型工程演示，不替代农业专家、植保机构或实验室检测；不提供农药、剂量、治疗方案或任何专业保证。对低置信度、非叶片、田间拍摄或训练分布之外的图片，结果尤其可能不可靠。
