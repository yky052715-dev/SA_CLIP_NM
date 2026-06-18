# SA-CLIP-NM

SA-CLIP-NM 是一个独立实现的工业图像异常检测工程。它继承 PatchCore 的正常特征记忆与最近邻检索范式，但不修改 Anomalib 1.1.1 或现有 PatchCore baseline。

当前实现的视觉主线包括：

```text
冻结的 CLIP ViT 中间层 patch token
类别特定、分层正常特征记忆库
随机采样或 k-center coreset
正常数据驱动的分层空间可靠度 R(c,l)
空间自适应最近邻检索 lambda(c,l)
中位数/MAD 正常分数校准
正常校准阈值与 oracle 指标分离
MVTec AD 图像级和像素级评估
结果汇总与可视化
```

文本分支尚未实现。后续文本提示、窗口级文本相似度和视觉-文本融合应作为独立可插拔分支增加，不能改变当前 visual-memory 主结果的评价协议。

## 1. 环境

目标服务器环境：

```text
Python 3.10.8
PyTorch 2.1.2
CUDA 12.1
Anomalib 1.1.1（仅用于已有 PatchCore baseline，不是本项目依赖）
GPU RTX 4090
```

强烈建议使用独立 Conda 环境，不要在现有 Anomalib/PatchCore 环境中安装本项目依赖。

### 自动创建环境

```bash
cd /root/autodl-tmp/iad_project/SA_CLIP_NM
bash scripts/create_conda_env.sh
conda activate sa_clip_nm
```

脚本会依次安装：

```text
Python 3.10.8
PyTorch 2.1.2 + CUDA 12.1
torchvision 0.16.2
transformers及其他项目依赖
SA-CLIP-NM editable package
```

如果环境名已经存在，脚本会直接停止，不会覆盖或删除已有环境。

### 手动创建环境

如果不使用脚本：

```bash
cd /root/autodl-tmp/iad_project/SA_CLIP_NM

conda create -y -n sa_clip_nm python=3.10.8 pip setuptools wheel
conda activate sa_clip_nm

python -m pip install --upgrade pip
python -m pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements-base.txt
python -m pip install -e . --no-deps
```

检查环境：

```bash
python scripts/check_environment.py
```

期望看到：

```text
python: 3.10.8
torch: 2.1.2+cu121
torch_cuda: 12.1
cuda_available: true
gpu: NVIDIA GeForce RTX 4090
```

再确认当前命令确实来自新环境：

```bash
which python
python -c "import sys; print(sys.executable)"
conda env list
```

不要在原有 PatchCore 环境中执行 `pip install -r requirements.txt`。

首次运行需要下载：

```text
openai/clip-vit-base-patch16
```

如果服务器不能访问 Hugging Face，请提前下载权重，并将配置中的 `model.checkpoint` 改成本地模型目录。

## 2. 数据目录

默认 MVTec AD 路径：

```text
/home/ubuntu/yyk/datasets/
```

目录示例：

```text
MVTecAD/
|-- bottle/
|   |-- train/good/
|   |-- test/good/
|   |-- test/broken_large/
|   `-- ground_truth/broken_large/
`-- ...
```

数据路径可以在 YAML 中修改：

```yaml
data:
  root: /home/ubuntu/yyk/datasets
```

## 3. 锁定的主实验协议

### CLIP token

实现使用 Hugging Face `CLIPVisionModel`：

```text
hidden_states[0]：patch embedding 输出
hidden_states[k]：经过第 k 个 Transformer block 后的输出
去除第一个 CLS token
默认不施加最终 LayerNorm
对每个 patch token 进行 L2 归一化
```

运行以下命令确认真实输出形状：

```bash
python scripts/inspect_clip_tokens.py
```

默认候选层：

```text
{3, 6, 9, 12}
```

主配置固定使用指定层集合，不根据测试异常标签自动选择最优层。单层实验通过 `--layers` 单独运行，只作为消融分析。

### 预处理

默认采用：

```text
RGB
直接 resize 到 224 x 224
CLIP mean/std
无随机 crop
无随机 flip
```

空间约束要求训练、校准和测试阶段使用完全一致的确定性几何变换。

### 正常数据划分

每类正常训练数据使用固定随机种子划分：

```text
80% memory set
20% normal calibration set
```

测试图像、异常标签和 GT mask 不参与记忆库、空间权重、MAD 参数或实际检测阈值的拟合。

## 4. 运行

先运行五类原型：

```bash
bash scripts/run_dev5.sh
```

或直接运行：

```bash
python run_experiment.py \
  --config configs/mvtec_dev5.yaml \
  --device cuda
```

全 15 类：

```bash
bash scripts/run_mvtec15_adaptive.sh
```

该脚本会将15类放在一次完整运行中，使每层的空间可靠度分位数
`\(\tau_l\)`由全部类别共同确定。不要把15类拆成15个独立的
`--categories`任务，否则单类别运行时自适应空间权重会退化。

如需自定义输出目录：

```bash
OUTPUT_DIR=outputs/mvtec15_run2 \
LOG_FILE=outputs/mvtec15_run2.log \
bash scripts/run_mvtec15_adaptive.sh
```

单层消融：

```bash
python run_experiment.py \
  --config configs/mvtec_dev5.yaml \
  --layers 3 \
  --output-dir outputs/ablation_block3 \
  --device cuda
```

指定类别：

```bash
python run_experiment.py \
  --config configs/mvtec_dev5.yaml \
  --categories bottle screw \
  --device cuda
```

注意：每次不同消融实验应使用不同的 `experiment.output_dir`，避免覆盖其他结果。

空间约束消融：

```bash
# 一次运行 adaptive、none 和 fixed 0.05 三组5类实验
bash scripts/run_dev5_spatial_ablation.sh

# 或分别手动运行

# 不使用空间约束
python run_experiment.py \
  --config configs/mvtec_dev5.yaml \
  --spatial-mode none \
  --output-dir outputs/ablation_spatial_none

# 所有类别和层使用固定 lambda
python run_experiment.py \
  --config configs/mvtec_dev5.yaml \
  --spatial-mode fixed \
  --fixed-lambda 0.05 \
  --output-dir outputs/ablation_spatial_fixed005

# 使用 R(c,l) 映射得到自适应 lambda(c,l)
python run_experiment.py \
  --config configs/mvtec_dev5.yaml \
  --spatial-mode adaptive \
  --lambda-max 0.10 \
  --output-dir outputs/ablation_spatial_adaptive
```

## 5. 输出

```text
outputs/sa_clip_nm_dev5/
|-- resolved_config.json
|-- spatial_reliability_summary.json
|-- metrics_summary.csv
|-- metrics_summary.json
|-- metrics_summary.md
|-- bottle/
|   |-- artifacts/
|   |   |-- split_manifest.json
|   |   |-- memory_bank.pt
|   |   |-- calibration_features.pt
|   |   |-- reliability.json
|   |   `-- calibration.json
|   |-- metrics.json
|   `-- visualizations/
`-- ...
```

指标明确区分：

```text
image_F1_calibrated / pixel_F1_calibrated
image_F1_oracle / pixel_F1_oracle
```

`calibrated` 指标使用正常校准集确定的固定阈值。`oracle` 指标使用测试标签寻找最佳阈值，只能用于参考比较，不能描述为实际部署性能。

## 6. 测试

无需真实 CLIP 权重和 MVTec 数据的烟雾测试：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python smoke_test.py
```

完整单元测试：

```bash
pytest -q
```

烟雾测试覆盖：

```text
正常数据划分
记忆库压缩
空间约束最近邻
空间可靠度
lambda 映射
MAD 校准
图像分数
calibrated/oracle 指标
迷你 MVTec 端到端管线
```

## 7. 推荐实验顺序

1. `bottle` 单类别端到端运行；
2. block 3/6/9/12 单层消融；
3. 五类固定四层融合；
4. `lambda=0`、固定 lambda、自适应 lambda 对比；
5. 无校准、均值/标准差、median/MAD 对比；
6. MVTec AD 全 15 类；
7. WinCLIP 对比；
8. VisA 外部验证；
9. 文本辅助分支；
10. 高分辨率或 coarse-to-fine 局部细化。

## 8. 当前限制

当前版本尚未包含：

```text
文本 prompt 分支
VisA 数据读取
AUPRO
FAISS 加速
高分辨率或多尺度局部细化
自动层选择
```

这些功能应在 visual-memory 主线结果稳定后逐项增加，并通过独立消融验证。
