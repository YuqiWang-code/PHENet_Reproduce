# PHENet 源码与指定文档的 GitHub 推送流程

仓库已经初始化，`origin` 已配置，当前分支为 `main`。允许提交的项目内容为 `.gitignore`、`AGENTS.md`、`models/`、`train_scripts/`、本文件和 PHENet 论文 PDF。以后更新这些内容后，在 PHENet 项目根目录依次执行以下“查看 → 暂存 → 提交 → 推送”四步：

```bash
git status
git add .gitignore AGENTS.md models train_scripts docs/Upload_gpt/server_and_dataset_info.md docs/Upload_gpt/PHENet_Toward_Robust_Fog-Resilient_Change_Detection_in_Remote_Sensing_Imagery_With_a_Physics-Guided_Height-Enhanced_Network.pdf
git commit -m "本次修改说明"
git push
```

将 `本次修改说明` 替换为能够准确概括本次变更的提交信息。执行 `git add` 前先通过 `git status` 检查文件范围，避免把数据集、模型权重、训练结果、日志或敏感配置误提交到 GitHub。论文 PDF 约 4.83 MB，低于 GitHub 单文件 100 MB 限制；但论文页面注明重新发布或再分发需要 IEEE 许可，推送到公开仓库前必须确认拥有分发授权。没有授权时，从 `git add` 命令中删除 PDF 路径，并在仓库文档中改为论文 DOI 或官方页面链接。

---

# 服务器与数据集信息

> 最后更新: 2026-09-03

---

## 服务器硬件配置

### GPU 信息

**数量:** 2x NVIDIA GeForce RTX 4090

| GPU ID | 型号 | 显存 | 温度 | 利用率 | 显存使用 | Driver版本 |
|--------|------|------|------|--------|----------|------------|
| 0 | RTX 4090 | 24564 MiB | 64°C | 63% | 14398 MiB / 24564 MiB | 570.211.01 |
| 1 | RTX 4090 | 24564 MiB | 61°C | 93% | 14513 MiB / 24564 MiB | 570.211.01 |

**CUDA 版本:** 12.8  
**cuDNN 版本:** 8.9.2

**当前GPU进程:**
```
GPU 0: python (PID: 3314799) - 14298 MiB
GPU 1: python (PID: 3454941) - 14484 MiB
```

### CPU 信息

**型号:** Intel(R) Core(TM) i9-14900KF  
**核心数:** 32 (24核心 × 2线程)  
**频率范围:** 800 MHz - 6000 MHz  
**架构:** x86_64

**缓存:**
- L1d: 896 KiB (24 instances)
- L1i: 1.3 MiB (24 instances)
- L2: 32 MiB (12 instances)
- L3: 36 MiB (1 instance)

### 内存信息

**总内存:** 64 GB (65665048 kB)  
**可用内存:** ~44 GB  
**已用内存:** ~11 GB  
**交换分区:** 2.0 GB (已用 647 MB)

### 磁盘信息

| 挂载点 | 设备 | 总容量 | 已用 | 可用 | 使用率 |
|--------|------|--------|------|------|--------|
| / | /dev/nvme0n1p2 | 1.8T | 1.6T | 137G | 93% ⚠️ |
| /data | /dev/nvme1n1p1 | 1.8T | 1.2T | 561G | 68% |
| /data2 | /dev/sda1 | 3.6T | 41G | 3.4T | 2% |
| /storage | storage | 5.4T | 1.2T | 4.0T | 24% |

⚠️ **注意:** 根分区使用率已达 93%，需要定期清理

### 操作系统

**系统:** Ubuntu 22.04.5 LTS (Jammy Jellyfish)  
**内核:** Linux 6.8.0-107-generic  
**架构:** x86_64

### 网络配置

**主机名:** 2024a  
**主网卡:** enp4s0  
**IPv4:** 172.18.232.141/24 (主), 172.18.232.151/24 (次)  
**IPv6:** 2001:da8:1010:a5d0:2c7f:6faf:6288:136c/64

---

## Python 环境

### PHENet 专用环境 ✅

**环境名称:** `phenet` (当前激活)  
**Python 版本:** 3.10.21  
**Python 路径:** `/home/yqwang/miniconda3/envs/phenet/bin/python`  
**环境管理:** Miniconda3  
**环境大小:** 5.7 GB

### PyTorch 配置

✅ **PyTorch 已安装并可用**

| 组件 | 版本 | 状态 |
|------|------|------|
| torch | 2.2.0+cu121 | ✅ |
| torchvision | 0.17.0+cu121 | ✅ |
| torchaudio | 2.2.0+cu121 | ✅ |
| CUDA 运行时 | 12.1 | ✅ |
| cuDNN | 8.9.02 (8902) | ✅ |

**CUDA 可用性:**
- `torch.cuda.is_available()`: ✅ True
- 检测到 GPU 数量: 2
- GPU 0: NVIDIA GeForce RTX 4090 (23.52 GB, Compute 8.9)
- GPU 1: NVIDIA GeForce RTX 4090 (23.53 GB, Compute 8.9)

**CUDA 张量测试:** ✅ 通过
```python
# 测试结果
shape : torch.Size([2, 3, 256, 256])
dtype : torch.float32
device: cuda:0
```

### PHENet 关键依赖

| 包名 | 版本 | 状态 | 说明 |
|------|------|------|------|
| torch | 2.2.0+cu121 | ✅ | 深度学习框架 |
| torchvision | 0.17.0+cu121 | ✅ | 视觉工具库 |
| torch-dct | 0.1.6 | ✅ | DCT变换（去雾核心） |
| opencv-python | 4.11.0.86 | ✅ | 图像处理 |
| numpy | 1.26.4 | ✅ | 数值计算 |
| scipy | 1.15.3 | ✅ | 科学计算 |
| PIL (pillow) | 12.3.0 | ✅ | 图像IO |
| matplotlib | 3.10.9 | ✅ | 可视化 |
| tqdm | 4.70.0 | ✅ | 进度条 |
| tensorboardX | 2.6.5 | ✅ | 训练日志 |
| tensorboard | - | ❌ | 需要安装 |
| timm | 1.0.29 | ✅ | 预训练模型库 |
| transformers | 4.46.3 | ✅ | Hugging Face模型库 |
| einops | 0.8.2 | ✅ | 张量操作工具 |
| thop | 0.1.1.post2209072238 | ✅ | FLOPs计算 |

⚠️ **transformers 已更新到 4.46.3 (兼容 PyTorch 2.2.0)**

**核心依赖验证:**
```bash
$ python -c "import torch, transformers, torch_dct, cv2, thop; print(torch.__version__, transformers.__version__)"
2.2.0+cu121 4.46.3
```

✅ 所有核心依赖可正常导入

### torch-dct 验证

✅ **DCT/IDCT GPU 测试通过**

PHENet 去雾模块依赖 DCT 变换，已验证可在 GPU 上正常运行：

```python
# 测试结果
device       : cuda
input shape  : torch.Size([1, 24, 64, 64])
DCT shape    : torch.Size([1, 24, 64, 64])
IDCT shape   : torch.Size([1, 24, 64, 64])
recon MAE    : 2.02e-07  # 重建误差极小
```

### 其他工具库

| 包名 | 版本 | 用途 |
|------|------|------|
| scikit-learn | 1.7.2 | 机器学习工具 |
| thop | 0.1.1.post2209072238 | FLOPs计算 |
| safetensors | 0.8.0 | 模型序列化 |
| huggingface_hub | 1.29.0 | 模型下载 |

### 环境变量

```bash
PATH=/home/yqwang/miniconda3/envs/phenet/bin:...
CUDA_HOME=               # 未设置
CUDA_VISIBLE_DEVICES=    # 未设置（可全部使用）
LD_LIBRARY_PATH=         # 未设置
```

### 已知问题

❌ **PHENet 模块导入失败**

当前在 `/home/yqwang/project/PHENet` 目录下无法直接导入项目模块：

```python
# 以下导入失败
from modeling.PHENet.PHENet import ...
from dataloaders.make_data_loader_heightmap import ...
from utils.loss import SegmentationLosses
```

**原因:** 项目未添加到 Python 路径

**解决方案 1 (临时):**
```bash
cd /home/yqwang/project/PHENet
export PYTHONPATH="${PYTHONPATH}:/home/yqwang/project/PHENet"
python models/train.py ...
```

**解决方案 2 (永久):**
```bash
cd /home/yqwang/project/PHENet
pip install -e .  # 需要有 setup.py
```

**解决方案 3 (推荐):**
```bash
# 训练时直接从 models/ 目录运行
cd /home/yqwang/project/PHENet/models
python train.py --backbone mobilenet --lr 0.01 ...
```

### 其他可用 Conda 环境

服务器上还有以下相关环境可供参考：

| 环境名 | 可能用途 |
|--------|----------|
| cd_mamba | 变化检测 + Mamba |
| changemamba | 变化检测 + Mamba |
| changerd | 变化检测相关 |
| transformer_cd | Transformer变化检测 |
| uabcd | UAB变化检测 |
| wemcd | WEM变化检测 |

共计 27 个 conda 环境，`phenet` 为当前项目专用环境。

---

## 数据集信息

### 总体概览

**数据集根路径:** `/storage/BCD-foggy`  
**总文件数:** 136,152 个 PNG 图像  
**数据集数量:** 4 个一级数据集

`/storage/BCD-foggy/` 下包含以下数据集：

```
/storage/BCD-foggy/
├── HH/                      (恶劣天气测试集)
├── LEVIR-CD+foggy/          (扩展训练集)
├── LEVIR-CD-foggy/          (主数据集，PHENet首选)
└── SYSU-CD-foggy/           (大规模训练集)
```

**所有图像格式:** PNG, 256×256, 8-bit RGB

### 1. LEVIR-CD-foggy (主数据集) ⭐

**路径:** `/storage/BCD-foggy/LEVIR-CD-foggy/`  
**状态:** ✅ 已就绪，推荐用于 PHENet 训练

#### 目录结构

```
LEVIR-CD-foggy/
├── list/
│   ├── train.txt
│   └── test.txt
├── train/
│   ├── A/               (时相1图像)
│   ├── B/               (时相2图像)
│   └── label/           (变化标签)
└── test/
    ├── A/               (时相1图像)
    ├── B/               (时相2图像)
    └── label/           (变化标签)
```

⚠️ **注意:** 探查日志中检测到 `val/` 目录，但与前面的目录结构存在矛盾，需要进一步确认。

#### 数据量统计

| Split | A | B | label | 总计 |
|-------|--:|--:|------:|-----:|
| train | 7,120 | 7,120 | 7,120 | 21,360 |
| test | 2,048 | 2,048 | 2,048 | 6,144 |
| val (待确认) | 2,048? | 2,048? | 2,048? | 6,144? |

**文件名一致性:**
- train: A/B/label 文件名完全对应 ✅
- test: A/B/label 文件名完全对应 ✅ (test/A 中额外有一个 fog.txt)

#### 文件示例

```
100_10.png, 100_11.png, 100_12.png, ...
184_16.png, 103_2.png, ...
```

**命名规范:** `{scene_id}_{patch_id}.png`

#### 高度图状态

❌ **当前无高度图**

PHENet 需要的完整结构：
```
train/
├── A/
├── B/
├── label/
├── A_heightmap/      ❌ 需要生成
└── B_heightmap/      ❌ 需要生成

test/
├── A/
├── B/
├── label/
├── A_heightmap/      ❌ 需要生成
└── B_heightmap/      ❌ 需要生成
```

### 2. LEVIR-CD+foggy (扩展数据集)

**路径:** `/storage/BCD-foggy/LEVIR-CD+foggy/`  
**状态:** ✅ 可用，数据量较大

#### 目录结构

```
LEVIR-CD+foggy/
├── list/
│   ├── train.txt
│   └── test.txt
├── train/
│   ├── A/
│   ├── B/
│   └── GT/
└── test/
    ├── A/
    ├── B/
    └── GT/
```

#### 数据量统计

| Split | A | B | GT | 总计 |
|-------|--:|--:|---:|-----:|
| train | 10,192 | 10,192 | 10,192 | 30,576 |
| test | 5,568 | 5,568 | 5,568 | 16,704 |
| **总计** | **15,760** | **15,760** | **15,760** | **47,280** |

#### 文件示例

```
638_00.png, 638_01.png, 638_02.png, ...
```

#### 高度图状态

❌ 当前无高度图，如需用于 PHENet 训练需另外生成。

### 3. SYSU-CD-foggy (大规模数据集)

**路径:** `/storage/BCD-foggy/SYSU-CD-foggy/`  
**状态:** ✅ 可用，包含完整 train/val/test 划分

#### 目录结构

```
SYSU-CD-foggy/
├── list/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── train/
│   ├── A/
│   ├── B/
│   └── GT/
├── val/
│   ├── A/
│   ├── B/
│   └── GT/
└── test/
    ├── A/
    ├── B/
    └── GT/
```

#### 数据量统计

| Split | A | B | GT | 总计 |
|-------|---:|---:|---:|-----:|
| train | 12,000 | 12,000 | 12,000 | 36,000 |
| val | 4,000 | 4,000 | 4,000 | 12,000 |
| test | 4,000 | 4,000 | 4,000 | 12,000 |
| **总计** | **20,000** | **20,000** | **20,000** | **60,000** |

#### 文件示例

```
00000.png, 00001.png, 00002.png, ...
```

#### 高度图状态

❌ 当前无高度图，如需用于 PHENet 训练需另外生成。

### 4. HH (恶劣天气测试集)

**路径:** `/storage/BCD-foggy/HH/`  
**状态:** ✅ 可用，专用于泛化评估

#### 目录结构

```
HH/
├── HH-fog-snow/      (雾+雪)
│   └── test/
│       ├── A/
│       ├── B/
│       └── label/
├── HH-normal-fog/    (正常+雾)
│   └── test/
│       ├── A/
│       ├── B/
│       └── label/
└── HH-normal-snow/   (正常+雪)
    └── test/
        ├── A/
        ├── B/
        └── label/
```

#### 数据量统计

| 子集 | A | B | label | 用途 |
|------|--:|--:|------:|------|
| HH-fog-snow | 152 | 152 | 152 | 雾雪组合测试 |
| HH-normal-fog | 152 | 152 | 152 | 正常-雾测试 |
| HH-normal-snow | 152 | 152 | 152 | 正常-雪测试 |

**总计:** 1,374 个文件 (3 × 152 × 3)

#### 文件示例

```
tile_000_000_256x256.png
tile_000_001_256x256.png
...
```

#### 用途

仅用于不同恶劣天气组合下的测试和泛化评估，无训练划分。

### 数据集对比总结

| 数据集 | 训练集 | 验证集 | 测试集 | 总图像对 | 高度图 | 推荐用途 |
|--------|-------:|-------:|-------:|---------:|--------|----------|
| **LEVIR-CD-foggy** | 7,120 | (待确认) | 2,048 | ~9,168 | ❌ | ⭐ PHENet首选 |
| LEVIR-CD+foggy | 10,192 | - | 5,568 | 15,760 | ❌ | 扩展训练 |
| SYSU-CD-foggy | 12,000 | 4,000 | 4,000 | 20,000 | ❌ | 大规模训练 |
| HH | - | - | 456 | 456 | ❌ | 泛化测试 |

### 高度图生成需求

⚠️ **所有数据集均无高度图！**

搜索关键词 (`*height*`, `*heightmap*`, `*dsm*`, `*dem*`, `*elevation*`, `*agl*`) 结果为空。

**PHENet 训练前必须生成高度图：**

```bash
# 快速验证（假高度图，5分钟）
python analyse/generate_fake_heightmaps.py --mode grayscale

# 正式训练（真实高度图，45分钟）
bash analyse/generate_heightmaps.sh --method midas
```

**优先级:**
1. 首先为 LEVIR-CD-foggy 生成高度图（数据量适中，7,120 + 2,048 对）
2. 根据需要为其他数据集生成

详细说明参考:
- [高度图快速参考](../docs/heightmap_quick_reference.md)
- [高度图生成方案](../docs/heightmap_generation_plan.md)

---

## 训练资源评估

### GPU 资源

✅ **充足的训练资源**
- 2x RTX 4090 (24GB each)
- 支持大批量训练 (建议 batch_size=8-16 per GPU)
- 支持分布式训练

### 内存资源

✅ **64GB 系统内存**
- 足够的数据预加载缓存
- 支持多进程数据加载 (num_workers=4-8)

### 磁盘资源

⚠️ **注意事项:**
- 根分区 (/) 空间紧张 (仅剩 137GB)
- 建议将训练输出保存到 `/data2` (3.4TB可用)
- 检查点和日志占用空间较大，定期清理

### 网络带宽

✅ **千兆以太网**
- 支持远程训练监控
- 支持 Tensorboard 远程访问

---

## 推荐配置

### 训练脚本参数

基于当前硬件配置，推荐的训练参数：

```bash
python models/train.py \
    --backbone resnet \
    --lr 0.007 \
    --batch-size 16 \           # 2x RTX 4090, 8 per GPU
    --gpu-ids 0,1 \
    --workers 8 \
    --epochs 100 \
    --dataset-root /storage/BCD-foggy/LEVIR-CD-foggy \
    --checkname phenet_exp1 \
    --save-dir /data2/checkpoints/phenet
```

### 环境变量设置

```bash
# CUDA 设备可见性
export CUDA_VISIBLE_DEVICES=0,1

# cuDNN 性能优化
export CUDNN_BENCHMARK=1

# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 监控建议

1. **GPU 监控:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **磁盘空间监控:**
   ```bash
   df -h / /data /data2
   ```

3. **训练监控:**
   ```bash
   tensorboard --logdir=/data2/checkpoints/phenet/runs
   ```

---

## PHENet 训练准备清单

### 环境就绪项 ✅

- [x] **phenet conda 环境已配置完成**
  - Python 3.10.21
  - PyTorch 2.2.0+cu121
  - CUDA 12.1, cuDNN 8.9.02
  - torch-dct 已安装并测试通过
  - transformers 4.46.3（已更新，兼容 PyTorch 2.2.0）
  - einops 0.8.2（新增）
  - 所有核心依赖已就绪并验证可导入

- [x] **GPU 资源可用**
  - 2x RTX 4090 (24GB each)
  - CUDA 设备正常识别
  - GPU 张量测试通过

### 立即行动项

- [ ] **安装 tensorboard（可选，tensorboardX 已安装）**
  ```bash
  conda activate phenet
  pip install tensorboard
  ```

- [ ] **确认 LEVIR-CD-foggy/val 目录是否真实存在**
  ```bash
  ls -ld /storage/BCD-foggy/LEVIR-CD-foggy/val
  find /storage/BCD-foggy/LEVIR-CD-foggy/val -maxdepth 2 -type d
  ```

- [ ] **验证 RDAH-Net checkpoint 可加载性**
  ```bash
  conda activate phenet
  cd /home/yqwang/project/PHENet
  python -c "
import torch
ckpt = torch.load('/home/yqwang/project/PHENet/pre_checkpoint/checkpoints-HK/best_model.pth')
print('Checkpoint keys:', ckpt.keys())
if 'state_dict' in ckpt:
    print('Model keys sample:', list(ckpt['state_dict'].keys())[:10])
"
  ```

- [ ] **为 LEVIR-CD-foggy 生成高度图** (最高优先级)
  - 先测试 1-10 张图像
  - 验证输出尺寸、通道、数值范围
  - 批量生成 train/test 高度图

### 中期任务

- [ ] 清理根分区磁盘空间 (当前 92% 使用率)
- [ ] PHENet dataloader 冒烟测试
- [ ] PHENet forward 冒烟测试
- [ ] 配置分布式训练环境（双GPU）

### 长期计划

- [ ] 为 LEVIR-CD+foggy 生成高度图 (扩展训练用)
- [ ] 为 SYSU-CD-foggy 生成高度图 (大规模训练用)
- [ ] 为 HH 测试集生成高度图 (泛化评估用)
- [ ] 设置自动备份策略

---

## PHENet 推荐训练流程

### 步骤 1: 确认 val 目录状态

```bash
ls -ld /storage/BCD-foggy/LEVIR-CD-foggy/val
readlink -f /storage/BCD-foggy/LEVIR-CD-foggy/val
```

### 步骤 2: 生成高度图

```bash
# 快速验证（假高度图）
cd /path/to/PHENet
python analyse/generate_fake_heightmaps.py \
    --dataset-root /storage/BCD-foggy/LEVIR-CD-foggy \
    --split train \
    --mode grayscale

# 正式训练（真实高度图）
bash analyse/generate_heightmaps.sh \
    --method midas \
    --dataset-root /storage/BCD-foggy/LEVIR-CD-foggy
```

### 步骤 3: Dataloader 测试

```bash
python models/test_dataloader.py \
    --dataset-root /storage/BCD-foggy/LEVIR-CD-foggy \
    --batch-size 8
```

### 步骤 4: 开始训练

```bash
python models/train.py \
    --backbone mobilenet \
    --lr 0.01 \
    --batch-size 16 \
    --gpu-ids 0,1 \
    --workers 8 \
    --epochs 100 \
    --dataset bcd-foggy \
    --dataset-root /storage/BCD-foggy/LEVIR-CD-foggy \
    --checkname phenet_levir_exp1 \
    --save-dir /data2/checkpoints/phenet
```

---

## 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-08-29 | 初始信息收集，创建文档 |
| 2026-08-31 | 合并详细数据集探查结果，添加4个数据集完整信息 |

---

## 预训练权重信息

### 目录路径

**预训练权重根目录:** `/home/yqwang/project/PHENet/pre_checkpoint/`  
**总磁盘占用:** 约 16 GB

### 目录结构

```
/home/yqwang/project/PHENet/pre_checkpoint/
├── checkpoints-HK/
│   └── best_model.pth          (63 MB)
├── checkpoints-Swiss/
│   └── best_model.pth          (63 MB)
├── checkpoints-track1/
│   └── 104best_model.pth       (63 MB)
├── HK_crop.rar                 (2.1 GB, RAR v5)
└── Swiss_crop.rar              (14 GB, RAR v5)
```

### RDAH-Net 预训练权重

PHENet 使用 RDAH-Net 进行高度图估计，当前已有 3 个预训练模型：

| 数据域 | 路径 | 大小 | 状态 |
|--------|------|-----:|------|
| HK | `checkpoints-HK/best_model.pth` | 63 MB | ✅ 已下载 |
| Swiss | `checkpoints-Swiss/best_model.pth` | 63 MB | ✅ 已下载 |
| Track1 | `checkpoints-track1/104best_model.pth` | 63 MB | ✅ 已下载 |

**文件格式:** 所有 `.pth` 文件识别为 `Zip archive data`（PyTorch 标准序列化格式）

### 数据压缩包

| 文件 | 大小 | 格式 | 用途 |
|------|-----:|------|------|
| HK_crop.rar | 2.1 GB | RAR v5 | HK 数据集裁剪版 |
| Swiss_crop.rar | 14 GB | RAR v5 | Swiss 数据集裁剪版 |

⚠️ **建议:** 暂时不要解压这两个压缩包，等确认生成 LEVIR-CD-foggy 高度图是否需要其中数据后再决定。

### 权重验证状态

❌ **未完成 checkpoint 内部结构检查**

由于 base 环境未安装 PyTorch，以下项目尚未验证：
- checkpoint 顶层对象类型
- 是否包含 `state_dict` / `model` / `model_state_dict`
- 参数 key 命名方式
- 与 RDAH-Net 官方代码的兼容性
- 三个 checkpoint 的网络结构是否一致

**验证方法:**
```python
import torch

# 检查 checkpoint 内容
ckpt = torch.load('/home/yqwang/project/PHENet/pre_checkpoint/checkpoints-HK/best_model.pth')
print(f"Top-level keys: {ckpt.keys()}")
print(f"Type: {type(ckpt)}")

# 检查模型结构
if 'state_dict' in ckpt:
    print(f"Model keys: {list(ckpt['state_dict'].keys())[:10]}")
```

### Depth Anything V2 状态

❓ **未找到 Depth Anything V2 权重**

在 `pre_checkpoint/` 目录中未发现 Depth Anything V2 相关文件。需要确认：
- PHENet 是否需要 Depth Anything V2
- 如需要，从哪里下载及如何集成

### 高度图生成准备状态

| 资源 | 状态 | 说明 |
|------|------|------|
| RDAH-Net 代码 | ❓ | 需要确认服务器上是否已有 |
| RDAH-Net 权重 | ✅ | 3 个预训练模型已下载 |
| Depth Anything V2 | ❓ | 需要确认是否需要 |
| 目标数据集 | ✅ | LEVIR-CD-foggy 已就绪 |
| PyTorch 环境 | ❌ | base 环境未安装 torch |

### 高度图生成前的准备步骤

在为 LEVIR-CD-foggy 生成高度图前，建议按以下顺序操作：

```bash
# 1. 激活有 PyTorch 的环境（或安装 PyTorch）
conda activate <env_with_pytorch>

# 2. 验证 checkpoint 可加载性
python -c "
import torch
ckpt = torch.load('/home/yqwang/project/PHENet/pre_checkpoint/checkpoints-HK/best_model.pth')
print('Checkpoint keys:', ckpt.keys())
"

# 3. 确认 RDAH-Net 代码位置
find /home/yqwang/project -name "*RDAH*" -type d

# 4. 冒烟测试（1-5张图像）
python generate_heightmap_test.py \
    --checkpoint /home/yqwang/project/PHENet/pre_checkpoint/checkpoints-HK/best_model.pth \
    --input /storage/BCD-foggy/LEVIR-CD-foggy/train/A \
    --output /tmp/heightmap_test \
    --num-samples 5

# 5. 检查输出
python -c "
import numpy as np
import cv2
hm = cv2.imread('/tmp/heightmap_test/100_10.png', -1)
print(f'Shape: {hm.shape}, dtype: {hm.dtype}')
print(f'Range: [{hm.min()}, {hm.max()}]')
print(f'NaN: {np.isnan(hm).any()}, Inf: {np.isinf(hm).any()}')
"
```

### 推荐 checkpoint 选择策略

对于 LEVIR-CD-foggy 数据集（建筑变化检测），建议：

1. **首选:** `checkpoints-track1/104best_model.pth`
   - Track1 通常对应竞赛主赛道，可能更通用
   
2. **备选:** `checkpoints-HK/best_model.pth`
   - HK 数据域可能包含城市建筑场景
   
3. **最后:** `checkpoints-Swiss/best_model.pth`
   - Swiss 可能更偏向自然地形

**验证方法:** 对比 3 个模型在相同 5 张图像上的输出质量，选择最佳的。

---

## 备注

- GPU 当前正在运行其他任务，训练前需要检查资源可用性
- 建议使用 `tmux` 或 `screen` 进行长时间训练
- 定期备份检查点到 `/data2` 或远程存储
- 高度图生成建议先小批量测试，验证无误后再批量处理
