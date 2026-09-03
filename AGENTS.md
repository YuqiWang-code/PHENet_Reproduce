# PHENet 复现项目代理指南

本文件适用于整个仓库。后续代理在读取、修改、验证或运行本项目时必须遵守以下约定。默认使用中文沟通，代码标识符、命令行参数和日志字段保持英文。

## 1. 项目目标与事实来源

项目目标是在 BCD-foggy 的 `LEVIR-CD-foggy`、`SYSU-CD-foggy`、`LEVIR-CD+foggy` 上复现 PHENet，并使用高度图增强双时相变化检测。

事实来源按以下优先级判断：

1. 当前 `models/` 与 `train_scripts/` 中实际代码。
2. checkpoint 内保存的 `args`、`metrics` 和 `epoch`。
3. `saved_models/Run1/PHENet_Run1_results.xlsx` 与对应 `train.log`。
4. `docs/Upload_gpt/PHENet_Toward_Robust_Fog-Resilient_Change_Detection_in_Remote_Sensing_Imagery_With_a_Physics-Guided_Height-Enhanced_Network.pdf`。
5. `docs/Upload_gpt/server_and_dataset_info.md`。该文件包含不同日期的服务器探查记录，部分“未完成”“无高度图”、双卡命令和旧路径已经过时；不得用旧段落覆盖当前代码和 checkpoint 事实。

发现论文、脚本、日志或 checkpoint 不一致时，必须明确列出差异及采用依据，禁止静默假设它们完全一致。

## 2. 目录职责

- `models/train.py`：训练、逐 epoch 测试集验证、模型统计、日志与 best checkpoint 管理。
- `models/modeling/PHENet.py`：MobileNetV2 双时相编码、HAMF、高度增强差异融合和 PGARM 辅助分支。
- `models/dataloaders/`：A/B、标签和 A/B 高度图的配对加载及同步增强。
- `models/utils/loss.py`：`current` 与 `bce_fg_dice` 两种变化检测损失。
- `models/utils/metrics.py`：基于完整混淆矩阵的 Recall、Precision、OA、F1、IoU、Kappa。
- `models/third_partys/RDAH-Net-main/`：Depth Anything V2 相对深度到 RDAH-Net 高度图的推理流程。
- `models/diagnose_*.py`：伪标签质量、伪标签变体和判别器条件通道诊断；这些脚本不训练模型且不应修改数据集。
- `models/test_change_losses.py`：损失公式与梯度的确定性测试。
- `train_scripts/`：服务器正式训练、高度图生成和消融任务入口。
- `saved_models/Run1/`：本地 Run1 日志、best 权重和结果汇总；不得当作源码目录批量提交。

第三方二进制模型 `models/third_partys/Depth-Anything-V2-Large-hf/model.safetensors` 和 MobileNetV2 的 `.pth` 很大，已经由 `.gitignore` 排除。不要把预训练权重、Run1 权重或数据集提交到 GitHub。

## 3. 服务器固定约定

- 项目根目录：`/home/yqwang/project/PHENet`
- Python：`/home/yqwang/miniconda3/envs/phenet/bin/python`
- 数据根目录：`/storage/BCD-foggy`
- 本轮结果根目录：`/storage/yqwang/PHENet/saved_models/Run1`
- RDAH-Net checkpoint：`/home/yqwang/project/PHENet/pre_checkpoint/checkpoints-track1/104best_model.pth`
- Depth Anything V2：`/home/yqwang/project/PHENet/models/third_partys/Depth-Anything-V2-Large-hf`

正式高度图生成和训练仅使用服务器物理 GPU 1。Shell 脚本内部必须设置 `CUDA_VISIBLE_DEVICES=1`；物理 GPU 1 在 Python 中映射为逻辑 `cuda:0`，所以参数使用 `--device cuda:0` 或 `--gpu-ids 0`。不要改成逻辑 `cuda:1`，不要启用 GPU 0 或双卡。运行前用 `nvidia-smi` 核实 GPU 1 的实时状态；文档中的历史显存占用和 PID 不能视为当前状态。

使用现有 `phenet` 环境，不要无故重建环境或升级 PyTorch。服务器记录的有效组合是 Python 3.10、PyTorch 2.2.0+cu121、torchvision 0.17.0+cu121、Transformers 4.46.3。`models/requirements.txt` 中固定的 torch 2.3.1 是环境声明差异，修改依赖前必须先说明兼容性影响。

## 4. 数据契约

每个训练或验证 split 必须具有：

```text
split/
├── A/
├── B/
├── label/ 或 GT/
├── A_heightmap/
└── B_heightmap/
```

- `LEVIR-CD-foggy` 使用 `label/`；另两个数据集使用 `GT/`。
- 三个实验均以 `train` 训练，并按项目要求使用完整 `test` split 逐 epoch 验证和选择 best；不要改用 SYSU 的 `val`。
- A、B、标签和两张高度图必须文件名完全对应；`fog.txt` 等非图像文件应被忽略。
- RGB 为 `[B,3,256,256]`，标签为 `[B,256,256]`，高度图为 `[B,1,256,256]`。
- 标签统一二值化为 0/1。uint16 高度图必须先保留为浮点数读取，不得转成 uint8。
- 训练几何增强必须同步应用于 A、B、标签和两张高度图；RGB 可另外做 Gaussian blur 和 color jitter。

## 5. 高度图流程及论文差异

当前正式入口是 `train_scripts/generate_heightmaps.sh`，实际流程为：

```text
RGB → 本地离线 Depth Anything V2 Large 相对深度 → RDAH-Net → 单通道 uint16 PNG
```

仅处理三个数据集的 `train`、`test` 下 A/B，输出到同级 `A_heightmap/`、`B_heightmap/`。默认跳过已有文件，只有明确要求重建时才使用 `--overwrite`。本地 Depth Anything 目录必须同时具有 `config.json`、`preprocessor_config.json`、`model.safetensors`；正式脚本设置 `HF_HUB_OFFLINE=1`，不得临时改成联网下载。

论文原文使用自建数据训练的预训练 Swin-UNet 估计高度；本复现使用 Depth Anything V2 + RDAH-Net，这是明确的复现偏差，报告结果时必须披露。禁止以灰度图、常数图或随机图冒充正式高度图。

## 6. 模型、损失与训练行为

- PHENet 输入为 `(image_a, image_b, height_a, height_b)`，输出第一个元素为两类变化 logits；训练态还输出两时相 PGARM 的 fog/clear/transmission/atmospheric-light 张量。
- 高度图进入网络后按样本 min-max 归一化。
- `current`：两类 CrossEntropy + 前景/背景平均 Dice。
- `bce_fg_dice`：使用 `logits[:,1]-logits[:,0]` 的 BCEWithLogits + 仅变化前景 Dice。
- PAC 总损失为 `change + 0.2*TV + 0.5*Dark + 0.5*AdvG`，与论文 λ1=0.2、λ2=0.5、λ3=0.5 一致。
- 默认 `pseudo_mode=frozen` 使用按初始化顺序构建但未训练的冻结 `ShallowCNN` 生成 Otsu 伪标签；`zero` 仅用于受控消融。不要把 `frozen` 描述成有预训练语义的特征提取器。
- 主网络和判别器均使用 Adam；主学习率默认 `1e-4`，`ReduceLROnPlateau(factor=0.9, patience=5)` 按验证损失调节，并同步判别器学习率。
- 新运行会删除目标数据集结果目录下已有的所有 `.pth`，并以写模式重建 `train.log`。运行前必须确认 `--save-dir` 和 `--dataset-name`，避免误删有效权重；需要保留旧结果时先使用新的 Run 名称。
- best 按测试集 F1 选择，只保留一个 `best_F1=0.xxxx.pth`。checkpoint 同时保存网络、优化器、调度器、判别器、metrics 和 args。

## 7. 当前脚本与 Run1 实际配置

- `train_levir_cd_foggy.sh`：当前为 300 epochs、默认 `current` loss。
- `train_sysu_cd_foggy.sh`：当前为 100 epochs、默认 `current` loss。
- `train_levir_cd_plus_foggy.sh`：当前为 100 epochs、默认 `current` loss。
- `train_all_datasets.sh` 名称虽为 “all”，当前只顺序运行 SYSU-CD-foggy 与 LEVIR-CD+foggy，不包含 LEVIR-CD-foggy。
- 两个 `ablate_*.sh` 均为 LEVIR 的 80-epoch 消融，结果写入 `/storage/yqwang/PHENet/saved_models/Ablations`。

论文实验设置为 Adam、`1e-4`、300 epochs、batch size 8；因此当前两个 100-epoch 脚本不等同于论文完整 300-epoch 设置。不要在未重新训练前称其为严格论文复现。

Run1 结果如下，数值来自 Excel、日志和 checkpoint 的交叉核对，指标单位均为百分数：

| 数据集 | Params(M) | FLOPs(G) | FPS | GPU Mem Allocated(MB) | Recall | Precision | OA | F1 | IoU | Kappa | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LEVIR-CD-foggy | 5.242 | 70.927 | 125.373 | 4861.28 | 72.3131 | 69.5936 | 96.9800 | 70.9273 | 54.9514 | 69.3352 | 31 |
| SYSU-CD-foggy | 5.242 | 70.927 | 124.021 | 4861.28 | 67.2110 | 77.3316 | 87.6212 | 71.9170 | 56.1487 | 64.0277 | 46 |
| LEVIR-CD+foggy | 5.242 | 70.927 | 121.988 | 4861.28 | 65.2694 | 65.2333 | 97.1677 | 65.2513 | 48.4245 | 63.7750 | 35 |

服务器权重：

- `/storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD-foggy/best_F1=0.7093.pth`
- `/storage/yqwang/PHENet/saved_models/Run1/SYSU-CD-foggy/best_F1=0.7192.pth`
- `/storage/yqwang/PHENet/saved_models/Run1/LEVIR-CD+foggy/best_F1=0.6525.pth`

必须保留以下来源说明：Run1 的 LEVIR checkpoint 内部 args 显示其原始任务是 `LEVIR-CD-foggy_bce-fg-dice`，`save_dir` 为 `.../Ablations`、`epochs=80`、`change_loss_mode=bce_fg_dice`，而不是当前正式 LEVIR 脚本的 300-epoch `current` 配置。SYSU 与 LEVIR-CD+ checkpoint 均为 100 epochs、`current` loss、`pseudo_mode=frozen`。因此三份 Run1 权重并非完全相同训练配置的横向对比。

## 8. 日志解析注意事项

日志开头记录 Params、FLOPs、FPS、参数显存和 CUDA allocated/reserved memory；Excel 的 `GPU Mem(MB)` 使用 `GPU Mem Allocated(MB)`。

当前 `models/train.py` 的 epoch 表头包含 `TV`，但数据行在 Dice 后直接写入 Dark、AdvG、Disc、ValLoss，遗漏了 `losses['tv']`。因此现有 Run1 日志表头与数据列错位。解析旧日志时必须从每行右侧取最后六项作为 `Recall, Precision, OA, F1, IoU, Kappa`；不要按表头左对齐读取。若修复未来日志，应在 Dice 后补写 TV，但不得篡改既有 Run1 原始日志。

## 9. 修改与验证规则

- 优先修复根因，不伪造数据、不吞异常、不写死实验结果。
- 保留路径参数化和项目相对路径，禁止重新引入开发机或旧服务器硬编码路径。
- 不修改原始 A/B/label/GT；生成高度图前先小样本验证，批量后检查数量、文件名、shape、dtype、有限值和范围。
- 修改数据加载、损失或模型后至少运行 Python 静态编译、`python models/test_change_losses.py`，并用合成数据验证 train/eval 前向、shape、有限值与反向梯度。
- 修改 RDAH-Net 后检查 checkpoint key/shape 严格兼容，并进行 256×256 冒烟推理。
- 修改 Shell 脚本后检查 Bash 语法、GPU 1 映射、Python 路径、数据路径、标签目录、epoch、save root 和引号。
- 修改 `models/` 或 `train_scripts/` 后运行 `python analyse/merge_code_to_txt.py`，更新 `docs/Upload_gpt/models_code_merged.txt`。
- 本地缺少服务器数据或 CUDA 时，只能声明静态/合成测试通过，不得声称服务器训练或推理已执行。
- 不覆盖无关用户修改，不执行 `git reset --hard` 等破坏性操作，不泄露 SFTP 密码、令牌或认证码。

