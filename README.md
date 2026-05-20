# Score\-Based Point Cloud Denoising \(ICCV\&\#39;21\)

📌 本仓库采用基于分数的生成模型实现高质量点云去噪，同时支持普通尺寸（≤ 50K 个点）和大规模（\&gt; 50K 个点）的点云数据处理。

📄 **论文链接**：[https://arxiv\.org/abs/2107\.10981](https://arxiv.org/abs/2107.10981)

📸 项目效果预览图：

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YmU4YTMxOTA5OTVhMjE5ZWQzOTFiODMwZTE5MDZiYmFfNzUxNjZjYjE2N2YxZTdhOTcyZTQ5NGNkMjA5ODcwNmFfSUQ6NzY0MTkzMjQ1NTg2MzUzNjgyMF8xNzc5Mjc2Mzk5OjE3NzkzNjI3OTlfVjM)

## 📋 目录

- 🔍 [1\. 项目介绍](https://www.doubao.cn)（含模型原理）

- ⚙️ [2\. 环境配置](https://www.doubao.cn)

- 📊[3\. 数据集准备](https://www.doubao.cn)（含数据预处理）

- 🚀 [4\. 去噪使用方法](https://www.doubao.cn)

- 📈 [5\. 模型训练](https://www.doubao.cn)

- 📝 [6\. 引用格式](https://www.doubao.cn)

- ❓ [7\. 常见问题（FAQ）](https://www.doubao.cn)

- ⚠️ [8\. 重要说明](https://www.doubao.cn)

- ✅ [9\. 模型推理验证](https://www.doubao.cn)

## 🔍 1\. 项目介绍

点云去噪是三维计算机视觉中的基础任务，核心目标是在去除受损点云噪声的同时，完整保留原始几何结构。本项目采用基于分数的生成模型对干净点云的分布进行建模，无需依赖手工设计的先验知识，即可实现高效的噪声去除。

🌟 **核心特点**：

- ✅ 支持双尺寸点云：兼顾普通尺寸（≤ 50K 个点）和大规模（\&gt; 50K 个点）点云去噪需求；

- ✅ 性能领先：在 PUNet 等标准点云去噪数据集上实现当前最优性能；

- ✅ 易用性强：提供简洁直观的测试、训练脚本，无需复杂配置即可上手；

- ✅ 稳定性高：兼容 PyTorch 1\.9\.0 和 CUDA 11\.1，经过充分测试，运行稳定。

### 1\.1 模型核心原理（分数匹配去噪）

🔬 本项目的核心是基于**分数匹配（Score Matching）**的生成模型，其去噪逻辑源于对“噪声点云分布”的数学建模，结合论文核心思想，拆解为以下4个关键步骤，通俗易懂且贴合技术本质：

1. **噪声分布建模（核心前提）**

      我们采集到的带噪声点云，其分布可看作是“干净点云分布 $p(x)$”与“噪声模型分布 $n$”的卷积，即 $(p * n)(x)$。其中，这个卷积分布的“模态（Mode）”就是我们需要恢复的干净点云表面——简单来说，带噪声点云的整体分布趋势，始终围绕着干净点云的几何结构波动。

2. **分数函数的核心作用**

      去噪的关键的是找到“从噪声点云回归到干净点云”的方向，而这个方向由**分数函数（Score Function）**决定。分数函数的本质是“对数概率函数的梯度”，即$\nabla_x \log (p * n)(x)$，它能直接指示：每个带噪声的点，朝着哪个方向移动，才能更接近干净点云的分布（也就是卷积分布的模态）。

3. **分数函数的神经网络估计**
实际测试时，我们无法直接获取干净点云的分布 $p(x)$，因此也无法直接计算分数函数。本项目通过设计专用神经网络，仅使用带噪声点云作为输入，即可精准估计出分数函数$\nabla_x \log (p * n)(x)$——这也是模型的核心创新点，无需干净点云标签，仅通过噪声数据就能完成训练。

4. **梯度上升去噪流程**

      得到估计的分数函数后，采用“梯度上升”策略实现去噪：迭代更新每个带噪声点的坐标，让每个点不断朝着分数函数指示的方向移动，逐步提升其在卷积分布 $(p * n)(x)$ 中的对数似然，最终收敛到干净点云的几何结构上。迭代次数可根据噪声水平调整（噪声越大，迭代次数越多，默认1\-2次即可满足多数场景）。

💡 补充说明：模型的训练核心是“分数匹配损失”，通过最小化模型估计的分数与真实分数之间的误差，让神经网络学会精准预测去噪方向。与传统去噪方法相比，该方法无需手动设计滤波规则或几何先验，能自适应不同噪声模型（如高斯噪声），在保留细节的同时实现更彻底的噪声去除。

## ⚙️ 2\. 环境配置

推荐使用 Conda 创建独立环境，避免依赖冲突。本项目代码已在以下环境中经过完整测试，可直接参考配置。

### 2\.1 推荐环境配置

|📦 依赖包|🔢 版本号|📝 说明|
|---|---|---|
|Python|3\.8|推荐版本，与 PyTorch 1\.9\.0 完美兼容|
|PyTorch|1\.9\.0|模型训练和推理的核心框架|
|CUDA|11\.1|需 GPU 加速（不推荐仅使用 CPU 运行）|
|point\_cloud\_utils|0\.18\.0|用于评估：加载网格并计算点到网格的距离|
|pytorch3d|0\.5\.0|用于评估：计算点到网格的距离|
|pytorch\-cluster|1\.5\.9|使用 fps（最远点采样）合并去噪后的补丁|
|tqdm / scipy / scikit\-learn|最新稳定版|用于进度条、数据处理和评估|
|pyyaml / easydict|最新稳定版|用于配置文件解析|
|tensorboard / pandas|最新稳定版|用于训练可视化和结果记录|

### 2\.2 安装方法

#### 方法 1：Conda 一键安装（推荐）

📥 项目提供预配置的 `env\.yml` 文件，可自动安装所有依赖，无需手动操作：

```bash

# 克隆仓库
git clone https://github.com/luost26/score-denoise.git
cd score-denoise

# 创建并激活环境
conda env create -f env.yml
conda activate score-denoise

```

#### 方法 2：手动安装

🔧 若需自定义环境，可按以下步骤手动安装依赖：

```bash

# 创建 Conda 环境
conda create --name score-denoise python=3.8
conda activate score-denoise

# 安装 PyTorch 1.9.0 + CUDA 11.1
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

# 安装基础依赖
conda install -c conda-forge tqdm scipy scikit-learn pyyaml easydict tensorboard pandas

# 安装 point_cloud_utils（用于评估）
conda install -c conda-forge point_cloud_utils==0.18.0

# 安装 PyTorch3d（用于评估）
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c pytorch3d pytorch3d==0.5.0

# 安装 pytorch-cluster（用于 fps 采样）
conda install -c pyg pytorch-cluster==1.5.9
      
```

### 2\.3 环境验证

✅ 安装完成后，运行以下命令验证环境是否配置成功：

```bash

# 检查 PyTorch 版本和 CUDA 可用性
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 是否可用:', torch.cuda.is_available())"

# 检查关键依赖是否安装成功
python -c "import point_cloud_utils; import pytorch3d; import torch_cluster; print('所有依赖安装成功！')"
      
```

若未报错，则说明环境配置成功，可以正常运行项目。

## 📊 3\. 数据集准备

本项目提供论文中使用的测试数据集，可通过以下链接下载，下载后解压至指定目录即可使用。同时，为提升去噪效果、适配模型输入要求，需对输入点云（无论是官方数据集还是自定义数据集）进行必要的数据预处理，具体步骤如下。

### 3\.1 数据集下载

⚠️ **注意**：数据集下载链接为境外地址，当前可能无法直接访问，建议使用合规网络环境尝试。

    📥 下载链接：[Google Drive](https://drive.google.com/drive/folders/1--MvLnP7dsBgBZiu46H0S32Y1eBa_j6P?usp=sharing)

### 3\.2 数据集解压

下载 `data\.zip` 文件后，将其解压至项目的`data/` 目录下，最终目录结构如下：

```bash

score-denoise/
├── data/                  # 数据集目录
│   ├── PUNet/             # PUNet 数据集（论文测试用）
│   │   ├── 10000_poisson/ # 10K 点云数据
│   │   └── 50000_poisson/ # 50K 点云数据
│   └── ... (其他数据集文件)
├── datasets/              # 数据集相关脚本
├── models/                # 模型定义目录
└── ... (其他项目文件)
     
```

### 3\.3 自定义数据集

🔧 若需使用自定义数据集进行去噪或训练，请遵循以下规则：

- 📄 点云文件需为`\.xyz` 格式（每行包含 3 个浮点数，分别代表点的 x、y、z 坐标）；

- 📈 训练时，需将数据集按照提供的 PUNet 数据集目录结构组织（参考 `data/PUNet/`）；

- 🧪 测试时，可直接使用`test\_single\.py` 或 `test\_large\.py` 脚本，指定输入和输出路径即可（详见第 4 节）。

### 3\.4 数据预处理（关键步骤）

🔍 数据预处理是提升点云去噪效果的核心前提，其核心目标是：去除无效数据、统一数据格式、降低噪声干扰，确保输入模型的点云数据符合训练/推理要求。无论是官方数据集还是自定义数据集，建议按以下步骤完成预处理，预处理后的数据可直接用于模型训练或去噪测试。

#### 3\.4\.1 预处理核心目的

1\.  去除无效干扰：剔除点云中的离群点、冗余点和缺失值，避免此类数据影响模型对分数函数的估计；

    2\.  统一数据规格：将不同来源、不同尺度的点云数据标准化，确保模型输入的一致性；

    3\.  优化数据分布：调整点云密度，减少局部点云过密或过疏的情况，提升模型去噪的均匀性；

    4\.  适配模型输入：确保点云点数、坐标范围符合模型要求，避免因数据不规范导致的训练失败或去噪效果不佳。

#### 3\.4\.2 具体预处理步骤（附代码示例）

以下步骤基于 Python 实现，依赖 scipy、numpy 等基础库（已在环境配置中安装），可直接整合到自定义数据集的处理脚本中，也可单独运行预处理脚本对单个点云文件进行处理。

```python

# 导入所需库
import numpy as np
from scipy import stats
import point_cloud_utils as pcu

# 1. 读取点云数据（.xyz格式）
def read_xyz(file_path):
    """读取.xyz格式点云文件，返回坐标数组"""
    try:
        pc = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
        # 确保点云为3维坐标（x,y,z），剔除无效列
        if pc.shape[1] > 3:
            pc = pc[:, :3]
        return pc
    except Exception as e:
        print(f"读取点云文件失败：{e}")
        return None

# 2. 去除缺失值和异常值（离群点）
def remove_outliers(pc, z_threshold=3):
    """
    去除点云中的离群点和缺失值
    pc: 点云坐标数组 (N, 3)
    z_threshold: Z-score阈值，默认3（超过3倍标准差视为离群点）
    """
    # 去除缺失值（包含NaN的点）
    pc = pc[~np.isnan(pc).any(axis=1)]
    # 去除无穷大值
    pc = pc[~np.isinf(pc).any(axis=1)]
    # 基于Z-score去除离群点
    z_scores = stats.zscore(pc, axis=0)
    pc = pc[(np.abs(z_scores) < z_threshold).all(axis=1)]
    return pc

# 3. 点云标准化（统一坐标范围）
def normalize_pc(pc):
    """将点云坐标标准化到[-1, 1]范围，保留相对几何结构"""
    # 计算点云的边界框
    min_coord = pc.min(axis=0)
    max_coord = pc.max(axis=0)
    # 平移到原点
    pc_translated = pc - (min_coord + max_coord) / 2
    # 缩放至[-1, 1]
    scale_factor = np.max(max_coord - min_coord) / 2
    if scale_factor == 0:
        return pc_translated
    pc_normalized = pc_translated / scale_factor
    return pc_normalized

# 4. 点云重采样（统一点数，适配模型输入）
def resample_pc(pc, target_num=10000):
    """
    点云重采样，统一点数
    target_num: 目标点数（默认10000，可根据模型设置调整）
    """
    current_num = pc.shape[0]
    if current_num == target_num:
        return pc
    # 点数过多：使用最远点采样（FPS）降采样
    elif current_num > target_num:
        # 使用pytorch-cluster的fps函数，或point_cloud_utils的采样函数
        indices = pcu.farthest_point_sampling(pc, target_num)
        return pc[indices]
    # 点数过少：使用线性插值补点（保留几何结构）
    else:
        # 计算补点数量，均匀插值
        补点数量 = target_num - current_num
        # 随机选择现有点作为插值起点
        idx = np.random.choice(current_num, 补点数量, replace=True)
        # 微小扰动生成新点，避免与原有点重合
        perturbations = np.random.normal(0, 0.001, (补点数量, 3))
        new_points = pc[idx] + perturbations
        # 合并原有点和新点
        pc_resampled = np.vstack((pc, new_points))
        return pc_resampled

# 5. 预处理主函数（整合所有步骤）
def preprocess_pc(input_path, output_path, target_num=10000):
    """
    点云预处理主函数
    input_path: 输入.xyz文件路径
    output_path: 预处理后文件保存路径
    target_num: 目标点数
    """
    # 读取点云
    pc = read_xyz(input_path)
    if pc is None or pc.shape[0] < 100:  # 过滤点数过少的无效点云
        print("无效点云文件，点数过少")
        return
    # 执行预处理步骤
    pc = remove_outliers(pc)
    pc = normalize_pc(pc)
    pc = resample_pc(pc, target_num)
    # 保存预处理后的点云
    np.savetxt(output_path, pc, fmt='%.6f', delimiter=' ')
    print(f"预处理完成，保存至：{output_path}，点数：{pc.shape[0]}")

# 示例：处理单个点云文件
if __name__ == "__main__":
    input_file = "data/custom/noisy.xyz"  # 输入带噪声点云路径
    output_file = "data/custom/preprocessed_noisy.xyz"  # 预处理后保存路径
    preprocess_pc(input_file, output_file, target_num=10000)
     
```

#### 3\.4\.3 预处理注意事项

- 📌 离群点去除：Z\-score阈值可根据点云噪声水平调整，噪声较大时可适当降低阈值（如2\.5），避免误删有效点；

- 📏 标准化：若需保留原始点云的真实尺度，可跳过标准化步骤，或在去噪后反向缩放恢复原始坐标；

- 📈 重采样：训练时建议将所有点云统一到相同点数（如10000或50000），与模型设置的point\_num参数保持一致；测试时可根据点云实际尺寸调整，无需强制统一；

- 🔧 批量处理：若需处理大量点云文件，可编写循环调用preprocess\_pc函数，批量完成预处理；

- 💡 真实场景数据：对于3D扫描仪等真实设备采集的点云，建议先进行去重处理（去除重复坐标的点），再执行上述预处理步骤，进一步提升去噪效果。

补充说明：官方PUNet数据集已完成基础预处理，可直接用于模型训练和测试；若需进一步优化效果，可对其再次执行上述预处理步骤（重点是去除离群点和重采样）。

## 🚀 4\. 去噪使用方法

本项目提供 3 个测试脚本，分别对应不同使用场景：复现论文结果、普通尺寸点云去噪、大规模点云去噪，所有脚本均支持命令行参数配置，灵活便捷。

### 4\.1 复现论文结果

📝 若需复现论文中 PUNet 数据集上的去噪结果，可运行以下命令，参数与论文设置完全一致。

#### PUNet 数据集（10K 点）

```bash
# 噪声水平 σ=0.01，迭代 1 次
python test.py --dataset PUNet --resolution 10000_poisson --noise 0.01 --niters 1

# 噪声水平 σ=0.02，迭代 1 次
python test.py --dataset PUNet --resolution 10000_poisson --noise 0.02 --niters 1

# 噪声水平 σ=0.03，迭代 2 次
python test.py --dataset PUNet --resolution 10000_poisson --noise 0.03 --niters 2
      
```

#### PUNet 数据集（50K 点）

```bash

# 噪声水平 σ=0.01，迭代 1 次
python test.py --dataset PUNet --resolution 50000_poisson --noise 0.01 --niters 1

# 噪声水平 σ=0.02，迭代 1 次
python test.py --dataset PUNet --resolution 50000_poisson --noise 0.02 --niters 1

# 噪声水平 σ=0.03，迭代 2 次
python test.py --dataset PUNet --resolution 50000_poisson --noise 0.03 --niters 2
      
```

### 4\.2 普通尺寸点云去噪（≤ 50K 点）

📌 对于普通尺寸点云（不超过 50,000 个点），使用 `test\_single\.py` 脚本，可指定输入输出路径，或直接运行查看快速示例。

```bash

# 基础用法（指定输入和输出路径）
python test_single.py --input_xyz <输入文件路径> --output_xyz <输出文件路径>

# 快速示例（使用项目默认测试文件）
python test_single.py
```

💡 说明：将 `\&lt;输入文件路径\&gt;` 替换为你的带噪声点云文件路径（例如`data/custom/noisy\.xyz`），`\&lt;输出文件路径\&gt;` 替换为去噪结果的保存路径（例如`results/denoised\.xyz`）。

### 4\.3 大规模点云去噪（\&gt; 50K 点）

📌 对于大规模点云（超过 50,000 个点），使用 `test\_large\.py` 脚本，该脚本采用基于补丁的去噪策略，避免出现内存不足（OOM）问题。

```bash

# 基础用法（指定输入和输出路径）
python test_large.py --input_xyz <输入文件路径> --output_xyz <输出文件路径>

# 快速示例（使用项目默认大规模测试文件）
python test_large.py
      
```

💡 说明：对于超大规模点云（例如 \&gt; 100K 点），可调整 `\-\-patch\_size` 参数（默认值：50000），平衡运行速度和去噪质量——补丁尺寸越小，内存占用越低，但计算时间可能增加。

### 4\.4 关键参数说明

|⚙️ 参数|📋 描述|📊 默认值|
|---|---|---|
|\-\-dataset|数据集名称（仅在 test\.py 中使用）|PUNet|
|\-\-resolution|点云分辨率（例如 10000\_poisson）|10000\_poisson|
|\-\-noise|噪声水平（σ 值）|0\.01|
|\-\-niters|去噪迭代次数|1|
|\-\-input\_xyz|输入带噪声点云文件路径|默认测试文件|
|\-\-output\_xyz|去噪结果保存路径|results/denoised\.xyz|
|\-\-patch\_size|大规模点云去噪的补丁尺寸（仅 test\_large\.py 可用）|50000|

## 📈 5\. 模型训练

若需从零开始训练基于分数的去噪模型，使用 `train\.py` 脚本即可，脚本中包含可调整参数，可根据自身需求修改。

### 5\.1 基础训练命令

```bash

python train.py
      
```

### 5\.2 可调整参数

🔧 `train\.py`中的关键可调整参数（可直接在脚本中修改）：

- `batch\_size`：训练批次大小（默认：32；根据 GPU 内存调整）；

- `lr`：学习率（默认：1e\-4；推荐范围：1e\-4 \~ 1e\-3）；

- `epochs`：训练轮数（默认：100；足以实现收敛）；

- `noise\_levels`：训练使用的噪声水平列表（默认：\[0\.01, 0\.02, 0\.03\]）；

- `point\_num`：每个点云的点数（默认：10000）；

- `save\_freq`：模型 checkpoint 保存频率（默认：每 10 轮保存一次）。

### 5\.3 训练注意事项

- 🖥️ 硬件要求：训练需至少 16GB 内存的 GPU（推荐：24GB 及以上），对应批次大小 32、每个样本 10K 点；

- 💾 模型保存：训练后的模型 checkpoint 会保存至 `pretrained/`目录；

- 📊 训练可视化：可使用 TensorBoard 查看训练过程，命令：`tensorboard \-\-logdir=logs`；

- 🔧 微调模型：若需在自定义数据集上微调模型，修改脚本中的 `dataset\_path` 参数，指向自定义数据集目录即可。

## 📝 6\. 引用格式

若在你的研究中使用了本项目的代码或方法，请引用以下论文：

```latex

@InProceedings{Luo_2021_ICCV,
    author    = {Luo, Shitong and Hu, Wei},
    title     = {Score-Based Point Cloud Denoising},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4583-4592}
}
      
```

## ❓ 7\. 常见问题（FAQ）

### Q1：为什么 PCNet 测试集的 P2M 结果会出现波动？

A：P2M（点到网格）结果可能会因 GPU 架构和 PyTorch 版本不同而出现波动，但这种波动与 CD（倒角距离）指标存在强线性相关性，不会影响本研究的主要结论。

### Q2：如何解决“内存不足（OOM）”错误？

A：有三种解决方法：

- 🔧 减小 `train\.py` 中的批次大小（例如从 32 改为 16）；

- 📌 对于大规模点云，使用 `test\_large\.py` 并减小 `\-\-patch\_size` 参数；

- 🖥️ 使用内存更大的 GPU（推荐：≥ 24GB）。

### Q3：无法导入 pytorch3d 或 point\_cloud\_utils 怎么办？

A：确保安装了正确版本的依赖。对于 pytorch3d，需先安装 fvcore 和 iopath（详见 2\.2 节）；若问题仍未解决，尝试通过 Conda 重新安装相关依赖。

### Q4：如何可视化去噪后的点云？

A：可使用 CloudCompare、MeshLab 或 PyVista 等工具打开 `\.xyz` 文件。例如，在 CloudCompare 中，直接将去噪后的 `\.xyz` 文件拖拽至软件中即可查看。

### Q5：该方法可用于真实场景中的带噪声点云吗？

A：可以。本方法具有通用性，可应用于真实场景中的带噪声点云（例如 3D 扫描仪扫描的数据）。建议先对输入点云进行预处理，去除离群点，以获得更好的去噪效果。

## ⚠️ 8\. 重要说明

- 📚 学术用途：本代码仅用于学术研究，商业用途请联系论文作者；

- 💾 预训练模型：本仓库不包含预训练模型，需使用 `train\.py` 自行训练，或联系作者获取预训练权重；

- 🔧 问题反馈：若遇到 bug 或有疑问，请在仓库中提交 issue，或通过邮件联系作者；

- 🖥️ 系统兼容性：代码已在 Ubuntu 18\.04/20\.04 中测试通过；Windows 用户可能需要手动安装部分依赖（例如 pytorch3d）。

## ✅ 9\. 模型推理验证

模型推理验证是确保去噪效果、验证模型稳定性和泛化能力的关键步骤，核心目标是通过定量指标和定性观察，验证模型输出的去噪点云是否符合预期，是否保留原始几何结构、有效去除噪声。本章节将详细介绍推理验证的完整流程、核心指标、实现方法及注意事项，适配本项目的双尺寸点云去噪场景。

### 9\.1 推理验证核心目的

- 📊 定量评估：通过标准化指标，量化去噪效果（噪声去除程度、几何结构保留精度），验证模型性能是否达到论文水平或预期目标；

- 🧪 定性观察：直观对比带噪声点云、去噪后点云及干净点云（若有），检查是否存在过度平滑、细节丢失或噪声残留问题；

- 🔍 稳定性验证：验证模型在不同噪声水平、不同点云尺寸、不同几何结构下的推理一致性，避免出现极端场景下的性能退化；

- ⚙️ 参数调试：通过验证结果，优化去噪参数（如迭代次数、补丁尺寸），进一步提升特定场景下的去噪效果。

### 9\.2 核心验证指标（贴合论文标准）

结合论文《Score\-Based Point Cloud Denoising》及点云去噪领域通用标准，推荐使用以下 4 个核心指标，涵盖噪声去除、几何保留、精度误差三个维度，可直接通过项目已安装的依赖（point\_cloud\_utils、pytorch3d）实现计算。

#### 9\.2\.1 定量指标说明

- **倒角距离（Chamfer Distance, CD）**：核心指标，衡量去噪后点云与干净点云（或标准网格）之间的平均距离，值越小，去噪精度越高、几何结构保留越好。计算公式为：$CD(P, Q) = \frac{1}{|P|}\sum_{p \in P} \min_{q \in Q} \|p - q\|_2 + \frac{1}{|Q|}\sum_{q \in Q} \min_{p \in P} \|q - p\|_2$，其中 $P$ 为去噪后点云，$Q$ 为干净点云/标准网格。

- **点到网格距离（Point\-to\-Mesh Distance, P2M）**：辅助指标，计算去噪后点云每个点到原始干净网格（若有）的最短距离，平均值越小，说明去噪点云与真实几何结构的偏差越小，常用于论文结果复现和性能对比。

- **均方根误差（Root Mean Square Error, RMSE）**：衡量去噪后点云与干净点云的坐标偏差，值越小，噪声去除越彻底、坐标精度越高，适合用于定量对比不同去噪方法的效果。

- **结构相似度（Structural Similarity, SSIM）**：衡量去噪后点云与干净点云的几何结构相似度，取值范围 \[0,1\]，越接近 1，说明几何细节保留越完整，避免过度去噪导致的结构失真。

#### 9\.2\.2 指标计算代码示例

以下代码可直接整合到推理脚本中，批量计算去噪结果的各项指标，依赖 point\_cloud\_utils 和 pytorch3d，与项目环境完全兼容：

```python
# 导入所需库
import numpy as np
import point_cloud_utils as pcu
from pytorch3d.loss import chamfer_distance
import torch

# 读取点云/网格数据（适配.xyz点云、.ply网格）
def load_data(noisy_path, denoised_path, clean_mesh_path=None, clean_pc_path=None):
    """
    读取推理验证所需数据
    noisy_path: 带噪声点云路径（.xyz）
    denoised_path: 去噪后点云路径（.xyz）
    clean_mesh_path: 干净网格路径（.ply，用于计算P2M）
    clean_pc_path: 干净点云路径（.xyz，用于计算CD、RMSE、SSIM）
    """
    # 读取带噪声点云和去噪点云
    noisy_pc = np.loadtxt(noisy_path, delimiter=' ', dtype=np.float32)[:, :3]
    denoised_pc = np.loadtxt(denoised_path, delimiter=' ', dtype=np.float32)[:, :3]
    # 读取干净数据（根据实际情况选择网格或点云）
    clean_mesh = pcu.load_mesh_vf(clean_mesh_path) if clean_mesh_path else None
    clean_pc = np.loadtxt(clean_pc_path, delimiter=' ', dtype=np.float32)[:, :3] if clean_pc_path else None
    return noisy_pc, denoised_pc, clean_mesh, clean_pc

# 计算核心验证指标
def calculate_metrics(denoised_pc, clean_mesh=None, clean_pc=None):
    """
    计算去噪结果的各项验证指标
    denoised_pc: 去噪后点云（N, 3）
    clean_mesh: 干净网格（顶点数组, 
```

> （注：文档部分内容可能由 AI 生成）
