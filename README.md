# 3D-
# Score\-Based Point Cloud Denoising \(ICCV\&\#39;21\) 

📌 本仓库是论文《Score\-Based Point Cloud Denoising》（ICCV 2021）的官方实现，采用基于分数的生成模型实现高质量点云去噪，同时支持普通尺寸（≤ 50K 个点）和大规模（\&gt; 50K 个点）的点云数据处理。

📄 **论文链接**：[https://arxiv\.org/abs/2107\.10981](https://arxiv.org/abs/2107.10981)

📸 项目效果预览图：

![Image](https://internal-api-drive-stream.larkoffice.com/space/api/box/stream/download/authcode/?code=NzllMDFmZjYxN2I3YzBiZThiYWZiYTQxZDQ4MzIyYjVfMzA5NjEzOWJhODIzYTNjYWI5YjQ2YmUxNDg2ZGRiMGVfSUQ6NzY0MTgyMzU2MDcyMTI4ODE1Nl8xNzc5MjUzMTU3OjE3NzkzMzk1NTdfVjM)

## 📋 目录

- 🔍 [1\. 项目介绍](https://www.doubao.cn)（含模型原理）

- ⚙️ [2\. 环境配置](https://www.doubao.cn)

- 📊[3\. 数据集准备](https://www.doubao.cn)

- 🚀 [4\. 去噪使用方法](https://www.doubao.cn)

- 📈 [5\. 模型训练](https://www.doubao.cn)

- 📝 [6\. 引用格式](https://www.doubao.cn)

- ❓ [7\. 常见问题（FAQ）](https://www.doubao.cn)

- ⚠️ [8\. 重要说明](https://www.doubao.cn)

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

      去噪的关键的是找到“从噪声点云回归到干净点云”的方向，而这个方向由**分数函数（Score Function）**决定。分数函数的本质是“对数概率函数的梯度”，即 $\nabla_x \log (p * n)(x)$，它能直接指示：每个带噪声的点，朝着哪个方向移动，才能更接近干净点云的分布（也就是卷积分布的模态）。

3. **分数函数的神经网络估计**

      实际测试时，我们无法直接获取干净点云的分布 $p(x)$，因此也无法直接计算分数函数。本项目通过设计专用神经网络，仅使用带噪声点云作为输入，即可精准估计出分数函数 $\nabla_x \log (p * n)(x)$——这也是模型的核心创新点，无需干净点云标签，仅通过噪声数据就能完成训练。

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

本项目提供论文中使用的测试数据集，可通过以下链接下载，下载后解压至指定目录即可使用。

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

💡 说明：将 `\&lt;输入文件路径\&gt;` 替换为你的带噪声点云文件路径（例如 `data/custom/noisy\.xyz`），`\&lt;输出文件路径\&gt;` 替换为去噪结果的保存路径（例如`results/denoised\.xyz`）。

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

📅 最后更新：2026年5月
