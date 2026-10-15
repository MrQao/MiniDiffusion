# MiniDiffusion

从零实现的轻量 DDPM 教学实验：32×32 CIFAR-10、残差 U-Net、正弦时间嵌入、跳跃连接与 1,000 步线性 beta 日程（0.0001→0.02），通过 MSE 学习预测高斯噪声。属于无条件生成，不包含 CLIP 文本条件。

## 文件与运行

`miniDiffusion.py` 包含模型、训练和采样；`mini_diffusion.pth` 为复制的已有检查点；`results/` 保留原生成样例，生成新图会继续写入该目录。CIFAR-10 位于 `data/cifar10`。默认 MPS 可用时使用 MPS，否则 CPU。

```bash
python miniDiffusion.py --help
python miniDiffusion.py sample
python miniDiffusion.py mse
python miniDiffusion.py recover
# 训练会覆盖 mini_diffusion.pth，请先备份
python miniDiffusion.py train --epochs 10
```

新副本增加命令入口，导入不再创建数据集或自动采样；去掉未使用的 Transformers 依赖。无条件采样改为完整 1,000 步；恢复模式在 t=50 加噪后从 t=50 开始逆过程，修正原入口的时间步偏移。当前采样入口仍初始化 CIFAR-10 训练加载器，因此无本地数据时也会下载数据。

原历史报告 t=50 单步 x0 估计 PSNR 29.50 dB、SSIM 0.928；这不等于完整反向链采样质量。本次未重跑定量评测，未测 FID/KID，不宣称高质量生成。

## 环境安装

建议 Python 3.11，在项目根目录执行（所有相对路径均以项目根目录为基准）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

核心版本依据原实验环境整理；并非跨平台锁文件。首次运行可能下载数据与预训练模型，需要网络与足够磁盘空间。原环境为 Apple Silicon，其他平台需选择适配硬件的 PyTorch 安装包。本次拆分未重新训练或验证完整模型指标。

## 本地资源与 GitHub

本地保留本项目对应的数据及权重副本，不引用原 MiniCLIPTest，也不使用符号链接。`.gitignore` 排除数据、权重、运行产物、缓存和 IDE 配置；这些文件不会随普通 Git 提交上传。其他设备需要下载基础模型/数据，并另行获取自有训练检查点。可通过 GitHub Releases 或 Git LFS 管理权重；请确认数据和基础模型的许可后再分发。未擅自添加开源许可证，请在公开前自行选择代码许可证。

上传到新建空仓库时，可在本目录执行：

```bash
git init
git add .
git status --short
# 确认暂存内容后
git commit -m "Initial commit"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

数据与权重仅保留在本地，Git 仓库只包含代码、依赖与说明文件。
