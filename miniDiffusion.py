import math

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torch.utils.data import random_split

import random
from torch.utils.data import Sampler
from torch.utils.data import Subset

import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()
        half_dim = 64

        freq = torch.exp(
            -math.log(10000) *
            torch.arange(half_dim) / half_dim
        )
        self.register_buffer("freq", freq)

    def forward(self, t):
        args = t[:, None] * self.freq[None, :]  # [B, 64]

        emb = torch.cat([
            torch.sin(args),
            torch.cos(args)
        ], dim=1)  # [B, 128]
        return emb

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.time_linear = nn.Linear(128, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.act = nn.SiLU()

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1
            )

    def forward(self, x, t_emb):
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        t = self.time_linear(t_emb)
        t = t[:,:,None,None]
        x = x + t
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)
        return x

class MiniUNet(nn.Module):
    def __init__(self):
        super(MiniUNet, self).__init__()

        self.time_emb = TimeEmbedding()
        self.block1 = Block(3, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.up_block1 = Block(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.up_block2 = Block(128, 64)

        self.out = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )

        shortcut = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        # 32×32
        x1 = self.block1(x, t_emb)
        # [B, 64, 32, 32]

        x = self.pool(x1)

        # 16×16
        x2 = self.block2(x, t_emb)
        # [B, 128, 16, 16]

        x = self.pool(x2)

        # 8×8
        x3 = self.block3(x, t_emb)
        # [B, 256, 8, 8]

        x = self.up1(x3)
        # [B, 128, 16, 16]

        x = torch.cat([x, x2], dim=1)
        # [B, 256, 16, 16]

        x = self.up_block1(x, t_emb)
        # [B, 128, 16, 16]

        x = self.up2(x)
        # [B, 64, 32, 32]

        x = torch.cat([x, x1], dim=1)
        # [B, 128, 32, 32]

        x = self.up_block2(x, t_emb)
        # [B, 64, 32, 32]

        x = self.out(x)
        # [B, 3, 32, 32]

        return x

def add_noise(x0, t):
    alpha_bar_t = alpha_bars[t]
    alpha_bar_t = alpha_bar_t[:, None, None, None]

    noise = torch.randn_like(x0)

    x_t = (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1 - alpha_bar_t) * noise
    )

    return x_t, noise

def train():
    for epoch in range(epochs):
        total_loss = 0
        for x0, labels in train_loader:
            start = time.time()

            optimizer.zero_grad()
            x0 = x0.to(device)
            B = x0.shape[0]
            t = torch.randint(
                0,
                T,
                (B,),
                device=device
            )
            x_t, noise = add_noise(x0, t)
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss.item()}")
            print("backward time:", time.time() - start)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"--------------- epoch {epoch} finished -----------------------------------")
        print(f"epoch: {epoch}, avg_loss: {avg_loss}")
        print("====================================================================")
    torch.save(model.state_dict(), "mini_diffusion.pth")

def test(x, T):
    model.eval()
    with torch.no_grad():
        for i in reversed(range(T)):
            t = torch.tensor([i], device=device)
            pred_noise = model(x, t)
            beta_t = betas[i]  # 标量
            alpha_t = alphas[i]  # 标量
            alpha_bar_t = alpha_bars[i]  # 标量

            mean = 1 / torch.sqrt(alpha_t) * (
                    x - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise
            )
            if i > 0:
                posterior_variance = (
                        beta_t
                        * (1 - alpha_bars[i - 1])
                        / (1 - alpha_bar_t)
                )
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_variance) * noise
            else:
                x = mean

    show_image(x)

def image_recover():
    model.eval()

    x0, _ = next(iter(train_loader))
    T = 50

    # 只拿第一张
    x0 = x0[:1].to(device)

    t = torch.tensor([T], device=device)

    x_t, noise = add_noise(x0, t)

    test(x_t, T + 1)
    show_image(x0)

def show_image(x, name="generated"):
    x = (x + 1) / 2
    x = x.clamp(0, 1)

    image = x[0].detach().permute(1, 2, 0).cpu()

    # 创建保存目录
    os.makedirs("results", exist_ok=True)

    # 生成时间戳（精确到微秒）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 自动生成文件名
    filename = f"results/{name}_{timestamp}.png"

    # 显示并保存
    plt.figure()
    plt.imshow(image)
    plt.axis("off")

    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()

    print(f"图片已保存：{filename}")

def load_model():
    model.load_state_dict(
        torch.load("mini_diffusion.pth", map_location=device)
    )
    model.eval()

def mse_test():
    model.eval()

    x0, _ = next(iter(train_loader))
    x0 = x0.to(device)

    with torch.no_grad():
        for step in [50, 300, 700, 999]:
            t = torch.full(
                (x0.shape[0],),
                step,
                device=device,
                dtype=torch.long
            )

            x_t, noise = add_noise(x0, t)

            pred_noise = model(x_t, t)

            loss = F.mse_loss(pred_noise, noise)

            print(f"t={step}, loss={loss.item():.6f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mini DDPM on CIFAR-10")
    parser.add_argument("mode", choices=["train", "sample", "mse", "recover"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    global device, model, T, betas, alphas, alpha_bars, batch_size, epochs, optimizer, train_loader
    #init
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MiniUNet().to(device)

    T = 1000
    betas = torch.linspace(0.0001, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    batch_size = 64
    epochs = 10

    optimizer = torch.optim.Adam([
            {
                "params": model.parameters(),
                "lr": 1e-3
            }
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    epochs = args.epochs
    if args.mode == "train":
        train()
    else:
        load_model()
        if args.mode == "sample":
            test(torch.randn(1, 3, 32, 32, device=device), T)
        elif args.mode == "mse":
            mse_test()
        else:
            image_recover()

if __name__ == "__main__":
    main()
