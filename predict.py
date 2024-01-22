from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy import io

from network import DDPM, ContextUnet

def predict_mnist():
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128
    save_dir = './data/diffusion_output/prediction/'
    path = './data/diffusion_output/model_19.pth'
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(path))
    ddpm.to(device)
    ddpm.eval()

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    n_sample = 5
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=5)

    with torch.no_grad():
        x, c = next(iter(dataloader))
        x = x.to(device)
        x_gen, x_gen_store = ddpm.prediction(n_sample, (1, 28, 28), c, device, guide_w= 2.0)

        x_all = torch.cat([x_gen, x])
        grid = make_grid(x_all*-1 + 1, nrow=n_sample)
        save_image(grid, save_dir + "prediction.png")
        print('saved image at ' + save_dir)


if __name__ == "__main__":
    predict_mnist()