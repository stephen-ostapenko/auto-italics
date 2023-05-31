#!/bin/python3

import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

import sys
import numpy as np
import shutil

from tqdm import tqdm

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, ImageChops
from PIL.Image import Resampling

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as tr
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from enum import Enum
import gc
from copy import deepcopy

from recover_svg import process_dir

torch.set_num_threads(8)

IMAGE_SIZE = int(sys.argv[1])
CHANNELS_CNT = 1

RE_FONTS_PATH = f"fonts-re-{IMAGE_SIZE}/"
IT_FONTS_PATH = f"fonts-it-{IMAGE_SIZE}/"


def clear_gpu(model):
    model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()


def train_val_test_split():
    np.random.seed(24)
    
    fonts = list(filter(lambda f: "i" not in f, os.listdir("fonts-src")))
    val_cnt = len(fonts) // 15
    test_cnt = len(fonts) // 15
    
    fonts = np.array(fonts)
    np.random.shuffle(fonts)
    return list(fonts[: -(val_cnt + test_cnt)]), list(fonts[-(val_cnt + test_cnt) : -test_cnt]), list(fonts[-test_cnt :])


def retrieve_glyphs(fonts):
    re_glyphs, it_glyphs = [], []
    for font in fonts:
        re = sorted(os.listdir(f"{RE_FONTS_PATH}/{font.replace('.otf', '')}"))
        it = sorted(os.listdir(f"{IT_FONTS_PATH}/{font.replace('.otf', '')}i"))
        assert(re == it)
        
        for glyph in re:
            img = Image.open(f"{RE_FONTS_PATH}/{font.replace('.otf', '')}/{glyph}")
            re_glyphs.append(np.array(img))
            
        for glyph in it:
            img = Image.open(f"{IT_FONTS_PATH}/{font.replace('.otf', '')}i/{glyph}")
            it_glyphs.append(np.array(img))
            
    return re_glyphs, it_glyphs


transforms = tr.Compose([
    tr.ToTensor(),
    #tr.Normalize(channel_mean, channel_std),
])


class Mode(Enum):
    train = 0
    test = 1
    val = 2


class GlyphDataset(Dataset):
    def __init__(self, re, it, mode):
        assert(len(re) == len(it))
        self.re = re
        self.it = it
        
        self.mode = mode

    def __len__(self):
        return len(self.re)
    
    def __getitem__(self, index):
        re = transforms(self.re[index])
        it = transforms(self.it[index])
        
        return re, it


def correct_picture(pic, lower_bound = 0.0, upper_bound = 0.8):
    res = deepcopy(pic)
    
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if (res[i][j] > upper_bound):
                res[i][j] = 1
            elif (res[i][j] < lower_bound):
                res[i][j] = 0
            elif (lower_bound == upper_bound):
                res[i][j] = 0
            else:
                res[i][j] = (res[i][j] - lower_bound) / (upper_bound - lower_bound)
    
    return res


from torchmetrics import MeanSquaredError, MeanAbsoluteError, StructuralSimilarityIndexMeasure
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


criterion = nn.MSELoss()


def val(model, loader, criterion):
    model.eval()
    
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)
    acc = BinaryAccuracy().to(device)
    fsc = BinaryF1Score().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    
    loss_l, mse_l, mae_l, acc_l, fsc_l, ssim_l = [], [], [], [], [], []
    
    with torch.no_grad():
        for re, it in tqdm(loader):
            re = re.to(device)
            it = it.to(device)
            
            out = model(re)
            
            loss_l.append(criterion(out, it).item())
            
            mse_l.append(mse(out, it).item())
            mae_l.append(mae(out, it).item())
            
            correcred_it = it.round().type(torch.int)
            
            acc_l.append(acc(out, correcred_it).item())
            fsc_l.append(fsc(out, correcred_it).item())
            
            ssim_l.append(ssim(out, it).item())
    
    loss = np.mean(loss_l)
    mse = np.mean(mse_l)
    mae = np.mean(mae_l)
    acc = np.mean(acc_l)
    fsc = np.mean(fsc_l)
    ssim = np.mean(ssim_l)
    
    return loss, mse, mae, acc, fsc, ssim


def create_model_and_optimizer(model_class, model_params, lr = 1e-3, device = None):
    model = model_class(**model_params)
    model = model.to(device)
    
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
    
    optimizer = torch.optim.Adam(params, lr)
    return model, optimizer


def make_predictions(model, loader):
    shutil.rmtree("./samples-png-in")
    os.mkdir("./samples-png-in")

    model.eval()
    
    with torch.no_grad():
        cur_idx = 0
        for batch_re, batch_it in tqdm(loader):
            batch_re = batch_re.to(device)
            batch_it = batch_it.to(device)
            
            for re, it in zip(batch_re, batch_it):
                re = re.unsqueeze(0)
                it = it.unsqueeze(0)
                
                out = model(re)
                
                ini_pic = re.detach().cpu().squeeze().numpy()
                res_pic = correct_picture(out.detach().cpu().squeeze().numpy())
                tgt_pic = it.detach().cpu().squeeze().numpy()
                
                res_pic = Image.fromarray(np.uint8(res_pic * 255), mode = "L")
                res_pic.save(f"./samples-png-in/{cur_idx}.png")
                cur_idx += 1


class simple_net(nn.Module):
    def __init__(self, input_size, num_layers, hidden_sizes, activations, dropouts, output_size):
        super(simple_net, self).__init__()
        
        flat = ("flat", nn.Flatten())
        in_to_hid = ("in2hid", nn.Linear(input_size, hidden_sizes))
        
        head = [
            (f"act_last", nn.ReLU()),
            ("hid2out", nn.Linear(hidden_sizes, output_size)),
            ("sigmoid", nn.Sigmoid())
        ]
        
        self.net = [flat, in_to_hid, *head]
        self.net = nn.Sequential(OrderedDict(self.net))
    
    def forward(self, inp):
        return torch.reshape(self.net(inp), (-1, CHANNELS_CNT, IMAGE_SIZE, IMAGE_SIZE))


def create_simple_net(device):
    hidden_size = None
    if (IMAGE_SIZE == 64):
        hidden_size = 2 ** 14
    elif (IMAGE_SIZE == 128):
        hidden_size = 2 ** 12
    elif (IMAGE_SIZE == 256):
        hidden_size = 2 ** 10

    model, optimizer = create_model_and_optimizer(
        simple_net,
        {
            "input_size": IMAGE_SIZE ** 2,
            "num_layers": 0,
            "hidden_sizes": hidden_size,
            "activations": 0,
            "dropouts": 0,
            "output_size": IMAGE_SIZE ** 2
        },
        lr = 1e-4,
        device = device
    )

    MODEL_NAME = f"simple-2-layer-model-for-{IMAGE_SIZE}"
    print(MODEL_NAME)
    
    checkpoint = torch.load(f"./{MODEL_NAME}", map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        stride = 2 if (in_channels != out_channels) else 1

        if (in_channels != out_channels):
            self.shortcut = nn.Sequential(OrderedDict([
                ("downsample_conv", nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size = 1, stride = 2,
                    bias = False
                )),
                ("downsample_norm", nn.BatchNorm2d(out_channels))
            ]))
        
        else:
            self.shortcut = nn.Identity()
        
        self.activation = nn.ReLU()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size = 3, stride = stride,
            padding = 1, dilation = 1,
            groups = 1, bias = False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size = 3, stride = 1,
            padding = 1, dilation = 1,
            groups = 1, bias = False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride


    def forward(self, x):
        residual = self.shortcut(x)
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.activation(out)

        return out


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.blocks = nn.Sequential(OrderedDict([
            ("residual1", ResidualBlock(in_channels, out_channels)),
            ("residual2", ResidualBlock(out_channels, out_channels))
        ]))


    def forward(self, x):
        return self.blocks(x)


class MyResNet64(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_size = 64
        self.channels_cnt = 1

        self.conv1 = nn.Conv2d(
            self.channels_cnt, 64,
            kernel_size = 3, stride = 1,
            padding = 1, dilation = 1,
            groups = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size = 3, stride = 2,
            padding = 1, dilation = 1
        )

        self.layers = nn.Sequential(OrderedDict([
            #("resnet1", ResNetLayer(64, 64)),
            ("resnet2", ResNetLayer(64, 128)),
            ("resnet3", ResNetLayer(128, 256)),
            ("resnet4", ResNetLayer(256, 512))
        ]))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = 512 * 4 * 4, out_features = self.image_size ** 2, bias = True)

        
    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out = self.layers(out)

        out = self.flatten(out)
        out = self.fc(out)

        return torch.reshape(out, (-1, self.channels_cnt, self.image_size, self.image_size))


class MyResNet128(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_size = 128
        self.channels_cnt = 1

        self.conv1 = nn.Conv2d(
            self.channels_cnt, 64,
            kernel_size = 3, stride = 1,
            padding = 1, dilation = 1,
            groups = 1, bias = False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size = 3, stride = 2,
            padding = 1, dilation = 1
        )

        self.layers = nn.Sequential(OrderedDict([
            ("resnet1", ResNetLayer(64, 128)),
            ("resnet2", ResNetLayer(128, 256)),
            ("resnet3", ResNetLayer(256, 512))
        ]))

        self.avgpool = nn.AvgPool2d(kernel_size = 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = 512 * 4 * 4, out_features = self.image_size ** 2, bias = True)

        
    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out = self.layers(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return torch.reshape(out, (-1, self.channels_cnt, self.image_size, self.image_size))


def create_resnet(device):
    model, optimizer = create_model_and_optimizer(
        MyResNet64 if (IMAGE_SIZE == 64) else MyResNet128,
        {},
        lr = 1e-4,
        device = device
    )

    MODEL_NAME = f"resnet-based-model-for-{IMAGE_SIZE}"
    print(MODEL_NAME)
    
    checkpoint = torch.load(f"./{MODEL_NAME}", map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


if (__name__ == "__main__"):
    train_fonts, val_fonts, test_fonts = train_val_test_split()
    print(len(train_fonts), len(val_fonts), len(test_fonts))

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    BATCH_SIZE = 256

    test_ds = GlyphDataset(*retrieve_glyphs(test_fonts), Mode.test)
    test_dl = DataLoader(
        test_ds,
        batch_size = BATCH_SIZE,
        shuffle = False,
        drop_last = False,
        num_workers = 0
    )

    # model = create_simple_net(device)
    model = create_resnet(device)

    loss, mse, mae, acc, fsc, ssim = val(model, test_dl, criterion)
    print(
        f"loss: {'{:.6f}'.format(loss)}; " + \
        f"mse: {'{:.6f}'.format(mse)}; " + \
        f"mae: {'{:.6f}'.format(mae)}; " + \
        f"acc: {'{:.3f}'.format(acc)}; " + \
        f"fsc: {'{:.3f}'.format(fsc)}; " + \
        f"ssim: {'{:.3f}'.format(ssim)} ",
        end = "\n", flush = True
    )

    make_predictions(model, test_dl)

    clear_gpu(model)

    print("svg commands:")

    mean, std, q_0_2, q_0_8 = process_dir("./samples-png-in/", "./samples-svg-out")
    print(
        f"mean: {'{:.3f}'.format(mean)}; " + \
        f"std: {'{:.3f}'.format(std)}; " + \
        f"0.2 quantile: {'{:.3f}'.format(q_0_2)}; " + \
        f"0.8 quantile: {'{:.3f}'.format(q_0_8)}; ",
        end = "\n", flush = True
    )
