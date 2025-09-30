from torch.cuda.amp import autocast,GradScaler
import pickle
from CTC import SignLanguageAligner
import os
import numpy as np
import pandas as pd
import torchvision
import  torch
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from my_generators import  MyGeneratorResnet
from train_generators import GeneratorResnet
import random
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import torch
import torch.nn as nn
import importlib
import faulthandler
import numpy as np
import torchvision

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.load_slowfast_pkl = True
        self.model, self.optimizer = self.loading()

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict
    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)
    def load_data(self):
        print("Loading dataset")
        self.feeder = import_class(self.arg.feeder)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size=['K5', 'P2', 'K5', 'P2'],
                                             dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading dataset finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            # batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            batch_size=1,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
            pin_memory=True,
            worker_init_fn=self.init_fn,
        )

    def loading(self):
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
            load_pkl=self.load_slowfast_pkl,
            slowfast_config=self.arg.slowfast_config,
            slowfast_args=self.arg.slowfast_args
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        self.kernel_sizes = model.conv1d.kernel_size
        print(self.kernel_sizes)
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(0)
        model = convert_model(model)
        model.cuda()
        return model

    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed) + worker_id)


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


# 新加
# 反向钩子：捕获目标层的梯度
# 前向钩子：捕获目标层的输出特征
# 这两个钩子用于获取模型内部信息，指导对抗样本生成
# Getting the gradient
def backward_hook(module, grad_in, grad_out):
    global back_grad
    back_grad = grad_out[0].clone().detach()


# Get feature layer
def forward_hook(module, input, output):
    global back_fea
    back_fea = output.detach()


def find_layer(module, last_conv=None):
    for child in module.children():
        if isinstance(child, nn.Conv3d):
            last_conv = child
        last_conv = find_layer(child, last_conv)
    return last_conv


# trans_incep：调整Inception v3输入尺寸
# data_transform：标准图像预处理流程
# normalize：ImageNet数据集标准化
def trans_incep(x):
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x

def chunk_generate(netG,img,grad_box,chunk_size=1):
    chunks=[]
    adv_0_chunk=[]
    adv_00_chunk = []
    for i in range(0,img.size(0),chunk_size):
        img_chunk=img[i:i+chunk_size]
        grad_chunk=grad_box[i:i+chunk_size]
        with autocast(enabled=True):
            adv, adv_0, adv_00 =netG(img_chunk,grad_chunk)
            chunks.append(adv.float())
            adv_0_chunk.append(adv_0.float())
            adv_00_chunk.append(adv_00.float())
            del img_chunk,grad_chunk,adv,adv_0,adv_00
            torch.cuda.empty_cache()
    return torch.cat(chunks,dim=0),torch.cat(adv_0_chunk,dim=0),torch.cat(adv_00_chunk,dim=0)
def total_variation(images):
    """计算总变分损失，促进空间平滑性"""
    # 仅计算mask区域的TV损失
    # diff_i = torch.abs(masked_images[:, :, :, 1:] - masked_images[:, :, :, :-1])
    # diff_j = torch.abs(masked_images[:, :, 1:, :] - masked_images[:, :, :-1, :])
    #
    # return torch.sum(diff_i) + torch.sum(diff_j)
    # 空间维度 (高度和宽度)
    diff_h = torch.abs(images[..., 1:, :, :] - images[..., :-1, :, :])
    diff_w = torch.abs(images[..., :, 1:, :] - images[..., :, :-1, :])

    # 时间维度 (帧间变化)
    diff_t = torch.abs(images[:, 1:, ...] - images[:, :-1, ...])

    return torch.sum(diff_h) + torch.sum(diff_w) + torch.sum(diff_t)
def gen_img(img,url,to_01=False, normalize=False):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    torchvision.utils.save_image(img,url, normalize=normalize)
def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM攻击实现
    :param image: 原始输入图像
    :param epsilon: 扰动大小
    :param data_grad: 输入的梯度
    :return: 对抗样本
    """
    # 收集数据梯度的符号
    sign_data_grad = data_grad.sign()
    # 创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 将像素值裁剪到[0,1]范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def generate_fgsm(model, device, images,img_number, labels,label_lgt, epsilon):
    """
    生成FGSM对抗样本
    :param model: 目标模型
    :param device: CPU/GPU
    :param images: 原始图像
    :param labels: 真实标签
    :param epsilon: 扰动大小
    :return: 对抗样本
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images.requires_grad = True

    # 前向传播
    outputs = model(images, img_number, label=labels, label_lgt=label_lgt)
    loss = model.criterion_calculation(outputs,labels,label_lgt)

    # 反向传播，获取梯度
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data

    # 调用FGSM攻击
    perturbed_images = fgsm_attack(images, epsilon, data_grad)

    return perturbed_images


def generate_mifgsm(model, device, images, labels, epsilon=0.03, alpha=0.01, num_iter=10, decay=1.0):
    """
    生成MI-FGSM对抗样本
    :param model: 目标模型
    :param device: CPU/GPU
    :param images: 原始图像
    :param labels: 真实标签
    :param epsilon: 最大扰动大小
    :param alpha: 步长
    :param num_iter: 迭代次数
    :param decay: 动量衰减因子
    :return: 对抗样本
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # 初始化动量
    momentum = torch.zeros_like(images).detach().to(device)

    # 初始化对抗样本
    perturbed_images = images.clone().detach()

    for _ in range(num_iter):
        perturbed_images.requires_grad = True

        # 前向传播
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # 反向传播，获取梯度
        model.zero_grad()
        loss.backward()
        grad = perturbed_images.grad.data

        # 更新动量
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1)
        grad = grad / grad_norm.view(-1, 1, 1, 1)
        momentum = decay * momentum + grad

        # 更新对抗样本
        perturbed_images = perturbed_images.detach() + alpha * momentum.sign()
        delta = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(images + delta, 0, 1).detach()

    return perturbed_images


def generate_pgd(model, device, images, labels, epsilon=0.03, alpha=0.01, num_iter=10):
    """
    生成PGD对抗样本
    :param model: 目标模型
    :param device: CPU/GPU
    :param images: 原始图像
    :param labels: 真实标签
    :param epsilon: 最大扰动大小
    :param alpha: 步长
    :param num_iter: 迭代次数
    :return: 对抗样本
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # 随机初始化扰动
    perturbed_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()

    for _ in range(num_iter):
        perturbed_images.requires_grad = True

        # 前向传播
        outputs = model(perturbed_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # 反向传播，获取梯度
        model.zero_grad()
        loss.backward()
        grad = perturbed_images.grad.data

        # 更新对抗样本
        perturbed_images = perturbed_images.detach() + alpha * grad.sign()

        # 投影到epsilon邻域内
        delta = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
        perturbed_images = torch.clamp(images + delta, 0, 1).detach()

    return perturbed_images
if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    lam_1 = 0.0000001
    lam_2 = 0.0000003
    lam_3 = 0.0000001
    eps = args.eps
    # 模式选择：TK=True使用TopK方法选择重要区域，否则使用区间选择
    tk = args.tk
    epochs = args.epochs
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 存储中间特征和梯度
    back_fea = torch.tensor([]).to(device)
    back_grad = torch.tensor([]).to(device)
    # Model                                                            weights/soft_eps10_res50_tk0.95.pth
    model, opt = processor.loading()
    processor.load_data()
    split = 'dev'
    loader = processor.data_loader[split]
    eps=0.01
    for i, data in enumerate(loader):
        # 外层循环：训练轮数
        # 内层循环：遍历数据加载器
        # 15.1 数据准备和目标设置
        img = data[0].to(device)
        img_number = data[1]
        print(f"number:{img_number}")
        gt = data[2].to(device)
        label_lgt = data[3].to(device)
        adv=generate_fgsm(model,device,img,img_number,gt,label_lgt,eps)
        for i in range(img.shape[0]):
            ss = name[0:name.index(name.split("_")[-1]) - 1]
            name = name + "/1"
            saveurl = f'/media/cv3/store1/postgraduate/y2023/WLF/datasets/ph_adv/newAttack/{split}/{name}'
            if not os.path.exists(saveurl):
                os.makedirs(saveurl)
            for j in range(img.shape[1]):
                img1 = adv[j]
                gen_img(img1, '{}/{}.avi_pid0_fn{:06d}-0.png'.format(saveurl, ss, j))





