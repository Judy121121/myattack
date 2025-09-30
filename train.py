# 1. 导入库和设置随机种子
import argparse
import os
import numpy as np
import pandas as pd
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from train_generators import GeneratorResnet
import random

# 随机种子设置：确保实验可复现性，所有随机操作都被固定
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# 关键参数：
# train_dir：ImageNet数据集路径
# model_type：攻击的目标模型（res50或incv3）
# eps：扰动预算（像素值变化范围）
# target：目标攻击类别（-1表示非目标攻击）
# tk：选择重要区域的比例
# 正则化系数：
# lam_1：稀疏性正则化系数
# lam_2：掩码二值化正则化系数
parser = argparse.ArgumentParser(description='Training EGS_TSSA for generating sparse adversarial examples')
parser.add_argument('--train_dir', default='/media/cv3/store2/imagenet/', help='path to imagenet training set')
parser.add_argument('--model_type', type=str, default='res50', help='Model against GAN is trained: incv3, res50')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--batch_size', type=int, default=8, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2.25e-5, help='Initial learning rate for adam')
parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint')
parser.add_argument('--tk', type=float, default=0.6, help='path to checkpoint')

# stage I/media/cv3/store2/imagenet/
lam_1 = 0.00
lam_2 = 0.00001

## stage II
# lam_1 = 0.0001
# lam_2 = 0.0003

args = parser.parse_args()
eps = args.eps
print(args)
# 模式选择：TK=True使用TopK方法选择重要区域，否则使用区间选择
TK = True
if TK == True:
    tk = args.tk
else:
    choose = [0., 0.5]

epochs = args.epochs
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 存储中间特征和梯度
back_fea = torch.tensor([]).to(device)
back_grad = torch.tensor([]).to(device)


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


# Model
if args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
    # 注册钩子到特定层：
    model.Mixed_7c.register_forward_hook(forward_hook)
    model.Mixed_7c.register_full_backward_hook(backward_hook)
elif args.model_type == 'res50':
    model = torchvision.models.resnet50(pretrained=True)
    model.layer4[-1].register_forward_hook(forward_hook)
    model.layer4[-1].register_full_backward_hook(backward_hook)

model = model.to(device)
# 模型设为评估模式
model.eval()

# 不同模型的输入尺寸要求：
# ResNet50：缩放256→裁剪224
# Inception v3：缩放300→裁剪299
# filterSize和stride：用于划分图像网格
# Input dimensions
if args.model_type in ['res50']:
    scale_size = 256
    img_size = 224
    filterSize = 8
    stride = 8
else:
    scale_size = 300
    img_size = 299
    filterSize = 13
    stride = 13
# x_box
# 网格索引计算
# 网格划分：将图像划分为P×Q个网格
# 索引计算：计算每个网格对应的像素索引
# 批处理扩展：为整个批次创建索引张量
# 输出形状：[batch_size, num_grids, grid_pixels]
P = np.floor((img_size - filterSize) / stride) + 1
P = P.astype(np.int32)
Q = P
index = np.ones([P * Q, filterSize * filterSize], dtype=int)
tmpidx = 0
for q in range(Q):
    plus1 = q * stride * img_size
    for p in range(P):
        plus2 = p * stride
        index_ = np.array([], dtype=int)
        for i in range(filterSize):
            plus = i * img_size + plus1 + plus2
            index_ = np.append(index_, np.arange(plus, plus + filterSize, dtype=int))
        index[tmpidx] = index_
        tmpidx += 1
index = torch.LongTensor(np.tile(index, (args.batch_size, 1, 1))).to(device)

# Generator生成器初始化eps参数：
# 扰动预算（归一化到[0,1]范围）
# 支持从检查点恢复训练
if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True, eps=eps / 255.)
else:
    netG = GeneratorResnet(eps=eps / 255.)
if args.checkpoint != '':
    netG.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0'))
netG.to(device)

# Optimizer优化器设置
# Adam优化器用于训练生成器
# 学习率通过命令行参数设置
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))


#  图像预处理

# trans_incep：调整Inception v3输入尺寸
# data_transform：标准图像预处理流程
# normalize：ImageNet数据集标准化
def trans_incep(x):
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x


# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size, antialias=True),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


# 11. 数据加载
# 加载ImageNet训练集
# 使用DataLoader进行批处理
train_set = datasets.ImageFolder(args.train_dir, data_transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
#                                            pin_memory=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                           pin_memory=True)
train_size = len(train_set)
print('Training data size:', train_size)


# Adv Loss
# 12. 对抗损失函数
# C&W损失函数：用于生成对抗样本
# 支持目标攻击（tar=True）和非目标攻击
# 通过最大化目标类/最小化原始类概率实现攻击
def CWLoss(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).to(device).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(1000).type(torch.cuda.FloatTensor)[target.long()].to(device))

    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other - real, kappa))
    else:
        return torch.sum(torch.max(real - other, kappa))


criterion = CWLoss


# Get the most important area
# 13. 区域选择函数
# grad_topk：选择梯度最大的前k个区域
# grad_choose：选择梯度排名在特定区间的区域
# 两个函数都生成二进制掩码，指示哪些区域可以修改
def grad_topk(grad, index, filterSize, Tk):
    k = int(((img_size / filterSize) ** 2) * Tk)
    box_size = filterSize * filterSize
    for i in range(len(grad)):
        tmp = torch.take(grad[i], index[i])
        norm_tmp = torch.norm(tmp, dim=-1)
        g_topk = torch.topk(norm_tmp, k=k, dim=-1)
        top = g_topk.values.max() + 1
        norm_tmp_k = norm_tmp.put_(g_topk.indices, torch.FloatTensor([top] * k).to(device))
        norm_tmp_k = torch.where(norm_tmp_k == top, 1., 0.)
        tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
        grad[i] = grad[i].put_(index[i], tmp_bi)
    return grad


# Get the zone area of interest
def grad_choose(grad, index, filterSize, choose):
    box_size = filterSize * filterSize
    for i in range(len(grad)):
        tmp = torch.take(grad[i], index[i])
        norm_tmp = torch.norm(tmp, dim=-1)
        norm_UD = torch.argsort(norm_tmp, descending=True)
        norm_len = len(norm_tmp)
        choose_ch = [int(norm_len * choose[0]), int(norm_len * choose[1])]
        choose_index = norm_UD[choose_ch[0]:choose_ch[1]]
        norm_0 = torch.zeros_like(norm_tmp).detach().to(device)
        norm_0[choose_index] = 1
        norm_tmp_k = norm_0
        tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
        grad[i] = grad[i].put_(index[i], tmp_bi)
    return grad


# Training
print(
    'Label: {} \t Model: {} \t Dataset: {} \t Saving instances: {}'.format(args.target, args.model_type,
                                                                           args.train_dir, epochs))
if TK == True:
    now = 'TK-{}_TG-{}_eps-{}_S-{}_Q-{}_K-{}-box-{}/'.format(args.model_type, args.target, eps, lam_1, lam_2, tk,
                                                             filterSize)
else:
    now = 'CH-{}_TG-{}_eps-{}_S-{}_Q-{}_CH-{}_{}-box-{}/'.format(args.model_type, args.target, eps, lam_1, lam_2,
                                                                 choose[0], choose[1], filterSize)
# 14. 输出目录设置
# 创建输出目录和图片保存目录
# 初始化日志数据结构
# 计算日志记录间隔（iterp）和总迭代次数（i_len）
now_pic = now + 'pictures/'
if not os.path.exists(now):
    os.mkdir(os.path.join(now))
    os.mkdir(os.path.join(now_pic))

out_csv = pd.DataFrame([])
FR_white_box = []
tra_loss, norm_0, norm_1, norm_2, test = [], [], [], [], []
iterp = 2000 // args.batch_size
i_len = train_size // (iterp * args.batch_size)
out_csv['id'] = [i for i in range(i_len * (epochs + 1))]
# 15. 训练主循环

for epoch in range(epochs):
    FR_wb, FR_wb_epoch = 0, 0
    for i, (img, gt) in enumerate(train_loader):
        # 外层循环：训练轮数
        # 内层循环：遍历数据加载器
        # 15.1 数据准备和目标设置
        img = img.to(device)
        gt = gt.to(device)
        # 根据攻击类型（目标/非目标）设置标签
        # 执行前向传播和反向传播获取梯度
        if args.target == -1:
            # # 非目标攻击
            img_in = normalize(img.clone().detach())
            out = model(img_in)
            label = out.argmax(dim=-1).detach()
            out_wb = label.clone().detach()
            out.backward(torch.ones_like(out))
        else:
            # 目标攻击
            out = torch.LongTensor(img.size(0))
            out.fill_(args.target)
            label = out.to(device)

            out_tmp = model(normalize(img.clone().detach()))
            out_tmp.backward(torch.ones_like(out_tmp))
            out_wb = label.clone().detach()
        # 15.2 生成器训练
        # 准备生成器训练
        # 计算梯度特征图
        # 生成区域选择掩码（grad_box）
        # 使用生成器产生对抗样本和相关输出
        netG.train()
        optimG.zero_grad()

        # # 获取结构化掩码
        grad = back_grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        grad_fea = (grad * back_fea).sum(dim=1)
        resize = transforms.Resize((img_size, img_size), antialias=True)
        G_F = resize(grad_fea).reshape(len(img), 1, img_size, img_size)

        if TK == True:
            grad_box = grad_topk(G_F, index, filterSize, tk)
        else:
            grad_box = grad_choose(G_F, index, filterSize, choose)
        # 生成对抗样本
        adv, adv_inf, adv_0, adv_00, grad_img = netG(img, grad_box)
        adv_img = adv.clone().detach()
        adv_test = adv.clone().detach()
        # 15.3 对抗样本评估
        # 评估对抗样本的攻击成功率（FR）
        # 计算对抗损失
        adv_out = model(normalize(adv))
        adv_out_to_wb = adv_out.clone().detach()

        if args.target == -1:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) != out_wb).item()
            # Untargeted Attack
            loss_adv = criterion(adv_out, label)
        else:
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) == out_wb).item()
            # Targeted Attack
            loss_adv = criterion(adv_out, label, tar=True)

        FR_wb += FR_wb_tmp
        FR_wb_epoch += FR_wb_tmp
        # 15.4 损失计算和优化
        # 总损失 = 对抗损失 + 稀疏损失 + 二值化损失
        # 反向传播和优化器更新

        # L1稀疏正则
        loss_spa = torch.norm(adv_0, 1)
        bi_adv_00 = torch.where(adv_00 < 0.5, torch.zeros_like(adv_00), torch.ones_like(adv_00) * grad_box)
        # 二值化损失
        loss_qua = torch.sum((bi_adv_00 - adv_00) ** 2)
        loss = loss_adv + lam_1 * loss_spa + lam_2 * loss_qua
        print(f"loss_adv{loss_adv},loss_spa {loss_spa},loss_qua {loss_qua}")
        loss.backward()
        optimG.step()

        adv_loss = loss_adv
        spa1 = lam_1 * loss_spa
        spa2 = lam_2 * loss_qua
        # 15.5 日志记录和输出
        # 定期记录和打印训练指标
        # 包括各种范数（L0、L1、L2、L∞）和损失值
        if i % iterp == 0:
            # 计算各种指标
            FR = FR_wb / (iterp * args.batch_size)
            FR_wb = 0
            adv_0_img = torch.where(adv_0 < 0.5, torch.zeros_like(adv_0), torch.ones_like(adv_0)).clone().detach()
            l0 = (torch.norm(adv_0_img.clone().detach(), 0) / args.batch_size).item()
            l1 = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), 1) / args.batch_size).item()
            l2 = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), 2) / args.batch_size).item()
            linf = (torch.norm(adv_0_img.clone().detach() * adv_inf.clone().detach(), p=np.inf)).item()
            # 保存指标
            tra_loss.append(loss.item())
            FR_white_box.append(FR)
            norm_0.append(l0)
            norm_1.append(l1)
            norm_2.append(l2)
            print('\n', '#' * 20)
            print('l0:', l0, 'l1:', l1, 'l2:', l2, 'linf:', linf, '\n',
                  'loss: %.4f' % loss.item(), 'adv: %.4f' % adv_loss.item(), 'spa1: %.4f' % spa1.item(),
                  'spa2:%.4f' % spa2.item(), '\n',
                  args.model_type, ':', FR)

        if epochs < 21:
            try:
                out_csv['tra_loss'] = pd.Series(tra_loss)
                out_csv['norm_0'] = pd.Series(norm_0)
                out_csv['norm_1'] = pd.Series(norm_1)
                out_csv['norm_2'] = pd.Series(norm_2)
                out_csv[args.model_type] = pd.Series(FR_white_box)
                loss_csv = now + "model-{}_eps-{}_lr-{}_S-{}_Q-{}.csv".format(args.model_type, eps, args.lr, lam_1,
                                                                              lam_2)
                out_csv.to_csv(loss_csv)
            except:
                pass

            if i in [0,100,200,300,400]:
                vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_adv{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(grad_img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_grad_img{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(adv_img - img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_noise{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_org{}.png'.format(epoch, i))
    FR_wb_ep_mean = FR_wb_epoch / train_size
    print('running:{} | FR-{}:{}\n'.format(epoch, args.model_type, FR_wb_ep_mean))
    start, end = int(epoch) * i_len, int(epoch + 1) * i_len
    N0 = np.mean(norm_0[start:end])
    N1 = np.mean(norm_1[start:end])
    try:
        print('loss:{}--L0:{}--L1:{}--L2:{}\n'.format(tra_loss[-1], N0, N1, np.mean(norm_2[start:end])))
    except:
        pass

    save_path = now + 'GN_{}_{}_{}.pth'.format(args.target, args.model_type, epoch)
    torch.save(netG.state_dict(), os.path.join(save_path))

out_csv['tra_loss'] = pd.Series(tra_loss)
out_csv['norm_0'] = pd.Series(norm_0)
out_csv['norm_1'] = pd.Series(norm_1)
out_csv['norm_2'] = pd.Series(norm_2)
out_csv[args.model_type] = pd.Series(FR_white_box)
# 16. 最终保存
loss_csv = now + "model-{}_eps-{}_lr-{}_S-{}_Q-{}.csv".format(args.model_type, eps, args.lr, lam_1, lam_2)
out_csv.to_csv(loss_csv)
print("Training completed...")
