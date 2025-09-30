import argparse
import os
import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from test_generators import GeneratorResnet

# 命令行参数解析
parser = argparse.ArgumentParser(description='testing EGS_TSSA for generating sparse adversarial examples')
parser.add_argument('--test_dir', default='/media/cv3/store2/imagenet/', help='path to imagenet testing set')
parser.add_argument('--model_type', type=str, default='res50', help='Model against GAN is tested: incv3, res50')
parser.add_argument('--model_t', type=str, default='vgg16', help='Model')  # 转移攻击的目标模型
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')  # 扰动预算
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')  # 目标攻击类别
parser.add_argument('--batch_size', type=int, default=1, help='Number of testig samples/batch')
parser.add_argument('--checkpoint', type=str, default='weights/soft_eps255_res50_tk0.873.pth',
                    help='path to checkpoint')  # 生成器权重
parser.add_argument('--tk', type=float, default=0.873, help='path to checkpoint')  # TopK比例

if __name__ == '__main__':
    args = parser.parse_known_args()[0]
    eps = args.eps
    tk = args.tk
    choose = [0., 0.6]  # 备选的区域选择区间
    print('eps:', eps)
    # GPU  设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 用于存储梯度和特征的全局变量
    back_fea = torch.tensor([]).to(device)
    back_grad = torch.tensor([]).to(device)


    # Getting the gradient
    # 2. 梯度钩子函数
    # 梯度钩子函数 - 获取反向传播的梯度
    def backward_hook(module, grad_in, grad_out):
        global back_grad
        back_grad = grad_out[0].clone().detach()  # 捕获并保存梯度


    # # 特征钩子函数 - 获取前向传播的特征
    def forward_hook(module, input, output):
        global back_fea
        back_fea = output.detach()  # 捕获并保存梯度


    # Model 3. 模型加载和设置
    # # 加载目标模型（白盒模型）
    if args.model_type == 'incv3':
        model = torchvision.models.inception_v3(weights=True)
        model.Mixed_7c.register_forward_hook(forward_hook)  # 注册前向钩子
        model.Mixed_7c.register_full_backward_hook(backward_hook)  # 注册反向钩子
    elif args.model_type == 'res50':
        model = torchvision.models.resnet50(weights=True)
        model.layer4[-1].register_forward_hook(forward_hook)
        model.layer4[-1].register_full_backward_hook(backward_hook)
    #  加载转移模型（黑盒模型）
    if args.model_t == 'dense161':
        model_t = torchvision.models.densenet161(weights=True)
    elif args.model_t == 'vgg16':
        model_t = torchvision.models.vgg16(weights=True)
    elif args.model_t == 'incv3':
        model_t = torchvision.models.inception_v3(weights=True)
    elif args.model_t == 'res50':
        model_t = torchvision.models.resnet50(weights=True)
    # 将模型转移到设备并设置为评估模式
    model_t = model_t.to(device)
    model_t.eval()

    model = model.to(device)
    model.eval()

    # Input dimensions 4. 输入尺寸和网格索引计算
    # 根据模型类型设置输入尺寸
    if args.model_type in ['res50']:
        scale_size = 256  # 缩放尺寸
        img_size = 224  # 裁剪尺寸
        filterSize = 8  # 网格大小
        stride = 8  # 网格步长
    else:
        scale_size = 300
        img_size = 299
        filterSize = 13
        stride = 13
    # x_box  计算网格数量
    P = np.floor((img_size - filterSize) / stride) + 1
    P = P.astype(np.int32)
    Q = P  # 网格在高度和宽度方向数量相同
    # 计算每个网格的像素索引
    index = np.ones([P * Q, filterSize * filterSize], dtype=int)
    tmpidx = 0
    for q in range(Q):
        plus1 = q * stride * img_size  # 行偏移量
        for p in range(P):
            plus2 = p * stride  # 列偏移量
            index_ = np.array([], dtype=int)
            for i in range(filterSize):
                plus = i * img_size + plus1 + plus2
                index_ = np.append(index_, np.arange(plus, plus + filterSize, dtype=int))
            index[tmpidx] = index_
            tmpidx += 1
    # 将索引扩展为批处理形式并转移到设备
    index = torch.LongTensor(np.tile(index, (args.batch_size, 1, 1))).to(device)

    # Generator 5. 生成器加载
    if args.model_type == 'incv3':
        netG = GeneratorResnet(inception=True, eps=eps / 255.)
    else:
        netG = GeneratorResnet(eps=eps / 255.)  # 扰动预算归一化
    # 加载预训练权重
    netG.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0'))
    netG.to(device)
    netG.eval()  # 设置为评估模式

    # 6. 数据预处理和加载
    # 数据预处理流程
    data_transform = transforms.Compose([
        transforms.Resize(scale_size, antialias=True),  # 调整尺寸
        transforms.CenterCrop(img_size),  # 中心裁剪
        transforms.ToTensor(),  # 转为张量
    ])


    # Inception模型专用的尺寸调整函数
    def trans_incep(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return x


    # ImageNet归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    # 归一化函数
    def normalize(t):
        t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
        t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
        t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

        return t


    # 加载测试集
    test_set = datasets.ImageFolder(args.test_dir, data_transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2,
    #                                           pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                              pin_memory=True)
    test_size = len(test_set)
    print('test data size:', test_size)


    # 7. 重要区域选择函数
    # 基于TopK选择重要区域
    def grad_topk(grad, index, filterSize, Tk=tk):
        k = int(((img_size / filterSize) ** 2) * Tk)  # 计算要选择的网格数量
        box_size = filterSize * filterSize  # 每个网格的像素数
        for i in range(len(grad)):
            # 提取每个网格的特征
            tmp = torch.take(grad[i], index[i])
            # 计算每个网格特征的范数
            norm_tmp = torch.norm(tmp, dim=-1)
            # 选择范数最大的k个网格
            g_topk = torch.topk(norm_tmp, k=k, dim=-1)
            # 创建二进制掩码标识重要区域
            top = g_topk.values.max() + 1
            norm_tmp_k = norm_tmp.put_(g_topk.indices, torch.FloatTensor([top] * k).to(device))
            norm_tmp_k = torch.where(norm_tmp_k == top, 1., 0.)
            # 扩展掩码到像素级别
            tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
            grad[i] = grad[i].put_(index[i], tmp_bi)
        return grad


    # 基于区间选择重要区域
    def grad_choose(grad, index, filterSize, choose):
        box_size = filterSize * filterSize
        for i in range(len(grad)):
            tmp = torch.take(grad[i], index[i])
            norm_tmp = torch.norm(tmp, dim=-1)
            # 按范数排序
            norm_UD = torch.argsort(norm_tmp, descending=True)
            norm_len = len(norm_tmp)
            # 计算选择区间
            choose_ch = [int(norm_len * choose[0]), int(norm_len * choose[1])]
            choose_index = norm_UD[choose_ch[0]:choose_ch[1]]
            # 创建二进制掩码
            norm_0 = torch.zeros_like(norm_tmp).detach().to(device)
            norm_0[choose_index] = 1
            norm_tmp_k = norm_0
            # 扩展掩码到像素级别
            tmp_bi = torch.as_tensor(norm_tmp_k.repeat_interleave(box_size)) * 1.0
            grad[i] = grad[i].put_(index[i], tmp_bi)
        return grad


    now = '{}TO{}_eps-{}-K-{}/'.format(args.model_type, args.model_t, eps, tk)
    now_pic = now + 'pictures/'
    if not os.path.exists(now):
        os.mkdir(os.path.join(now))
        os.mkdir(os.path.join(now_pic))
    #  初始化评估指标
    l0, l1, l2, linf = 0, 0, 0, 0
    FR_bb_epoch, FR_wb_epoch = 0, 0  # 黑盒和白盒攻击成功率
    for i, (img, gt) in enumerate(test_loader):
        img = img.to(device)
        gt = gt.to(device)
        # 白盒模型前向传播
        if 'inc' in args.model_type or 'xcep' in args.model_type:
            out = model(normalize(trans_incep(img.clone().detach())))

        else:
            out = model(normalize(img.clone().detach()))
        label = out.argmax(dim=-1).clone().detach()
        out_wb = label.clone().detach()
        out.backward(torch.ones_like(out))
        # 黑盒模型前向传播
        if 'inc' in args.model_t or 'xcep' in args.model_t:
            out_bb = model_t(normalize(trans_incep(img.clone().detach())))
        else:
            out_bb = model_t(normalize(img.clone().detach()))

        # 计算结构化掩码
        grad = back_grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        grad_fea = (grad * back_fea).sum(dim=1)
        resize = transforms.Resize((img_size, img_size), antialias=True)
        G_F = resize(grad_fea).reshape(len(img), 1, img_size, img_size)
        # grad_box = grad_choose(G_F, index, filterSize, choose)
        # 选择重要区域
        grad_box = grad_topk(G_F, index, filterSize, tk)
        # 生成对抗样本
        adv, adv_inf, adv_0, adv_00, grad_img = netG(img, grad_box)
        adv_img = adv.clone().detach()
        adv_test = adv.clone().detach()
        # 白盒模型评估对抗样本
        if 'inc' in args.model_type or 'xcep' in args.model_type:
            adv_out = model(normalize(trans_incep(adv.clone().detach())))

        else:
            adv_out = model(normalize(adv.clone().detach()))

        adv_out_to_wb = adv_out.clone().detach()
        # 黑盒模型评估对抗样本
        if 'inc' in args.model_t or 'xcep' in args.model_t:
            adv_out_to_bb = model_t(normalize(trans_incep(adv_test.clone().detach())))
        else:
            adv_out_to_bb = model_t(normalize(adv_test.clone().detach()))
        # 计算攻击成功率
        if args.target == -1:  # 非目标攻击
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) != out_wb).item()
            FR_bb_tmp = torch.sum(adv_out_to_bb.argmax(dim=-1) != out_bb.argmax(dim=-1)).item()

        else:  # 目标攻击
            FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) == out_wb).item()
            FR_bb_tmp = torch.sum(adv_out_to_bb.argmax(dim=-1) == out_bb.argmax(dim=-1)).item()
        # 累计攻击成功率
        FR_wb_epoch += FR_wb_tmp
        FR_bb_epoch += FR_bb_tmp
        # 计算扰动指标
        l0 += torch.norm(adv_0.clone().detach(), 0).item()  # L0范数（非零元素数量）
        l1 += torch.norm(adv_0.clone().detach() * adv_inf.clone().detach(), 1).item()  # L1范数
        l2 += torch.norm(adv_0.clone().detach() * adv_inf.clone().detach(), 2).item()  # L2范数
        linf = (torch.norm(adv_0.clone().detach() * adv_inf.clone().detach(), p=np.inf)).item()  # L∞范数
        # 定期保存示例图像
        if i in [201, 1001, 2001, 3001, 4001]:
            vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True),
                              now_pic + 'adv{}.png'.format(i))
            vutils.save_image(vutils.make_grid(grad_img, normalize=True, scale_each=True),
                              now_pic + 'grad_img{}.png'.format(i))
            vutils.save_image(vutils.make_grid(adv_img - img, normalize=True, scale_each=True),
                              now_pic + 'noise{}.png'.format(i))
            vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True),
                              now_pic + 'org{}.png'.format(i))
    # 计算平均指标
    FR_wb_ep_mean = FR_wb_epoch / test_size
    FR_bb_ep_mean = FR_bb_epoch / test_size
    print('FR-{}:{} | FR-{}:{}\n'.format(args.model_type, FR_wb_ep_mean, args.model_t,
                                         FR_bb_ep_mean))

    try:
        print('L0:{}--L1:{:.4f}--L2:{:.4f}--Linf:{:.4f}\n'.format(int(l0 / test_size), l1 / test_size, l2 / test_size,
                                                                  linf))
    except:
        pass
