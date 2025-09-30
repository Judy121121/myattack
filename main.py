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
def compute_loss_with_sequence_term(
        x: torch.Tensor,
        x_adv: torch.Tensor,
        mu: float,
        alpha: float,
        lambd: float
) -> torch.Tensor:
    """
    计算完整损失函数（包含序列差分项）：
    -μ·D_KL(f(x)||f(x+η)) + (α/2)·||η||² + λ·Σ||r_t - r_{t+1}||₁

    参数说明：
        model: 序列图像模型（输出包含时间步维度的log_softmax结果）；
        x: 原始序列图像输入，形状$(batch_size, T, 3, 224, 224)$（需提前添加batch维度）；
        eta: 扰动项，形状与$x$一致（$(batch_size, T, 3, 224, 224)$），requires_grad=True；
        mu: KL散度项权重；
        alpha: 扰动项权重；
        lambd: 序列差分项权重。
    返回：
        total_loss: 总损失（标量）。
    """
    # 1. 计算加扰动后的输入（形状保持一致）
    eta=x_adv-x# 形状：(batch_size, T, 3, 224, 224)

    # 2. 计算模型输出（原始输入与扰动输入）

    # 3. KL散度项（D_KL(f(x)||f(x+η))）
    # 注：F.kl_div(input=log_q, target=p) 计算 D_KL(p||q)，其中 p=exp(f_x)（原始分布），q=exp(f_xeta)（扰动分布）
    kl_div = F.kl_div(x_adv, torch.exp(x), reduction="batchmean")
    kl_term = -mu * kl_div  # 公式中的“-μ·D_KL”

    # 4. 扰动项L2范数（修正：对所有维度求平均）
    # 正确计算：||η||² = (1/(B*T*C*H*W)) * sum(η²)，即torch.mean(η²)
    eta_norm_sq = torch.mean(eta ** 2)  # 对batch、时间步、通道、高度、宽度所有维度求平均
    eta_term = (alpha / 2) * eta_norm_sq

    # 5. 序列差分项（λ·Σ||r_t - r_{t+1}||₁）  # 扰动输入的输出：(batch_size, T, num_classes)
    r_diff = eta[:, :-1, :] - eta[:, 1:, :]  # 相邻时间步差：(batch_size, T-1, num_classes)
    # 对时间步（T-1）、类别（num_classes）求和，再batch平均
    l1_norm = torch.mean(torch.sum(torch.abs(r_diff), dim=(1, 2)))
    sequence_term = lambd * l1_norm

    # 6. 总损失
    total_loss = kl_term + eta_term + sequence_term
    return total_loss


def compute_frame_importance(model, input_seq, img_number, label, label_lgt, k=50):
    """
    计算关键帧重要性并返回梯度张量，只有关键帧保留原始梯度

    Args:
        model: 手语识别模型
        input_seq: 输入视频序列 [batch, n_frames, C, H, W]
        img_number: 帧数信息
        label: 标签信息
        label_lgt: 标签长度信息
        k: 选择的关键帧数量

    Returns:
        torch.Tensor: 梯度张量，形状与input_seq相同(1, n, 3, 224, 224)
                      只有关键帧保留原始梯度，其他帧梯度为0
    """
    # 确保模型处于训练模式以计算梯度
    model.train()

    # 确保输入需要梯度计算
    input_seq = input_seq.clone().detach()
    input_seq.requires_grad = True

    # 前向计算
    output = model(input_seq, img_number, label=label, label_lgt=label_lgt)
    loss = model.criterion_calculation(output, label, label_lgt)

    # 反向传播获取梯度
    model.zero_grad()
    loss.backward()

    # 检查梯度是否存在
    if input_seq.grad is None:
        raise RuntimeError("没有计算梯度，请检查模型是否可训练或输入是否需要梯度")

    # 获取原始梯度 (形状: [1, n, 3, 224, 224])
    grads = input_seq.grad.data.clone()

    # 计算每帧的梯度重要性 (每帧梯度的L2范数)
    # 将每帧梯度展平为向量，然后计算L2范数
    frame_importance = torch.norm(grads.view(grads.size(1), -1), p=2, dim=1)  # 形状: [n]

    # 获取关键帧索引 (选择重要性最高的k帧)
    _, topk_indices = torch.topk(frame_importance, k)

    # 创建全零梯度张量，形状与原始梯度相同
    result_grads = torch.zeros_like(grads)

    # 将关键帧位置的原始梯度值复制到结果中
    result_grads[:, topk_indices] = grads[:, topk_indices]

    return result_grads.detach()
def gen_img(img,url,to_01=False, normalize=False):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    torchvision.utils.save_image(img,url, normalize=normalize)
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
    lam_3 = 0.000005
    eps = args.eps
    # 模式选择：TK=True使用TopK方法选择重要区域，否则使用区间选择
    tk = args.tk
    epochs = args.epochs
    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 存储中间特征和梯度
    # back_fea = torch.tensor([]).to(device)
    # back_grad = torch.tensor([]).to(device)
    # Model                                                            weights/soft_eps10_res50_tk0.95.pth
    model, opt = processor.loading()
    # layer = find_layer(model.conv2d)
    # layer =model.conv2d.s4.pathway0_res0.branch2.b
    # layer1 =model.conv2d.s3_fuse.conv_s2f
    # layer.register_forward_hook(forward_hook)
    # layer.register_full_backward_hook(backward_hook)
    model = model.to(device)
    # 模型设为评估模式
    model.eval()

    # 不同模型的输入尺寸要求：
    # ResNet50：缩放256→裁剪224
    # Inception v3：缩放300→裁剪299
    # filterSize和stride：用于划分图像网格
    # Input dimensions
    scaler = GradScaler()
    accumulation_steps = 1  # 梯度累积步数
    scale_size = 256
    img_size = 224
    filterSize = 8
    stride = 8
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
    # Generator生成器初始化eps参数：
    # 扰动预算（归一化到[0,1]范围）
    # 支持从检查点恢复训练
    index_temp=index
    netG = GeneratorResnet(eps=eps)
    # netG = MyGeneratorResnet(eps=eps / 255.)
    if args.checkpoint != '':
        netG.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0'))
    netG.to(device)
    # Optimizer优化器设置
    # Adam优化器用于训练生成器
    # 学习率通过命令行参数设置
    optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
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


    processor.load_data()
    split = 'train'
    loader = processor.data_loader[split]
    train_size = len(loader)
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
    now = 'TK-{}_TG-{}_eps-{}_S-{}_Q-{}_K-{}-box-{}/'.format(args.model_type, args.target, eps, lam_1, lam_2, tk,
                                                             filterSize)
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
    # with open('/media/cv3/store1/postgraduate/y2023/WLF/NLA-SLR/keypoints/phtrain.pkl', 'rb') as f:
    #     name2all_keypoints = pickle.load(f)
    # 15. 训练主循环
    for epoch in range(epochs):
        FR_wb, FR_wb_epoch = 0, 0
        for i, data in enumerate(loader):
            # 外层循环：训练轮数
            # 内层循环：遍历数据加载器
            # 15.1 数据准备和目标设置
            img = data[0].to(device)
            if len(img[0])>220:
                print('length  too big')
                continue
            img_in=img.clone().detach()
            img_number = data[1]
            print(f"number:{img_number}")
            gt = data[2].to(device)
            label_lgt = data[3].to(device)
            grad_import=compute_frame_importance(model,img_in,img_number,gt,label_lgt,k=int(img_number*0.5))
            index = torch.LongTensor(np.tile(index_temp, (img_number, 1, 1))).to(device)
            # keypoint=data[5][0].to(0)
            with autocast():
                model.eval()
                netG.train()
                adv, adv_0, adv_00, grad_img,x_inf= netG(img[0], grad_import[0])

                torch.cuda.empty_cache()
                adv_img = adv.clone().detach()
                # 15.3 对抗样本评估
                # 评估对抗样本的攻击成功率（FR）
                # 计算对抗损失
                # adv_out = model(normalize(adv))
                model.train()
                adv_out = model(adv.unsqueeze(0), img_number, label=gt, label_lgt=label_lgt)
                model.eval()
                # adv_out_to_wb = adv_out.clone().detach()
                adv_out_to_wb = adv_out['sequence_logits'][0]
                # del adv_out
                # torch.cuda.empty_cache()
                with torch.no_grad():
                    aligner = SignLanguageAligner()
                    label_agliner = aligner.align(adv_out_to_wb, gt)
                # FR_wb_tmp = torch.sum(adv_out_to_wb.argmax(dim=-1) != out_wb).item()
                # Untargeted Attack
                # loss_adv = criterion(adv_out_to_wb, label_agliner)/img_number.to(device)
                loss_adv = model.criterion_calculation(adv_out,gt,label_lgt)
                # loss_diff=total_variation(adv)
                # loss_diff=compute_loss_with_sequence_term(img,adv.unsqueeze(0),1,1,1)
                del adv_out_to_wb
                torch.cuda.empty_cache()
                # loss_adv=sequence_cw_loss(adv_out_to_wb,label_agliner)
                # L1稀疏正则
                loss_spa = torch.norm(adv_0, 1)
                bi_adv_00 = torch.where(adv_00 < 0.5, torch.zeros_like(adv_00), torch.ones_like(adv_00) * grad_import)
                # 二值化损失
                loss_qua = torch.sum((bi_adv_00 - adv_00) ** 2)
                # print(f"loss_adv{loss_adv},loss_spa {loss_spa},loss_qua {loss_qua},loss_diff {loss_diff}")
                print(f"loss_adv{loss_adv},loss_spa {loss_spa},loss_qua {loss_qua}")
                loss = -loss_adv+lam_1 * loss_qua
                # del  adv_0, adv_00
                torch.cuda.empty_cache()
                scaled_loss = scaler.scale(loss / accumulation_steps)

                # 反向传播
                scaled_loss.backward()

                # 梯度累积更新
                if (i + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    scaler.unscale_(optimG)
                    # torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)

                    # 更新权重
                    scaler.step(optimG)
                    scaler.update()
                    optimG.zero_grad()

                    # 清理缓存
                    torch.cuda.empty_cache()

            # 15.5 日志记录和输出
            # 定期记录和打印训练指标
            # 包括各种范数（L0、L1、L2、L∞）和损失值
            if i in [0,1000,2000,3000,4000,5000]:
                for j in range(x_inf.shape[0]):
                    img1 = x_inf[j]
                    url='/media/cv3/store1/postgraduate/y2023/WLF/EGS-TSSA-main/TK-SLR_TG--1_eps-0.1_S-1e-07_Q-3e-07_K-0.6-box-8/my/'
                    gen_img(img1,  '{}/ep{}-inf{}.png'.format(url,epoch, j))
                vutils.save_image(vutils.make_grid(adv_img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_adv{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(grad_img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_grad_img{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(adv_img - img, normalize=True, scale_each=True),
                                  now_pic + 'ep{}_noise{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(adv_0, normalize=True, scale_each=True),
                                  now_pic + 'ep{}-adv0{}.png'.format(epoch, i))
                vutils.save_image(vutils.make_grid(adv_00, normalize=True, scale_each=True),
                                  now_pic + 'ep{}-adv00{}.png'.format(epoch, i))

            print(f"第{epoch}轮，第{i}次，loss：{loss}")
        FR_wb_ep_mean = FR_wb_epoch / train_size
        print('running:{} | FR-{}:{}\n'.format(epoch, args.model_type, FR_wb_ep_mean))
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

