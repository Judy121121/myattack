import os
from collections import OrderedDict
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import utils
from train_generators import GeneratorResnet
from modules.sync_batchnorm import convert_model
from my_generator_test import  MyGeneratorResnet
import yaml;
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
def backward_hook(module, grad_in, grad_out):
    global back_grad
    back_grad = grad_out[0].clone().detach()


# Get feature layer
def forward_hook(module, input, output):
    global back_fea
    back_fea = output.detach()
import torch.nn as nn

def find_layer(module, last_conv=None):
    for child in module.children():
        if isinstance(child, nn.Conv3d):
            last_conv = child
        last_conv = find_layer(child, last_conv)
    return last_conv
def get_name(name):
    #ph
    name=name[0]
    location = name.find('/') + 1
    name = name[location:]
    location = name.find('/') + 1
    name = name[location:]
    str1 = name.split('/')[-1]
    location = name.rfind('/')
    name = name[:location]
    return name, str1
def gen_img(img,url,to_01=False, normalize=False):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    torchvision.utils.save_image(img,url, normalize=normalize)
from torch.cuda.amp import autocast,GradScaler
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
    model, opt = processor.loading()
    layer = find_layer(model.conv2d)
    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)
    model = model.to(device)
    model.train()

    # Input dimensions 4. 输入尺寸和网格索引计算
    # 根据模型类型设置输入尺寸
    scale_size = 256
    img_size = 224
    filterSize = 8
    stride = 8
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
    index_temp = index
    # index = torch.LongTensor(np.tile(index, (args.batch_size, 1, 1))).to(device)

    netG = GeneratorResnet(eps=eps / 255.)
    # netG = MyGeneratorResnet(eps=eps / 255.)
    if args.checkpoint != '':
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


    processor.load_data()
    split = 'dev'
    loader = processor.data_loader[split]
    test_size = len(loader)
    print('test data size:', test_size)


    # 7. 重要区域选择函数
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

    for i, data in enumerate(loader):
        name=data[4][0].split("|")[0]
        img = data[0].to(device)
        img_in = img.clone().detach()
        img_number = data[1]
        print(f"number:{img_number}")
        gt = data[2].to(device)
        label_lgt = data[3].to(device)
        index = torch.LongTensor(np.tile(index_temp, (img_number, 1, 1))).to(device)
        keypoint = data[5][0].to(0)
        with autocast():
            out = model(img_in, img_number, label=gt, label_lgt=label_lgt)
            label = gt.detach()
            out_wb = label.clone().detach()
            # todo
            out['sequence_logits'][0].backward(torch.ones_like(out['sequence_logits'][0]))
            del out
            torch.cuda.empty_cache()
            # 黑盒模型前向传播
            # 计算结构化掩码
            netG.train()
            grad = back_grad.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            grad_fea = (grad * back_fea).sum(dim=1)
            resize = transforms.Resize((img_size, img_size), antialias=True)
            G_F = resize(grad_fea).reshape(len(img[0]), 1, img_size, img_size)

            with torch.no_grad():
                grad_box = grad_topk(G_F.float(), index, filterSize, tk)
                del G_F, grad_fea, grad
                torch.cuda.empty_cache()
                # 生成对抗样本
                adv, adv_0, adv_00, grad_img,x_inf = netG(img[0], grad_box)
            # adv, adv_0, adv_00 = chunk_generate(netG, img[0], grad_box)
            for i in range(img.shape[0]):
                ss = name[0:name.index(name.split("_")[-1]) - 1]
                name = name + "/1"
                saveurl = f'/media/cv3/store1/postgraduate/y2023/WLF/datasets/ph_adv/newAttack/{split}/{name}'
                if not os.path.exists(saveurl):
                    os.makedirs(saveurl)
                for j in range(img.shape[1]):
                    img1 = adv[j]
                    gen_img(img1, '{}/{}.avi_pid0_fn{:06d}-0.png'.format(saveurl, ss, j))
            del adv,adv_0, adv_00
            torch.cuda.empty_cache()
