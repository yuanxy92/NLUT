import os
import numpy as np
import time
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
from torch.utils import data
from parameter import *
from utils.losses import *
from PIL import Image
import torch.utils.data as data
import net
from nlut_models import *

import os
import numpy as np

from parameter import cuda, Tensor, device
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print(f'now device is {device}')


# def train_transform():
#     transform_list = [
#         transforms.Resize(size=(512, 512)),
#         # transforms.Resize(size=(256, 256)),
#         transforms.RandomCrop(256),
#         transforms.ToTensor()
#     ]
#     return transforms.Compose(transform_list)

def train_transform():
    transform_list = [
        transforms.Resize(size=(400, 400)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class FlatFolderDataset2(data.Dataset):
    def __init__(self, root_hr, root_lr, transform):
        super(FlatFolderDataset2, self).__init__()
        self.root_hr = root_hr
        self.root_lr = root_lr
        # self.paths_hr = os.listdir(self.root_hr)
        # self.paths_lr = os.listdir(self.root_lr)
        # sorted(self.paths_hr)
        # sorted(self.paths_lr)
        self.paths = os.listdir(self.root_lr)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        path_hr = os.path.join(self.root_hr, path)
        path_lr = os.path.join(self.root_lr, path)
        img_hr = Image.open(path_hr).convert('RGB')
        img_lr = Image.open(path_lr).convert('RGB')
        img_hr = self.transform(img_hr)
        img_lr = self.transform(img_lr)
        return img_hr, img_lr, path_hr, path_lr

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset2'

def adjust_learning_rate(optimizer, iteration_count, opt):
    """Imitating the original implementation"""
    # lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    lr = opt.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def train(setting):

def train(opt):
    train_dataset = FlatFolderDataset2(opt.style_dir, opt.content_dir, train_transform())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)

    model = NLUTNet400(opt.model, dim=opt.dim).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
        for p in model.parameters()) / 1000000.0))

    # VGG
    vgg = net.vgg
    vgg.load_state_dict(torch.load(opt.vgg))
    encoder = net.Net(vgg)
    encoder.to(device)
    encoder.eval()

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("--------no checkpoint found---------")
    
    mseloss = nn.MSELoss()
    model.train()
    TVMN_temp = TVMN(opt.dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # if opt.resume:
    #     if os.path.isfile(opt.resume):
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    log_c = []
    log_s = []
    log_mse = []
    Time = time.time()

    losses = AverageMeter()
    c_losses = AverageMeter()
    s_losses = AverageMeter()
    mse_losses = AverageMeter()
    tv_losses = AverageMeter()
    mn_losses = AverageMeter()

    # -----------------------training------------------------
    # for i in range(opt.start_iter, opt.max_iter):
    i = 0
    for epoch_id in range((opt.max_iter - opt.start_iter) // len(train_dataset) // opt.batch_size):
        for batch, (style_images, content_images, path_hr, path_lr) in enumerate(train_dataloader):
            adjust_learning_rate(optimizer, iteration_count=i, opt=opt)
    
            content_images = content_images.to(device)
            style_images = style_images.to(device)
        
            stylized, st_out, others = model(
                content_images, content_images, style_images, TVMN=TVMN_temp)

            tvmn = others.get("tvmn")
            mn_cons = opt.lambda_smooth * \
                (tvmn[0]+10*tvmn[2]) + opt.lambda_mn*tvmn[1]

            # loss_c, loss_s = encoder(content_images, style_images, stylized)

            # loss_c = loss_c.mean()
            # loss_s = loss_s.mean()
            # loss_mse = mseloss(content_images, stylized)
            # loss_style = opt.content_weight*loss_c + \
            #     opt.style_weight*loss_s + mn_cons  # +tv_cons
            
            loss_style = mseloss(style_images, stylized) * 100 + mn_cons * 40  # +tv_cons

            # optimizer update
            optimizer.zero_grad()
            loss_style.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()

            # update loss log
            # log_c.append(loss_c.item())
            # log_s.append(loss_s.item())
            # log_mse.append(loss_mse.item())

            losses.update(loss_style.item())
            # c_losses.update(loss_c.item())
            # s_losses.update(loss_s.item())
            # mse_losses.update(loss_mse.item())
            mn_losses.update(mn_cons.item())

            i = i + 1

            # save image
            if i % opt.print_interval == 0:

                output_name = os.path.join(opt.save_dir, "%06d.jpg" % i)

                output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized.cpu(), st_out.cpu()),  # refined_out
                                        # output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized_rgb.cpu()), #refined_out
                                        # color_stylized.cpu(), another_content.cpu(), another_real_stylized.cpu()),
                                        0)
                save_image(output_images, output_name, nrow=opt.batch_size)
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print("iter %d   time/iter: %.2f  lr: %.6f loss_mn: %.4f losses: %.4f " % 
                    (i,(time.time()-Time)/opt.print_interval, current_lr,mn_losses.avg, losses.avg))
                Time = time.time()

            if (i + 1) % opt.save_model_interval == 0 or (i + 1) == opt.max_iter:
                # state_dict = model.module.state_dict()
                state_dict = model.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))

                state = {'iter': i, 'state_dict': state_dict,
                        'optimizer': optimizer.state_dict()}
                torch.save(state, opt.resume)
                torch.save(state, "./"+opt.save_dir+"/"+str(i)+"_style_lut.pth")


if __name__ == "__main__":
    opt = parser.parse_args()
    opt.content_dir = '/data/hdd/Data/Metalens/MetalensSR_20241220/lr_images'
    opt.style_dir = '/data/hdd/Data/Metalens/MetalensSR_20241220/hr_images'
    opt.save_dir = './data/metacam/experiments'
    opt.lr = 0.001
    train(opt)
    
