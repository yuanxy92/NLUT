import torch
from nlut_models import *
from PIL import Image
from utils.losses import *
from parameter_finetuning import *
import torch.nn as nn
from torchvision.utils import save_image, save_image_newsize
import time
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print(f'now device is {device}')

def train_transform2():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def get_lut2(opt, model, TVMN, original, example):
    content_tf2 = train_transform2()
    content_images = content_tf2(Image.open(
        original).convert('RGB')).unsqueeze(0).to(device)
    style_images = content_tf2(Image.open(
        example).convert('RGB')).unsqueeze(0).to(device)

    content_images = content_images.repeat(1, 1, 1, 1)
    style_images = style_images.repeat(1, 1, 1, 1)

    stylized, st_out, others = model(
        content_images, content_images, style_images, TVMN=TVMN)

    LUT = others.get("LUT")
    return LUT[:1]


def draw_img2(original, dst, LUT):
    content_tf2 = train_transform2()
    target = content_tf2(Image.open(original).convert(
        'RGB')).unsqueeze(0).to(device)

    TrilinearInterpo = TrilinearInterpolation()
    img_res = TrilinearInterpo(LUT, target)
    img_out = img_res+target

    save_image(img_out, dst, nrow=1)

def draw_img_newsize(original, dst, LUT):
    content_tf2 = train_transform2()
    target = content_tf2(Image.open(original).convert(
        'RGB')).unsqueeze(0).to(device)

    TrilinearInterpo = TrilinearInterpolation()
    img_res = TrilinearInterpo(LUT, target)
    img_out = img_res+target

    save_image_newsize(img_out, dst, 100, nrow=1)


if __name__ == '__main__':
    opt = parser.parse_args()
    # Sample 100 random numbers from the range 1 to 26,000
    opt.pretrained = './data/metacam/experiments/17999_style_lut.pth'
    model = NLUTNet2(opt.model, dim=opt.dim).to(device)
    TVMN = TVMN(opt.dim).to(device)
    model.eval()
    checkpoint = torch.load(opt.pretrained)
    opt.start_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])

    imagenames = os.listdir('/data/hdd/Data/Metalens/MetalensSR_20241220/lr_images')
    luts = []

    for imagename in imagenames:
        original = f'/data/hdd/Data/Metalens/MetalensSR_20241220/lr_images/{imagename}'
        example = f'/data/hdd/Data/Metalens/MetalensSR_20241220/hr_images/{imagename}'
        dst = f'/data/hdd/Data/Metalens/MetalensSR_20241220/lr_images_corrected/{imagename}'
        lut = get_lut2(opt, model, TVMN, original, example)
        draw_img_newsize(original, dst, lut)

    # random_numbers = random.sample(range(1, 26001), 100)
    # for num in random_numbers:
    #     original = f'/data/hdd/Data/Metalens/MetalensSR_20241220/lr_images/{imagenames[num]}'
    #     example = f'/data/hdd/Data/Metalens/MetalensSR_20241220/hr_images/{imagenames[num]}'
    #     dst = f'./data/metacam/test/{imagenames[num]}'
    #     lut = get_lut2(opt, model, TVMN, original, example)
    #     draw_img2(original, dst, lut)

    
    




