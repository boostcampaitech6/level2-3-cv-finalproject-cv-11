import io
import os.path as osp
import time

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale
from rembg import remove

from models.afwm import AFWM
from models.networks import load_checkpoint
from options.test_options import TestOptions

from celery.app.base import Celery
import redis
from PIL import Image

app = Celery('app', broker='redis://10.0.3.6:6379/0', backend='redis://10.0.3.6:6379/1')

r = redis.Redis(host='10.0.3.6', port=6379, db=1)

@app.task
def real_to_wrap(sketch_id, sketc_id):
# if __name__ == "__main__":
    print(sketch_id)
    print(sketc_id)

    start_epoch, epoch_iter = 1, 0

    warp_model = AFWM(16)
    load_checkpoint(warp_model, 'checkpoints/warp_viton.pth')
    warp_model.eval()
    warp_model.cuda()
    data_path = '/home/rtboa/DCI-VTON-Virtual-Try-On/VITON-HD/test'
    im_name = 'image/00055_00.jpg'

    labels = {
        0: ['background', [0, 10]],
        1: ['hair', [1, 2]],
        2: ['face', [4, 13]],
        3: ['upper', [5, 6, 7]],
        4: ['bottom', [9, 12]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [16]],
        8: ['right_leg', [17]],
        9: ['left_shoe', [18]],
        10: ['right_shoe', [19]],
        11: ['socks', [8]],
        12: ['noise', [3, 11]]
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    num_samples = 0
    with torch.no_grad():
        for epoch in range(start_epoch, 2):
            epoch_start_time = time.time()
            iter_start_time = time.time()

            semantic_nc = 13
            fine_width = int(512 / 256 * 192)
            fine_height = 512


            
            # input1
            # c_paired = data['cloth'].cuda()
            c_bytes = r.get(f'task:{sketch_id}:result')
            c_paired = Image.open(io.BytesIO(c_bytes)).convert('RGB')
            # c_paired = Image.open(osp.join(data_path, 'cloth/00006_00.jpg')).convert('RGB')
            c_paired = transforms.Resize(fine_width, interpolation=2)(c_paired)

            # cm_paired = data['cloth_mask']
            img = remove(c_paired)
            cm_paired = Image.new('RGB', img.size, (255,255,255))
            width, height = img.size
            pixel_data = img.load()

            for y in range(height):
                for x in range(width):
                    rr, g, b, alpha = pixel_data[x, y]
                    if alpha == 0:
                        cm_paired.putpixel((x, y), (0,0,0))
                    else:
                        cm_paired.putpixel((x, y), (255,255,255))

            transform_g = Grayscale(num_output_channels=1)
            cm_paired = transform_g(cm_paired)
            # cm_paired = cm_paired.convert("RGB")
            cm_paired = transforms.Resize(fine_width, interpolation=0)(cm_paired)

            c_paired = transform(c_paired).cuda() # [-1,1]

            cm_array = np.array(cm_paired)
            cm_array = (cm_array >= 128).astype(np.float32)
            cm_paired = torch.from_numpy(cm_array)  # [0,1]
            cm_paired.unsqueeze_(0)
            
            cm_paired = torch.FloatTensor((cm_paired.numpy() > 0.5).astype(float)).cuda()

            # input2
            # parse_agnostic = data['parse_agnostic'].cuda()
            parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
            im_parse_pil_big = Image.open(osp.join(data_path, parse_name))
            im_parse_pil = transforms.Resize(fine_width, interpolation=0)(im_parse_pil_big)
            parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

            image_parse_agnostic = Image.open(
                osp.join(data_path, parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
            image_parse_agnostic = transforms.Resize(fine_width, interpolation=0)(image_parse_agnostic)
            parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
            image_parse_agnostic = transform(image_parse_agnostic.convert('RGB'))

            parse_agnostic_map = torch.FloatTensor(20, fine_height, fine_width).zero_()
            parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
            new_parse_agnostic_map = torch.FloatTensor(semantic_nc, fine_height, fine_width).zero_()
            for i in range(len(labels)):
                for label in labels[i][1]:
                    new_parse_agnostic_map[i] += parse_agnostic_map[label]

            parse_agnostic = new_parse_agnostic_map.cuda()

            # densepose = data['densepose'].cuda()
            densepose_name = im_name.replace('image', 'image-densepose')
            densepose_map = Image.open(osp.join(data_path, densepose_name))
            densepose_map = transforms.Resize(fine_width, interpolation=2)(densepose_map)
            densepose_map = transform(densepose_map)  # [-1,1]
            densepose = densepose_map.cuda()

            c_paired = c_paired.unsqueeze(0)
            cm_paired = cm_paired.unsqueeze(0)
            parse_agnostic = parse_agnostic.unsqueeze(0)
            densepose = densepose.unsqueeze(0)

            pre_clothes_mask_down = F.interpolate(cm_paired, size=(256, 192), mode='nearest')
            input_parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='nearest')
            clothes_down = F.interpolate(c_paired, size=(256, 192), mode='bilinear')
            densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')


            # input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
            input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)
            flow_out = warp_model(input2, clothes_down)
            warped_cloth, last_flow = flow_out
            warped_mask = F.grid_sample(pre_clothes_mask_down, last_flow.permute(0, 2, 3, 1),
                                        mode='bilinear', padding_mode='zeros')

            N, _, iH, iW = c_paired.size()
            if iH != 256:
                last_flow = F.interpolate(last_flow, size=(iH, iW), mode='bilinear')
                warped_cloth = F.grid_sample(c_paired, last_flow.permute(0, 2, 3, 1),
                                                mode='bilinear', padding_mode='border')
                warped_mask = F.grid_sample(cm_paired, last_flow.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros')

            for j in range(warped_cloth.shape[0]):
                e = warped_cloth
                f = warped_mask
                combine = e[j].squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode('.jpg', bgr)
                img_bytes = img_encoded.tobytes()
                
                mask_img = f[j].squeeze(0).cpu().numpy()
                mask_img = (mask_img * 255).astype(np.uint8)
                _, mask_img_encoded = cv2.imencode('.jpg', mask_img)
                mask_img_bytes = mask_img_encoded.tobytes()

                r.set(f'task:{sketc_id}:result2', img_bytes)
                r.set(f'task:{sketc_id}:result3', mask_img_bytes)
                cv2.imwrite('img.jpg', bgr)
                cv2.imwrite('mask_img.jpg', mask_img)