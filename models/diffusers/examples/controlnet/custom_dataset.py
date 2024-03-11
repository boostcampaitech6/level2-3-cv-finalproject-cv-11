import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/junseok/controlnet_dataset/captions.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx_list = sorted(os.listdir('/home/junseok/controlnet_dataset/train'))
        item = self.data[idx_list[idx]]

        filename = item
        prompt = item

        source = cv2.imread('/home/junseok/controlnet_dataset/train/sketch/' + filename)
        target = cv2.imread('/home/junseok/controlnet_dataset/train/image/' + filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    

    