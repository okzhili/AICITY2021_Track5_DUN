import argparse
import json
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from lib.config import cfg
from lib.modeling import build_model
import os


class MyDataSet(Dataset):

    def __init__(self, X, transform=None):
        self.img = X
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = self.transform(self.img[idx])
        return image


def get_roi_image(root,img_path, box):
    img = Image.open(root + img_path)
    ori_img = img.crop((box[0], box[1], box[2] + box[0], box[3] + box[1]))
    return ori_img


def get_features(root,dic, model):
    data_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frames = dic['frames']
    boxes = dic['boxes']
    roi_images = []

    for i in range(len(boxes)):
        roi_images.append(get_roi_image(root,frames[i], boxes[i]))
    datas = MyDataSet(roi_images, data_transform)
    loader = torch.utils.data.DataLoader(datas, batch_size=30, shuffle=False, pin_memory=True, num_workers=4)
    features = []
    for i, (im) in enumerate(loader):
        im = im.cuda()
        with torch.no_grad():
            feature = model(im)
        features.append(feature)
    features = torch.cat(features)
    features = torch.nn.functional.normalize(features)
    return features


parser = argparse.ArgumentParser(description="ReID Baseline Inference")
parser.add_argument(
    "--config_file", default="./configs/debug.yml", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('-track_file', type=str)
parser.add_argument('-save_file', type=str)
parser.add_argument('-root', type=str)
args = parser.parse_args()
root = args.root
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()
f = open(args.track_file)
t = json.load(f)
save_dict = {}
a = tqdm(total=len(t.keys()))
model = build_model(cfg, 1)
model.load_param(cfg.TEST.WEIGHT)
model.cuda()
model.eval()
for key in t.keys():
    a.update(1)
    features = get_features(root,t[key], model)
    save_dict[key] = features
pickle.dump(save_dict, open(args.save_file, 'wb'))