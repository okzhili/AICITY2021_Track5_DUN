import json
import os
from typing import List, Any

import torch
from torch import nn
import pytorch_lightning as pl
import timm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import T_co
import numpy as np
import nltk
from tqdm import tqdm
import pickle
from torchmetrics import RetrievalMRR
from pytorch_metric_learning.losses import CircleLoss



class TrainDataset(Dataset):
    def __init__(self, feature_file, glove_file, annotation_json, size, LV, LS, num_mul=5, mode='all'):
        super(TrainDataset, self).__init__()
        self.num_mul = num_mul
        # self.feature_file = feature_file
        self.LV = LV
        self.LS = LS
        self.anno = json.load(open(annotation_json, 'r'))
        train_vids, val_vids, all_vids = [], [], []
        for k, v in self.anno.items():
            all_vids.append(k)
            if v['frames'][0].split('/')[1] in ['S01', 'S04']:
                train_vids.append(k)
            else:
                val_vids.append(k)
        if mode == 'train':
            self.vids = train_vids
        if mode == 'val':
            self.vids = val_vids
        if mode == 'all':
            self.vids = all_vids

        self.glove = np.load(glove_file, allow_pickle=True).tolist()
        self.feature_db = pickle.load(open(feature_file, 'rb'))
        for k, v in self.feature_db.items():
            self.feature_db[k] = v.cpu()

    def __getitem__(self, index):
        index = index % len(self.vids)
        vid = self.vids[index]
        frames = self.anno[vid]['frames']
        boxes = self.anno[vid]['boxes']
        nl = self.anno[vid]['nl']
        vf = self.feature_db[vid]
        if len(vf) > self.LV:
            vf = vf[torch.linspace(0, len(vf) - 1, self.LV).round().long()]
            lv = self.LV
        else:
            em = torch.zeros(self.LV, 2048)
            lv = len(vf)
            em[0:lv] = vf
            vf = em
        # # vf = vf[torch.linspace(0, len(vf) - 1, self.LV).round().long()]
        # vf = vf[torch.randint(low=0, high=len(vf), size=(self.LV,)).sort()[0]]
        sentence = np.random.choice(nl)
        words = nltk.word_tokenize(sentence.lower())
        sf = np.zeros(shape=(self.LS, 300), dtype=np.float32)
        for i, w in enumerate(words):
            if i >= self.LS:
                break
            try:
                sf[i] = self.glove[w]
            except KeyError:
                pass
        ls = min(len(words), self.LS)
        return vf, sf, lv, ls, dict(vid=vid, sentence=sentence, ls=ls, frames=len(frames))

    def __len__(self):
        return len(self.vids)


class TestVisualDataset(Dataset):
    def __init__(self, test_file,annotation_json, LV):
        super(TestVisualDataset, self).__init__()
        self.feature_db = pickle.load(open(test_file, 'rb'))
        for k, v in self.feature_db.items():
            self.feature_db[k] = v.cpu()
        self.LV = LV
        self.anno = json.load(open(annotation_json, 'r'))
        self.vids = list(self.anno.keys())

    def __getitem__(self, index) -> T_co:
        index = index % len(self)
        vid = self.vids[index]
        frames = self.anno[vid]['frames']
        boxes = self.anno[vid]['boxes']
        vf = self.feature_db[vid]
        if len(vf) > self.LV:
            vf = vf[torch.linspace(0, len(vf) - 1, self.LV).round().long()]
            lv = self.LV
        else:
            em = torch.zeros(self.LV, 2048)
            lv = len(vf)
            em[0:lv] = vf
            vf = em
        return vf, lv, dict(vid=vid, frames=len(frames))

    def __len__(self):
        return len(self.vids)


class TestSentenceDataset(Dataset):
    def __init__(self, annotation_json, glove_file, LS, index):
        super(TestSentenceDataset, self).__init__()
        self.LS = LS
        self.index=index
        self.anno = json.load(open(annotation_json, 'r'))
        self.vids = list(self.anno.keys())
        self.glove = np.load(glove_file, allow_pickle=True).tolist()

    def __getitem__(self, index) -> T_co:
        vid = self.vids[index]
        nl = self.anno[vid]
        sentence = nl[self.index]
        words = nltk.word_tokenize(sentence.lower())
        sf = np.zeros(shape=(self.LS, 300), dtype=np.float32)
        for i, w in enumerate(words):
            if i >= self.LS:
                break
            try:
                sf[i] = self.glove[w]
            except KeyError:
                pass
        ls = min(len(words), self.LS)
        return sf, dict(ls=ls, vid=vid, nl=nl)

    def __len__(self):
        return len(self.vids)


class Baseline(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Baseline, self).__init__()
        self.__dict__.update(kwargs)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.gru1 = nn.GRU(input_size=self.d_v, hidden_size=self.d_v_h, bidirectional=True)
        self.gru2 = nn.GRU(input_size=self.d_s, hidden_size=self.d_s_h, bidirectional=True)
        self.fc1 = nn.Linear(2 * self.d_v_h, self.final_dim)
        self.fc2 = nn.Linear(2 * self.d_s_h, self.final_dim)
        if self.use_bn:
            self.norm1 = nn.BatchNorm1d(self.final_dim)
            self.norm2 = nn.BatchNorm1d(self.final_dim)
        self.mrr_metric = RetrievalMRR()
        # self.circle_loss = TripletLoss()
        self.circle_loss = CircleLoss()

    def forward(self, *args, **kwargs):
        return None

    def visual_forward(self, x, lv):
        _, x = self.gru1(pack_padded_sequence(x.transpose(0, 1), lv.cpu(), False, False))
        x = x.transpose(0, 1).flatten(1)
        x = self.fc1(x)
        if self.use_bn:
            x = self.norm1(x)
        return x

    def text_forward(self, s, ls):
        _, s = self.gru2(pack_padded_sequence(s.transpose(0, 1), ls.cpu(), False, False))
        s = s.transpose(0, 1).flatten(1)
        s = self.fc2(s)
        if self.use_bn:
            s = self.norm2(s)
        return s

    def training_step(self, bc, idx, **kwargs):
        v, s, lv, ls, z = bc
        v = self.visual_forward(v, lv)
        s = self.text_forward(s, ls)
        # loss = torch.relu(n - p + self.margin).sum()
        # loss = self.circle_loss(s, v)+self.circle_loss(v,s)
        loss = self.circle_loss(torch.cat([v, s]), torch.cat(
            [torch.arange(len(v), device=self.device), torch.arange(len(v), device=self.device)]))
        return dict(loss=loss, v=v.detach(), s=s.detach())

    def training_epoch_end(self, outputs: List[Any]) -> None:
        v = torch.cat([o['v'] for o in outputs])
        s = torch.cat([o['s'] for o in outputs])
        mat = []
        for si in torch.chunk(s, len(s) // self.batch_size + 1, ):
            mat.append(F.cosine_similarity(si[:, None, :], v[None, :, :], dim=-1))
        mat = torch.cat(mat)
        r = mat.argmax(-1)
        r1 = (r == torch.arange(len(mat), device=self.device)).float().mean()
        print('train r1', r1)

    def validation_step(self, bc, idx, **kwargs):
        v, s, lv, ls, z = bc
        v = self.visual_forward(v, lv)
        s = self.text_forward(s, ls)
        # loss = torch.relu(n - p + self.margin).sum()
        # loss = self.circle_loss(s, v)+self.circle_loss(v,s)
        loss = self.circle_loss(torch.cat([v, s]), torch.cat(
            [torch.arange(len(v), device=self.device), torch.arange(len(v), device=self.device)]))
        return dict(v_loss=loss, v=v.detach(), s=s.detach())

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        v = torch.cat([o['v'] for o in outputs])
        s = torch.cat([o['s'] for o in outputs])
        mat = []
        for si in torch.chunk(s, len(s) // self.batch_size + 1, ):
            mat.append(F.cosine_similarity(si[:, None, :], v[None, :, :], dim=-1))
        mat = torch.cat(mat)
        mrr = self.mrr_metric(
            torch.arange(len(mat), device=self.device)[:, None].expand(len(mat), len(mat)).flatten(),
            mat.flatten(),
            torch.eye(len(mat), device=self.device).long().bool().flatten()
        )
        r = mat.argmax(-1)
        r1 = (r == torch.arange(len(mat), device=self.device)).float().mean()
        print('r1_cos', r1)
        print('mrr_cos', mrr)
        self.log_dict(dictionary=dict(r1_cos=r1, mrr_cos=mrr))
        mat = []
        for si in torch.chunk(s, len(s) // self.batch_size + 1, ):
            mat.append(torch.norm(si[:, None, :] - v[None, :, :], dim=-1))
        mat = torch.cat(mat)
        mrr = self.mrr_metric(
            torch.arange(len(mat), device=self.device)[:, None].expand(len(mat), len(mat)).flatten(),
            -mat.flatten(),
            torch.eye(len(mat), device=self.device).long().bool().flatten()
        )
        r = mat.argmin(-1)
        r1 = (r == torch.arange(len(mat), device=self.device)).float().mean()
        print('r1_p2', r1)
        print('mrr_p2', mrr)
        self.log_dict(dictionary=dict(r1_p2=r1, mrr_p2=mrr))

    def train_dataloader(self):
        return DataLoader(TrainDataset(self.train_file, self.glove_file,
                                       self.train_json,
                                       self.size, self.LV, self.LS),
                          num_workers=self.num_workers,
                          batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TrainDataset(self.train_file, self.glove_file,
                                       self.train_json,
                                       self.size, self.LV, self.LS, mode='val'),
                          num_workers=self.num_workers,
                          batch_size=self.batch_size, shuffle=False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sdl = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-5, max_lr=1e-3, step_size_up=1000, step_size_down=1000,
                                                cycle_momentum=False)
        return [opt], [{
            'scheduler': sdl,
            'interval': 'step'}
        ]


def train(args):
    model = Baseline(**args)
    ckpt = pl.callbacks.ModelCheckpoint(filename=args['name'] + '-{epoch:002d}-{mrr_cos:.6f}', monitor='mrr_cos',
                                        mode='max')
    trainer = pl.Trainer(callbacks=[ckpt, ],
                         gpus=[0],
                         max_epochs=args['max_epochs'],
                         deterministic=True,
                         profiler=True,
                         check_val_every_n_epoch=20,
                         # limit_train_batches=10,
                         )
    trainer.fit(model)


def test(args):
    model = Baseline.load_from_checkpoint(args['ckpt'], **args)
    model.cuda().eval()
    v_loader = DataLoader(TestVisualDataset(args['test_file'], args['test_v_json'], args['LV']),
                          batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False)
    vv, vids = [], []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(v_loader)):
            v, lv, z = batch
            v = model.visual_forward(v.cuda(), lv)
            vv.append(v)
            vids.extend(z['vid'])
        vv = torch.cat(vv)
        vids = np.array(vids)
        vv = torch.nn.functional.normalize(vv)

        for i in range(3):
            zz = []
            ss = []
            s_loader = DataLoader(TestSentenceDataset(args['test_s_json'], args['glove_file'], args['LS'],i),
                                  batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=False)

            for idx, batch in enumerate(tqdm(s_loader)):
                s, z = batch
                s = model.text_forward(s.cuda(), z['ls'])
                ss.append(s)
                zz.extend(z['vid'])
            ss = torch.cat(ss)
            ss = torch.nn.functional.normalize(ss)
            save_path = 'mat/'+str(args['save_num'])+'_'+str(i)+'.pkl'
            pickle.dump((ss, vv, zz, vids), open(save_path, 'wb'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NL')
    parser.add_argument('-train_file', type=str)
    parser.add_argument('-root', type=str)
    parser.add_argument('-use_bn', action='store_true')
    parser.add_argument('-epoch', type=int)
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-ckpt',type=str)
    parser.add_argument('-save_num', type=str)
    parser.add_argument('-test_file',type=str)
    # parser.add_argument('testname', nargs='+')
    args = parser.parse_args()
    if args.test:
        setting = dict(
            LV=300,
            LS=20,
            max_epochs=args.epoch,
            backbone='regnety_002',
            d_v_h=256,
            d_s_h=128,
            d_s=300,
            d_v=2048,
            size=224,
            final_dim=64,
            dropout_rate=0.25,
            margin=0.2,
            name='circle-loss-reid',
            batch_size=32,
            num_workers=16,
            neg_mul=1,
            test_file=args.test_file,
            use_bn=args.use_bn,
            train_json=os.path.join(args.root, 'data/train-tracks.json'),
            test_v_json=os.path.join(args.root, 'data/test-tracks.json'),
            test_s_json=os.path.join(args.root, 'data/test-queries.json'),
            ckpt = args.ckpt,
            # img_root='/home/dataset_nl/AIC21_Track5_NL_Retrieval/',
            # train_file='/home/dataset_nl/AIC21_Track5_NL_Retrieval/train/train.pkl',
            # train_json='/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/train-tracks.json',
            # test_v_json='/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/test-tracks.json',
            # test_s_json='/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/test-queries.json',
            glove_file='data/6B.300d.npy',
            save_num=args.save_num)
        test(setting)
    else:
        setting = dict(
            LV = 300,
            LS = 20,
            max_epochs = args.epoch,
            backbone = 'regnety_002',
            d_v_h = 256,
            d_s_h = 128,
            d_s = 300,
            d_v = 2048,
            size = 224,
            final_dim = 64,
            dropout_rate = 0.25,
            margin = 0.2,
            name = 'circle-loss-reid',
            batch_size = 32,
            num_workers = 16,
            neg_mul = 1,
            use_bn = args.use_bn,
            train_file = args.train_file,
            train_json = os.path.join(args.root, 'data/train-tracks.json'),
            test_v_json = os.path.join(args.root, 'data/test-tracks.json'),
            test_s_json = os.path.join(args.root, 'data/test-queries.json'),
            # img_root='/home/dataset_nl/AIC21_Track5_NL_Retrieval/',
            # train_file='/home/dataset_nl/AIC21_Track5_NL_Retrieval/train/train.pkl',
            # train_json='/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/train-tracks.json',
            # test_v_json='/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/test-tracks.json',
            # test_s_json='/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/test-queries.json',
            glove_file = 'data/6B.300d.npy',)
        train(setting)
    #-train_file /home/dataset_nl/AIC21_Track5_NL_Retrieval/train/train.pkl -root /home/dataset_nl/AIC21_Track5_NL_Retrieval -use_bn -epoch 60
