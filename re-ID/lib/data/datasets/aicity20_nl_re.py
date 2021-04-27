# encoding: utf-8

import glob
import json
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET


from .bases import BaseImageDataset


class AICity20_nl_re(BaseImageDataset):
    """
      ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   333 |    36935 |        36
  query    |   333 |     1052 |        ?
  gallery  |   333 |    18290 |        ?
  ----------------------------------------

    """

    def __init__(self, root='', verbose=True, **kwargs):
        super(AICity20_nl_re, self).__init__()
        if verbose:
            print("=> AI CITY 2020 data loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train,self.query,self.gallery = self._process_dir()

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self):
        dataset=[]
        id=0
        f = open('/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/test-tracks.json')
        t = json.load(f)
        for key in t.keys():
            for frame in t[key]['frames']:
                dataset.append(('/home/dataset_nl/AIC21_Track5_NL_Retrieval/'+frame,id,-1))
            id += 1
        f = open('/home/dataset_nl/AIC21_Track5_NL_Retrieval/data/train-tracks.json')
        t = json.load(f)
        for key in t.keys():
            for frame in t[key]['frames']:
                dataset.append(('/home/dataset_nl/AIC21_Track5_NL_Retrieval/' + frame, id, -1))
            id += 1
        # img_paths = glob.glob(osp.join(img_dir, '*.jpg'))
        # for img_path in img_paths:
        #     dataset.append((img_path, -1, -1))
        return dataset,[],[]

if __name__ == '__main__':
    dataset = AICity20(root='AIC/AIC21_Track2_ReID')
