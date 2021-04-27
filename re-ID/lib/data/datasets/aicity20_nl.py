# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET


from .bases import BaseImageDataset


class AICity20_nl(BaseImageDataset):
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
        super(AICity20_nl, self).__init__()
        dataset_dir = '/home/dataset_nl/AIC21_Track5_NL_Retrieval/train/S01/c001/img1'
        query = self._process_dir(dataset_dir)
        if verbose:
            print("=> AI CITY 2020 data loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train = []
        self.query = query
        self.gallery = []

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, img_dir):
        dataset=[]
        img_paths = glob.glob(osp.join(img_dir, '*.jpg'))
        for img_path in img_paths:
            dataset.append((img_path, -1, -1))
        return dataset

if __name__ == '__main__':
    dataset = AICity20(root='AIC/AIC21_Track2_ReID')
