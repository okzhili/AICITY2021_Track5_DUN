# encoding: utf-8

import glob
import re
import os
import os.path as osp
import xml.etree.ElementTree as ET


from .bases import BaseImageDataset


class AICity20(BaseImageDataset):
    """
      ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   333 |    36935 |        36
  query    |   333 |     1052 |        ?
  gallery  |   333 |    18290 |        ?
  ----------------------------------------

    """
    dataset_dir = 'AIC21_Track2_ReID'
    dataset_aug_dir = 'AIC_crop'
    def __init__(self, root='', verbose=True, **kwargs):
        super(AICity20, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_aug_dir = osp.join(root, self.dataset_aug_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_aug_dir = osp.join(self.dataset_aug_dir, 'image_train')

        self.list_train_path = osp.join(self.dataset_dir, 'name_train.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'name_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'name_test.txt')

        self.train_label_path = osp.join(self.dataset_dir, 'train_label.xml')


        self._check_before_run()

        train = self._process_dir(self.train_dir, self.list_train_path, self.train_label_path, relabel=False)
        query = self._process_dir(self.query_dir, self.list_query_path, None)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path, None)

        train += self._process_dir(self.train_aug_dir, self.list_train_path, self.train_label_path, relabel=False)
        # train+=self._process_dir_dg(self.dataset_dg)
        train = self.relabel(train)
        if verbose:
            print("=> AI CITY 2020 data loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.train_tracks = self._read_tracks(os.path.join(self.dataset_dir, 'train_track.txt'))
        self.test_tracks = self._read_tracks(os.path.join(self.dataset_dir, 'test_track.txt'))

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir_dg(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid= int(img_path.split('/')[-2])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid= int(img_path.split('/')[-2])
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid+1000, -1))

        return dataset

    def _process_dir(self, img_dir, list_path, label_path, relabel=False,double=True):
        dataset = []
        if label_path:
            tree = ET.parse(label_path, parser=ET.XMLParser(encoding='utf-8'))
            objs = tree.find('Items')
            for obj in objs:
                image_name = obj.attrib['imageName']
                img_path = osp.join(img_dir, image_name)
                pid = int(obj.attrib['vehicleID'])
                camid = int(obj.attrib['cameraID'][1:])
                dataset.append((img_path, pid, camid))
                # if double:
                #     dataset.append((img_path, pid, camid))
                #dataset.append((img_path, camid, pid))
            if relabel: dataset = self.relabel(dataset)
        else:
            with open(list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    img_path = osp.join(img_dir, line)
                    pid = 0
                    camid = 0
                    dataset.append((img_path, pid, camid))
                    # if double:
                    #     dataset.append((img_path, pid, camid))
        return dataset

if __name__ == '__main__':
    dataset = AICity20(root='AIC/AIC21_Track2_ReID')
