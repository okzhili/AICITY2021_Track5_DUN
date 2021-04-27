### Prepare Data
- Download CityFlowV2-ReID and VehicleX
, rename them to 'AIC21_Track2_ReID' and 'AIC21_Track2_ReID_Simulation' respectively.
- Weakly Supervised crop Augmentation. Crop vehicle in image via weakly supervised method, 
a vehicle ReID pretrain model is needed to generate attention map. If you dont want to train

````
# first temporary comment aicity20.py line 49 # train += self._process_dir(self.train_aug_dir, self.list_train_path, self.train_label_path, relabel=False)
# to make sure only original data is used
# step 1: train inital model
python tools/train.py --config_file='configs/aicity21.yml' \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('resnet50_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
OUTPUT_DIR "('xx')"

# step2: use inital model to crop vehicles
python tools/aicity20/weakly_supervised_crop_aug.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.MODEL_TYPE "baseline" \
DATASETS.TRAIN "('aicity20',)" \
DATASETS.TEST "('aicity20',)" \
DATALOADER.SAMPLER 'softmax' \
DATASETS.ROOT_DIR "('/home/zxy/data/ReID/vehicle')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('xx/final.pth')"

# AIC20_ReID_cropped will be saved at './output/aicity20/0326-search/augmix/'
# dont forget to uncomment aicity20.py line 49 # train += self._process_dir(self.train_aug_dir, self.list_train_path, self.train_label_path, relabel=False)

````
- after all works have be done, data folder should look like
````
--AIC21_Track2_ReID
--AIC21_Track2_ReID_Simulation
--AIC_crop
````

## Download pretrain model
We use [ResNet-ibn](https://github.com/XingangPan/IBN-Net) as backbone.
Download ImageNet pretrain model at [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) 


## Train
```
python tools\train.py --config_file='configs/aicity21.yml'
```
## get_feature
use get_frame.py get the frames of video
```
python get_vision_feature --config_file='configs/aicity21.yml' -track_file AIC21_Track5_NL_Retrieval/data/train-tracks.json -save_file train.pkl -root xx
```
