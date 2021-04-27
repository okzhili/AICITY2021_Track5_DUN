import numpy as np
import sys
sys.path.append('.')
from lib.data.datasets.aicity20 import AICity20
from tools.aicity20.submit import write_result_with_track


if __name__ == '__main__':
    dataset = AICity20('../../AIC/AIC21_Track2_ReID')
    distmat_path = ['../../data/ens/1/distmat.npy',
                    '../../data/ens/2/distmat.npy',
                    '../../data/324resnext101416/distmat.npy',
                    '../../data/324_resnext101320/distmat.npy',
                    '../../data/323_resnext320/distmat.npy',
                    '../../data/323_resnext416/distmat.npy',
                    '../../data/330_senet154320/distmat.npy'
                    ]
    #cam_distmat = np.load('./output/aicity20/0407-ReCamID/distmat_submit.npy')
    #ori_distmat = np.load('./output/aicity20/0409-ensemble/ReTypeID/distmat_submit.npy')
    distmat = []
    for path in distmat_path:
        distmat.append(np.load(path))
    distmat = sum(distmat) / len(distmat)
    #distmat = distmat - 0.1 * cam_distmat - 0.1 * ori_distmat

    indices = np.argsort(distmat, axis=1)
    write_result_with_track(indices, '../../data/ens', dataset.test_tracks,name='326')

