import numpy as np
import cv2
import torch.multiprocessing as mp
mp.set_start_method('spawn')
import h5py

f = h5py.File('../../train_b8_LD37.h5','r')

for key in f.keys():
    print(f[key].name,f[key].shape)
    # for i in range(1,100):
    # cv2.imshow('1.jpg',f[key][i,0,...])
    # cv2.waitKey(0)
