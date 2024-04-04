import cv2
import gzip
import os
import numpy as np


def load_gzip(data_folder, data_name):
    assert 'gz' in data_name and ('labels' in data_name or 'images' in data_name)
    # rb : binary, unzip data to binary
    with gzip.open(os.path.join(data_folder, data_name), 'rb') as file:
        # why offset 8 for labels and 16 for images ?
        datas = np.frombuffer(file.read(), np.uint8, offset=8 if 'labels' in data_name else 16)
    return datas


def cv_show(mat):
    cv2.imshow('test', mat=mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
