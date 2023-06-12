import os
from src.data.data_preparation import get_data

path = "/home/hesaam/Hesaam/Data/physionet.org/files/"
save_path = "data/twoClasses"
type_run = 'execution'
save_path = os.path.join(save_path, type_run)
get_data(path, save_path=save_path, long = False, normalization = 0,subjects_list=range(1,106), n_classes=2, type_run=type_run)