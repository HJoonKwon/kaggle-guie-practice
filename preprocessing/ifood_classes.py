from collections import OrderedDict
import os

def get_ifood_2019_class_dict(data_root_dir: str) -> OrderedDict:
    IFOOD_2019_CLASS_DICT = OrderedDict()
    with open(os.path.join(data_root_dir, "class_list.txt")) as f:
        for line in f.readlines():
            line = line.strip()
            k = int(line.split(" ")[0])
            v = str(line.split(" ")[1])
            IFOOD_2019_CLASS_DICT[k] = v

    return IFOOD_2019_CLASS_DICT