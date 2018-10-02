import os
import random
from shutil import copyfile

classes = []
# src_dir = "Fire_Classification_dataset/refined_dataset_multilabeled_0801"
# dst_dir = "Fire_Classification_dataset/refined_dataset_sepset"
src_dir = "../Dataset/BLDC/BLDC_color/raw"
dst_dir = "../Dataset/BLDC/BLDC_color"

train_ratio = 0.8
val_ratio = 0
test_ratio = 0.2

if (train_ratio + val_ratio + test_ratio != 1):
    print("set configuration is not collect!")
    exit()

for (dirpath, dirnames, filenames) in os.walk(src_dir):
    classes.extend(dirnames)
    break

if not os.path.isdir(dst_dir):
    os.mkdir(dst_dir)

setdict = [
    ["train", train_ratio],
    ["val", val_ratio],
    ["test", test_ratio]
]

setdict = dict(setdict)

for set in setdict.keys():
    set_dst_dir = dst_dir + "/" + set
    if not os.path.isdir(set_dst_dir):
        os.mkdir(set_dst_dir)

for classname in classes:

    files = []
    for dirpath, dirnames, filenames in os.walk(src_dir + "/" + classname):
        for filename in filenames:
            files.append(filename)

    setnum_dict = []
    for set in setdict.keys():
        # set별로 각 클래스 폴더 생성
        set_dst_dir = dst_dir + "/" + set
        dst_path = set_dst_dir + "/" + classname
        print(dst_path)
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)

        # set별로 데이터 개수 결정
        setnum_dict.append([set, int(len(files) * setdict[set])])

    # 랜덤 샘플링으로, 각 개수만큼 샘플링해서 복사
    setnum_dict = dict(setnum_dict)
    for set in setdict.keys():
        datas = random.sample(files, setnum_dict[set])
        for data in datas:
            copyfile(src_dir + "/" + classname + "/" + data, dst_dir + "/" + set + "/" + classname + "/" + data)
            files.remove(data)



