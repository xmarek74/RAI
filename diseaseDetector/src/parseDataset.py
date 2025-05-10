#imports
import os
import random
from pathlib import Path
import shutil
from tqdm import tqdm


#set up constants and variables
source = "../PlantVillage-Dataset/raw/color"
classes = os.listdir(source)
target = "../public"
dirs = ["train", "val"]
subdirs = ["healthy", "diseased"]
healthy = []
diseased = []

#create directory structure for training and validation
for dir in dirs:
    for subdir in subdirs:
        Path(f"{target}/{dir}/{subdir}").mkdir(parents=True, exist_ok=True)

#start sorting the images to their respective directories
#https://tqdm.github.io/docs/tqdm/#parameters
for dir in tqdm(classes, "Creating dataset, please wait...", ncols=60, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}", colour="green"):
    product, disease = dir.split("___")
    label = ""
    if (disease.strip() == "healthy"):
        label = "healthy"
    else:
        label = "diseased"
    #load files in dir
    images = os.listdir(os.path.join(source, dir))
    #shuffle images in dir to avoid bias
    random.shuffle(images)
    #split to dirs -> TRAIN/VAL = 8/2
    splittedFiles = {
        "train" : images[:int(len(images) * 0.8 )],
        #from 0.8 length of images to the end
        "val" : images[int(len(images) * 0.8 ):]
    }
    #each image has to be unique so they don't overload
    index = 0
    for split, files in splittedFiles.items():
        for name in files:
            src = os.path.join(os.path.join(source, dir), name)
            suffix = os.path.splitext(name)[1]
            finalName = f"{product}_{disease}_{index}_{suffix}"
            dst = os.path.join(target, split, label, finalName)
            shutil.copyfile(src, dst)
            index += 1
print("Dataset finalized at", Path(target).resolve())
