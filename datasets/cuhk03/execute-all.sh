# this script calls the other scripts which do data extraction, rescaling,
# random 1260 - 100 train and test splits, data augmentation

# data extraction
echo 'extracting the data from .mat file '
matlab14a  -r "run('preprocessing.m'); exit;"

# rescaling the images
echo 'Rescaling the images to 160x60 '
th rescaling.lua

# random 1260 - 100 train and test split
echo 'Random 1260 - 100 train and test split '
matlab14a  -r "run('randomSplit_1260_100.m'); exit;"

# augment the training data
echo 'Augment the training data '
th dataAugmentation.lua
