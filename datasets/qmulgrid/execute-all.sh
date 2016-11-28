# this script calls the other scripts which do data rescaling, folder creation,
# random 125 - 125 train and test splits, additional gallery

# rescaling the images
echo 'Rescaling the images to 160x60 '
th rescaling.lua

# Folder creation based on person identities
echo 'Folder creation based on person identities '
matlab14a  -r "run('foldering.m'); exit;"

# random 125 - 125 train and test split
echo 'Random 125 - 125 train and test split '
matlab14a  -r "run('randomTrainAndTestSplit.m'); exit;"

# augment the training data
echo 'Augment the training data '
th dataAugmentation.lua
