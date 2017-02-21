[![DOI](https://zenodo.org/badge/69573427.svg)](https://zenodo.org/badge/latestdoi/69573427)
# Person Re-Identification with Normalized correlation matching layer

Deep neural network model introducing new novel matching layer called **'Normalized correlation'** layer. This repository contain information about the datasets used, implementation code. The paper titled **"Deep Neural Networks with Inexact Matching for Person Re-Identification"** is accepted in NIPS-2016. You can find the paper [here](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf)

##Datasets
|Dataset name | Description|
|-------------|------------|
|CUHK03 (Labeled & Detected) |A collection of 13,164 images of 1360 people captured from 6 different surveillance cameras, with each person observed by 2 cameras with disjoint views. <br><br> The dataset comes with manual (**"Labeled" dataset**) and algorithmically (**"Detected" dataset**) labeled pedestrian bounding boxes. The dataset can be obtained from [here](http://www.ee.cuhk.edu.hk/~rzhao/)|
|CUHK01|A mid-sized dataset with 9884 images of 971 identities (4 images per identitiy). The dataset can be obtained from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)|
|QMULGRID | A small sized dataset with 250 identities observed from 2 cameras (2 images per identity). There are 775 unmatched identities (1 image per identity) to be included as part of Gallery images during test run. Dataset can be downloaded from [here](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html)|

More dataset links can be found [here](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html) 

##Software prerequisites
The code development has been done in the environment as mentioned below:

#####Operating system
Ubuntu 14.04 LTS

#####Software packages used

1. Torch (with packages (nn, cunn, cutorch, image) installed by default, as well as some special packages such as matio)
2. Matlab (R2014a version) - for some data preprocessing scripts

#####GPUs & drivers used
NVIDIA-SMI 352.41     Driver Version: 352.41  with GeForce GTX TITAN GPUs

##Preprocessing steps
The data-preprocessing scripts are written mainly using Lua (torch), Matlab.
###CUHK03 (labeled & detected)

`cd` to the folder `datasets/cuhk03`

Follow the steps below:

1. Download the CUHK03 dataset after filling out a google form available in [Rui Zhao's Homepage](http://www.ee.cuhk.edu.hk/~rzhao/)
2. Place the file (cuhk-03.mat) in the folder ./datasets/cuhk03
3. Execute the script "execute-all.sh"

If everything goes well, there will be 4 folders (**"labeled", "labeled_testsets", "detected", "detected_testsets"**) available in "datasets/cuhk03" folder.

###CUHK01 Test100

`cd` to the folder `datasets/cuhk01_test100`

Follow the steps below:

1. Refer to [this page](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) to get the CAMPUS person re-id dataset
2. Extract the zip file (CAMPUS.zip) in the folder ./datasets/cuhk01_test100 (now, this folder should contain a folder named 'campus')
3. Execute the script "execute-all.sh"

If everything goes well, there will be 2 folders (**"train", "test"**) available in "datasets/cuhk01_test100" folder.

###CUHK01 Test486

`cd` to the folder `datasets/cuhk01_test486`

Follow the steps below:

1. Refer to [this page](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) to get the CAMPUS person re-id dataset
2. Extract the zip file (CAMPUS.zip) in the folder ./datasets/cuhk01_test486 (now, this folder should contain a folder named 'campus')
3. Execute the script "execute-all.sh"

If everything goes well, there will be 2 folders (**"train", "test"**) available in "datasets/cuhk01_test486" folder.

###QMULGRID

`cd` to the folder `datasets/qmulgrid`

Follow the steps below:

1. Download the QMULGRID dataset from [here](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html)
2. Extract the zip file (underground_reid.zip) in the folder ./datasets/qmulgrid (now, this folder should contain a folder named 'underground_reid')
3. Execute the script "execute-all.sh"

If everything goes well, there will be 3 folders (**"train", "test", "additionalgallery"**) available in "datasets/qmulgrid" folder.

##Options

###For Training

The options used during training are available in the file "src/opts.lua"

####1. Change the dataset for training

```lua
opt.dataset = 'cuhk03'; -- cuhk03  |  others
opt.dataType = 'detected' -- labeled | detected | ''
--------------
opt.datasetname = 'cuhk03' -- cuhk03 | cuhk01_test100 | cuhk01_test486 | qmulgrid
```
The options `opt.dataset` and `opt.dataType` should be changed carefully according to the dataset. 
#####for CUHK03
```lua
opt.dataset = 'cuhk03'; -- cuhk03
opt.dataType = 'detected' -- labeled | detected
--------------
opt.datasetname = 'cuhk03'
```
#####for CUHK01 (test-100 & test-486), QMULGRID
```lua
opt.dataset = 'others';
opt.dataType = ''
--------------
opt.datasetname = 'cuhk01_test100' -- cuhk01_test100 | cuhk01_test486 | qmulgrid
```

####2. Change the model type
```lua
opt.modelType = 'normxcorr' -- normxcorr | cin+normxcorr
```
####3. Change the Number of GPUs for training
```lua
opt.GPU = 1  -- default GPU to hold the original copy of model
opt.nGPUs = 3 -- the total number of GPUs to be used during training 
```

_The other options should be left as it is (Reason: they are for future use / error handling is not proper for them)_. 

###For Testing
Use the file 'doallTest.lua' for testing the model for any particular trained model.

All you have to do is, to change the options in the file 'doallTest.lua'. Note that, 'doallTest.lua' file does not depend on 'opts.lua'.

####Dataset options

```lua
opt.dataset = 'cuhk03'; -- cuhk03  |  others
opt.datasetname = 'cuhk03'  -- cuhk03 | cuhk01_test100  | cuhk01_test486   | qmulgrid
opt.datapath = '../datasets/' .. opt.datasetname .. '/' 
opt.dataType = 'detected' 
```
The meaning of these options is as same as the training options mentioned above. As per these options, the data will be read from appropriate 
dataset directory, tests will be carried out for 10 times and average CMC (Cumulative Matching Characteristics) will be calculated at the end of the tests.

####Model path specification

```lua
MODEL_PATH = '<absolute/relative path>' -- exact path of the model
```
The test log will be created in the same path of the model.

_The other options should be left as it is (Reason: they are for future use / error handling is not proper for them)_. 

##Execution
To execute a particular file (`doall.lua`,`doallTest.lua`), first `cd` to the folder `./src` folder.

####Training

After making sure that the [preprocessing](https://github.com/InnovArul/personreid_normxcorr#preprocessing-steps) of the data is done correctly and setting all the [options](https://github.com/InnovArul/personreid_normxcorr#for-training) accordingly, execute the file `doall.lua` as,

```sh
th doall.lua
```
The trained model(s) will be stored (after every epoch) in the folder './scratch'. 

####Testing

After setting the [options](https://github.com/InnovArul/personreid_normxcorr#for-testing) for test in 'doallTest.lua' file, execute it as,

```sh
th doallTest.lua
```

##Important Code files

|File path | purpose |
|----------|---------|
|src/doall.lua | the main file to be executed for training|
|src/doallTest.lua | the test execution file to get CMC (Cumulative Matching Characteristics) percentages for a particular model and dataset)|
|src/model_\<xxxx>.lua | defines a particular type of model|
|src/trainMultiGPU.lua | training subroutine using Multi GPUs|
|src/dataForTests.lua | data reading procedures for 'test'; reads the data depending on the options set in `doallTest.lua`|
|src/data.lua  | data reading procedures for training and validation; reads the data depending on the options set in `opts.lua`|
|src/log.lua|logging helper functions. copied from [here](https://github.com/rxi/log.lua)|
|src/rankScores.lua|to rank the gallery images based on Softmax scores|
|src/doallTest.lua| code to carry out test for a particular trained model and dataset |
|src/doall.lua| code to carry out training for a particular dataset and type of model sepcified in `opts.lua`|
|src/loss.lua| defines loss/objective/criterion function (negative log likelihood - NLL) used during training |
|src/opts.lua|contains options for training|
|src/test.lua | testing subroutine to test for probe images and rank the gallery based on softmax classifier scores (applicable for both validation and testing)|
|src/utilities.lua|contains miscellaneous multi-purpose utility functions that are used in other code files|
|src/modules/* | contains code files for Normalized correlation Matching layer (`NormCrossMapCorrelation.lua`), Ahmed at al.'s Cross Input Neighborhood layer (`CrossInputNeighborhood.lua`), Parallel Multi GPU model training package (`DataParallelTableForSiamese.lua`) inspired and adapted from Facebook's torch [`DataParallelTable.lua`] (https://github.com/torch/cunn/blob/master/DataParallelTable.lua) and CUDA C++ implementation of modules|
-----------------------

###Citation 

```
@inproceedings{subramaniam2016deep,
  title={Deep Neural Networks with Inexact Matching for Person Re-Identification},
  author={Subramaniam, Arulkumar and Chatterjee, Moitreya and Mittal, Anurag},
  booktitle={Advances In Neural Information Processing Systems},
  pages={2667--2675},
  year={2016}
}
```

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-86890539-2', 'auto');
  ga('send', 'pageview');

</script>
