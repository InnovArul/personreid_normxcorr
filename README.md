# personreid_normxcorr
## repository is under construction and will be available soon
### contact Arulkumar (arul.csecit@ymail.com) for further details
Deep neural network model introducing new novel matching layer called 'Normalized correlation' layer. This repository contain information about the datasets used, implementation code. The paper titled "Deep Neural Networks with Inexact Matching for Person Re-Identification" is accepted in NIPS-2016. You can find the paper [here](http://papers.nips.cc/paper/6367-deep-neural-networks-with-inexact-matching-for-person-re-identification.pdf)

##Datasets
|Dataset name | Description|
|-------------|------------|
|CUHK03 (Labeled & Detected) |A collection of 13,164 images of 1360 people captured from 6 different surveillance cameras, with each person observed by 2 cameras with disjoint views. <br><br> The dataset comes with manual (**"Labeled" dataset**) and algorithmically (**"Detected" dataset**) labeled pedestrian bounding boxes. The dataset can be obtained from [here](http://www.ee.cuhk.edu.hk/~rzhao/)|
|CUHK01|A mid-sized dataset with 9884 images of 971 identities (4 images per identitiy). The dataset can be obtained from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)|
|QMULGRID | A small sized dataset with 250 identities observed from 2 cameras (2 images per identity). There are 775 unmatched identities (1 image per identity) to be included as part of Gallery images during test run. Dataset can be downloaded from [here](http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html)|

More dataset links can be found [here](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html) 

##Code files

|File path | purpose |
|----------|---------|
|src/doall.lua | the main file to be executed for training|
|src/doallTest.lua | the test execution file to get CMC (Cumulative Matching Characteristics) percentages for a particular model and dataset)|
|src/model_xxxx.lua | defines a particular type of model|
|src/train.lua | training subroutine|
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
