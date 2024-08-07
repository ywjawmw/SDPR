# SDPR: Prescription Recommendation with Syndrome Differentiation
This is our Pytorch implementation for the paper:

## Introduction
We propose a new four-partite graph modeling paradigm that enables prescription recommendation models to represent the four steps of SD. Based on this paradigm, we design a prescription recommendation model, SDPR, which includes four modules corresponding to the steps of SD. We introduce a graph neural network-based entity representation module to enhance symptom and herb representation. To address the challenges of modeling syndrome and therapeutic method information, we propose a symptom set aggregator and an herb set aggregator. Our pre-training strategy for inducing syndrome and the therapeutic method-aware contrastive learning framework can model the complex relationships among symptoms, syndromes, therapeutic methods, and herbs. Finally, we integrate these modules into a multi-task learning framework to complete the SD analysis. Experiments on a public prescription recommendation dataset demonstrate that our model accurately recommends herbs and retrieves existing prescriptions, showing that the four-partite graph paradigm enhances precise herb prescribing.

## Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.10.0
* numpy == 1.17.4
* scipy == 1.21.6
* temsorboardX == 2.0

## Usage
The hyperparameter search range and optimal settings have been clearly stated in the codes (see the 'utils' dict in parser.py).
* Train

```
bash CLEPR_train.sh 
```

* Pretrain Train File

\weights-CLEPR\Herb\CLEPR_norm_CLEPR_l2\64-128\32\date_2022-09-18_60_ori_emb_seed1234\model.pkl
  
Retraining: 
```
bash CLEPR_train.sh
```
the parameters should be notice:
* --attention 0
* --use_S1 0
* --alg_type  CLEPR(S1)
* --batch_size 1024

* Test
```
bash CLEPR_test.sh 
```

Some important hyperparameters:
* `lrs`
  * It indicates the learning rates. 
  * The learning rate is default as 2e-5.

* `mess_dropouts`
  * It indicates the message dropout ratio, which randomly drops out the outgoing messages. 
  * The message dropout is default as '[0.0,0.0]'.

* `steps`
  * It indicates the length of the hard-negative sample.
  * We search it in {2, 4, 5, 6, 8, 10, 16, 24}.
  
* `max_step_lens`
  * It indicates the search length of the hard-negative sample.
  * We search it in {16, 24, 25, 26, 27, 28, 29}.
  
* `ts`
  * temperature parameter.
  * We search it in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.

* `hard_neg`
  * It indicates that whether utilizing the hard-negative sample.
  
* `use_S1`
  * It indicates that whether using the first stage to pretrain.
  

## Dataset
We provide one public TCM dataset: TCM. \data\Herb
* `train_id.txt`
  * Train file.
  * Each line is 'symptom ID1 symptom ID2 ... \t herb ID1 herb ID2 ...\n'.
  * Every observed interaction means symptom s once interacted herb h, symptom set sc once interacted prescription p
  
* `valid_id.txt`
  * Validation file.
  * Each line is 'symptom ID1 symptom ID2 ...\t herb ID1 herb ID2 ...\n'.

* `test_id.txt`
  * Test file.
  * Each line is 'symptom ID1 symptom ID2 ...\t herb ID1 herb ID2 ...\n'.
* `x.npz`
 * entity interaction matrix
