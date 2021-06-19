---
title: "Model Compression"
date: 2021-06-19T18:39:43+09:00
draft: true

tags: [
    "markdown",
    "machine learning"
]
---

## Hyper-parameter Optimization
###  HP categories to be optimized
1. Table of principal HPs grouped by role and domain

| hp-parameter | example | 
| ------------------ | ---------------------------| 
|continuous         | learning rate | 
|Discrete               |poolSize, kernal size,  sliding, hcHidden, data batch, numEpoch |  
|categorical         | Choice of optimizer, Activation function |  
| binary                   | whether to early stop | 

###   Existing Methods
#### summary
Table. 1: comparison of common HPO algorithm (n is the number of hyper-parameter values and k is the number of hyper-parameters)

| HPO methods| Strengths | Limitations | Time Complexity |
| -------- | -------- | -------------------------------------------------------------------- |------------------------|
| Grid Search     | Simple     | Time consuming; Only efficient with categorical HPs     |O(n^k)|
| Random Search | More efficient than GS| Not consider previous results; Not efficient with conditional HPs. | O(n) |
| Gradient-based models |Fast convergence speed for continuous HPs  | Only support continuous HPs; May only detect local optimums |  O(n^k)|
| BO-GP |Fast convergence speed for continuous HPs  | Poor capacity for parallelization; Not efficient with conditional HPs. |O(n^3)  |
| SMAC |Efficient with all types of HPs  |Poor capacity for parallelization  | O(nlog(n)) |
| BO-TPE |Efficient with all types of HPs;  Keep conditional dependencies.  | Poor capacity for parallelization| O(nlog(n)) |
| Hyperband | Enable parallelization | Not efficient with conditional HPs; Require subsets with small budgets to be representative |O(nlog(n))  |
| BOHB |Efficient with all types of HPs   | Require subsets with small budgets to be representative. | O(nlog(n)) |
| GA |Efficient with all types of HPs; Not require good initialization  | Poor capacity for parallelization. | O(n^2)|
| PSO | Efficient with all types of HPs; Enable parallelization| Require proper initialization | O (nlogn)  |

 Yang+, [On hyperparameter optimization of machine learning algorithms: Theory and practice](https://www.sciencedirect.com/science/article/abs/pii/S0925231220311693?casa_token=vCBDB7uhEc8AAAAA:LRUfJSQMmt_BE7qPpbxzXVDrCQDhapL5vkaQTGFgDp9DldPnfSPncqoMVBYkTV--rFDY1vqu7fo), Neurocomputing 2020

####  Grid search  
Grid Search performs an exhaustive search through a manually specified subset of the hyperparameter space defined in the searchspace file.
https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

 #### Random Search
 Random Search might be surprisingly effective despite its simplicity. We suggest using Random Search as a baseline when no knowledge about the prior distribution of hyper-parameters is available.
 Bergstra+, [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf), Journal of Machine Learning Research Vol.13, pp. 281-305, 2012

#### Gradient-based optimization 
 First randomly selecting a data point, then moves towards the opposite direction of the largest gradient to locate the next data point. Therefore, a local optimum can be reached after convergence. The local optimum is also the global optimum for convex functions.

#### Bayesian optimization
- summary: BO determines the future evaluation points based on the previously obtained results. To determine the next hyper-parameter configuration, BO uses two key components: a surrogate model and an acquisition function.
- procedure 
  - 1. Build a probabilistic surrogate model of the objective function. 
  - 2. Detect the optimal hyper-parameter values on the surrogate model.
  - 3. Apply these hyper-parameter values to the real objective function to evaluate them.
  - 4. Update the surrogate model with new results. 
  - 5. Repeat steps 2–4 until the maximum number of iterations is reached.

- BO-GP
https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
This is a sequential model-based optimization approach with Gaussian Process as the surrogate.

- SMAC
https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf
SMAC is based on Sequential Model-Based Optimization (SMBO). It adapts the most prominent previously used model class (Gaussian stochastic process models) and introduces the model class of random forests to SMBO in order to handle categorical parameters.
- BO-TPE (Tree-structured Parzen Estimator)
https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach. SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model. 
The TPE approach models P(x|y) and P(y) where x represents hyperparameters and y the associated evaluation matric. P(x|y) is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior with non-parametric densities.

- Hyperband
https://arxiv.org/pdf/1603.06560.pdf
Hyperband tries to use limited resources to explore as many configurations as possible and returns the most promising ones as a final result. 
The basic idea is to generate many configurations and run them for a small number of trials. The half least-promising configurations are thrown out, the remaining are further trained along with a selection of new configurations. The size of these populations is sensitive to resource constraints (e.g. allocated search time).

- BOHB(Bayesian Optimization Hyper Band)
https://arxiv.org/pdf/1807.01774.pdf
BO is an abbreviation for “Bayesian Optimization” and HB is an abbreviation for “Hyperband”.
BOHB relies on HB to determine how many configurations to evaluate with which budget, 
**but it replaces the random selection of configurations at the beginning of each HB iteration by a model-based search (Bayesian Optimization).** 
Once the desired number of configurations for the iteration is reached, the standard successive halving procedure is carried out using these configurations. We keep track of the performance of all function evaluations g(x, b) of configurations x on all budgets b to use as a basis for our models in later iterations.
HB part
Follow HyperBand's way of choosing the budgets and continue to use SuccessiveHalving.
<img width="528" alt="Screen Shot 2021-06-16 at 2.14.03.png (83.2 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/16/103347/c87f716d-4824-4b72-9651-b3999f6cbf13.png">
It is initialized with a set of configurations, a minimum and maximum budget, and a scaling parameter η. In the first stage all configurations are evaluated on the smallest budget (line3). 
The losses are then sorted and only the best 1/η configurations are kept in the set C (line 4). 
For the following stage, the budget is increased by a factor of η (line 5). 
This is repeated until the maximum budget for a single configuration is reached (line 2).
BO part
Tree Parzen Estimator(TPE): uses a KDE (kernel density estimator) to model the densities.
<img width="236" alt="Screen Shot 2021-06-16 at 2.58.08.png (20.8 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/16/103347/0cd1f35c-4c0e-4262-a298-39a06e0f6bb0.png">

Multidimensional KDE is used to guide the selection of configurations for the next iteration. The sampling procedure (using Multidimensional KDE to guide selection) is summarized by the pseudocode below.
 <img width="423" alt="Screen Shot 2021-06-16 at 2.45.58.png (121.9 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/16/103347/4b3ac567-d4ce-4ff8-ad5e-3c8a6104795c.png">
To fit useful KDEs, we require a minimum number of data points Nmin; this is set to d + 1 for our experiments, where d is the number of hyperparameters. To build a model as early as possible, we do not wait until Nb = |Db|, where the number of observations for budget b is large enough to satisfy q · Nb ≥ Nmin. Instead, after initializing with Nmin + 2 random configurations, we choose the best and worst configurations respectively to model the two densities.
<img width="375" alt="Screen Shot 2021-06-16 at 2.59.51.png (24.2 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/16/103347/6f7ac176-35a9-4863-b165-aaf4e2a72c4e.png">
Workflow
<img width="1280" alt="bohb_6.jpg (124.3 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/16/103347/34be100d-35df-4a82-8310-a858a7350a9d.jpg">
This image shows the workflow of BOHB. Here we set max_budget = 9, min_budget = 1, eta = 3, others as default. In this case, s_max = 2, so we will continuously run the {s=2, s=1, s=0, s=2, s=1, s=0, …} cycle. In each stage of SuccessiveHalving (the orange box), we will pick the top 1/eta configurations and run them again with more budget, repeating the SuccessiveHalving stage until the end of this iteration. At the same time, we collect the configurations, budgets and final metrics of each trial and use these to build a multidimensional KDE model with the key “budget”.

[opensourced code](https://github.com/automl/HpBandSter)

#### Metaheuristic algorithms
- GA (genetic algorithm)    
- PSO (Particle Swarm Optimization)

## Model Compression
### Introduction
<img width="1359" alt="Screen Shot 2021-06-15 at 15.04.00.png (1.2 MB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/15/103347/588bd8eb-63e4-49e1-bb86-341ce1542cb3.png">
Choudhary+, [A comprehensive survey on model compression and acceleration](https://link.springer.com/content/pdf/10.1007/s10462-020-09816-7.pdf), Artifcial Intelligence Review 2020

Table 2: summary of different methods for network compression.
| Method                | strengths                                       | limitations                                                                                       |
| --------------- | --------------------------------- | ------------------------------------------------------------- |
| Knowledge Distill  | Can downsize a network regardless of the structural difference between the teacher and the student network   | can only be applied  to classification tasks with softmax loss function |
| Low-Rank Factorization      | standard pipepline    | performed layer by layer, cannot perform global parameter compression     |
|Data quantization | significantly reduced memory usage and float-point operations   | Quantized weights make neural networks harder to converge    |
| Pruning    | Can improve the inference time and model size vs accuracy tradeoff for a given architecture | Generally, does not help as much as switching to a better architecture |  


Cheng+, [Model Compression and Acceleration for Deep Neural Networks: The Principles, 
Progress, and Challenges](https://ieeexplore.ieee.org/document/8253600), Vol.35, pp.126-136, 2018



### Low-Rank Factorization
In low-rank factorization, a weight matrix A with m × n dimension and having rank r is replaced by smaller dimension matrices. 
In feed-forward NN and CNN, singular value decomposition (SVD) is a common and popular factorization scheme for reducing the number of parameters. 
SVD factorize the original weight matrix into three smaller matrices, replacing the original weight matrix. For any matrix A ∈ ℝm×n , there exists a factorization, A = U S V^(T) . Where, U ∈ ℝ^(m×r) , S ∈ ℝ^(r×r) , and V^T ∈ ℝ^(r×n) . S is a diagonal matrix with the singular values on the diagonal, U and V are orthogonal matrices. 
Each entry in the S is larger than the next diagonal entry. When reducing the model size is necessary, low- rank factorization techniques help by factorizing a large matrix into smaller matrices. 
Yu+, [On Compressing Deep Models by Low Rank and Sparse Decomposition](https://ieeexplore.ieee.org/document/8099498)

### Knowledge Distillation 
Knowledge Distilled: The basic idea of KD is to distill knowledge from a large teacher model into a small one by learning the class distributions output by the teacher via softened softmax.

Mishra+, [Apprentice: using knowledge distillation techniques to improve low-precision network accuracy](https://openreview.net/forum?id=B1ae1lZRb&noteId=B1e0XDKXf),  ICLR2018

### data quantization

In quantization, we represent weights by reducing the number of bits required per weight to store each weight. This idea can also be further extended to represent gradient and activation in the quantized form. The weights can be quantized to 16-bit, 8-bit, 4-bit or even with 1-bit (which is a particular case of quantization, in which weights are represented with binary values only, known as weight binarization).

our model is implemented by scala, the  parameter is  in[ double](https://www.tutorialspoint.com/scala/scala_data_types.htm): 64 bit, float: 32 bit, integer: 32 bit.

- QAT Quantizer
 [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf), CVPR2018

  - Weights are quantized before they are convolved with the input. If batch normalization is used for the layer, the batch normalization parameters are “folded into” the weights before quantization.
Activations are quantized at points where they would be during inference, e.g. after the activation function is applied to a convolutional or fully connected layer’s output, or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.

- BNN Quantizer
[Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)

### pruning
pruning:  remove parameters which have least effect on the accuracy of the network, which can reduce model complexity and mitigate the over-ﬁtting issue.

- Weight pruning: In unimportant weight connection pruning, we prunes (zeros out) the
weight connections if they are below some predefned threshold (Han et al. 2015) or if
they are redundant.
Han+, [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626), NIPS 2015

- Neuron pruning: Instead of removing the weights one by one, which is a timeconsuming process, we can also remove the individual neurons if they are redundant
(Srinivas and Babu 2015). In that case, all the incoming and outgoing connection of the
neuron will also be removed. There are many other ways of removing individual weight
connections or neurons.
Srinivas+, [Data-free parameter pruning for Deep Neural Networks](https://arxiv.org/abs/1507.06149), BMVC 2015

- Filter pruning: In flter pruning, flters are ranked according to their importance, and the
least important (least ranking) flters are removed from the network. The importance of
the flters can be calculated by L1/L2 norm (Li et al. 2017) or some other methods like
their infuence on the error.
Li+, [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710), ICLR 2017
- Layer pruning: Similarly, from a very deep network, some of the layers can also be
pruned (Chen and Zhao 2018).
Chen+, [Shallowing Deep Networks: Layer-Wise Pruning Based on Feature Representations](https://ieeexplore.ieee.org/document/8485719), IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 41, Iss. 12, pp. 3048 - 3056, 2019

### criteria
The stsndard criteria to measure the quality of model compression and acceleration are the compression rate and the speedup rate.
<img width="1217" alt="Screen Shot 2021-06-15 at 15.27.31.png (350.7 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/15/103347/2b991ac8-6664-4b99-952c-386cb8437cb1.png">

table source:  Choudhary+, [A comprehensive survey on model compression and acceleration](https://link.springer.com/content/pdf/10.1007/s10462-020-09816-7.pdf), Artifcial Intelligence Review 2020

## Relation to current baseline model 

### Current Model

| Model | Configuration | average Accuracy on same dataset|
| -------- | -------- | -------- |
| Tsukijinet |   4 layer CNN      |   78.06%   |
| Tukijinet2021（reproduced model） |4 layer CNN, with batch normalization  |78.76%  |

- original tsukijinet
<img width="480" alt="conf_2di-CNN_eng.png (79.0 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/15/103347/8f5441cb-3e3b-473a-9dba-b9c204435458.png">

```
fs:/records/documents/researches/2016/master201701/tsukiji/submission/m2thesis_tsukiji_submission.pdf
```
- tusikijinet2021
<img width="502.08" alt="tsukijinet2021.png (143.4 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/06/16/103347/f9676819-b5a6-4c0c-af07-545492a6c48d.png">

### Proposed Methods
- Rough Ideas:
1. to improve accuracy -> HPO process -> latency / energy cost increased ->  Model Compression


- HPO process
   * HPO  -> improve accuracy -> Model Compression (considered lantency, power cost) 
   * HPO: treat random search as a base method, apply Baysian Optimization(HPBO)
- Model Compression: 
  - quantization
  - pruning

- Metrics
  - Accuracy
  - Latency
  - power consumption
  - number of parameters

## discussion
- hyper-parameter optimization
    - please post hyperband paper at IoT
    - other category
        - differentiable model search
        - gradient based method
- model compression
    - pruning
        - drop out in the inference part
    - data quantization
        - discussed in NeurIPS at last year
- mxnet implementation does not employ the sparsification
- necessity of batch normalization
- 78% is proper?
    - 81% is achived 
    - depends on the window size
- current first strategy
    - hyper-parameter optimization for performance improvement -> model compression
        - sequential approach
            - acceptable way for first trial on our lab.
- **continue to invesitgate the hyper parameter search with gradient-based method**
- metrics
    - the number of parameter
    - actual calcuration time
    - power consumption
- create the sketch of first trial
    - simplest way is prefered

