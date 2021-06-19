---
title: "Hyper Parameter Optimization"
date: 2021-06-19T21:24:38+09:00
draft: false
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


