---
title: "Model Compression"
date: 2021-06-19T18:39:43+09:00
draft: false

tags: [
    "markdown",
    "machine learning"
]
---


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



