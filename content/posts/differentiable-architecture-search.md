---
title: "Differentiable Architecture Search"
date: 2021-07-11T00:44:32+09:00
draft: false
tags: [
    "markdown",
    "machine learning"
]
---
## NAS: General Problem Setup
### Overview
<img width="936" alt="Screen Shot 2021-07-06 at 17.59.43.png (86.0 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/06/103347/e1161571-e2ea-4ebc-8890-258efeff760f.png">

- source: Elksen et al., [Neural Architecture Search: A survey](https://arxiv.org/pdf/1808.05377.pdf), 2018 

- Search Space: Defines which architecture can be represented in principle, 
- Search Strategy: Detail on how to explore search space
  - often exponentially large or even unbounded  
- Performance Estimation Strategy: estimating an architecture's performance 
  - standard training and validation of architecture on data may be computionally expensive and limits number of architectures that can be explored
 


### Cell based Search Space:

-  the NASNet search space (Zoph et al. 2018) defines the architecture of a conv net as the same cell getting repeated multiple times and each cell contains several operations predicted by the NAS algorithm. 
- a small directed acyclic graph representing a feature transformation
-   NASNet search space: Learns 2 types of cells:
    - Normal Cell: Input and output feature maos have same dimension
    - Reduction Cell: Output feature map has width the height reduced by half 
   
  <img width="464" alt="Screen Shot 2021-07-07 at 3.17.55.png (103.1 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/07/103347/9347b543-2f0b-4f1b-ab5d-f3dd88710791.png">

- Zoph et al., [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf), CVPR 2018
#### pros 
- A well-designed cell module enables transferability between datasets. 
-  easy to scale down or up the model size by adjusting the number of cell repeats.

#### cons
- arrangement of operation is restricted
- Suppose CNN can be obtained by stacking the same Cell, and RNN can be obtained by recursive connection of the same Cell


## Comparision of different methods

<img width="776" alt="Screen Shot 2021-07-07 at 2.21.50.png (184.1 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/07/103347/b1f1172b-573d-4b8c-9299-347f5f04b0d3.png">

- Comparison with state-of-the-art architectures on ImageNet (mobile setting)

Table source: Chen +,[ Progressive Differentiable Architecture Search:Bridging the Depth Gap between Search and Evaluation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Progressive_Differentiable_Architecture_Search_Bridging_the_Depth_Gap_Between_Search_ICCV_2019_paper.pdf), ICCV 2019 

|  | Reinforcement Learning | Evolution Algorithm | Differentiable Search |
| --------------| -------- | -------- | -------- |
|Computation cost | High    | High    | Low    |
|Search space | Large     | Large    | Restricted    |


## Differentiable Architecture Search: Gradient-based method
Liu et al., [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf), ICLR 2019

### Overview of DARTS
<img width="773" alt="Screen Shot 2021-07-06 at 19.36.03.png (250.4 kB)"  src="https://img.esa.io/uploads/production/attachments/14973/2021/07/06/103347/2e333080-419f-43c9-8030-bb020b07d183.png">

- (a) Operations on the edges are initially unknown
- (b)  Continuous relaxation of the search space by placing a mixture of candidate opeartions on each edge.
- (c) Joint optimization of the mixing probabilities and the network weights by solving a bilevel optimization problem.
- (d) Inducing the final architecture  from the learned mixing probabilities

### Continuous relaxation and optimization
#### Relaxation
- The operation mixing weights for a pair of nodes $(i, j)$ are parameterized by a vector  in $\alpha^{(i, j)}$ of dimension $ |\mathcal O|$
- The method relaxes categorical chioce of a particular operation as a softmax over all operations

$$  \overline{o} ^{(i,j)}(x) = \sum_{o\in \mathcal O}\frac{exp(\alpha_{o}^{ij})} { \sum_{o^{'} \in \mathcal O} exp(\alpha_{o^{'} } ^{(i,j)} ) }o(x)$$

- This task of architecture search then reduces  to learning a set of continuous variables  $\alpha = \lbrace\alpha^{(i, j)} \rbrace$
- At the end of search, obatain discrete architecture  by replacing each mixed operation $  { \overline{o} }^{(i,j)} $ with the most likely operation, i.e. $o^{(i,j)} = \underset{o \in \mathcal O}{argmax} \ \alpha_{o}^{(i, j)} $ 

#### Optimization
- After relaxation, our goal is to  jointly learn architecture $\alpha$ and the weights $w$ within all the mixed operations (e.g. weight of the convolution filters)
- The goal for archiutecture is to find $\alpha^{\*}$ that minimizes the validation loss $\mathcal L_{train} (w^{\*}, \alpha^{\*})$, 
- where the weights $ w^{*} $ assiciated with the architecture are obtained by minimizing the train loss $w^{\*} = \underset{w}{argmin}  \mathcal L_{train} (w, \alpha^{\*}) $

- This implies a bilevel optimization problem with $\alpha$ as the upper-level variable and $w$ as the lower-level variable 

$$ \underset{\alpha}{min} {\mathcal L}_{val}(w^{*}(\alpha), \alpha) $$

$$ s.t. \\  w^{\*} (\alpha) = \underset{w}{argmin}  \mathcal L_{train} (w, \alpha)  $$


### Approximate Architecture Gradient
- evalute the gradient exactly can be prohibitive due to the inner optimizartion is expensive.
- an approximation scheme as follows:

$$ \nabla_{\alpha} {\mathcal L_{val}(w^{\*}(\alpha), \alpha)}\approx \nabla_{\alpha}{\mathcal L_{val}(w - \xi \nabla_{w}{\mathcal L_{train} (w,\alpha), \alpha} )}$$

- the idea is to approximate $w^{\*}(\alpha)$ by adapting $w$ using only a single training step using learning rate $\xi$ without solving the inner optimization $w^{\*} (\alpha) = \underset{w}{argmin}  \mathcal L_{train} (w, \alpha) $ completely by training until convergence. 

- The procedure is outlined as Algotithm 1:

<img width="952" alt="Screen Shot 2021-07-06 at 20.20.46.png (127.1 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/06/103347/442ebfa7-e3cd-4ea0-a0bf-04a36c6e82e2.png">

- Apply chain rule to the approximate architecture  gradient can get:
$$\color{red}{\nabla_{\alpha}}{\mathcal L_{val}(w - \xi \nabla_{w}{\mathcal L_{train} (w,\color{red}{\alpha}), \color{red}{\alpha}} )} = \color{blue}{\nabla_{\alpha}} {\mathcal L_{val}(w^{'},\color{blue}{\alpha})} - \xi \nabla_{\alpha, w}^{2} {\mathcal L_{train}}(w,\alpha)\cdot \color{blue}{\nabla_{w^{'}}} {\mathcal L_{val}}(\color{blue}{w^{'}},\alpha)$$

- where $w^{'} = w - \xi \nabla_{w}\mathcal L_{train} (w,\alpha)$ denoted the weights for a one-step forward model.
---
*Induction:*
$$ \nabla_{\alpha} f(g_{1}(\alpha), g_{2}(\alpha))=\color{blue}{D_{1}f(g_{1}(\alpha), g_{2}(\alpha))}   \cdot \color {red}{\nabla_{\alpha} g_{1}(\alpha)}  +\color{blue}{D_{2}f(g{1}(\alpha), g_{2}(\alpha))}  \cdot \color {red}{\nabla_{\alpha} g_{2}(\alpha)} $$

-  $f(\cdot,\cdot) = {\mathcal L{val}}  (\cdot, \cdot) $
-  $g_{1}(\alpha) = w^{'} = w - \xi \nabla{w}\mathcal L{train} (w,\alpha)$
- $g_{2}(\alpha)  = \alpha$

*Differentiate:*
- $\nabla_{\alpha}  g_{1}(\alpha) = - \xi \nabla{\alpha, w}^{2} \mathcal L{train} (w,\alpha)$
- $\nabla_{\alpha} g_{2}(\alpha) = 1$
- $D{1}f(g_{1}(\alpha), g_{2}(\alpha)) =  \color{blue}{\nabla{w^{'}}} {\mathcal L{val}}(\color{blue}{w^{'}},\alpha)$
- $D{2}f(g_{1}(\alpha), g_{2}(\alpha)) = \color{blue}{\nabla{\alpha}} {\mathcal L{val}(w^{'},\color{blue}{\alpha})}$
---

- the expression above contains an expensive matrix-vector product in its second term. Reduce it using finite difference approximation. Let $\epsilon$ be a small scalar $\epsilon = \frac {0.01} { {\lVert \color{blue}{\nabla{\alpha}} {\mathcal L{val}(w^{'},\color{blue}{\alpha})} \rVert}_{ 2 }    }$
-  and  $w^{\pm} = w \pm \epsilon  \color{blue}{\nabla_{w^{'}}} {\mathcal L_{val}}(\color{blue}{w^{'}},\alpha) $
 - Then 

    $$\nabla_{\alpha, w}^{2} {\mathcal L_{train}}(w,\alpha)\cdot \color{blue}{\nabla_{w^{'}}} {\mathcal L_{val}}(\color{blue}{w^{'}},\alpha) \approx \frac {\nabla_{\alpha} {\mathcal L_{train}} (w^{+}, \alpha) - \nabla_{\alpha} {\mathcal L_{train}} (w^{-}, \alpha) } {2 \epsilon}  $$

- the Complexity reduced from $O(\|\alpha \| \|w\|)$ to $O(\|\alpha \| +  \|w\|)$

---
*Induction:*
*We know the Taylor series* :
$$ f(x) = f(x_0) +  \frac {f^{'}(x_0)}{ 1!} (x-x_0)+ \frac {f^{''}(x_0)}{ 2!} (x-x_0)^2 + \cdots$$

*Let  $ x =  x_0 + hA$* and $ x =  x_0 - hA $, we can induct the following expressions:
- $f(x_{0} + hA)  = f(x_0) + \frac {f^{'} (x_0) }{ 1!}hA + \cdots $
- $ f(x_{0} - hA) = f(x_0) -\frac {f^{'} (x_0) }{ 1!}hA  + \cdots $  
- $f^{'} (x_{0} )\cdot A \approx  \frac {f(x_{0} + hA )-f(x_{0} - hA )} { 2h } $

*Then replace items by :*
- $h = \epsilon$
- $A =  \color{blue}{\nabla_{w^{'}}} {\mathcal L_{val}}(\color{blue}{w^{'}},\alpha) $
- $x_{0} = w$
- $f(\cdot, \cdot) = \nabla_{\alpha} {\mathcal L_{train}(\cdot, \cdot)}$
---


### experiment and result 
#### experiment setting
- following operations are included in $\mathcal O$
   - 3 × 3 [dilated separable convolution ](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
   - 5 × 5 dilated separable convolution
   - 3 × 3 [depth-wise separable convolution](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
   - 5 × 5 depth-wise separable convolution 
  - 3 × 3 max pooling
  - 3 × 3 average pooling, 
  - no connection (zero)
  - and a skip connection (identity)

<img width="820" alt="Screen Shot 2021-07-07 at 3.14.07.png (230.3 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/07/103347/757635ac-0f07-4366-bea7-d63e44047246.png">

#### result
- CNN <img width="814" alt="Screen Shot 2021-07-07 at 3.33.52.png (354.8 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/07/103347/9ced0fd7-4ed0-4b28-a7d7-00d2d29b2841.png">
- RNN <img width="813" alt="Screen Shot 2021-07-07 at 3.34.58.png (278.3 kB)" src="https://img.esa.io/uploads/production/attachments/14973/2021/07/07/103347/ae3c5a6f-0f3d-468e-aa60-04bace3fd142.png">

- The performanc by ENAS(RL method) looks similar to the NAS, I will explore it next time.
- Pham et al., [Efficient Neural Architecture Search via Parameter Sharing ](https://arxiv.org/pdf/1802.03268.pdf), ICML 2018


### Conclusion

- DARTS is able to greatly reduce the cost of GPU hours. Their experiments for searching for CNN cells have 7 Nodes  and only took 1.5 days with a single GPU. 
- However, it suffers from the high GPU memory consumption issue due to its continuous representation of network architecture.

## Candidate NAS for my research
- Random Search (as baseline)
- DARTS (Gradient)
- ENAS (RL+ parameter sharing)
## Possible future direction
- Search efficiency
- Moving towards less constrained Search Space
- Designing efficient architectures: automated scaling, pruning and quantization (model compression techniques metioned at  last bw meeting)

## Material To learn about NAS:
[1] https://lilianweng.github.io/lil-log/2020/08/06/neural-architecture-search.html#evolutionary-algorithms
[2] [https://hangzhang.org/ECCV2020/](https://hangzhang.org/ECCV2020/)

## Engineering part
- The implementation of DARTS is available at https://github.com/quark0/darts
- Apply it on my current research domain to see the performance?
