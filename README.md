# DeepT - Multi-norm zonotopes & Transformer certification

We present DeepT, a novel method for certifying Transformer networks based on 
abstract interpretation. The key idea behind DeepT is our new multi-norm Zonotope
abstract domain, an extension of the classical Zonotope designed to handle L1 
and L2-norm bound perturbations. 


## Requirements

DeepT requires a GNU/Linux system with:
- An NVIDIA GPU with 
  - at least 11 GB of RAM for the scaled down version of the experiments
  - at least 24 GB of RAM for the full reproduction of the experiments
- Installed CUDA drivers supporting CUDAlib 10.


## Installation

1. Go into the `Robustness-Verification-for-Transformers/reproduce_results` 
directory: `cd Robustness-Verification-for-Transformers/reproduce_results`
2. Run `install.sh` script to install Miniconda, create an environment with the dependencies
installed and download the data used in the experiments: `./install.sh`

## Getting started

To run the DeepT and certify a Transformer network against a synonym attack, follow 
these instructions:

1. Install the DeepT library (as well as the CROWN baseline) as described 
   in the `Installation` section.
2. Go into the `Robustness-Verification-for-Transformers/reproduce_results` 
   directory: `cd Robustness-Verification-for-Transformers/reproduce_results`
3. Run the script `get_results_synonym_attack.sh`, which will produce the statistics in
   stdout.

To run the DeepT and certify a Transformer network against a L2 attack on the embeddings
for a Transformer with 6 layers, follow these instructions:

1. Install the DeepT library (as well as the CROWN baseline) as described 
   in the `Installation` section.
2. Go into the `Robustness-Verification-for-Transformers/getting_started` 
   directory: `cd Robustness-Verification-for-Transformers/getting_started`
3. Run the script `get_results_few_samples.sh`, which will produce the statistics in
   stdout.

## Step-by-Step Instructions to Reproduce Article Results

As running all experiments can take some time, we first give step-by-step instructions to reproduce
a scaled down version of our results, after which we will give step-by-step instructions to
reproduce all results.

**Note: Tables 1, 2, 3, 4, 5 here correspond to Tables 1, 4, 5, 13, 6 in the paper.**

### **Step-by-Step Instructions to Reproduce a faster and scaled down version of the Article Results**

**Preliminary step**: go into the `Robustness-Verification-for-Transformers/reproduce_results_scaled_down` 
directory.

#### Obtaining the (raw) results for each table

**All tables**

To compute **results for all Tables**, run the script `get_results_all_tables_scaled_down.sh`, which will store the results in 
the `Robustness-Verification-for-Transformers/results` directory, organized per experiment / Table.

**Table 1, 2, 3**

To compute **the results for a particular Table**, for example Table 3, run the 
script `get_results_table3_scaled_down.sh`, which will store the results for that experiment / Table 
in a subdirectory of the `Robustness-Verification-for-Transformers/results` directory.

**Table 4, 5**

Both tables show the results of an ablation study, therefore **you must first compute 
the results for Table 1**. Only once they **are fully computed**, can you run either 
`get_results_table4_scaled_down.sh` (for Table 4) or `get_results_table5_scaled_down.sh` (for Table 5).

#### Obtaining a summary of the data for each table

1. Run `open_notebook.sh` script to start Jupyter. 
2. Select the `Analysis (scaled down).ipynb`.
3. Run all the cells in the notebook.


#### Reproducing the synonym attack results

Run the script `get_results_synonym_attack.sh`, which will produce the statistics in stdout. 113 out of 127 sentences should be certified.

#### Expected results

We note the expected results for the scaled down experiments at the end of this README.


### **Step-by-Step Instructions to Reproduce the Article Results**

**Preliminary step**: go into the `Robustness-Verification-for-Transformers/reproduce_results` 
directory.

#### Obtaining the (raw) results for each table

**All tables**

To compute **results for all Tables**, run the script `get_results_all_tables.sh`, which will store the results in 
the `Robustness-Verification-for-Transformers/results` directory, organized per experiment / Table.

**Table 1, 2, 3**

To compute **the results for a particular Table**, for example Table 3, run the 
script `get_results_table3.sh`, which will store the results for that experiment / Table 
in a subdirectory of the `Robustness-Verification-for-Transformers/results` directory.

**Table 4, 5**

Both tables show the results of an ablation study, therefore **you must first compute 
the results for Table 1**. Only once they **are fully computed**, can you run either 
`get_results_table4.sh` (for Table 4) or `get_results_table5.sh` (for Table 5).


#### Obtaining a summary of the data for each table

1. Run `open_notebook.sh` script to start Jupyter. 
2. Select the `Analysis.ipynb`.
3. Run all the cells in the notebook.

#### Reproducing the synonym attack results

Run the script `get_results_synonym_attack.sh`, which will produce the statistics in stdout. 113 out of 127 sentences should be certified.

#### Expected results

The reference results for the full experiments can be consulted in the paper (Tables 1, 4, 5, 13, 6).

## Other platforms

The installation instructions and the scripts to obtain the results are for GNU/Linux. 
DeepT can be used on other platforms, provided the dependencies are met.
Feel free to write a pull request with working scripts for your favourite platform.

# Datasets

Our code depends on the following datasets which are downloaded by our installation scripts:

## SST

The files for the SST dataset [1] originate from 2 sources:

1. The TXT files can be downloaded from the
   [Stanford NLP Sentiment Analysis](https://nlp.stanford.edu/sentiment/index.html) 
   page using 
   [this link](https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip).
2. The TSV files can be downloaded from the 
   [GLUE benchmark](https://gluebenchmark.com/tasks)
   page using
   [this link](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip).

## Yelp

The Yelp dataset is available at [https://www.yelp.com/dataset](https://www.yelp.com/dataset).

The Yelp reviews polarity dataset is constructed by Xiang Zhang 
(xiang.zhang@nyu.edu) from the above dataset. It is first used as a
text classification benchmark in [2].


## References

[1] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D.
Manning, Andrew Y. Ng, and Christopher Potts. 2013. Recursive Deep
Models for Semantic Compositionality Over a Sentiment Treebank. In
EMNLP. ACL, 1631–1642.

[2] Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional 
Networks for Text Classification. Advances in Neural Information Processing 
Systems 28 (NIPS 2015).

## Expected results for the scaled down experiment


### DeepT-Fast vs CROWN-BaF

```
num_layers    p    Method    min_eps    avg_eps    timing    memory
0    3    1    CROWN-BaF    0.035840    0.328545    4.743803    2.348787e+08
1    3    1    DeepT-Fast    0.036953    0.342852    20.010851    2.462264e+09
2    3    2    CROWN-BaF    0.006323    0.059529    4.063299    2.315046e+08
3    3    2    DeepT-Fast    0.006577    0.061335    19.298236    2.443348e+09
4    3    inf    CROWN-BaF    0.000616    0.005845    4.025043    2.319634e+08
5    3    inf    DeepT-Fast    0.002402    0.006836    17.950831    2.414454e+09
6    6    1    CROWN-BaF    0.069883    0.298320    9.461189    5.313574e+08
7    6    1    DeepT-Fast    0.094609    0.613359    57.213571    4.658533e+09
8    6    2    CROWN-BaF    0.012344    0.052624    9.595476    4.611898e+08
9    6    2    DeepT-Fast    0.015859    0.106309    58.984073    4.648561e+09
10    6    inf    CROWN-BaF    0.001193    0.005087    9.155682    4.602286e+08
11    6    inf    DeepT-Fast    0.002123    0.010806    58.695221    4.659872e+09
12    12    1    CROWN-BaF    0.031250    0.033066    29.927394    8.069895e+08
13    12    1    DeepT-Fast    0.357812    0.448750    110.576533    4.662540e+09
14    12    2    CROWN-BaF    0.006245    0.006732    29.228716    1.144277e+09
15    12    2    DeepT-Fast    0.074375    0.093919    115.870781    4.667326e+09
16    12    inf    CROWN-BaF    0.000605    0.000655    27.410248    1.342709e+09
17    12    inf    DeepT-Fast    0.007324    0.009245    112.149634    4.667327e+09
```

### DeepT-Fast vs DeepT-Precise vs CROWN-Backward (p = inf)

```
num_layers    p    Method    min_eps    avg_eps    timing    memory
0    3    inf    CROWN-Backward    0.037910    0.037910    161.832892    2559889920
1    3    inf    DeepT-Fast    0.034980    0.034980    22.218244    6412910592
2    3    inf    DeepT-Precise    0.036914    0.036914    1375.276368    3629949952
3    6    inf    CROWN-Backward    0.034141    0.034141    592.294316    3903038464
4    6    inf    DeepT-Fast    0.033125    0.033125    52.560159    6454697984
5    6    inf    DeepT-Precise    0.034355    0.034355    3441.530187    3637649408
```

### DeepT-Fast vs CROWN-BaF vs CROWN-Backward (p = 1 and p = 2)

```
num_layers    p    Method    min_eps    avg_eps    timing    memory
0    3    1    CROWN-BaF    0.400937    0.752969    3.341010    9.118848e+07
1    3    1    CROWN-Backward    0.450625    0.901562    132.717805    2.055295e+09
2    3    1    DeepT-Fast    0.447188    0.839844    19.501237    4.721367e+09
3    3    2    CROWN-BaF    0.093906    0.174375    3.413080    9.180083e+07
4    3    2    CROWN-Backward    0.099766    0.190820    125.865152    2.051402e+09
5    3    2    DeepT-Fast    0.097500    0.178438    19.500627    4.748480e+09
6    6    1    CROWN-BaF    0.461563    0.752969    8.851718    1.426790e+08
7    6    1    CROWN-Backward    0.502813    0.858906    441.793114    3.156439e+09
8    6    1    DeepT-Fast    0.483125    0.812813    45.975986    4.957322e+09
9    6    2    CROWN-BaF    0.104844    0.167344    8.687577    1.414830e+08
10    6    2    CROWN-Backward    0.112656    0.184297    451.633752    3.158009e+09
11    6    2    DeepT-Fast    0.107891    0.177617    45.843470    4.958825e+09
```


### Impact of the softmax sum constraint

```
    num_layers    p    Method    min_eps    avg_eps    timing    memory
0    3    1    DeepT-Fast-With-Constraint    0.036953    0.342852    20.010851    2.462264e+09
1    3    1    DeepT-Fast-Without-Constraint    0.037715    0.343232    19.202335    2.423966e+09
2    3    2    DeepT-Fast-With-Constraint    0.006577    0.061335    19.298236    2.443348e+09
3    3    2    DeepT-Fast-Without-Constraint    0.006509    0.061301    18.467304    2.405997e+09
4    3    inf    DeepT-Fast-With-Constraint    0.002402    0.006836    17.950831    2.414454e+09
5    3    inf    DeepT-Fast-Without-Constraint    0.000624    0.005947    18.240911    2.379198e+09
6    6    1    DeepT-Fast-With-Constraint    0.094609    0.613359    57.213571    4.658533e+09
7    6    1    DeepT-Fast-Without-Constraint    0.088750    0.600026    54.795363    4.659621e+09
8    6    2    DeepT-Fast-With-Constraint    0.015859    0.106309    58.984073    4.648561e+09
9    6    2    DeepT-Fast-Without-Constraint    0.014941    0.105036    56.822147    4.649803e+09
10    6    inf    DeepT-Fast-With-Constraint    0.002123    0.010806    58.695221    4.659872e+09
11    6    inf    DeepT-Fast-Without-Constraint    0.001422    0.010097    55.280240    4.658774e+09
12    12    1    DeepT-Fast-With-Constraint    0.357812    0.448750    110.576533    4.662540e+09
13    12    1    DeepT-Fast-Without-Constraint    0.360000    0.448646    108.116962    4.663212e+09
14    12    2    DeepT-Fast-With-Constraint    0.074375    0.093919    115.870781    4.667326e+09
15    12    2    DeepT-Fast-Without-Constraint    0.074688    0.093828    114.400204    4.667800e+09
16    12    inf    DeepT-Fast-With-Constraint    0.007324    0.009245    112.149634    4.667327e+09
17    12    inf    DeepT-Fast-Without-Constraint    0.007358    0.009230    109.006677    4.667800e+09
```

### Impact of the norm ordering in the dot product abstract transformer

```
num_layers    p    Method    min_eps    avg_eps    timing    memory
0    3    1    DeepT-Fast-Linf-First    0.036953    0.342852    20.010851    2.462264e+09
1    3    1    DeepT-Fast-Lp-First    0.038008    0.343379    22.931275    2.462315e+09
2    3    2    DeepT-Fast-Linf-First    0.006577    0.061335    19.298236    2.443348e+09
3    3    2    DeepT-Fast-Lp-First    0.006577    0.061335    22.328354    2.443348e+09
4    6    1    DeepT-Fast-Linf-First    0.094609    0.613359    57.213571    4.658533e+09
5    6    1    DeepT-Fast-Lp-First    0.097344    0.608490    71.524694    4.658548e+09
6    6    2    DeepT-Fast-Linf-First    0.015859    0.106309    58.984073    4.648561e+09
7    6    2    DeepT-Fast-Lp-First    0.015859    0.106100    73.836048    4.648798e+09
8    12    1    DeepT-Fast-Linf-First    0.357812    0.448750    110.576533    4.662540e+09
9    12    1    DeepT-Fast-Lp-First    0.355625    0.445104    138.921459    4.662743e+09
10    12    2    DeepT-Fast-Linf-First    0.074375    0.093919    115.870781    4.667326e+09
11    12    2    DeepT-Fast-Lp-First    0.074219    0.093659    146.141137    4.667326e+09
```