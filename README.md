- [2017-08-17](#orgb9d65ce)
- [2017-08-15](#org7746a3b)
- [2017-08-14](#orga719344)
- [2017-08-13](#org2aceb5b)
- [2017-08-12](#orgc2d0fe6)

This is a repository for research on indoor localization based on wireless fingerprinting techniques. For more details, please visit [XJTLU SURF project home page](http://kyeongsoo.github.io/research/projects/indoor_localization/index.html).


<a id="orgb9d65ce"></a>

# 2017-08-17

-   Implement [a new program](./python/bf_classification.py), which calculates accuracies separately for building and floor classification, to investigate the hierarchical nature of the classification problem at hand; the deep-learning-based place recognition system described in [the key paper](https://arxiv.org/abs/1611.02049v2) does not take into account this and carries out classification based on flattened labels (i.e., (building, floor) -> 'building-floor'). We are now considering two options to guarantee 100% accuracy for the building classification:
    -   Hierarchical classifier with a tree structure and multiple classifiers and data sets, which is a conventional approach and a reference for this investigation;
    -   One classifier with a weighted loss function (e.g., [this](https://www.jstage.jst.go.jp/article/imt/10/3/10_488/_pdf)).


<a id="org7746a3b"></a>

# 2017-08-15

-   Today, we further simplified the building/floor classification system by removing a hidden layer from the classifier (therefore no dropout), resulting in the configuration of '520-64-4-13' (including input and output layers) with loss=7.050603e-01 and accuracy=9.234923e-01 ([results](./results/indoor_localization_deep_learning_out_20170815-203448.org)). This might mean that the 4-dimensional data from the SAE encoder (64-4) can be linearly separable. Due to training of SAE encoder weights for the combined system, however, it needs further investigation.


<a id="orga719344"></a>

# 2017-08-14

-   We investigated whether a couple of strong RSSs in a fingerprint dominate the classification performance in building/floor classification. After many trials with different configurations, we could obtain more than 90% accuracies with the stacked-autoencoder (SAE) having 64-4-64 hidden layers (i.e., just 4 dimension) and the classifier having just one 128-node hidden layer (the results are [here](./results/indoor_localization_deep_learning_out_20170814-184009.org)). This implies that a small number of RSSs from access points (APs) deployed in a building/floor can give enough information for the building/floor classification; the localization on the same floor, by the way, would be quite different, where RSSs from possibly many APs have a significant impact on the localization performance.


<a id="org2aceb5b"></a>

# 2017-08-13

-   We finally obtained [more than 90% accuracies](./results/indoor_localization_deep_learning.org) from [this version](./python/indoor_localization_deep_learning.py), which are comparable to the results of [the key paper](https://arxiv.org/abs/1611.02049v2); refer to the [multi-class clarification example](https://keras.io/getting-started/sequential-model-guide/#compilation) for classifier parameter settings.
-   We [replace the activation functions of the hidden-layer from 'tanh' to 'relu'](./python/indoor_localization-2.ipynb) per the second answer to [this question](https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer). The results are [here](./results/indoor_localization-2_20170813.csv). Compared to the case with 'tanh', however, the results seem to not improve (a bit in line with the gut-feeling suggestions from [this](https://datascience.stackexchange.com/questions/10048/what-is-the-best-keras-model-for-multi-class-classification)).


<a id="orgc2d0fe6"></a>

# 2017-08-12

-   We first tried [a feed-forward classifier with just one hidden layer](./python/indoor_localization-1.ipynb) per the comments from [this](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw). The results are [here](./results/indoor_localization-1_20170812.csv) (\* *nh*: number of hidden layer nodes, *dr*: [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) rate, *loss*: [categorical cross-entropy](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.categorical_crossentropy), *acc*: accuracy \*).
