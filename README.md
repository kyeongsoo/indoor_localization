This is a repository for research on indoor localization based on wireless fingerprinting techniques. For more details, please visit [XJTLU SURF project home page](http://kyeongsoo.github.io/research/projects/indoor_localization/index.html).


# 2018-01-03
  * A summary of the publications based on the work in this repository:
    * Kyeong Soo Kim, Sanghyuk Lee, and Kaizhu Huang "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting," *submitted to Big Data Analytics*, Dec. 5, 2017. ([arXiv](https://arxiv.org/abs/1712.01990))
        * [Building/floor estimation and floor-level coordinates estimation of a given location](./python/scalable_indoor_localization.py)
    * Kyeong Soo Kim, Ruihao Wang, Zhenghang Zhong, Zikun Tan, Haowei Song, Jaehoon Cha, and Sanghyuk Lee, "Large-scale location-aware services in access: Hierarchical building/floor classification and location estimation using Wi-Fi fingerprinting based on deep neural networks," *Proc. Fiber Optics in Access Networks (FOAN) 2017*, Munich, Germany Nov. 7, 2017. ([arXiv](https://arxiv.org/abs/1710.00951))
	    * [Multi-label classification of buildings and floors](./python/bf_multi-label_classification.py)

# 2017-08-18
  * Implement [a multi-label classifier](./python/bf_multi-label_classification.py) to address the issues described on 2017-08-17: 3 building and 5 floor identifiers are [one-hot](https://en.wikipedia.org/wiki/One-hot) encoded into an 8-dimensional vector (e.g., '001|01000') and classified with different class weights (e.g., 30 for buidlings and 1 for floors); the resulting one-hot-encoded vector is split into 3-dimensional building and 5-dimensional floor vectors and the index of a maximum value of each vector is returned as a classified class ([results](./results/bf_multi-label_classification_out_20170819-010852.org)).
      * Still, need to optimize parameters a lot.

# 2017-08-17
  * Implement [a new program](./python/bf_classification.py), which calculates accuracies separately for building and floor classification, to investigate the hierarchical nature of the classification problem at hand; the deep-learning-based place recognition system described in the key paper<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> does not take into account this and carries out classification based on flattened labels (i.e., (building, floor) -> 'building-floor'). We are now considering two options to guarantee 100% accuracy for the building classification:
      * Hierarchical classifier with a tree structure and multiple classifiers and data sets, which is a conventional approach and a reference for this investigation.
      * One classifier with a weighted loss function<sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup>. In our case, however, the loss function does not give a closed-form gradient function, which forces us to use evolutionary algorithms (e.g., [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)) for training of neural network weights or [multi-label classification with different class weights](https://github.com/fchollet/keras/issues/741) (i.e., higher weights for buildings in our case).

# 2017-08-15
  * Today, we further simplified the building/floor classification system by removing a hidden layer from the classifier (therefore no dropout), resulting in the configuration of '520-64-4-13' (including input and output layers) with loss=7.050603e-01 and accuracy=9.234923e-01 ([results](./results/indoor_localization_deep_learning_out_20170815-203448.org)). This might mean that the 4-dimensional data from the SAE encoder (64-4) can be linearly separable. Due to training of SAE encoder weights for the combined system, however, it needs further investigation.

# 2017-08-14
  * We investigated whether a couple of strong RSSs in a fingerprint dominate the classification performance in building/floor classification. After many trials with different configurations, we could obtain more than 90% accuracies with the stacked-autoencoder (SAE) having 64-4-64 hidden layers (i.e., just 4 dimension) and the classifier having just one 128-node hidden layer ([results](./results/indoor_localization_deep_learning_out_20170814-184009.org)). This implies that a small number of RSSs from access points (APs) deployed in a building/floor can give enough information for the building/floor classification; the localization on the same floor, by the way, would be quite different, where RSSs from possibly many APs have a significant impact on the localization performance.

# 2017-08-13
  * We finally obtained [more than 90% accuracies](./results/indoor_localization_deep_learning.org) from [this version](./python/indoor_localization_deep_learning.py), which are comparable to the results of the key paper <sup><a id="fnr.1.100" class="footref" href="#fn.1">1</a></sup> based on the [UJIIndoorLoc Data Set](https://archive.ics.uci.edu/ml/datasets/ujiindoorloc); refer to the [multi-class clarification example](https://keras.io/getting-started/sequential-model-guide/#compilation) for classifier parameter settings.
  * We [replace the activation functions of the hidden-layer from 'tanh' to 'relu'](./python/indoor_localization-2.ipynb) per the second answer to [this question](https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer) ([results](./results/indoor_localization-2_20170813.csv)). Compared to the case with 'tanh', however, the results seem to not improve (a bit in line with the gut-feeling suggestions from [this](https://datascience.stackexchange.com/questions/10048/what-is-the-best-keras-model-for-multi-class-classification)).

# 2017-08-12
  * We first tried [a feed-forward classifier with just one hidden layer](./python/indoor_localization-1.ipynb) per the comments from [this](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) ([results](./results/indoor_localization-1_20170812.csv)). (\* *nh*: number of hidden layer nodes, *dr*: [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) rate, *loss*: [categorical cross-entropy](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.categorical_crossentropy), *acc*: accuracy \*).

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> M. Nowicki and J. Wietrzykowski, "Low-effort place recognition with WiFi fingerprints using deep learning," arXiv:1611.02049v2 [cs.RO] [(arXiv)](https://arxiv.org/abs/1611.02049v2)

<sup><a id="fn.2" class="footnum" href="#fnr.2">2</a></sup> T. Yamashita et al., "Cost-alleviative learning for deep convolutional neural network-based facial part labeling," *IPSJ Transactions on Computer Vision and Applications*, vol. 7, pp. 99-103, 2015. [(DOI)](http://doi.org/10.2197/ipsjtcva.7.99)
