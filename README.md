# Indoor Localization

This is a repository for research on indoor localization based on wireless
fingerprinting techniques.

For more details, please
visit
[project home page](http://kyeongsoo.github.io/research/projects/indoor_localization/index.html).

* 20170812: We first
  try
  [a feed-forward classifier with just one hidden layer](indoor_localization-1.ipynb) per
  the comments
  from
  [this](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) and
  [this](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw). Below
  are the results (nh: number of hidden layer nodes, dr: dropout rate):
  * nh=16 & dr=0.00: loss=1.234082e+00, accuracy=7.533753e-01	
  * nh=16 & dr=0.20: loss=1.492540e+00, accuracy=7.353735e-01
  * nh=16 & dr=0.50: loss=1.369105e+00, accuracy=7.488749e-01
  * nh=32 & dr=0.00: loss=1.026828e+00, accuracy=8.001800e-01
  * nh=32 & dr=0.20: loss=1.256892e+00, accuracy=7.560756e-01
  * nh=32 & dr=0.50: loss=1.296779e+00, accuracy=7.659766e-01
  * nh=64 & dr=0.00: loss=1.033471e+00, accuracy=7.947795e-01
  * nh=64 & dr=0.20: loss=9.504665e-01, accuracy=8.235824e-01
  * nh=64 & dr=0.50: loss=1.162159e+00, accuracy=7.830783e-01
  * nh=128 & dr=0.00: loss=1.055724e+00, accuracy=8.001800e-01
  * nh=128 & dr=0.20: loss=1.023210e+00, accuracy=8.343834e-01
  * nh=128 & dr=0.50: loss=1.475645e+00, accuracy=7.389739e-01
  * nh=256 & dr=0.00: loss=1.222873e+00, accuracy=7.947795e-01
  * nh=256 & dr=0.20: loss=9.960540e-01, accuracy=8.280828e-01
  * nh=256 & dr=0.50: loss=1.053368e+00, accuracy=8.127813e-01
  * nh=512 & dr=0.00: loss=1.344286e+00, accuracy=8.100810e-01
  * nh=512 & dr=0.20: loss=1.255742e+00, accuracy=7.983798e-01
  * nh=512 & dr=0.50: loss=1.346962e+00, accuracy=8.001800e-01

* 20170813: Results
  after
  [replacing the activation functions of the hidden-layer from 'tanh' to 'relu'](indoor_localization-2.ipynb) per
  the second answer
  to
  [this qustion](https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer):
  * nh=16 & dr=0.00: loss=1.283800e+00, accuracy=7.416742e-01
  * nh=16 & dr=0.20: loss=1.673727e+00, accuracy=6.759676e-01
  * nh=16 & dr=0.50: loss=1.772957e+00, accuracy=6.444644e-01
  * nh=32 & dr=0.00: loss=1.505690e+00, accuracy=7.335734e-01
  * nh=32 & dr=0.20: loss=1.572381e+00, accuracy=7.155716e-01
  * nh=32 & dr=0.50: loss=1.882560e+00, accuracy=6.975698e-01
  * nh=64 & dr=0.00: loss=1.423406e+00, accuracy=7.470747e-01
  * nh=64 & dr=0.20: loss=1.680462e+00, accuracy=7.182718e-01
  * nh=64 & dr=0.50: loss=2.209932e+00, accuracy=6.966697e-01
  * nh=128 & dr=0.00: loss=1.386000e+00, accuracy=8.046805e-01
  * nh=128 & dr=0.20: loss=1.776332e+00, accuracy=7.704770e-01
  * nh=128 & dr=0.50: loss=1.947697e+00, accuracy=7.407741e-01
  * nh=256 & dr=0.00: loss=2.466667e+00, accuracy=7.227723e-01
  * nh=256 & dr=0.20: loss=3.134137e+00, accuracy=6.525653e-01
  * nh=256 & dr=0.50: loss=2.839252e+00, accuracy=6.894689e-01
  * nh=512 & dr=0.00: loss=3.212590e+00, accuracy=7.020702e-01
  * nh=512 & dr=0.20: loss=2.673689e+00, accuracy=7.263726e-01
  * nh=512 & dr=0.50: loss=3.308757e+00, accuracy=6.903690e-01
  
  Compared to the case with 'tanh', however, the results seem to not improve.
