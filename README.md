# indoor_localization

This is a repository for research on indoor localization based on wireless
fingerprinting techniques.

For more details, please
visit
[project home page](http://kyeongsoo.github.io/research/projects/indoor_localization/index.html).

* 20170812: Results
  with [1-hidden-layer classifier](indoor_localization-1.ipynb) (nh: number of
  hidden layer nodes, dr: dropout rate)
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
