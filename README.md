## Forecasting the Disturbance Storm Time Index with Bayesian Deep Learning<br>
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/dst.svg)](https://zenodo.org/badge/latestdoi/507188155)


## Authors
Yasser Abduallah, Jason T. L. Wang, Prianka Bose, Genwei Zhang, Firas Gerges, and Haimin Wang

## Abstract

The disturbance storm time (Dst) index is an important and useful measurement in space weather research. It has been used to characterize the size and intensity of a geomagnetic storm. A negative Dst value means that the Earth’s magnetic field is weakened, which happens during storms. Here, we present a novel deep learning method, called the Dst Transformer (or DSTT for short), to perform short-term, 1-6 hour ahead, forecasting of the Dst index based on the solar wind parameters provided by the NASA Space Science Data Coordinated Archive. The Dst Transformer combines a multi-head attention layer with Bayesian inference, which is capable of quantifying both aleatoric uncertainty and epistemic uncertainty when making Dst predictions. Experimental results show that the proposed Dst Transformer outperforms related machine learning methods in terms of the root mean square error and R-squared. Furthermore, the Dst Transformer can produce both data and model uncertainty quantification results, which can not be done by the existing methods. To our knowledge, this is the first time that Bayesian deep learning has been used for Dst index forecasting.


Please note that starting Binder might take some time to create and start the image.

Please also note that the execution time in Binder varies based on the availability of resources. The average time to run the notebook is 10-15 minutes, but it could be more.

For the latest updates of the tool refer to https://github.com/deepsuncode/Dst-prediction

## Installation on local machine

|Library | Version   | Description  |
|---|---|---|
|keras| 2.6.0 | Deep learning API|
|numpy| 1.21.5| Array manipulation|
|scikit-learn| 1.0.1| Machine learning|
|sklearn| latest| Tools for predictive data analysis|
|matlabplot| 3.4.3| Visutalization tool|
| pandas|1.3.4| Data loading and manipulation|
| seaborn | 0.11.2| Visualization tool|
| scipy|1.7.1| Provides algorithms for optimization and statistics|
| tensorboard| 2.7.0 | Provides the visualization and tooling needed for machine learning|
| tensorflow-gpu| 2.6.1| Deep learning tool for high performance computation |
|tensorflow-probability | 0.14.1| For probabilistic models|
