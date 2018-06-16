# MDFA-DeepLearning
MDFA-DeepLearning is a library for building machine learning applications on large numbers of multivariate time series (or sequential) data, with a heavy emphasis on noisy (non)stationary data. 

One might want to develop predictive models in multivariate time series data 
using MDFA-DeepLearning if the time series exhibit any of the following properties:
1) High-Dimensionality
2) Difficult to forecast using traditional model-based methods (VARIMA/GARCH) or traditional
deep learning methods (RNN/LSTM, component decompositions, etc)   
3) Emphasis needed on out-of-sample real-time signal extraction and forecasting
4) Regime changing is a common occurrence 


The MDFA-DeepLearning approach differs from most machine learning methods in time series analysis in that an emphasis on real-time feature extraction is utilized where the features extractors are build using the multivariate direct filter approach (MDFA). The motivation behind this coupling of MDFA with machine learning is that, while many time series decomposition methodologies exist (from empirical mode decomposition to stochastic component analysis methods), all of these rely on either in-sample decompositions (useless for future data), or assumptions about the boundary values, neither of which are attractive when fast, real-time out-of-sample predictions are the emphasis.  

Furthermore, simply applying standard recurrent neural networks for step-ahead forecasting or 
signal extraction directly on the noisy data is a useless exercise - the recurrent networks will only learn noise, producing signals and forecasts of little to no value. 

MDFA-DeepLearning requires both the MDFA-Toolkit package for constructing the time series feature extractors and the Eclipse Deeplearning4j (dl4j) library for the deep recurrent neural network constructors. The dl4j library is freely available at github.com/deeplearning4j, but is included in the build of this package using Gradle (or Maven) as the dependency management tool. 

The back-end for the dl4j package will depend on your computational infrastructure, but is available 
on a local basis using CPUs, or can take advantage of GPUs using CUDA libraries. In this package I have included a reference to both (assuming a standard linux64 architechture. 

The back-end used for the novel feature extraction multivariate direct filter approach (MDFA)

For the build and package management, we use a Gradle wrapper, and either Eclipse, IntelliJ, or NetBeans are recommended for the IDEs.

Examples of how to setup and use this package are found in the examples folder under the src directory, and more detailed versions will be continuously added on the author's main blog www.imetricablog.com. 