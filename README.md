# MDFA-DeepLearning
MDFA-DeepLearning is a library for building machine learning applications on time series (or sequential) data, with a
heavy emphasis on large data sets, where 

The MDFA-DeepLearning approach differs from most machine learning methods in time series analysis in that an emphasis 
on real-time feature extraction is utilized where the features extractors are build using the multivariate direct filter
approach (MDFA). The motivation behind this coupling of MDFA with machine learning is that, while many time series
decomposition methodologies exist (from empirical mode decomposition to stochastic component analysis methods), real-time


MDFA-DeepLearning requires both the MDFA-Toolkit package for constructing the time series feature extractors and 
the dl4j package for the deep recurrent neural network constructors. 

The back-end used for the 
Deep learning in financial time series using the multivariate direct filter approach (MDFA)

For the build and package management, we use a Gradle wrapper, and either Eclipse, IntelliJ, or NetBeans are recommended for
the IDEs.
