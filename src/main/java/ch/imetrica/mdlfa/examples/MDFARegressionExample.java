package ch.imetrica.mdlfa.examples;

import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.imetrica.mdlfa.learning.RecurrentMdfaRegression;
import ch.imetrica.mdlfa.util.TimeSeriesFile;


/**
 * This example demonstrates the use of the MDFA-LSTM network for prediction of a 
 * target signal on AAPL daily open price (found in the resources folder). 
 * In this case, for each daily observation the network should output a univariate 
 * signal value that is relatively "close" in the MSE to a given target signal 
 * at each observation. This target signal is built using a symmetric filter at a certain 
 * frequency, extracting low frequency components that are ideal for entering and exiting 
 * market positions on a frequency of several days, depending on the length of the local 
 * trends/cycles.
 * 
 * For the labeling, since we are interested in a target signal, we utilize the SymmetricLabelizer
 * found in ch.imetrica.mdlfa.labeling, which takes a target time series (AAPL) that has been 
 * transformed appropriately, and constructs the target signal for a given frequency using
 * the symmetric filter of a fixed length which we'll call LN = 2*L-1 for some L. 
 * Since the symmetric filter uses information L-1 steps into the future and L steps in 
 * the present past, all values except the first L-1 and final L-1 the target time series can be used. 
 * These values will be set as null in the TimeSeriesEntry objects. 
 * 
 * The frequency used in extracting the target signal will be taken as the frequency corresponding
 * to the first MDFABase object in the MDFA feature extraction set. The target signal will be computed 
 * over the entire target time series in the initiation using the SymmetricLabelizer, and then the 
 * train and test collections will be extraction from this target series and signal pairing in the 
 * DataSetIterator, where the features will be the K number of MDFA features plus time series data, 
 * and the label will be the output of the target signal, for each observation. 
 * 
 * In this example, the network is built as a multilayered LSTM with 212 hidden nodes on each 
 * of the 2 hidden layers. The input at each time step will be K+1 feature extracted values plus 
 * target series, and the output of the network in this case will be simply be an
 * approximation to the target signal, and thus in the form of a recurrent regression formulation. 
 * The basic RNN settings in this case require tanh activation functions and an output loss function 
 * given by the MSE, with an output activation in the form of an identity function. 
 * Other parameters into the LSTM include a Nesterov updating rule which assigns a learning rate 
 * parameter and a momentum for an updating rule during the optimizing. These values are left to 
 * default values. Depending on the normalization of the data, the amount of data, and quality of
 * the feature extractors, these LSTM hyperparamter values could require adjustment. Finally, we 
 * include an L2 regularization rule on the weights and a gradientNormalizer/clipping rule 
 * in the case of growing gradients in the backpropagation in time algorithm included in dl4j. 
 * These are typically standard practices in dealing with LSTMs. Other options including dropout
 * and mean-variance weight normalization have been left out of this experiment.  
 *  
 * Along with the output, we will also be outputing the original DateTime stamp in String format
 * for comparison with the original target signal. 
 *  
 */

public class MDFARegressionExample {
	
    private static final Logger LOGGER = LoggerFactory.getLogger(MDFARegressionExample.class);
    
    public static void main(String[] args) throws Exception {
	    	
	    
    	String[] dataFiles = new String[1];
		dataFiles[0] = "src/main/resources/stockDailyData/AAPL.daily.csv";
		TimeSeriesFile fileInfo = new TimeSeriesFile("yyyy-MM-dd", "Index", "Open");
    	
    	int miniBatchSize = 100;
    	int totalTrainExamples = 1500;
    	int totalTestExamples = 300;
		int timeStepLength = 60;
		
		int nHiddenLayers = 1;
		int nHidden = 212;
		
		int nEpochs = 400;
		int seed = 123;
		int iterations = 40;
		double learningRate = .001;
		double gradientNormThreshold = 10.0;
		
		IUpdater updater = new Nesterovs(learningRate, .4);
		
		
		RecurrentMdfaRegression myNet = new RecurrentMdfaRegression();
    	myNet.setTrainingTestData(dataFiles, fileInfo, 
    			                  miniBatchSize, totalTrainExamples, 
    			                  totalTestExamples, timeStepLength);
    	
    	
		myNet.buildNetworkLayers(nHiddenLayers, nHidden, 
				RecurrentMdfaRegression.setNeuralNetConfiguration(seed, iterations, learningRate, gradientNormThreshold, 0, updater));

		myNet.setupUserInterface();
    	myNet.train(nEpochs);
    
    	myNet.printPredicitions();
    	myNet.plotBatches(10);
    }
}





