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
 * 
 *  
 *  
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
		int nHidden = 300;
		
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
    	
    	//myNet.normalizeData();
    	
		myNet.buildNetworkLayers(nHiddenLayers, nHidden, 
				RecurrentMdfaRegression.setNeuralNetConfiguration(seed, iterations, learningRate, gradientNormThreshold, 0, updater));

		//myNet.setupUserInterface();
    	myNet.train(nEpochs);
    
    	myNet.printPredicitions();
    	myNet.plotBatches(10);
    }
}





