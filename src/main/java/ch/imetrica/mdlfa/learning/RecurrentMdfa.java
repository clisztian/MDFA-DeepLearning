package ch.imetrica.mdlfa.learning;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.imetrica.mdfa.mdfa.MDFABase;
import ch.imetrica.mdlfa.dataiterator.MDFADataSetIterator;
import ch.imetrica.mdlfa.features.MDFAFeatureExtraction;
import ch.imetrica.mdlfa.util.TimeSeriesFile;

/**
 * 
 * The recurrent MDfA approach is a method for
 * extracting signals in financial time series
 * using a Recurrent neural network coupled with 
 * the multisignal DFA. 
 * 
 * The multisignal DFA provides a real-time feature
 * engineering and extraction interface on a time series
 * while the recurrent neural network learns the  
 * 
 * 
 * 
 * @author lisztian
 *
 */
public class RecurrentMdfa {

	private MDFAFeatureExtraction featureExtractor;
	private MDFADataSetIterator trainData;
	private MDFADataSetIterator testData;
	private MultiLayerNetwork mdfaLSTMNetwork;
	
	private static final Logger log = LoggerFactory.getLogger(RecurrentMdfa.class);

	
	
	public RecurrentMdfa() {
		
		featureExtractor = new MDFAFeatureExtraction(6);
	}
	
	public RecurrentMdfa setTrainingTestData(String[] file, 
			                             TimeSeriesFile fileInfo, 
			                             int miniBatchSize,
			                             int totalTrainExamples,
			                             int totalTestExamples,
			                             int timeStepLength) throws Exception {
		
		setTrainData(new MDFADataSetIterator(file, fileInfo,
				featureExtractor.getFeatureExtractors(),
                miniBatchSize, 
                totalTrainExamples, 
                timeStepLength));
		
		setTestData(new MDFADataSetIterator(file, fileInfo,
				featureExtractor.getFeatureExtractors(),
                miniBatchSize, 
                totalTestExamples, 
                timeStepLength));
	
		return this;
	}
	
	public void normalizeData() {
		
    	DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              
        trainData.reset();
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);
	}
	
	
	public static NeuralNetConfiguration.ListBuilder setNeuralNetConfiguration(int seed, int iterations, 
				double learningRate, double gradientNormThreshold, double biasInit, IUpdater updater) {
		
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.seed(seed);
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        builder.biasInit(biasInit);
        builder.updater(updater);
        builder.gradientNormalization(GradientNormalization.ClipL2PerLayer);
        builder.gradientNormalizationThreshold(gradientNormThreshold);
		
        
        return builder.list();
	}
	
	public void buildNetworkLayers(int nHiddenLayers, int nHidden, NeuralNetConfiguration.ListBuilder listBuilder) {
		
		
		int numInput = featureExtractor.getNumberOfFeatures();
		
		listBuilder.layer(0, new GravesLSTM.Builder()
	            .nIn(numInput).nOut(nHidden)
	            .activation(Activation.SIGMOID)
	            .l2(0.0001)
	            .weightInit(WeightInit.XAVIER)
	            .build());
		
		for(int layerIndex = 0; layerIndex < nHiddenLayers; layerIndex++) {
			
			listBuilder.layer(layerIndex+1, new GravesLSTM.Builder()
					.nIn(nHidden).nOut(nHidden)
					.activation(Activation.SIGMOID)
					.l2(0.0001)
					.weightInit(WeightInit.XAVIER)
					.build());	
		}
		
		listBuilder.layer(nHiddenLayers + 1, new RnnOutputLayer.Builder()
				.lossFunction(LossFunctions.LossFunction.MCXENT)
				.activation(Activation.SOFTMAX)
				.l2(0.0001)
				.weightInit(WeightInit.XAVIER)
				.nIn(nHidden).nOut(2)
				.build());
		
		listBuilder.pretrain(false).backprop(true);
		MultiLayerConfiguration conf = listBuilder.build();
		mdfaLSTMNetwork = new MultiLayerNetwork(conf);
		mdfaLSTMNetwork.init();
		mdfaLSTMNetwork.setListeners(new ScoreIterationListener(20));
        
	}
	

	

	public void train(int nEpochs) {
		
		String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
        	
        	mdfaLSTMNetwork.fit(trainData);

            //Evaluate on the test set:
            Evaluation evaluation = mdfaLSTMNetwork.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
            System.out.println(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testData.reset();
            trainData.reset();
        }

        log.info("----- Example Complete -----");
        System.out.println("----- Example Complete -----");
		
		
	}
	

	public MDFADataSetIterator getTrainData() {
		return trainData;
	}

	public void setTrainData(MDFADataSetIterator trainData) {
		this.trainData = trainData;
	}

	public MDFADataSetIterator getTestData() {
		return testData;
	}

	public void setTestData(MDFADataSetIterator testData) {
		this.testData = testData;
	}


    public static void main(String[] args) throws Exception {
	    	
	    
    	String[] dataFiles = new String[1];
		dataFiles[0] = "src/test/resources/testSeries1.csv";
		TimeSeriesFile fileInfo = new TimeSeriesFile("yyyy-MM-dd", "Index", "Open");
    	
    	int miniBatchSize = 40;
    	int totalTrainExamples = 500;
    	int totalTestExamples = 200;
		int timeStepLength = 40;
		
		int nHiddenLayers = 1;
		int nHidden = 68;
		
		int nEpochs = 40;
		int seed = 123;
		int iterations = 40;
		double learningRate = .01;
		double gradientNormThreshold = 10.0;
		
		double biasInit = 1.0;
		IUpdater updater = new Nesterovs();
		
		
    	RecurrentMdfa myNet = new RecurrentMdfa();
    	myNet.setTrainingTestData(dataFiles, fileInfo, 
    			                  miniBatchSize, totalTrainExamples, 
    			                  totalTestExamples, timeStepLength);
    	
    	myNet.normalizeData();
    	
		myNet.buildNetworkLayers(nHiddenLayers, nHidden, 
    			setNeuralNetConfiguration(seed, iterations, learningRate, gradientNormThreshold, biasInit, updater));
    	
	    
    	myNet.train(nEpochs);
    
    	
    }
	
	    
	    
}
