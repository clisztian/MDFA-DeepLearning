package ch.imetrica.mdlfa.learning;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
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
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.ui.UIUtils;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.imetrica.mdfa.mdfa.MDFABase;
import ch.imetrica.mdlfa.dataiterator.MDFADataSetIterator;
import ch.imetrica.mdlfa.dataiterator.MDFARegressionDataSetIterator;
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
 * while the recurrent neural network learns to approximate
 * a signal produced by a symmetric filter. The quality
 * of the learning procedure depends on multiple factors, with 
 * the most influencial factor being the definition of
 * the feature extractors. 
 * 
 * 
 * 
 * @author lisztian
 *
 */
public class RecurrentMdfaRegression {

	private MDFAFeatureExtraction featureExtractor;
	private MDFARegressionDataSetIterator trainData;
	private MDFARegressionDataSetIterator testData;
	private MultiLayerNetwork mdfaLSTMNetwork;
	private DataNormalization normalizer;
	
	UIServer uiServer;
	
	private static final Logger log = LoggerFactory.getLogger(RecurrentMdfa.class);

	
	
	public RecurrentMdfaRegression() {
		
		featureExtractor = new MDFAFeatureExtraction(6);
	}
	
	public RecurrentMdfaRegression setTrainingTestData(String[] file, 
			                             TimeSeriesFile fileInfo, 
			                             int miniBatchSize,
			                             int totalTrainExamples,
			                             int totalTestExamples,
			                             int timeStepLength) throws Exception {
		
		setTrainData(new MDFARegressionDataSetIterator(file, fileInfo,
				featureExtractor.getFeatureExtractors(),
                miniBatchSize, 
                totalTrainExamples, 
                timeStepLength));
		
		setTestData(new MDFARegressionDataSetIterator(file, fileInfo,
				featureExtractor.getFeatureExtractors(),
                miniBatchSize, 
                totalTestExamples, 
                timeStepLength));
	
		return this;
	}
	
	public void normalizeData() {
		
    	normalizer = new NormalizerStandardize();
    	normalizer.fitLabel(true);
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
        builder.updater(updater);
        builder.gradientNormalization(GradientNormalization.ClipL2PerLayer);
        builder.gradientNormalizationThreshold(gradientNormThreshold);
		
        
        return builder.list();
	}
	
	
	public void buildNetworkLayers(int nHiddenLayers, int nHidden, NeuralNetConfiguration.ListBuilder listBuilder) {
		
		
		int numInput = featureExtractor.getNumberOfFeatures() + 1;
		int numOutput = 1;
		
		listBuilder.layer(0, new GravesLSTM.Builder()
	            .nIn(numInput).nOut(nHidden)
	            .activation(Activation.TANH)
	            .l2(0.0001)
	            .weightInit(WeightInit.XAVIER)
	            .build());
		
		for(int layerIndex = 0; layerIndex < nHiddenLayers; layerIndex++) {
			
			listBuilder.layer(layerIndex+1, new GravesLSTM.Builder()
					.nIn(nHidden).nOut(nHidden)
					.activation(Activation.TANH)
					.l2(0.0001)
					.weightInit(WeightInit.XAVIER)
					.build());	
		}
		
		listBuilder.layer(nHiddenLayers + 1, new RnnOutputLayer.Builder()
				.lossFunction(LossFunctions.LossFunction.MSE)
				.activation(Activation.IDENTITY)
				.l2(0.0001)
				.weightInit(WeightInit.XAVIER)
				.nIn(nHidden).nOut(numOutput)
				.build());
		
		listBuilder.pretrain(false).backprop(true);
		MultiLayerConfiguration conf = listBuilder.build();
		mdfaLSTMNetwork = new MultiLayerNetwork(conf);
		mdfaLSTMNetwork.init();
		mdfaLSTMNetwork.setListeners(new ScoreIterationListener(20));
        
	}
	
	public void setupUserInterface() {
		
		UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;
        mdfaLSTMNetwork.setListeners(new StatsListener(statsStorage, listenerFrequency));
        uiServer.attach(statsStorage);
		
	}
	
	public void train(int nEpochs) {
		
		String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
        	
        	mdfaLSTMNetwork.fit(trainData);

            testData.reset();
            trainData.reset();
        	
        	//Run regression evaluation on our single column input
            RegressionEvaluation evaluation = new RegressionEvaluation(1);
            
            while(testData.hasNext()) {
            	
            	DataSet testMe = testData.next();
                INDArray features = testMe.getFeatureMatrix();
                INDArray lables = testMe.getLabels();
                INDArray predicted = mdfaLSTMNetwork.output(features, false);
                evaluation.evalTimeSeries(lables, predicted);
            }
            System.out.println("Epoch: " + i + " " + evaluation.stats());
            
            testData.reset();
            trainData.reset();
        }

        log.info("----- Example Complete -----");
        System.out.println("----- Example Complete -----");
	}
	

	public void plotData() {
		
		
		DataSet trainSample = trainData.next();
		DataSet testSample = testData.next();

		mdfaLSTMNetwork.rnnTimeStep(trainSample.getFeatureMatrix());
        INDArray predicted = mdfaLSTMNetwork.rnnTimeStep(testSample.getFeatureMatrix());

//        //Revert data back to original values for plotting
//        normalizer.revert(trainSample);
//        normalizer.revert(testSample);
//        normalizer.revertLabels(predicted);

        INDArray testFeatures = testSample.getLabels();

        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, testFeatures,  "Test data");
        createSeries(c, predicted,  "Predicted test data");			
        
        plotDataset(c);
        
	}
	
	
	public void plotBatches(int nbatches) {
		
		trainData.reset();
		testData.reset();
		
		for(int i = 0; i < nbatches; i++) {
			if(trainData.hasNext()) {
				plotData();
			}
		}
	}
	
    private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Symmetric signal value";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        UIUtils.centerFrameOnScreen(f);
        f.setVisible(true);
    }
	
	
	public MDFARegressionDataSetIterator getTrainData() {
		return trainData;
	}

	public void setTrainData(MDFARegressionDataSetIterator trainData) {
		this.trainData = trainData;
	}

	public MDFARegressionDataSetIterator getTestData() {
		return testData;
	}

	public void setTestData(MDFARegressionDataSetIterator testData) {
		this.testData = testData;
	}


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
    			setNeuralNetConfiguration(seed, iterations, learningRate, gradientNormThreshold, 0, updater));

		//myNet.setupUserInterface();
    	myNet.train(nEpochs);
    
    	myNet.printPredicitions();
    	myNet.plotBatches(10);
    }
    
    public void printPredicitions() {
    	
    	trainData.reset();
    	
    	while(trainData.hasNext()) {
    		
    		DataSet trainSample = trainData.next();
    		INDArray predicted = mdfaLSTMNetwork.rnnTimeStep(trainSample.getFeatureMatrix());
    		INDArray output = trainSample.getLabels();
    		String[][] nextDates = trainData.getNextDates();
    		
//    		normalizer.revertLabels(predicted);
//    		normalizer.revertLabels(output);
    		
    		System.out.println(nextDates[0].length + " " + predicted.shape()[0] + " " + predicted.shape()[1] + " " + predicted.shape()[2]);
    		
    		int nRows = predicted.shape()[2];
    		for (int i = 0; i < nRows; i++) {
                System.out.println(nextDates[0][i] + ", " + output.getDouble(i) + " " + predicted.getDouble(i));
            }
    		System.out.println("Next set..\n");
    	}
    }
    
    
    private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, String name) {
        
    	int nRows = data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i, data.getDouble(i));
        }
        seriesCollection.addSeries(series);
    }
    
    
	
	    
	    
}

