package ch.imetrica.mdlfa.learning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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

	
	    //Random number generator seed, for reproducibility
	    public static final int seed = 12345;
	    //Number of iterations per minibatch
	    public static final int iterations = 35;
	    //Number of epochs (full passes of the data)
	    public static final int nEpochs = 80;
	    //Number of data points
	    public static final int nSamples = 25;
	    //Network learning rate
	    public static final double learningRate = 0.0001;

	    public static void main(String[] args) {
	        //Generate the training data
	        DataSet trainingData = getTrainingData();
	        trainingData.shuffle();
	        System.out.println(trainingData);
	        System.out.println();
	        DataSet testData = getTestData();
	        System.out.println(testData);

	        //Create the network
	        int numInput = 1;
	        int numOutputs = 1;
	        int nHidden = 30;

	        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
	            builder.seed(seed);
	            builder.iterations(iterations);
	            builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
	            builder.learningRate(learningRate);
	            builder.regularization(true);
	            builder.updater(Updater.RMSPROP);
	            builder.gradientNormalization(GradientNormalization.ClipL2PerLayer);
	            builder.gradientNormalizationThreshold(0.00001);

	        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

	        listBuilder.layer(0, new GravesLSTM.Builder()
	        		            .nIn(numInput).nOut(nHidden)
	        		            .activation(Activation.TANH)
	        		            .l2(0.0001)
	        		            .weightInit(WeightInit.XAVIER)
	        		            .build());
	        listBuilder.layer(1, new GravesLSTM.Builder()
	        					.nIn(nHidden).nOut(nHidden)
	        					.activation(Activation.TANH)
	        					.l2(0.0001)
	        					.weightInit(WeightInit.XAVIER)
	        					.build());
	        listBuilder.layer(2, new RnnOutputLayer.Builder()
	        					.lossFunction(LossFunctions.LossFunction.MSE)
	        					.activation(Activation.IDENTITY)
	        					.l2(0.0001)
	        					.weightInit(WeightInit.XAVIER)
	        					.nIn(nHidden).nOut(numOutputs)
	        					.build());
	        listBuilder.pretrain(false).backprop(true);

	        MultiLayerConfiguration conf = listBuilder.build();
	        MultiLayerNetwork net = new MultiLayerNetwork(conf);
	        net.init();
	        //net.setListeners(new HistogramIterationListener(1));

	        INDArray output;
	        //Train the network on the full data set
	        for( int i = 0; i < nEpochs; i++ ) {
	            // train the model
	            net.fit(trainingData);
	            output = net.rnnTimeStep(trainingData.getFeatureMatrix());
	            System.out.println(output);
	            net.rnnClearPreviousState();
	        }

	        System.out.println("Result on training data: ");
	        System.out.println(net.rnnTimeStep(trainingData.getFeatureMatrix()));
	        System.out.println(trainingData.getFeatureMatrix());

	        System.out.println();

	        System.out.println("Result on test data: ");
	        System.out.println(net.rnnTimeStep(testData.getFeatureMatrix()));
	        System.out.println(testData.getFeatureMatrix());


	    }

	    /*
	        Generate the training data. The sequence to train is out = 1, 2, 3, ..., 100.
	        This corresponds to having as input the sequence seq = 0, 1, 2, ..., 99, so for this
	        training data set the input attribute sequence is seq and the class/target attribute is out.
	        The RNN should then be able to predict 101, 102, ... given the input 100, 101, ...
	        That is: the last output is the next input.
	     */
	    private static DataSet getTrainingData() {
	        double[] seq = new double[nSamples];
	        double[] out = new double[nSamples];
	        // seq is 0, 1, 2, 3, .., nSamples-1
	        for (int i= 0; i < nSamples; i++) {
	            if(i == 0)
	                seq[i] = 0;
	            else
	                seq[i] = seq[i-1] + 1;
	        }
	        // out is the next seq input
	        for(int i = 0; i < nSamples; i++) {
	            if (i != (nSamples - 1))
	                out[i] = seq[i + 1];
	            else
	                out[i] = seq[i] + 1;
	        }
	        // Scaling to [0, 1] based on the training output
	        int min = 1;
	        int max = nSamples;
	        for(int i = 0; i < nSamples; i++) {
	            seq[i] = (seq[i] - min)/(max - min);
	            out[i] = (out[i] - min)/(max - min);
	        }

	        INDArray seqNDArray = Nd4j.create(seq, new int[]{nSamples,1});
	        INDArray inputNDArray = Nd4j.zeros(1,1,nSamples);
	        inputNDArray.putRow(0, seqNDArray.transpose());

	        INDArray outNDArray = Nd4j.create(out, new int[]{nSamples,1});
	        INDArray outputNDArray = Nd4j.zeros(1,1,nSamples);
	        outputNDArray.putRow(0, outNDArray.transpose());

	        DataSet dataSet = new DataSet(inputNDArray, outputNDArray);
	        return dataSet;
	    }

	    private static DataSet getTestData() {
	        int testLength = nSamples;
	        double[] seq = new double[testLength];
	        double[] out = new double[testLength];
	        for (int i= 0; i < testLength; i++) {
	            if(i == 0)
	                seq[i] = 25;
	            else
	                seq[i] = seq[i-1] + 1;
	        }
	        // out is the next seq input
	        for(int i = 0; i < testLength; i++) {
	            if (i != (testLength - 1))
	                out[i] = seq[i + 1];
	            else
	                out[i] = seq[i] + 1;
	        }

	        // Scaling to [0, 1] using same normalization as training data's
	        int min = 1;
	        int max = nSamples;
	        for(int i = 0; i < nSamples; i++) {
	            seq[i] = (seq[i] - min)/(max - min);
	            out[i] = (out[i] - min)/(max - min);
	        }

	        INDArray seqNDArray = Nd4j.create(seq, new int[]{testLength,1});
	        INDArray inputNDArray = Nd4j.zeros(1,1,testLength);
	        inputNDArray.putColumn(0, seqNDArray);

	        INDArray outNDArray = Nd4j.create(out, new int[]{testLength,1});
	        INDArray outputNDArray = Nd4j.zeros(1,1,testLength);
	        outputNDArray.putColumn(0, outNDArray);

	        DataSet dataSet = new DataSet(inputNDArray, outputNDArray);
	        return dataSet;
	    }
	
	
	
}
