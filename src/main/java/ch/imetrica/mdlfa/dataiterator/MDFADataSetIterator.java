package ch.imetrica.mdlfa.dataiterator;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import ch.imetrica.mdfa.datafeeds.CsvFeed;
import ch.imetrica.mdfa.mdfa.MDFABase;
import ch.imetrica.mdfa.series.MultivariateFXSeries;
import ch.imetrica.mdfa.series.TargetSeries;
import ch.imetrica.mdfa.series.TimeSeries;
import ch.imetrica.mdfa.series.TimeSeriesEntry;
import ch.imetrica.mdlfa.labeling.SymmetricLabelizer;
import ch.imetrica.mdlfa.util.TimeSeriesFile;


/**
 * DataSetIterator for financial time series. 
 * The iterator accepts a path to a director of .csv
 * files containing the file(s) of financial time 
 * series.
 * 
 * 
 * @author lisztian
 *
 */
public class MDFADataSetIterator implements DataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private MultivariateFXSeries fxSeries;
	
	private int miniBatchSize;
	private int currentFileListIndex = 0;
	private int totalExamples = 0;
	private String[] datasetInputPaths;
	private TimeSeries<int[]> myLabels;
	
	private int timeStepLength;
	private CsvFeed marketFeed; //;  = new CsvFeed(dataFiles, "Index", "Open");
	
	private int numberOfFeatures;
	private TimeSeriesFile fileInfo;
	
	private MDFABase[] anyMDFAs;

	private DataSetPreProcessor myPreprocessor;
	
	public MDFADataSetIterator(String[] dataInputPaths,
			                   TimeSeriesFile fileInfo,
			                   MDFABase[] anyMDFAs,
			                   int miniBatchSize, 
			                   int totalExamples, 
			                   int timeStepLength) throws Exception {

		this.fileInfo = fileInfo;
		this.datasetInputPaths = dataInputPaths;	
		
		this.setMiniBatchSize(miniBatchSize);
		this.setTotalExamples(totalExamples);
		this.setAnyMDFAs(anyMDFAs);
		this.setNumberOfFeatures(anyMDFAs.length);
		this.setTimeStepLength(timeStepLength);
		
		CsvFeed marketFeed = new CsvFeed(datasetInputPaths, 
				                         fileInfo.getDateTimeIndexName(),
				                         fileInfo.getPriceColumnName());
		
		fxSeries = new MultivariateFXSeries(anyMDFAs, fileInfo.getDateTimeFormat());
		fxSeries.addSeries(new TargetSeries(1.0, true, "TargetSeries"));
		fxSeries.setWhiteNoisePrefilters(36);
		
		
		int initializeLength = anyMDFAs[0].getSeriesLength();
		
        for(int i = 0; i < initializeLength; i++) {
			
			TimeSeriesEntry<double[]> observation = marketFeed.getNextMultivariateObservation();
			fxSeries.addValue(observation.getDateTime(), observation.getValue());
		}
        
        //fxSeries.computeAllFilterCoefficients();	
        fxSeries.chopFirstObservations(fxSeries.getSeries(0).getCoefficientSet(0).length);	
        
        int symmetricLength = fxSeries.getSeries(0).getCoefficientSet(0).length;

		while(true) {
			
			TimeSeriesEntry<double[]> observation = marketFeed.getNextMultivariateObservation();
			
			if(observation == null) break;
			
			fxSeries.addValue(observation.getDateTime(), observation.getValue());
			
		}
		
		SymmetricLabelizer labeler = new SymmetricLabelizer(anyMDFAs[0].getLowPassCutoff(), 
															symmetricLength)
				                                           .computeSymmetricFilter();
		
		setMyLabels(labeler.labelTimeSeriesInt(fxSeries.getSeries(0).getTargetSeries()));
		fxSeries.chopFirstObservations(symmetricLength);
	
		for(int i = 0; i < symmetricLength; i++) {
			myLabels.remove(0);
		}
		
	}
	
	
	
	@Override
	public boolean hasNext() {		
		return currentFileListIndex + miniBatchSize + timeStepLength < 
				 (fxSeries.size() - anyMDFAs[0].getFilterLength()) ;
	}

	@Override
	public DataSet next() {
		
		return next(miniBatchSize);
	}

	@Override
	public DataSet next(int miniBatchSize) {
		
		this.miniBatchSize = miniBatchSize;
		int index;
		INDArray input = Nd4j.zeros(new int[]{ miniBatchSize, numberOfFeatures, timeStepLength } );
		INDArray inputMask = Nd4j.ones( new int[]{ miniBatchSize, timeStepLength } );
		INDArray labels = Nd4j.zeros(new int[]{ miniBatchSize, 2, timeStepLength } );
		INDArray labelsMask = Nd4j.zeros(new int[]{ miniBatchSize, timeStepLength } ); // labels are always used
		
		
		for(int i = 0; i < miniBatchSize; i++) {
			
		  for(int t = 0; t < timeStepLength; t++) {
			  
			  index = currentFileListIndex + i;
			  double[] features = fxSeries.getSignalValue(index + t);
			  int[] label = myLabels.get(index + t).getValue();
		  
			  if(label != null) {
				  
				  for(int k = 0; k < features.length; k++) {
					  input.putScalar(i, k, t, features[k]);
				  }
				  labels.putScalar(new int[]{ i, 0, t }, label[0]);
				  labels.putScalar(new int[]{ i, 1, t }, label[1]);
				  
				  labelsMask.putScalar(new int[]{ i, t }, 0);
				  
				  if(t == timeStepLength-1) {
					  
					  System.out.println(features[0] + ", " + features[1] + ", " + features[2]   
							  + ", " + label[0] + ", " + label[1]);    
							 
				  }
				  
			  }
			  

		  }
		  labelsMask.putScalar(new int[]{ i, timeStepLength -1 }, 1);
		}
		
		
		this.currentFileListIndex += miniBatchSize;
		
		
		return new DataSet( input, labels, inputMask, labelsMask );
	}

	@Override
	public int totalExamples() {
		return this.totalExamples;
	}

	@Override
	public int inputColumns() {
		return numberOfFeatures;
	}

	@Override
	public int totalOutcomes() {
		return 2;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		this.currentFileListIndex = 0;
	}

	@Override
	public int batch() {
		return miniBatchSize;
	}

	@Override
	public int cursor() {
		return this.currentFileListIndex;
	}

	@Override
	public int numExamples() {
		return totalExamples;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		this.setMyPreprocessor(preProcessor); 
		
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return myPreprocessor;
	}

	@Override
	public List<String> getLabels() {
		return null;
	}



	public int getMiniBatchSize() {
		return miniBatchSize;
	}



	public void setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
	}



	public int getTimeStepLength() {
		return timeStepLength;
	}



	public void setTimeStepLength(int timeStepLength) {
		this.timeStepLength = timeStepLength;
	}



	public TimeSeriesFile getFileInfo() {
		return fileInfo;
	}



	public void setFileInfo(TimeSeriesFile fileInfo) {
		this.fileInfo = fileInfo;
	}



	public MDFABase[] getAnyMDFAs() {
		return anyMDFAs;
	}



	public void setAnyMDFAs(MDFABase[] anyMDFAs) {
		this.anyMDFAs = anyMDFAs;
	}



	public int getNumberOfFeatures() {
		return numberOfFeatures;
	}



	public void setNumberOfFeatures(int numberOfFeatures) {
		this.numberOfFeatures = numberOfFeatures;
	}



	public int getTotalExamples() {
		return totalExamples;
	}



	public void setTotalExamples(int totalExamples) {
		this.totalExamples = totalExamples;
	}



	public TimeSeries<int[]> getMyLabels() {
		return myLabels;
	}

	public MultivariateFXSeries getFXSeries() {
		return fxSeries;
	}

	public void setMyLabels(TimeSeries<int[]> myLabels) {
		this.myLabels = myLabels;
	}



	public DataSetPreProcessor getMyPreprocessor() {
		return myPreprocessor;
	}



	public void setMyPreprocessor(DataSetPreProcessor myPreprocessor) {
		this.myPreprocessor = myPreprocessor;
	}

	
	
}
