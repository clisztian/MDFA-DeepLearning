package ch.imetrica.mdlfa.dataiterator;

import java.util.List;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import ch.imetrica.mdfa.datafeeds.CsvFeed;
import ch.imetrica.mdfa.mdfa.MDFABase;
import ch.imetrica.mdfa.series.MultivariateFXSeries;
import ch.imetrica.mdfa.series.TargetSeries;
import ch.imetrica.mdfa.series.TimeSeries;
import ch.imetrica.mdfa.series.TimeSeriesEntry;
import ch.imetrica.mdlfa.labeling.SymmetricLabelizer;
import ch.imetrica.mdlfa.util.TimeSeriesFile;

/**
 * This MDFA dataSet iterator computes a time-series regression
 * on the symmetric signal which are the labels and the features
 * are the target series and M different features of the time
 * series defined in MDFABase 
 * 
 * @author lisztian
 *
 */
public class MDFARegressionDataSetIterator implements DataSetIterator {

	
	private MultivariateFXSeries fxSeries;
	
	private int miniBatchSize;
	private int currentFileListIndex = 0;
	private int totalExamples = 0;
	private String[] datasetInputPaths;
	private TimeSeries<Double> myLabels;
	
	private int timeStepLength;
	private CsvFeed marketFeed; //;  = new CsvFeed(dataFiles, "Index", "Open");
	
	private int numberOfFeatures;
	private TimeSeriesFile fileInfo;
	
	private MDFABase[] anyMDFAs;

	private DataSetPreProcessor myPreprocessor;
	
	
	public MDFARegressionDataSetIterator(String[] dataInputPaths,
            TimeSeriesFile fileInfo,
            MDFABase[] anyMDFAs,
            int miniBatchSize, 
            int totalExamples, 
            int timeStepLength) {
		
		
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
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public DataSet next() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DataSet next(int num) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int totalExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int inputColumns() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int batch() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int cursor() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int numExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return null;
	}



	public int getMiniBatchSize() {
		return miniBatchSize;
	}



	public void setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
	}



	public int getCurrentFileListIndex() {
		return currentFileListIndex;
	}



	public void setCurrentFileListIndex(int currentFileListIndex) {
		this.currentFileListIndex = currentFileListIndex;
	}



	public int getTotalExamples() {
		return totalExamples;
	}



	public void setTotalExamples(int totalExamples) {
		this.totalExamples = totalExamples;
	}



	public int getTimeStepLength() {
		return timeStepLength;
	}



	public void setTimeStepLength(int timeStepLength) {
		this.timeStepLength = timeStepLength;
	}



	public int getNumberOfFeatures() {
		return numberOfFeatures;
	}



	public void setNumberOfFeatures(int numberOfFeatures) {
		this.numberOfFeatures = numberOfFeatures;
	}



	public MDFABase[] getAnyMDFAs() {
		return anyMDFAs;
	}



	public void setAnyMDFAs(MDFABase[] anyMDFAs) {
		this.anyMDFAs = anyMDFAs;
	}
	
	
	

}
