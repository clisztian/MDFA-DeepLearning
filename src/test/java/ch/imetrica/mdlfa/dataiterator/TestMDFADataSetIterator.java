package ch.imetrica.mdlfa.dataiterator;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import ch.imetrica.mdfa.datafeeds.CsvFeed;
import ch.imetrica.mdfa.mdfa.MDFABase;
import ch.imetrica.mdfa.series.MultivariateFXSeries;
import ch.imetrica.mdlfa.util.TimeSeriesFile;

public class TestMDFADataSetIterator {

	final double eps = .00000001;
	
	@Test
	public void testMDFAIteratorInitialize() throws Exception {
		
		/* Create market feed */
		String[] dataFiles = new String[1];
		dataFiles[0] = "/home/lisztian/mdfaData/AAPL.daily.csv";

		TimeSeriesFile fileInfo = new TimeSeriesFile("yyyy-MM-dd", "Index", "Open");
		
				
		/* Create some MDFA sigEx processes */
		MDFABase[] anyMDFAs = new MDFABase[3];
		
		anyMDFAs[0] = (new MDFABase()).setLowpassCutoff(Math.PI/20.0)
				.setI1(1)
				.setHybridForecast(.01)
				.setSmooth(.3)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-2.0)
				.setLambda(2.0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(40);
		
		anyMDFAs[1] = (new MDFABase()).setLowpassCutoff(Math.PI/10.0)
				.setBandPassCutoff(Math.PI/15.0)
				.setSmooth(.1)
				.setSeriesLength(400)
				.setFilterLength(40);
		
		anyMDFAs[2] = (new MDFABase()).setLowpassCutoff(Math.PI/5.0)
                .setBandPassCutoff(Math.PI/10.0)
                .setSmooth(.1)
                .setSeriesLength(400)
                .setFilterLength(40);
		
				
		MDFADataSetIterator dataSet = new MDFADataSetIterator(dataFiles, fileInfo, anyMDFAs, 40, 100, 60);
		
		for(int i = 0; i < dataSet.getMyLabels().size(); i++) {
			
			assertEquals(dataSet.getFXSeries().getSignal(i).getDateTime(), 
					dataSet.getMyLabels().get(i).getDateTime());
		}
		
		System.out.println(dataSet.getMyLabels().last().getDateTime() + " " + 
		dataSet.getFXSeries().getLatestSignalEntry().getDateTime());
		
		
		assertEquals(dataSet.getMyLabels().size(), dataSet.getFXSeries().size());
		
		
	}
	
	
	@Test
	public void testIteratorNext() throws Exception {
		
		/* Create market feed */
		String[] dataFiles = new String[1];
		dataFiles[0] = "/home/lisztian/mdfaData/AAPL.daily.csv";

		TimeSeriesFile fileInfo = new TimeSeriesFile("yyyy-MM-dd", "Index", "Open");
		
				
		/* Create some MDFA sigEx processes */
		MDFABase[] anyMDFAs = new MDFABase[3];
		
		anyMDFAs[0] = (new MDFABase()).setLowpassCutoff(Math.PI/20.0)
				.setI1(1)
				.setHybridForecast(.01)
				.setSmooth(.3)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-2.0)
				.setLambda(2.0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(40);
		
		anyMDFAs[1] = (new MDFABase()).setLowpassCutoff(Math.PI/10.0)
				.setBandPassCutoff(Math.PI/15.0)
				.setSmooth(.1)
				.setSeriesLength(400)
				.setFilterLength(40);
		
		anyMDFAs[2] = (new MDFABase()).setLowpassCutoff(Math.PI/5.0)
                .setBandPassCutoff(Math.PI/10.0)
                .setSmooth(.1)
                .setSeriesLength(400)
                .setFilterLength(40);
		
		
		
		int batchSize = 40; 
		int totalExamples = 100; 
		int numTimeSteps = 40;
		MDFADataSetIterator dataSet = new MDFADataSetIterator(dataFiles, 
				                               fileInfo, anyMDFAs, batchSize,
				                               totalExamples, numTimeSteps);
		
		assertTrue(dataSet.hasNext());
		
		for(int i = 0; i < 3; i++) {
		
			DataSet data = dataSet.next();
		}
        assertEquals(3, dataSet.getNumberOfFeatures());
        assertEquals(120, dataSet.cursor());

        
        dataSet.reset();
    	DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);              
        dataSet.reset();
        dataSet.setPreProcessor(normalizer);
        
        assertEquals(0, dataSet.cursor());
	}
	
	@Test
	public void testLabels() throws Exception {
		
		
		/* Create market feed */
		String[] dataFiles = new String[1];
		dataFiles[0] = "src/test/resources/testSeries1.csv";

		TimeSeriesFile fileInfo = new TimeSeriesFile("yyyy-MM-dd", "Index", "Open");
		
				
		/* Create some MDFA sigEx processes */
		MDFABase[] anyMDFAs = new MDFABase[3];
		
		anyMDFAs[0] = (new MDFABase()).setLowpassCutoff(Math.PI/4.0)
				.setI1(1)
				.setHybridForecast(.01)
				.setSmooth(.3)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-2.0)
				.setLambda(2.0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
		
		anyMDFAs[1] = (new MDFABase()).setLowpassCutoff(Math.PI/10.0)
				.setBandPassCutoff(Math.PI/15.0)
				.setSmooth(.1)
				.setSeriesLength(400)
				.setFilterLength(5);
		
		anyMDFAs[2] = (new MDFABase()).setLowpassCutoff(Math.PI/5.0)
                .setBandPassCutoff(Math.PI/10.0)
                .setSmooth(.1)
                .setSeriesLength(400)
                .setFilterLength(5);
		
		
		
		int batchSize = 1; 
		int totalExamples = 4; 
		int numTimeSteps = 100;
		MDFADataSetIterator dataSet = new MDFADataSetIterator(dataFiles, 
				                               fileInfo, anyMDFAs, batchSize,
				                               totalExamples, numTimeSteps);
	
		//40 filter length burnin + 40 symmetric filter
		assertEquals(320,dataSet.getFXSeries().size());
		assertTrue(dataSet.hasNext());
		
		DataSet set = dataSet.next();
		INDArray labels = set.getLabels();
		
		for(int i = 0; i < 100; i++) {
			assertEquals(1.0, labels.getDouble(0, 0, i), eps);
			assertEquals(0.0, labels.getDouble(0, 1, i), eps);
		}
		
		assertTrue(dataSet.hasNext());
		set = dataSet.next();
		labels = set.getLabels();
		
		for(int i = 0; i < 95; i++) {
			assertEquals(1.0, labels.getDouble(0, 0, i), eps);
			assertEquals(0.0, labels.getDouble(0, 1, i), eps);
		}
		
		assertEquals(2,dataSet.cursor());
		
		for(int i = 0; i < 100; i++) {			
			assertTrue(dataSet.hasNext());
			set = dataSet.next();
		}
		
		assertEquals(102,dataSet.cursor());

		for(int i = 0; i < 77; i++) {		
			assertTrue(dataSet.hasNext());
			set = dataSet.next();
		}
		
		labels = set.getLabels();
		for(int i = 0; i < 100; i++) {			
			assertEquals(0.0, labels.getDouble(0, 0, i), eps);
			assertEquals(1.0, labels.getDouble(0, 1, i), eps);
		}
		
	}
	


	

}
