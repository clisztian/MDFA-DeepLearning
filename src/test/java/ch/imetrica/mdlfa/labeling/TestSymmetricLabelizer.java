package ch.imetrica.mdlfa.labeling;

import org.junit.Test;

import ch.imetrica.mdfa.series.TargetSeries;
import ch.imetrica.mdfa.series.TimeSeries;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

public class TestSymmetricLabelizer {

	final double eps = .00000001;
	
	@Test
	public void testSymmetricLabelizer() {
		
		
		/* Create TargetSeries */
		DateTime dt = new DateTime(2018,4,4,15,24);
		TargetSeries target = new TargetSeries(new TimeSeries<Double>(), 1.0, false);
		
		DateTimeFormatter dtfOut = DateTimeFormat.forPattern("yyyy-MM-dd");
		
		for(int i = 0; i < 200; i++) {			
			
			target.addValue(dt.toString(), (double)i);
			dt = dt.plusDays(1);

		}
		for(int i = 0; i < 200; i++) {			
			
			target.addValue(dt.toString(), 200.0 - (double)i);
			dt = dt.plusDays(1);

		}
		
		
		SymmetricLabelizer labeler = new SymmetricLabelizer(Math.PI/4.0, 20)
											.computeSymmetricFilter();
		
		
		TimeSeries<double[]> label = labeler.labelTimeSeries(target);
		
		assertNull(label.get(5).getValue());
		assertNull(label.get(10).getValue());
		assertNull(label.get(14).getValue());
		assertNull(label.get(19).getValue());
		
		assertNull(label.get(381).getValue());
		assertNull(label.get(382).getValue());
		assertNull(label.get(390).getValue());
		assertNull(label.get(395).getValue());

		assertEquals(1.0, label.get(20).getValue()[0], eps);
		assertEquals(0.0, label.get(20).getValue()[1], eps);
		assertEquals(1.0, label.get(180).getValue()[0], eps);
		assertEquals(0.0, label.get(180).getValue()[1], eps);
		
		assertEquals(0.0, label.get(220).getValue()[0], eps);
		assertEquals(1.0, label.get(220).getValue()[1], eps);		
		assertEquals(0.0, label.get(220).getValue()[0], eps);
		assertEquals(1.0, label.get(220).getValue()[1], eps);
		
		assertEquals(0.0, label.get(380).getValue()[0], eps);
		assertEquals(1.0, label.get(380).getValue()[1], eps);		
		assertEquals(0.0, label.get(380).getValue()[0], eps);
		assertEquals(1.0, label.get(380).getValue()[1], eps);
		
//		for(int i = 0; i < label.size(); i++) {
//			
//			if(label.get(i).getValue() != null) {
//				System.out.println(i + ", " + label.get(i).getValue()[0] + " " + 
//						label.get(i).getValue()[1]);
//			}
//		}
	}

	
	public void printSimpleTimeSeries() {
		
		DateTime dt = new DateTime(2018,4,4,15,24);
		TargetSeries target = new TargetSeries(new TimeSeries<Double>(), 1.0, false);
		
		DateTimeFormatter dtfOut = DateTimeFormat.forPattern("yyyy-MM-dd");
		
		int maxPeak = 1000;
		for(int i = 0; i < maxPeak; i++) {			
			
			target.addValue(dt.toString(), (double)i);
			dt = dt.plusDays(1);
			
			System.out.println(dtfOut.print(dt) + ", " + (double)i);
		}
		for(int i = 0; i < maxPeak; i++) {			
			
			target.addValue(dt.toString(), maxPeak - (double)i);
			dt = dt.plusDays(1);

		    System.out.println(dtfOut.print(dt) + ", " + (maxPeak - (double)i));	
		}
		
	}
}
