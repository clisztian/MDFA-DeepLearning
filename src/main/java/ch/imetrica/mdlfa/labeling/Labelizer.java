package ch.imetrica.mdlfa.labeling;
import java.io.Serializable;

import ch.imetrica.mdfa.series.TargetSeries;
import ch.imetrica.mdfa.series.TimeSeries;



/**
 * 
 * Interface to create a labeled series for machine learning 
 * applications. 
 * 
 * There are three categories of labeling processes that this interface
 * will offer:
 * 
 * 1) Observational labeling: every time series observation is labeled
 * 2) Fixed Period labeling: every period (day, week, etc) is labeled
 * 3) Regime labeling: every regime change is labeled
 * 
 * 
 * 
 * 
 * @author Christian D. Blakely (clisztian@gmail.com)
 *
 */
public interface Labelizer extends Serializable {

	public enum LabelType {
		
		OBSERVATIONAL,  /* Every time series observation is labeled */
		FIXED_PERIOD,   /* Certain fixed-period length labeled */
		REGIME;		    /* Undetermined length of time labeled */
	}
	
	/**
	 * 
	 * With a target series provided, the routine takes either the 
	 * transformed price data (fractionally differenced data or the raw
	 * price data from the target series and constructs a labeled timeseries
	 * for each observation in the series. 
	 * 
	 * If the time series observation has no label, the value will be null
	 * at that timestamp observation 
	 * 
	 * 
	 * @param series A TargetSeries with original raw price data and 
	 * transformed data 
	 * @return
	 *   A TimeSeries with labels in the form of one-hot vectors for 
	 *   classifying observations
	 */
	public TimeSeries<double[]> labelTimeSeries(TargetSeries series);
	
    /**
     * Returns the LabelType of the given labelizer. 
     * 
     * @return
     *   A LabelType value of the given labeling rule
     */
	public LabelType getLabelType();
	
}
