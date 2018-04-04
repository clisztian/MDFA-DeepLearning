package ch.imetrica.mdlfa.labeling;

import ch.imetrica.mdfa.series.TargetSeries;
import ch.imetrica.mdfa.series.TimeSeries;


/**
 * 
 * This regime labelizer labels a time series sequence
 * and labels the final point with up to three difference
 * types of "regimes" that are characterized by the 
 * variance ratios of the original raw time series.
 * 
 * The three regimes are: 
 * 
 * 1) Trend
 * 2) Random walk
 * 3) Mean-reverting
 * 
 * 
 * @author lisztian
 *
 */
public class RegimeLabelizer implements Labelizer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	LabelType labelType = LabelType.REGIME;
	
	@Override
	public TimeSeries<double[]> labelTimeSeries(TargetSeries series) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LabelType getLabelType() {
		return labelType;
	}

}
