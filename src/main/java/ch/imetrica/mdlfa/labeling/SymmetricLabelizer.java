package ch.imetrica.mdlfa.labeling;

import ch.imetrica.mdfa.series.TargetSeries;
import ch.imetrica.mdfa.series.TimeSeries;
import ch.imetrica.mdfa.series.TimeSeriesEntry;

/**
 * 
 * This time series labeler constructs a labeled 
 * time series for deep learning applications based
 * on the help of a defined target symmetric filter.
 * For a given frequency <code> omega_0 \in (0, pi)</code>,
 * this labeler outputs at each time-series 
 * observation a one-hot vector of length two where the value of (1, 0)  
 * signals an uptrend in the underlying (price) series and (0, 1) in the 
 * case of a downtrend.
 *  
 * The constructor initializes a labelizer with 
 * a given target frequency and a symmetric filter length.
 * All observations in a given TargetSeries will then be labeled except for the
 * first and last L observations, where L is the filter length. These 
 * time series observations will be filled with null values 
 * 
 * 
 * @author lisztian
 *
 */
public class SymmetricLabelizer implements Labelizer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private LabelType labelType = LabelType.OBSERVATIONAL;
	
	private int L; 
	private double omega0;
	private double[] symmetricaCoeffs;
	
	public SymmetricLabelizer(double omega0, int L) {
		
		this.omega0 = omega0;
		this.L = L; 	
	}
		
	public SymmetricLabelizer computeSymmetricFilter() {
					
		double sum = 0; 
		symmetricaCoeffs = new double[L+1];
		symmetricaCoeffs[0] = omega0/Math.PI; 
		sum = symmetricaCoeffs[0];
		
		for(int i=1;i<=L;i++) {
			symmetricaCoeffs[i] = (1.0/Math.PI)*Math.sin(omega0*i)/(double)i; 
			sum = sum + symmetricaCoeffs[i];
		} 
		
		sum = sum+(sum-symmetricaCoeffs[0]);
		for(int i=0;i<=L;i++) {
			symmetricaCoeffs[i] = symmetricaCoeffs[i]/sum;
		}		
		
		return this;
	}
	

	@Override
	public TimeSeries<double[]> labelTimeSeries(TargetSeries series) {
		
		int N = series.size();
		double sum = 0;
		
		TimeSeries<double[]> label = new TimeSeries<double[]>();

		for(int i = 0; i < L; i++) {
			label.add(new TimeSeriesEntry<double[]>(series.getTargetDate(i), null));
		}
		
		for(int i = L-1; i < N - L; i++) {
			
			sum = 0.0;
			for(int l=0;l < L; l++) {
		    	sum = sum + symmetricaCoeffs[l]*series.getTargetValue(i+l);
		    } 
			
		    for(int l=1; l < L; l++) {
		    	sum = sum + symmetricaCoeffs[l]*series.getTargetValue(i-l);
		    }
		    
		    double[] output = {0.0, 0.0};
		    output[sum > 0 ? 0 : 1] = 1.0;
		    
		    label.add(new TimeSeriesEntry<double[]>(series.getTargetDate(i), output));
		}
		
		for(int i = N - L; i < N; i++) {
			label.add(new TimeSeriesEntry<double[]>(series.getTargetDate(i), null));
		}

		return label;
	}


    public TimeSeries<int[]> labelTimeSeriesInt(TargetSeries series) {
		
		int N = series.size();
		double sum = 0;
		
		TimeSeries<int[]> label = new TimeSeries<int[]>();

		for(int i = 0; i < L-1; i++) {
			label.add(new TimeSeriesEntry<int[]>(series.getTargetDate(i), null));
		}
		
		for(int i = L-1; i < N - L; i++) {
			
			sum = 0.0;
			for(int l=0;l < L; l++) {
		    	sum = sum + symmetricaCoeffs[l]*series.getTargetValue(i+l);
		    } 
			
		    for(int l=1; l < L; l++) {
		    	sum = sum + symmetricaCoeffs[l]*series.getTargetValue(i-l);
		    }
		    
		    int[] output = {0, 0};
		    output[sum > 0 ? 0 : 1] = 1;
		    
		    label.add(new TimeSeriesEntry<int[]>(series.getTargetDate(i), output));
		}
		
		for(int i = N - L; i < N; i++) {
			label.add(new TimeSeriesEntry<int[]>(series.getTargetDate(i), null));
		}

		return label;
	}

	
	
	@Override
	public LabelType getLabelType() {
		return labelType;
	}




	
}
