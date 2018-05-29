package ch.imetrica.mdlfa.features;

import ch.imetrica.mdfa.mdfa.MDFABase;

public class MDFAFeatureExtraction {

	
	private MDFABase[] anyMDFAs;
	
	/**
	 * Initiates a feature extraction object with a
	 * group of six features including two lowpass
	 * filters and 4 bandpass filters.
	 * 
	 * 
	 */
	public MDFAFeatureExtraction() {
		
    	
    	anyMDFAs = new MDFABase[6];
    	
    	anyMDFAs[0] = (new MDFABase()).setLowpassCutoff(Math.PI/6.0)
				.setI1(1)
				.setSmooth(.3)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-1.0)
				.setLambda(2.0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
    	
    	anyMDFAs[1] = (new MDFABase()).setLowpassCutoff(Math.PI/4.0)
				.setI1(1)
				.setSmooth(.3)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-3.0)
				.setLambda(2.0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
		
		anyMDFAs[2] = (new MDFABase()).setLowpassCutoff(Math.PI/20.0)
				.setBandPassCutoff(Math.PI/25.0)
				.setSmooth(.2)
				.setLag(-1.0)
				.setSeriesLength(400)
				.setFilterLength(5);
		
		anyMDFAs[3] = (new MDFABase()).setLowpassCutoff(Math.PI/15.0)
                .setBandPassCutoff(Math.PI/20.0)
                .setSmooth(.2)
                .setLag(-1.0)
                .setSeriesLength(400)
                .setFilterLength(5);
		
		anyMDFAs[4] = (new MDFABase()).setLowpassCutoff(Math.PI/10.0)
                .setBandPassCutoff(Math.PI/15.0)
                .setSmooth(.2)
                .setLag(-1.0)
                .setSeriesLength(400)
                .setFilterLength(5);
		
		anyMDFAs[5] = (new MDFABase()).setLowpassCutoff(Math.PI/5.0)
                .setBandPassCutoff(Math.PI/10.0)
                .setSmooth(.3)
                .setDecayStart(.1)
				.setDecayStrength(.2)
                .setLag(-2.0)
                .setSeriesLength(400)
                .setFilterLength(5);
		
	}
	
	
    public MDFAFeatureExtraction(int nFeatures) {
		
    	
    	anyMDFAs = new MDFABase[4];
    	
    	anyMDFAs[0] = (new MDFABase()).setLowpassCutoff(Math.PI/8.0)
				.setI1(1)
				.setSmooth(.2)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-3)
				.setLambda(2.0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
    	
    	anyMDFAs[1] = (new MDFABase()).setLowpassCutoff(Math.PI/10.0)
				.setI1(1)
				.setSmooth(.2)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-2)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
    	
    	anyMDFAs[2] = (new MDFABase()).setLowpassCutoff(Math.PI/4.0)
				.setI1(1)
				.setSmooth(.2)
				.setDecayStart(.1)
				.setDecayStrength(.2)
				.setLag(-1)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
    	
    	anyMDFAs[3] = (new MDFABase()).setLowpassCutoff(Math.PI/14.0)
				.setI1(1)
				.setSmooth(.2)
				.setDecayStart(.1)
				.setDecayStrength(.1)
				.setLag(0)
				.setAlpha(2.0)
				.setSeriesLength(400)
				.setFilterLength(5);
    	
//    	anyMDFAs[4] = (new MDFABase()).setLowpassCutoff(Math.PI/16.0)
//				.setI1(1)
//				.setSmooth(.2)
//				.setDecayStart(.1)
//				.setDecayStrength(.1)
//				.setLag(-1)
//				.setAlpha(2.0)
//				.setSeriesLength(400)
//				.setFilterLength(5);
//    	
//    	anyMDFAs[5] = (new MDFABase()).setLowpassCutoff(Math.PI/18.0)
//				.setI1(1)
//				.setSmooth(.2)
//				.setDecayStart(.1)
//				.setDecayStrength(.2)
//				.setLag(-2)
//				.setAlpha(2.0)
//				.setSeriesLength(400)
//				.setFilterLength(5);
		
	}
	
	/**
	 * Get access to the MDFA Feature extractors
	 * @return
	 */
	public MDFABase[] getFeatureExtractors() {
		return anyMDFAs;
	}

	public int getNumberOfFeatures() {
		
		return anyMDFAs.length;
	}
	
	
}
