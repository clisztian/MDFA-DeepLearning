package ch.imetrica.mdlfa.util;

public class TimeSeriesFile {

	private String dateTimeFormat; 
	private String dateTimeIndexName;
	private String priceColumnName;
	
	public TimeSeriesFile(String dateTimeFormat, 
			              String dateTimeIndexName, 
			              String priceColumnName) {
		
		this.setDateTimeFormat(dateTimeFormat);
		this.setDateTimeIndexName(dateTimeIndexName);
		this.setPriceColumnName(priceColumnName);
				
	}

	public String getDateTimeFormat() {
		return dateTimeFormat;
	}

	public void setDateTimeFormat(String dateTimeFormat) {
		this.dateTimeFormat = dateTimeFormat;
	}

	public String getDateTimeIndexName() {
		return dateTimeIndexName;
	}

	public void setDateTimeIndexName(String dateTimeIndexName) {
		this.dateTimeIndexName = dateTimeIndexName;
	}

	public String getPriceColumnName() {
		return priceColumnName;
	}

	public void setPriceColumnName(String priceColumnName) {
		this.priceColumnName = priceColumnName;
	}
	
	
	
}
