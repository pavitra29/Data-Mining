/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


/**
 *
 * @author Ashita, Pavitra, Ravi, Sandeep
 */
class PreProcessor {

	private final String trainingFile, outputFile;

	public PreProcessor(String trainingFile, String outputFile) {
		this.trainingFile = trainingFile;
		this.outputFile = outputFile;
	}

	protected final Instances readInput() throws FileNotFoundException, IOException {
		//set attributes
		FastVector attributes = new FastVector();
		//defining each attribute
		//age is continuous
		attributes.addElement(new Attribute("age"));
		//adding categorical values for workclass attribute
		FastVector workClassValues = new FastVector();
		workClassValues.addElement("Private");
		workClassValues.addElement("Self-emp-not-inc");
		workClassValues.addElement("Self-emp-inc");
		workClassValues.addElement("Federal-gov");
		workClassValues.addElement("Local-gov");
		workClassValues.addElement("State-gov");
		workClassValues.addElement("Without-pay");
		workClassValues.addElement("Never-worked");
		//defining workclass attribute with categorical values
		attributes.addElement(new Attribute("workclass", workClassValues));
		//adding other attributes similarly
		attributes.addElement(new Attribute("fnlwgt"));
		FastVector educationValues = new FastVector();
		educationValues.addElement("Bachelors");
		educationValues.addElement("Some-college");
		educationValues.addElement("11th");
		educationValues.addElement("HS-grad");
		educationValues.addElement("Prof-school");
		educationValues.addElement("Assoc-acdm");
		educationValues.addElement("Assoc-voc");
		educationValues.addElement("9th");
		educationValues.addElement("7th-8th");
		educationValues.addElement("12th");
		educationValues.addElement("Masters");
		educationValues.addElement("1st-4th");
		educationValues.addElement("10th");
		educationValues.addElement("Doctorate");
		educationValues.addElement("5th-6th");
		educationValues.addElement("Preschool");
		attributes.addElement(new Attribute("education", educationValues));
		attributes.addElement(new Attribute("education-num"));
		FastVector maritalValues = new FastVector();
		maritalValues.addElement("Married-civ-spouse");
		maritalValues.addElement("Divorced");
		maritalValues.addElement("Never-married");
		maritalValues.addElement("Separated");
		maritalValues.addElement("Widowed");
		maritalValues.addElement("Married-spouse-absent");
		maritalValues.addElement("Married-AF-spouse");
		attributes.addElement(new Attribute("marital-status", maritalValues));
		FastVector occupationValues = new FastVector();
		occupationValues.addElement("Tech-support");
		occupationValues.addElement("Craft-repair");
		occupationValues.addElement("Other-service");
		occupationValues.addElement("Sales");
		occupationValues.addElement("Exec-managerial");
		occupationValues.addElement("Prof-specialty");
		occupationValues.addElement("Handle-cleaners");
		occupationValues.addElement("Machine-op-inspct");
		occupationValues.addElement("Adm-clerical");
		occupationValues.addElement("Farming-fishing");
		occupationValues.addElement("Transport-moving");
		occupationValues.addElement("Priv-house-serv");
		occupationValues.addElement("Protective-serv");
		occupationValues.addElement("Armed-Forces");
		attributes.addElement(new Attribute("occupation", occupationValues));
		FastVector relationshipValues = new FastVector();
		relationshipValues.addElement("Wife");
		relationshipValues.addElement("Own-child");
		relationshipValues.addElement("Husband");
		relationshipValues.addElement("Not-in-family");
		relationshipValues.addElement("Other-relative");
		relationshipValues.addElement("Unmarried");
		attributes.addElement(new Attribute("relationship", relationshipValues));
		//race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
		FastVector raceValues = new FastVector();
		raceValues.addElement("White");
		raceValues.addElement("Asian-Pac-Islander");
		raceValues.addElement("Amer-Indian-Eskimo");
		raceValues.addElement("Other");
		raceValues.addElement("Black");
		attributes.addElement(new Attribute("race", raceValues));
		//sex: Female, Male.
		FastVector sexValues = new FastVector();
		sexValues.addElement("Female");
		sexValues.addElement("Male");
		attributes.addElement(new Attribute("sex", sexValues));
		attributes.addElement(new Attribute("capital-gain"));
		attributes.addElement(new Attribute("capital-loss"));
		attributes.addElement(new Attribute("hours-per-week"));
		//native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), 
		//India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, 
		//Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, 
		//Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, 
		//Holand-Netherlands.
		FastVector nativeCountryValues = new FastVector();
		nativeCountryValues.addElement("United-States");
		nativeCountryValues.addElement("Cambodia");
		nativeCountryValues.addElement("England");
		nativeCountryValues.addElement("Puerto-Rico");
		nativeCountryValues.addElement("Canada");
		nativeCountryValues.addElement("Germany");
		nativeCountryValues.addElement("Outlying-US(Guam-USVI-etc)");
		nativeCountryValues.addElement("India");
		nativeCountryValues.addElement("Japan");
		nativeCountryValues.addElement("Greece");
		nativeCountryValues.addElement("South");
		nativeCountryValues.addElement("China");
		nativeCountryValues.addElement("Cuba");
		nativeCountryValues.addElement("Iran");
		nativeCountryValues.addElement("Honduras");
		nativeCountryValues.addElement("Philippines");
		nativeCountryValues.addElement("Italy");
		nativeCountryValues.addElement("Poland");
		nativeCountryValues.addElement("Jamaica");
		nativeCountryValues.addElement("Vietnam");
		nativeCountryValues.addElement("Mexico");
		nativeCountryValues.addElement("Portugal");
		nativeCountryValues.addElement("Ireland");
		nativeCountryValues.addElement("France");
		nativeCountryValues.addElement("Dominician-Republic");
		nativeCountryValues.addElement("Laos");
		nativeCountryValues.addElement("Ecuador");
		nativeCountryValues.addElement("Taiwan");
		nativeCountryValues.addElement("Haiti");
		nativeCountryValues.addElement("Columbia");
		nativeCountryValues.addElement("Hungary");
		nativeCountryValues.addElement("Guatemala");
		nativeCountryValues.addElement("Nicaragua");
		nativeCountryValues.addElement("Scotland");
		nativeCountryValues.addElement("Thailand");
		nativeCountryValues.addElement("Yugoslavia");
		nativeCountryValues.addElement("El-Salvador");
		nativeCountryValues.addElement("Trinadad&Tobago");
		nativeCountryValues.addElement("Peru");
		nativeCountryValues.addElement("Hong");
		nativeCountryValues.addElement("Holand-Netherlands");
		attributes.addElement(new Attribute("native-country", nativeCountryValues));
		FastVector incomeValues = new FastVector();
		incomeValues.addElement("<=50K");
		incomeValues.addElement(">50K");
		attributes.addElement(new Attribute("income", incomeValues));
		Instances data = new Instances(new String("Data"), attributes, 0);
		data.setClassIndex(data.numAttributes() - 1);
		BufferedReader br = new BufferedReader(new FileReader(this.trainingFile));
		String line;
		while ((line = br.readLine()) != null) {
			String[] tokens = line.split(",");
			double[] values = new double[tokens.length];
			for (int i = 0; i < tokens.length; i++) {
				try {
					if (tokens[i].startsWith("\"")) {
						tokens[i] = tokens[i].substring(1, tokens[i].length());
					} else if (tokens[i].endsWith("\"")) {
						tokens[i] = tokens[i].substring(0, tokens[i].length() - 1);
					}
					values[i] = Double.parseDouble(tokens[i].trim());
				} catch (NumberFormatException e) {
					switch (i) {
					case 1:
						values[i] = (double) workClassValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 3:
						values[i] = (double) educationValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 5:
						values[i] = (double) maritalValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 6:
						values[i] = (double) occupationValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 7:
						values[i] = (double) relationshipValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 8:
						values[i] = (double) raceValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 9:
						values[i] = (double) sexValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 13:
						values[i] = (double) nativeCountryValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					case 14:
						values[i] = (double) incomeValues.indexOf(tokens[i].trim());
						if (values[i] == -1) {
							values[i] = Instance.missingValue();
						}
						break;
					}
				}
			}
			data.add(new Instance(1.0, values));
		}
		br.close();
		return data;
	}

	protected final void writeOutput(Instances data) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(this.outputFile));
		bw.write(data.toString());
		bw.close();
	}

	protected final void smote() throws Exception{
		Instances newData = null;
		Instances newDatasm = null;
		try {
			Instances data = this.readInput();
			ReplaceMissingValues filter = new ReplaceMissingValues();
			SMOTE sm = new SMOTE();
			sm.setPercentage(140);
			sm.setNearestNeighbors(7);
			filter.setInputFormat(data);
			newData = Filter.useFilter(data, filter);
			sm.setInputFormat(data);

			newDatasm = Filter.useFilter(newData, sm);
		} catch (FileNotFoundException ex) {
			System.out.println("Input file not found. Program terminated");
			System.exit(0);
		} catch (IOException ex) {
			System.out.println("Input file corrupted. Program terminated");
			System.exit(0);
		} catch (Exception ex) {
			Logger.getLogger(PreProcessor.class.getName()).log(Level.SEVERE, null, ex);
		}
		try {
			this.writeOutput(newDatasm);
		} catch (IOException ex) {
			System.out.println("Output file corrupted. Program terminated");
			System.exit(0);
		}

	}

	protected final void run() {
		Instances newData = null;

		try {
			Instances data = this.readInput();
			ReplaceMissingValues filter = new ReplaceMissingValues();
			filter.setInputFormat(data);
			newData = Filter.useFilter(data, filter);

		} catch (FileNotFoundException ex) {
			System.out.println("Input file not found. Program terminated");
			System.exit(0);
		} catch (IOException ex) {
			System.out.println("Input file corrupted. Program terminated");
			System.exit(0);
		} catch (Exception ex) {
			Logger.getLogger(PreProcessor.class.getName()).log(Level.SEVERE, null, ex);
		}
		try {
			this.writeOutput(newData);
		} catch (IOException ex) {
			System.out.println("Output file corrupted. Program terminated");
			System.exit(0);
		}
	}

}