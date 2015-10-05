
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.classifiers.trees.J48;

class AdaBoost {
	
	
	private final String train_file;

    public AdaBoost(String train_file) {
        this.train_file = train_file;
    }
	
	protected final AdaBoostM1 AdaBoostDTClassify() throws IOException
	{
	BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
	Instances traininstance = new Instances(traindata);
	traindata.close();
	//setting class attribute
	traininstance.setClassIndex(traininstance.numAttributes() - 1);
	AdaBoostM1 m1 = new AdaBoostM1();
	try
	{
		m1.setWeightThreshold(10);
		 m1.setClassifier(new J48());
	    m1.buildClassifier(traininstance);
		
		
	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}
	return m1;
	}
	
}
