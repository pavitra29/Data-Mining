import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;

class BaggingMethod {

	private final String train_file;

    public BaggingMethod(String train_file) {
        this.train_file = train_file;
    }
	
	protected final Bagging BaggingDTClassify() throws IOException
	{
	BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
	Instances traininstance = new Instances(traindata);
	traindata.close();
	//setting class attribute
	traininstance.setClassIndex(traininstance.numAttributes() - 1);
	Bagging b = new Bagging();
	b.setBagSizePercent(28);
	try
	{
		
	    b.setClassifier(new J48());
	    b.buildClassifier(traininstance);
		
		
	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}	
	return b;
	}
	
	protected final Bagging BaggingKNNClassify() throws IOException
	{
	BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
	Instances traininstance = new Instances(traindata);
	traindata.close();
	//setting class attribute
	traininstance.setClassIndex(traininstance.numAttributes() - 1);
	Bagging b = new Bagging();
	b.setBagSizePercent(28);
	try
	{
		IBk ibk = new IBk();
		ibk.setKNN(11);
	    b.setClassifier(ibk);
	    b.buildClassifier(traininstance);
		
		
	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}	
	return b;
	}
	
}