
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
*
* @author Pavitra
*/
class NaiveBayesClassifier {
	
	private final String train_file;

    public NaiveBayesClassifier(String train_file) {
        this.train_file = train_file;
    }
	
	protected final Classifier NBClassify() throws IOException
	{
	BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
	Instances traininstance = new Instances(traindata);
	traindata.close();
	//setting class attribute
	traininstance.setClassIndex(traininstance.numAttributes() - 1);
	Classifier cnaive = (Classifier)new NaiveBayes();
	try
	{
		
		cnaive.buildClassifier(traininstance);

	}
	catch(Exception e)
	{
		System.out.println("An exception occured: "+e);
	}
	return cnaive;
	}
}