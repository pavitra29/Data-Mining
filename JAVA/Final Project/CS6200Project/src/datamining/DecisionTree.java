import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;

class DecisionTree {
	
	private final String train_file;

    public DecisionTree(String train_file) {
        this.train_file = train_file;
    }
	
    protected final Classifier DTClassify() throws IOException
	{
		BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
		Instances traininstance = new Instances(traindata);
		traindata.close();
		//setting class attribute
		traininstance.setClassIndex(traininstance.numAttributes() - 1);
		Classifier cls = new J48();
		try
		{
			 // train classifier
			
			cls.buildClassifier(traininstance);
			// evaluate classifier and print some statistics
			
		}
		catch(Exception e)
		{
			System.out.println("An exception occured: "+ e);
		}
		return cls;
	}
}
