import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class knn {
	
	private final String train_file;
	
	public knn(String train_file){
		this.train_file = train_file;
	}
	
	public IBk knnclassifier() throws Exception{
		BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
		Instances traininstance = new Instances(traindata);
		traindata.close();
		traininstance.setClassIndex(traininstance.numAttributes() - 1);
		IBk c = new IBk();
	
		c.buildClassifier(traininstance);
        c.setKNN(11);
        
        
        return c;
	}

}
