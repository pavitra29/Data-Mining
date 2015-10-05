/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

/**
 *
 * @author RavitejaSomisetty
 */
public class SVM {
	
	private final String train_file;
	
	public SVM(String train_file){
		this.train_file = train_file;
	}

    protected final SMO SMOClassifier() throws Exception {
    	BufferedReader traindata = new BufferedReader(new FileReader(this.train_file));
    	Instances traininstance = new Instances(traindata);
    	traindata.close();
    	traininstance.setClassIndex(traininstance.numAttributes() - 1);
    	
        SMO smo = new SMO();
        smo.setOptions(weka.core.Utils.splitOptions("-C 0.1 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
        smo.buildClassifier(traininstance);
        
        return smo;
        
    }

}