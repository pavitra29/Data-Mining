/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import weka.core.Instances;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;

import java.io.FileReader;
import java.io.BufferedReader;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;

/**
*
* @author Ashita, Pavitra, Ravi, Sandeep
*/

public class Driver {

    /**
     * @param args the command line arguments
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        PreProcessor p = new PreProcessor("census-income.data",
                "census-income-preprocessed.arff");
   
        p.smote();
        
        PreProcessor p_test = new PreProcessor("census-income.test",
                "census-income-test-preprocessed.arff");
       
        p_test.run();
        
        BufferedReader traindata = new BufferedReader(new FileReader("census-income-preprocessed.arff"));
        BufferedReader testdata = new BufferedReader(new FileReader("census-income-test-preprocessed.arff"));
		Instances traininstance = new Instances(traindata);
		Instances testinstance = new Instances(testdata);
		
		traindata.close();
		testdata.close();
		traininstance.setClassIndex(traininstance.numAttributes() - 1);
		testinstance.setClassIndex(testinstance.numAttributes() - 1);
		int numOfAttributes = testinstance.numAttributes();
		int numOfInstances = testinstance.numInstances();
		
        NaiveBayesClassifier nb = new NaiveBayesClassifier("census-income-preprocessed.arff");
        Classifier cnaive = nb.NBClassify();
        
        DecisionTree dt = new DecisionTree("census-income-preprocessed.arff");
        Classifier cls = dt.DTClassify();
        
        AdaBoost ab = new AdaBoost("census-income-preprocessed.arff");
        AdaBoostM1 m1 = ab.AdaBoostDTClassify();
        
        BaggingMethod b = new BaggingMethod("census-income-preprocessed.arff");
        Bagging bag = b.BaggingDTClassify();
        
        SVM s = new SVM("census-income-preprocessed.arff");
        SMO svm = s.SMOClassifier();
        
        knn knnclass = new knn("census-income-preprocessed.arff");
        IBk knnc = knnclass.knnclassifier();
        
        Logistic log = new Logistic();
        log.buildClassifier(traininstance);
       
        int match = 0;
        int error = 0;
        int greater = 0;
        int less = 0;
        
		for(int i=0; i<numOfInstances; i++){
			String predicted = "";
			greater=0;
			less=0;
			double predictions[] = new double[8];
			
			double pred = cls.classifyInstance(testinstance.instance(i));
			predictions[0] = pred;
			
			double abpred = m1.classifyInstance(testinstance.instance(i));
			predictions[1] = abpred;
			
			double naivepred = cnaive.classifyInstance(testinstance.instance(i));
			predictions[2] = naivepred;
			
			double bagpred = bag.classifyInstance(testinstance.instance(i));
			predictions[3] = bagpred;
			
			double smopred = svm.classifyInstance(testinstance.instance(i));
			predictions[4] = smopred;
			
			double knnpred = knnc.classifyInstance(testinstance.instance(i));
			predictions[5] = knnpred;
			
			for(int j=0; j<6; j++){
				if((testinstance.instance(i).classAttribute().value((int) predictions[j])).compareTo(">50K") == 0)
					greater++;
				else
					less++;
			}
			
			if(greater>less)
				predicted = ">50K";
			else 
				predicted = "<=50K";
			
			if ((testinstance.instance(i).stringValue(numOfAttributes - 1)).compareTo(predicted) == 0)
				match++;
				
		    else
			    error++;
			
		}
		
		System.out.println("Correctly classified Instances: " + match);
		System.out.println("Misclassified Instances: " + error);
		
		double accuracy = (double)match/(double)numOfInstances * 100;
		double error_percent = 100 - accuracy;
		System.out.println("Accuracy: " + accuracy + "%");
		System.out.println("Error: " + error_percent + "%");
		
        
    }

}
