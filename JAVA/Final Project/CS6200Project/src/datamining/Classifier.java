/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


import java.io.IOException;
import weka.core.Instances;

/**
 *
 * @author RavitejaSomisetty
 */
public interface Classifier {
    abstract Instances getData() throws IOException;
    abstract Instances runClassifier() throws ArrayIndexOutOfBoundsException;
}
