import java.io.*;
import java.util.*;
import Jama.Matrix;

public class LearningCurve {
    public static void main(String[] args) throws IOException {
        String train_file_name = "train-1000-100.csv",test_file_name = "test-1000-100.csv",train_data,test_data,output_file;
        int training_rows = 0,training_cols = 0,testing_rows = 0,testing_cols = 0;

        BufferedReader CSVTrain = new BufferedReader(new FileReader(train_file_name));
        train_data = CSVTrain.readLine();
        BufferedReader CSVTest = new BufferedReader(new FileReader(test_file_name));
        test_data = CSVTest.readLine();

        ArrayList<String[]> train_array_list = new ArrayList<String[]>();
        ArrayList<String[]> test_array_list = new ArrayList<String[]>();
        
        training_cols = countColumns(train_data, training_cols);
        testing_cols = countColumns(test_data, testing_cols);
        training_rows = readData(CSVTrain, train_array_list, training_rows, training_cols);
        testing_rows = readData(CSVTest, test_array_list, testing_rows, testing_cols);

        double X_train[][] = new double[training_rows][training_cols];
        double Y_train[][] = new double[training_rows][1];
        double X_test[][] = new double[testing_rows][testing_cols];
        double Y_test[][] = new double[testing_rows][1];

        insertX0(training_rows, X_train);
        insertX0(testing_rows, X_test);

        createXYarray(train_array_list, training_cols, X_train, Y_train);
        createXYarray(test_array_list, testing_cols, X_test, Y_test);

        Matrix mx_train = Matrix.constructWithCopy(X_train);
        Matrix my_train = Matrix.constructWithCopy(Y_train);
        Matrix mx_test = Matrix.constructWithCopy(X_test);
        Matrix my_test = Matrix.constructWithCopy(Y_test);

        int lambda[] = {1,25,150};
        for(int val=0;val<lambda.length;val++) {
            output_file = "output-lambda-"+lambda[val] + train_file_name;
            PrintStream output = new PrintStream(new File(output_file));
            findAlgorithmPerformance(training_rows, training_cols, testing_rows, mx_train, my_train, mx_test, my_test, lambda[val], output);
        }
    }

    private static void findAlgorithmPerformance(int training_rows, int training_cols, int testing_rows, Matrix mx_train, Matrix my_train, Matrix mx_test, Matrix my_test, int lambda, PrintStream output) {
        ArrayList<Integer> training_index = new ArrayList<Integer>();
        for(int k=0;k<10;k++) {
            for (int i = 0; i < 1; i++) {
                for (int j = 1; j <= 10; j++) {
                    int training_size = 50 * j;
                    Matrix mx_sample = new Matrix(training_size, training_cols);
                    Matrix my_sample = new Matrix(training_size, 1);

                    randomSampling(training_cols, mx_train, my_train, training_size, mx_sample, my_sample, training_index);

                    Matrix w = ((((mx_sample.transpose().times(mx_sample)).plus(Matrix.identity(training_cols, training_cols).
                            times((double) lambda))).inverse()).times((mx_sample.transpose()).times(my_sample)));

                    Matrix E_train = calculateMSE(training_size, mx_sample, my_sample, w);
                    Matrix E_test = calculateMSE(testing_rows, mx_test, my_test, w);

                    output.append(Integer.toString(lambda)+",");
                    output.append(Double.toString(E_train.get(0, 0))+",");
                    output.append(Double.toString(E_test.get(0, 0))+",");
                    output.println();
                }
            }
        }
    }

    private static Matrix calculateMSE(double training_size, Matrix mx_sample, Matrix my_sample, Matrix w) {
        return ((((mx_sample.times(w).minus(my_sample)).transpose())
                            .times(mx_sample.times(w).minus(my_sample)))
                            .times(1 / training_size));
    }

    private static void randomSampling(int training_cols, Matrix mx_train, Matrix my_train, int training_size, Matrix mx_sample, Matrix my_sample, ArrayList<Integer> training_index) {
        for(int r=0;r<1000;r++)
        {
            training_index.add(r);
        }
        for (int i = 0; i < training_size; i++) {
            Collections.shuffle(training_index);
            my_sample.set(i, 0, my_train.get(training_index.get(i), 0));
            for (int col = 0; col < training_cols; col++) {
                mx_sample.set(i, col, mx_train.get(training_index.get(i), col));
            }
        }
    }

    private static void insertX0(int training_rows, double[][] x_train) {
        // adding extra column x0 = 1 to make the line have intercepts on both x and y axis
        for (int i = 0; i < training_rows; i++) {
            x_train[i][0] = 1.0;
        }
    }

    private static void createXYarray(ArrayList<String[]> train_array_list,
                                      int training_cols, double[][] X_train, double[][] Y_train) {
        for (int i = 0; i < train_array_list.size(); i++) {
            for (int x = 1; x < training_cols; x++) {
                X_train[i][x] = Double.parseDouble(train_array_list.get(i)[x - 1]);
            }
            Y_train[i][0] = Double.parseDouble(train_array_list.get(i)[training_cols - 1]);
        }
    }

    private static int countColumns(String train_data, int training_cols) {
        StringTokenizer train_token = new StringTokenizer(train_data, ",");
        while (train_token.hasMoreTokens()) {
            train_token.nextToken();
            training_cols++;
        }
        return training_cols;
    }

    private static int readData(BufferedReader CSVTrain,
                                ArrayList<String[]> train_array_list, int training_rows,
                                int training_cols) throws IOException {
        String line;
        while ((line = CSVTrain.readLine()) != null) {
            String[] data = new String[training_cols];
            String[] value = line.split(",", training_cols);
            for (int i = 0; i < value.length; i++) {
                data[i] = value[i];
            }
            training_rows++;
            train_array_list.add(data);
        }
        return training_rows;
    }
}
