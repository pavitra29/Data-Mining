import java.io.*;
import java.util.*;
import Jama.Matrix;

public class CrossValidation {
    public static void main(String[] args) throws IOException {
        String train_file_name = "150(1000)-100-train.csv", test_file_name = "test-1000-100.csv",train_data,test_data;
        int training_rows = 0,training_cols = 0, testing_rows = 0,testing_cols = 0,length;

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

        ArrayList<Matrix> x_folds = new ArrayList<Matrix>();
        ArrayList<Matrix> y_folds = new ArrayList<Matrix>();
        int kfolds=10;
        int rows_per_fold=training_rows/10;

        createKFolds(training_cols, mx_train, my_train, x_folds, y_folds, kfolds, rows_per_fold);

        length = (kfolds-1)*rows_per_fold;

        Matrix test_x_matrix=new Matrix(rows_per_fold,training_cols);
        Matrix test_y_matrix = new Matrix(rows_per_fold,1);
        Matrix errors=new Matrix(151,10);
        double x_result[][] = new double[length][training_cols];
        double y_result[][] = new double[length][1];


        String output_file = "output-cv-" + train_file_name;
        PrintStream output = new PrintStream(new File(output_file));
        findTestSetError(training_cols, x_folds, y_folds, kfolds, rows_per_fold, test_x_matrix, test_y_matrix, errors,
                x_result, y_result,output);

        optimalLambda(errors);
    }

    private static void findTestSetError(int training_cols, ArrayList<Matrix> x_folds, ArrayList<Matrix> y_folds, int kfolds,
                                         int rows_per_fold, Matrix test_x_matrix, Matrix test_y_matrix, Matrix errors,
                                         double[][] x_result, double[][] y_result, PrintStream output) {
        Matrix train_x_matrix;
        Matrix train_y_matrix;
        Matrix E_test_fold=new Matrix(0,0);
        for(int lambda=0;lambda<=150;lambda++){
            for (int i=0;i<kfolds;i++) {
                int row_so_far = 0;
                for (int j = 0; j < kfolds; j++) {
                    if (i != j) {
                        System.arraycopy(x_folds.get(j).getArray(), 0, x_result, row_so_far, x_folds.get(j).getRowDimension());
                        System.arraycopy(y_folds.get(j).getArray(), 0, y_result, row_so_far, y_folds.get(j).getRowDimension());
                        row_so_far = row_so_far + rows_per_fold;
                    }
                    else
                    {
                        test_x_matrix = x_folds.get(i);
                        test_y_matrix = y_folds.get(i);
                    }
                }
                train_x_matrix = Matrix.constructWithCopy(x_result);
                train_y_matrix = Matrix.constructWithCopy(y_result);

                Matrix w = ((((train_x_matrix.transpose().times(train_x_matrix)).plus(Matrix.identity(training_cols, training_cols).
                        times((double) lambda))).inverse()).times((train_x_matrix.transpose()).times(train_y_matrix)));

                E_test_fold = calculateMSE(rows_per_fold, test_x_matrix, test_y_matrix, w);
                errors.set(lambda,i,E_test_fold.get(0,0));

            }
            output.append(Double.toString(E_test_fold.get(0,0))+",");
            output.append(Integer.toString(lambda)+",");
            output.println();
        }
    }

    private static void optimalLambda(Matrix errors) {
        double minimum_error = (double)1000;
        int optimal_lambda = 0;
        for(int i=0;i<151;i++) {
            double average_error = 0;
            for (int j = 0; j < 10; j++) {
                average_error = average_error + errors.get(i, j);
            }
            average_error = average_error / 10;
            if (average_error < minimum_error) {
                minimum_error = average_error;
                optimal_lambda = i;
            }
        }
        System.out.println("Minimum error: "+ minimum_error);
        System.out.println("Optimal lambda: " + optimal_lambda);
    }

    private static void createKFolds(int training_cols, Matrix mx_train, Matrix my_train, ArrayList<Matrix> x_folds, ArrayList<Matrix> y_folds, int kfolds, int rows_per_fold) {
        for(int i=0;i<kfolds;i++){
            Matrix x_fold = new Matrix(rows_per_fold,training_cols);
            Matrix y_fold = new Matrix(rows_per_fold,1);
            for(int j=0;j<rows_per_fold;j++) {
                for (int col = 0; col < training_cols; col++) {
                    x_fold.set(j, col, mx_train.get((i * rows_per_fold + j), col));
                }
                y_fold.set(j,0,my_train.get((i * rows_per_fold + j),0));
            }
            x_folds.add(x_fold);
            y_folds.add(y_fold);
        }
    }

    private static Matrix calculateMSE(double testing_rows, Matrix mx_test, Matrix my_test, Matrix w) {
        return ((((mx_test.times(w).minus(my_test)).transpose()).times(mx_test.times(w).minus(my_test)))
                .times(1 / testing_rows));
    }

    private static void createXYarray(ArrayList<String[]> train_array_list, int training_cols, double[][] X_train, double[][] Y_train) {
        for (int i = 0; i < train_array_list.size(); i++) {
            for (int x = 1; x < training_cols; x++) {
                X_train[i][x] = Double.parseDouble(train_array_list.get(i)[x - 1]);
            }
            Y_train[i][0] = Double.parseDouble(train_array_list.get(i)[training_cols - 1]);
        }
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

    private static int countColumns(String train_data, int training_cols) {
        StringTokenizer train_token = new StringTokenizer(train_data, ",");
        while (train_token.hasMoreTokens()) {
            train_token.nextToken();
            training_cols++;
        }
        return training_cols;
    }

    private static void insertX0(int training_rows, double[][] x_train) {
        // adding extra column x0 = 1 to make the line have intercepts on both x and y axis
        for (int i = 0; i < training_rows; i++) {
            x_train[i][0] = 1.0;
        }
    }
}
