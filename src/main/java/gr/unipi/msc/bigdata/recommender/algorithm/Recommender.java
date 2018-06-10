package gr.unipi.msc.bigdata.recommender.algorithm;

import java.math.BigDecimal;
import java.math.RoundingMode;

public interface Recommender {

    void train();

    double[][] getPMatrix();

    double[][] getQMatrix();

    double[][] getInputMatrix();

    double[][] getPredictionsMatrix();

    default String getMSE() {
        BigDecimal error = new BigDecimal(0);
        double[][] inputMatrix = getInputMatrix();
        double[][] recommendationMatrix = getPredictionsMatrix();
        int counter = 0;

        for (int i = 0; i < inputMatrix.length; i++) {
            for (int j = 0; j < inputMatrix[0].length; j++) {
                if (inputMatrix[i][j] != 0) {
                    BigDecimal actualValue = new BigDecimal(inputMatrix[i][j]);
                    BigDecimal predictedValue = new BigDecimal(recommendationMatrix[i][j]);
                    BigDecimal errorValue = actualValue.subtract(predictedValue).pow(2);
                    error = error.add(errorValue);
                    counter++;
                }
            }
        }
        return error.divide(new BigDecimal(counter), RoundingMode.HALF_UP).setScale(10, RoundingMode.HALF_UP).toPlainString();
    }

}
