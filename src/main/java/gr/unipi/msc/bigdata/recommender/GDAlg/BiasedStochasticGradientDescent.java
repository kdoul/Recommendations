package gr.unipi.msc.bigdata.recommender.GDAlg;

import gr.unipi.msc.bigdata.recommender.models.TrainingSample;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Collections;

public class BiasedStochasticGradientDescent extends StochasticGradientDescent {
    double[] userBias, itemBias;
    double globalBias;

    public BiasedStochasticGradientDescent(double[][] R, int k, double l, double a, int iterations) {
        super(R, k, l, a, iterations);

        //initialize biases
        userBias = new double[users];
        itemBias = new double[items];
        globalBias = getMeanRating(R);
    }

    private double getMeanRating(double[][] R) {
        int counter = 0;
        float total = 0f;

        for (int i = 0; i < R.length; ++i) {
            for (int j = 0; j < R[0].length; ++j) {
                if (R[i][j] > 0) {
                    total += R[i][j];
                    counter++;
                }
            }
        }

        return total / (double) counter;
    }

    @Override
    public void run() {
        for (int z = 0; z < iterations; z++) {
            Collections.shuffle(trainingSamples);

            for (TrainingSample ts : trainingSamples) {
                int i = ts.getUser();
                int j = ts.getItem();
                double r = ts.getRating();
                double prediction = getPrediction(i, j);
                double e = r - prediction;

                userBias[i] += a * (e - l * userBias[i]);
                itemBias[j] += a * (e - l * userBias[j]);

                for (int y = 0; y < k; y++) {
                    P[i][y] += a * (e * Q[j][y] - l * P[i][y]);
                }

                for (int y = 0; y < k; y++) {
                    Q[j][y] += a * (e * P[i][y] - l * Q[j][y]);
                }
            }
        }
    }

    @Override
    public double getPrediction(int i, int j) {
        double unbiasedPrediction = super.getPrediction(i, j);

        return globalBias + userBias[i] + itemBias[j] + unbiasedPrediction;
    }

    @Override
    public double[][] getPredictionMatrix() {
        double[][] unbiasedPredictions = super.getPredictionMatrix();

        for (int i = 0; i < unbiasedPredictions.length; i++){
            for (int j = 0; j < unbiasedPredictions[0].length; j++){
                unbiasedPredictions[i][j] += userBias[i] + itemBias[j] +globalBias;
            }
        }
//        double[][] userBias2d, itemBias2d;
//        userBias2d = new double[userBias.length][1];
//        itemBias2d = new double[1][itemBias.length];
//
//        for (int i = 0; i < userBias.length; i++) {
//            userBias2d[i][0] = userBias[i];
//        }
//
////        for (int i = 0; i < itemBias.length; i++) {
//            itemBias2d[0] = itemBias;
//        //}
//
//        RealMatrix userBiasMatrix = MatrixUtils.createRealMatrix(userBias2d),
//                itemBiasMatrix = MatrixUtils.createRealMatrix(itemBias2d),
//                unbiasedPredictionsMatrix = MatrixUtils.createRealMatrix(unbiasedPredictions);
//
//        RealMatrix biasedPredictionsMatrix = userBiasMatrix
//                .add(itemBiasMatrix)
//                .add(unbiasedPredictionsMatrix)
//                .scalarAdd(globalBias);
//
//        return biasedPredictionsMatrix.getData();
        return unbiasedPredictions;
    }

}
