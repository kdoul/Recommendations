package gr.unipi.msc.bigdata.recommender.algorithm;

import java.util.Random;

public class GradientDescentRecommender implements Recommender {

    private final double[][] inputMatrix;
    private final int r, iterations, users, items;
    private final double rate, lambda;
    private double[][] P, Q;

    public GradientDescentRecommender(double[][] inputMatrix, int r, double rate, double lambda, int iterations) {
        this.inputMatrix = inputMatrix;
        this.r = r;
        this.rate = rate;
        this.lambda = lambda;
        this.iterations = iterations;
        this.users = inputMatrix.length;
        this.items = inputMatrix[0].length;
        this.P = new double[users][r];
        this.Q = new double[items][r];

        //Initialize P and Q
        Random rand = new Random();
        for (int i = 0; i < P.length; ++i) {
            for (int j = 0; j < P[0].length; ++j) {
                P[i][j] = rand.nextDouble() / (double) r;
            }
        }

        for (int i = 0; i < Q.length; ++i) {
            for (int j = 0; j < Q[0].length; ++j) {
                Q[i][j] = rand.nextDouble() / (double) r;
            }
        }
    }

    public void train() {
        for (int iter = 0; iter < iterations; ++iter) {
            double[][] predictionMatrix = new double[users][items];
            for (int i = 0; i < users; ++i) {
                for (int j = 0; j < items; ++j) {
                    for (int k = 0; k < r; ++k) {
                        predictionMatrix[i][j] += P[i][k] * Q[j][k];
                    }
                }
            }

            double[][] errorMatrix = new double[users][items];
            for (int i = 0; i < users; ++i) {
                for (int j = 0; j < items; ++j) {
                    if (inputMatrix[i][j] == 0f) {
                        errorMatrix[i][j] = 0f;
                    } else {
                        errorMatrix[i][j] = inputMatrix[i][j] - predictionMatrix[i][j];
                    }
                }
            }

            double[][] Pgradient = new double[users][r];
            for (int i = 0; i < users; ++i) {
                for (int j = 0; j < r; ++j) {
                    for (int k = 0; k < items; ++k) {
                        Pgradient[i][j] += errorMatrix[i][k] * Q[k][j];
                    }
                }
            }

            double[][] Qgradient = new double[items][r];
            for (int i = 0; i < items; ++i) {
                for (int j = 0; j < r; ++j) {
                    for (int k = 0; k < users; ++k) {
                        Qgradient[i][j] += errorMatrix[k][i] * P[k][j];
                    }
                }
            }

            double[][] Preg = new double[users][r];
            for (int i = 0; i < users; ++i) {
                for (int j = 0; j < r; ++j) {
                    Preg[i][j] = (1d - rate * lambda) * P[i][j] + rate * Pgradient[i][j];
                }
            }

            double[][] Qreg = new double[items][r];
            for (int i = 0; i < items; ++i) {
                for (int j = 0; j < r; ++j) {
                    Qreg[i][j] = (1d - rate * lambda) * Q[i][j] + rate * Qgradient[i][j];
                }
            }
            P = Preg;
            Q = Qreg;
        }
//        System.out.println("Unbiased Matrix:");
//        for (double[] row : prodMatrix) {
//            System.out.println(Arrays.toString(row));
//        }
//        System.out.println("Mean square error: " + getMSE(inputMatrix, prodMatrix));
    }

    @Override
    public double[][] getPredictionsMatrix() {
        double[][] predictionsMatrix = new double[users][items];
        for (int i = 0; i < users; ++i) {
            for (int j = 0; j < items; ++j) {
                for (int k = 0; k < 2; ++k) {
                    predictionsMatrix[i][j] += P[i][k] * Q[j][k];
                }
            }
        }

        return predictionsMatrix;
    }

    @Override
    public double[][] getPMatrix() {
        return P;
    }

    @Override
    public double[][] getQMatrix() {
        return Q;
    }

    @Override
    public double[][] getInputMatrix() {
        return inputMatrix;
    }


}
