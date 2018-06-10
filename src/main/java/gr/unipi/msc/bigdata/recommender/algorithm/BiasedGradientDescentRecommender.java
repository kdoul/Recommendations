package gr.unipi.msc.bigdata.recommender.algorithm;

import java.util.Random;

public class BiasedGradientDescentRecommender implements Recommender {

    private final double[][] inputMatrix;
    private final int r, iterations, users, items;
    private final double rate, lambda, globalBias;
    private final double[] userBias, itemBias;
    private double[][] P, Q;

    public BiasedGradientDescentRecommender(double[][] inputMatrix, int r, double rate, double lambda, int iterations) {
        this.inputMatrix = inputMatrix;
        this.r = r;
        this.rate = rate;
        this.lambda = lambda;
        this.iterations = iterations;
        this.users = inputMatrix.length;
        this.items = inputMatrix[0].length;
        this.P = new double[users][r];
        this.Q = new double[items][r];
        userBias = new double[users];
        itemBias = new double[items];
        globalBias = getMeanRating(inputMatrix);

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

    private static double getMeanRating(double[][] R) {
        int counter = 0;
        double total = 0f;

        for (double[] aR : R) {
            for (int j = 0; j < R[0].length; ++j) {
                if (aR[j] > 0) {
                    total += aR[j];
                    counter++;
                }
            }
        }

        return total / (double) counter;
    }

    @Override
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

            for (int i = 0; i < users; ++i) {
                for (int j = 0; j < items; ++j) {
                    if (inputMatrix[i][j] != 0)
                        predictionMatrix[i][j] += userBias[i] + itemBias[j] + globalBias;
                }
            }

            double[][] errorMatrix = new double[users][items];
            for (int i = 0; i < users; ++i) {
                for (int j = 0; j < items; ++j) {
                    if (inputMatrix[i][j] == 0f) {
                        errorMatrix[i][j] = 0f;
                    } else {
                        errorMatrix[i][j] = inputMatrix[i][j] - predictionMatrix[i][j];
                        userBias[i] += rate * (errorMatrix[i][j] - lambda * userBias[i]);
                        itemBias[j] += rate * (errorMatrix[i][j] - lambda * itemBias[j]);
                    }
                }
            }
//            System.out.println("User bias matrix:");
//            System.out.println(Arrays.toString(userBias));
//            System.out.println("Item bias matrix:");
//            System.out.println(Arrays.toString(itemBias));

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
                    Preg[i][j] = (1f - rate * lambda) * P[i][j] + rate * Pgradient[i][j];
                }
            }

            double[][] Qreg = new double[items][r];
            for (int i = 0; i < items; ++i) {
                for (int j = 0; j < r; ++j) {
                    Qreg[i][j] = (1f - rate * lambda) * Q[i][j] + rate * Qgradient[i][j];
                }
            }

            P = Preg;
            Q = Qreg;
        }
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

        for (int i = 0; i < users; ++i) {
            for (int j = 0; j < items; ++j) {
                predictionsMatrix[i][j] += userBias[i] + itemBias[j] + globalBias;
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
