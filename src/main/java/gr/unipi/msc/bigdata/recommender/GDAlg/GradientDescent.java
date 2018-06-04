package gr.unipi.msc.bigdata.recommender.GDAlg;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

public abstract class GradientDescent {
    double[][] R; //input matrix
    double[][] P,Q; //users to factors, items to factors matrices
    int users, items; //number of users and items
    double k; //number of latent factors
    double l; //regularization parameter
    double a; //convergence rate
    int iterations; //number of gd iterations

    public abstract void run();

    public GradientDescent(double[][] R, int k, double l, double a, int iterations){
        this.R=R;
        users = R.length;
        items = R[0].length;
        this.k = k;
        this.l = l;
        this.a = a;
        this.iterations = iterations;

        P = new double[users][k];
        Q = new double[items][k];

        // Initialize P and Q matrix
        Random rand = new Random();
        for (int i = 0; i < P.length; ++i) {
            for (int j = 0; j < P[0].length; ++j) {
                P[i][j] = rand.nextDouble() / (double) k;
            }
        }

        for (int i = 0; i < Q.length; ++i) {
            for (int j = 0; j < Q[0].length; ++j) {
                Q[i][j] = rand.nextDouble() / (double) k;
            }
        }

    }

    public double getMSE(){
        double[][] predictions = getPredictionMatrix();
        double error = 0;

        for (int i = 0; i < users; i++) {
            for (int j = 0; j < items; j++) {
                if (R[i][j] != 0f)
                error = R[i][j] - predictions[i][j];
            }
        }

        return Math.pow(error,2);
    }

    public double getPrediction(int i, int j){
        double[][] QTranspose = MatrixUtils.createRealMatrix(Q).transpose().getData(); //check
//        RealMatrix PMatrix = MatrixUtils.createRealMatrix(P);

        double sum = 0d;
        for (int z = 0; z < k; z++){
            sum += P[i][z] * QTranspose[z][j];
        }
        return sum;
    }

    public double[][] getPredictionMatrix(){
        double[][] prodMatrix = new double[R.length][R[0].length];
        for (int i = 0; i < R.length; ++i) {
            for (int j = 0; j < R[0].length; ++j) {
                for (int k = 0; k < 2; ++k) {
                    prodMatrix[i][j] += P[i][k] * Q[j][k];
                }
            }
        }

        return prodMatrix;
    }



}
