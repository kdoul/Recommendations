package gr.unipi.msc.bigdata.recommender;

import gr.unipi.msc.bigdata.recommender.GDAlg.GradientDescent;

import java.util.Arrays;

import static gr.unipi.msc.bigdata.recommender.App.myBiasedRecommender;
import static gr.unipi.msc.bigdata.recommender.App.myRecommender;

public class Recs {
    public static void main(String args[]){
        double R[][] = new double[][]{
                {5, 3, 0, 1},
                {4, 0, 0, 1},
                {1, 1, 0, 5},
                {1, 0, 0, 4},
                {0, 1, 5, 4},
        };
        App.Pair<double[][], double[][]> v = myBiasedRecommender(R, 2, 0.1f, 0.01f);

        App.Pair<double[][], double[][]> p = myRecommender(R, 2, 0.1f, 0.01f);

//
//        float U[][]=p.a,
//        V[][] = p.b;
//
//
//        float[][] prodMatrix = new float[R.length][R[0].length];
//        for (int i = 0; i < R.length; ++i) {
//            for (int j = 0; j < R[0].length; ++j) {
//                for (int k = 0; k < 2; ++k) {
//                    prodMatrix[i][j] += U[i][k] * V[j][k];
//                }
//            }
//        }
//        for (float[] row : prodMatrix) {
//            System.out.println(Arrays.toString(row));
//        }

//        GradientDescent gd = new StochasticGradientDescent(R, 2,0.01, 0.001, 200);
//        gd.run();
//        for (double[] row : gd.getPredictionMatrix()){
//            System.out.println(Arrays.toString(row));
//        }
//
//        gd = new BiasedStochasticGradientDescent(R, 2, 0.01, 0.001, 200);
//        gd.run();
//        for (double[] row : gd.getPredictionMatrix()){
//            System.out.println(Arrays.toString(row));
//        }
    }
}
