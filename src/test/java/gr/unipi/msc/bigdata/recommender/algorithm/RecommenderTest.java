package gr.unipi.msc.bigdata.recommender.algorithm;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

public class RecommenderTest {

    double R[][] = new double[][]{
            {5, 3, 0, 1},
            {4, 0, 0, 1},
            {1, 1, 0, 5},
            {1, 0, 0, 4},
            {0, 1, 5, 4},
    };

    @Test
    public void GradientDescentRecommenderTest() {
        Recommender gdr = new GradientDescentRecommender(R, 2, 0.1f, 0.01f, 100);
        gdr.train();
        System.out.println("--Unbiased algorithm results--");
        System.out.println("Input Matrix:");
        for (double[] row : gdr.getInputMatrix()) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println("Predicted Matrix:");
        for (double[] row : gdr.getPredictionsMatrix()) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println("Mean square error: " + gdr.getMSE());
        System.out.println("-----------------------------");
        Assert.assertTrue(Double.parseDouble(gdr.getMSE()) < 1d);
    }

    @Test
    public void BiasedGradientDescentRecommenderTest() {
        Recommender bgdr = new BiasedGradientDescentRecommender(R, 2, 0.1f, 0.01f, 100);
        bgdr.train();

        System.out.println("--Biased algorithm results--");
        System.out.println("Input Matrix:");
        for (double[] row : bgdr.getInputMatrix()) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println("Predicted Matrix:");
        for (double[] row : bgdr.getPredictionsMatrix()) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println("Mean square error: " + bgdr.getMSE());
        System.out.println("-----------------------------");
        Assert.assertTrue(Double.parseDouble(bgdr.getMSE()) < 0.001d);
    }

}