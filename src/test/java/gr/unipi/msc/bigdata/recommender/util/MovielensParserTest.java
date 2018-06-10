package gr.unipi.msc.bigdata.recommender.util;

import gr.unipi.msc.bigdata.recommender.algorithm.BiasedGradientDescentRecommender;
import gr.unipi.msc.bigdata.recommender.algorithm.Recommender;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

public class MovielensParserTest {

    File inputFile;

    @Before
    public void setUp() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        inputFile = new File(classLoader.getResource("ratings.csv").getFile());
    }

    @Test
    public void parseTest() throws Exception {
        CSVParser parser = new MovielensParser();
        double[][] parsedArray = parser.parse(inputFile);
        Assert.assertNotNull(parsedArray);

        System.out.println("Number of parsed users:");
        System.out.println(parsedArray.length);
        System.out.println("Number of parsed items:");
        System.out.println(parsedArray[0].length);

        Recommender rec = new BiasedGradientDescentRecommender(parsedArray, 3, 0.001f, 0.0001f, 10000);
        System.out.println(rec.getMSE());
        System.out.println(Arrays.toString(rec.getInputMatrix()[0]));
        System.out.println(Arrays.toString(rec.getPredictionsMatrix()[0]));

    }
}