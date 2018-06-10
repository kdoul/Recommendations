package gr.unipi.msc.bigdata.recommender.util;

import gr.unipi.msc.bigdata.recommender.algorithm.BiasedGradientDescentRecommender;
import gr.unipi.msc.bigdata.recommender.algorithm.GradientDescentRecommender;
import gr.unipi.msc.bigdata.recommender.algorithm.Recommender;
import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class CSVPrinterTest {

    double R[][] = new double[][]{
            {5, 3, 0, 1},
            {4, 0, 0, 1},
            {1, 1, 0, 5},
            {1, 0, 0, 4},
            {0, 1, 5, 4},
    };

    @Test
    public void printMatrixTest() throws Exception {
        MatrixPrinter csvPrinter = new CSVPrinter();
        csvPrinter.printMatrix(R);
        Assert.assertTrue(Files.exists(Paths.get("result.csv")));
        List<String> lines = Files.readAllLines(Paths.get("result.csv"));
        lines.forEach(System.out::println);
        Files.deleteIfExists(Paths.get("result.csv"));
    }

    @Test
    public void printMatrixFromImportTest() throws Exception {
        ClassLoader classLoader = getClass().getClassLoader();
        File inputFile = new File(classLoader.getResource("ratings.csv").getFile());
        MatrixPrinter csvPrinter = new CSVPrinter();
        CSVParser parser = new MovielensParser();
        double[][] parsedArray = parser.parse(inputFile);

//        for(double[] row : parsedArray){
//            System.out.println(Arrays.toString(row));
//        }

        Recommender rec = new GradientDescentRecommender(parsedArray, 2, 0.0001d, 0.00001d, 200);
        rec.train();
        System.out.println(rec.getMSE());
        csvPrinter.printMatrix(rec.getPredictionsMatrix());
        Assert.assertTrue(Files.exists(Paths.get("result.csv")));
        //List<String> lines = Files.readAllLines(Paths.get("result.csv"));
        //lines.forEach(System.out::println);
//        Files.deleteIfExists(Paths.get("result.csv"));
    }
}