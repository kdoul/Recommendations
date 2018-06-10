package gr.unipi.msc.bigdata.recommender;

import gr.unipi.msc.bigdata.recommender.algorithm.BiasedGradientDescentRecommender;
import gr.unipi.msc.bigdata.recommender.algorithm.Recommender;
import gr.unipi.msc.bigdata.recommender.util.CSVParser;
import gr.unipi.msc.bigdata.recommender.util.CSVPrinter;
import gr.unipi.msc.bigdata.recommender.util.MatrixPrinter;
import gr.unipi.msc.bigdata.recommender.util.MovielensParser;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.file.FileAlreadyExistsException;

public class App {
    public static void main(String args[]) {

        if (args.length != 5) {
            System.err.println("Please provide all the required parameters.");
            System.out.println("Params: inputFile numberOfLatentFactors rateOfApproach regularizationParam numberOfIterations.");
            System.exit(1);
        }

        System.out.println("Starting up recommender system...");
        String inputFile = args[0];
        String input = args[1];
        int latentFactors = Integer.parseInt(input);
        input = args[2];
        double rate = Double.parseDouble(input);
        input = args[3];
        double lambda = Double.parseDouble(input);
        input = args[4];
        int iterations = Integer.parseInt(input);
        CSVParser movieLensDataParser = new MovielensParser();
        double[][] inputMatrix = null;

        System.out.println("Parsing input file...");
        try {
            inputMatrix = movieLensDataParser.parse(new File(inputFile));
        } catch (IOException e) {
            System.err.println("Failed to read input file.");
            e.printStackTrace();
            System.exit(1);
        }

        Recommender bgdr = new BiasedGradientDescentRecommender(inputMatrix, latentFactors, rate, lambda, iterations);
        System.out.println("Training...");
        bgdr.train();

        System.out.println("Training complete over " + iterations + " iterations.");
        //System.out.println("Got an MSE of: " + bgdr.getMSE());
        System.out.println("Writing output file to disk...");

        MatrixPrinter csvPrinter = new CSVPrinter();
        try {
            csvPrinter.printMatrix(bgdr.getPredictionsMatrix());
        } catch (FileAlreadyExistsException | FileNotFoundException | UnsupportedEncodingException e) {
            System.err.println("Error writing output.csv.");
            e.printStackTrace();
        }

    }
}
