package gr.unipi.msc.bigdata.recommender.util;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

public class CSVPrinter implements MatrixPrinter {
    @Override
    public void printMatrix(double[][] matrix) throws FileNotFoundException, UnsupportedEncodingException {
        try (PrintWriter writer = new PrintWriter("result.csv", "UTF-8")) {
            writer.println("userId,movieId,rating");
            for (int i = 0; i < matrix.length; ++i) {
                for (int j = 0; j < matrix[0].length; j++) {
                    writer.println(i + "," + j + "," + matrix[i][j]);
                }
            }
        }
    }
}
