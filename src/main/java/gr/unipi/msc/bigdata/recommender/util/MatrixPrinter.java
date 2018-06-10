package gr.unipi.msc.bigdata.recommender.util;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileSystemNotFoundException;

public interface MatrixPrinter {

    void printMatrix(double[][] matrix) throws FileAlreadyExistsException, FileSystemNotFoundException, FileNotFoundException, UnsupportedEncodingException;
}
