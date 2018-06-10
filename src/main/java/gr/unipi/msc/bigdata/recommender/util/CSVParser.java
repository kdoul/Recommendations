package gr.unipi.msc.bigdata.recommender.util;

import java.io.File;
import java.io.IOException;

public interface CSVParser {

    double[][] parse(File file) throws IOException;
}
