package gr.unipi.msc.bigdata.recommender.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class MovielensParser implements CSVParser {
    @Override
    public double[][] parse(File file) throws IOException {
        List<Line> linesInFile = new ArrayList<>();
        Map<Integer, Integer> userIDtoNumberMap = new HashMap<>(),
                movieIDtoNumberMap = new HashMap<>();

        int numberOfUsers = 0,
                numberOfMovies = 0;

        BufferedReader br = new BufferedReader(new FileReader(file));
        StringTokenizer st = null;
        String row;
        int counter=0;
        br.readLine(); //Skip first line
        while ((row = br.readLine()) != null) {
            counter++;

            st = new StringTokenizer(row, ",");
                int userNumber = Integer.parseInt(st.nextToken());
                int movieNumber = Integer.parseInt(st.nextToken());
                double rating = Double.parseDouble(st.nextToken());

                Line line = new Line(userNumber, movieNumber, rating);
                linesInFile.add(line);

                if (!userIDtoNumberMap.containsKey(userNumber)) {
                    userIDtoNumberMap.putIfAbsent(userNumber, numberOfUsers);
                    numberOfUsers++;
                }

                if (!movieIDtoNumberMap.containsKey(movieNumber)) {
                    movieIDtoNumberMap.putIfAbsent(movieNumber, numberOfMovies);
                    numberOfMovies++;
                }
        }

        double[][] theMatrix = new double[numberOfUsers][numberOfMovies];

        linesInFile.forEach(line -> {
            theMatrix[userIDtoNumberMap.get(line.user)][movieIDtoNumberMap.get(line.movie)] = line.rating;
        });

        return theMatrix;
    }

    class Line {
        final int user, movie;
        final double rating;

        Line(int user, int movie, double rating) {
            this.user = user;
            this.movie = movie;
            this.rating = rating;
        }
    }
}
