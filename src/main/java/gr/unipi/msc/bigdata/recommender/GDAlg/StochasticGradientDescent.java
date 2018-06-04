package gr.unipi.msc.bigdata.recommender.GDAlg;

import gr.unipi.msc.bigdata.recommender.models.TrainingSample;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StochasticGradientDescent extends GradientDescent{
    List<TrainingSample> trainingSamples; //list of samples to train the sgd against

    public StochasticGradientDescent(double[][] R, int k, double l, double a, int iterations) {
        super(R, k, l, a, iterations);

        trainingSamples = new ArrayList<>();
        for (int i = 0; i < R.length; ++i) {
            for (int j = 0; j < R[0].length; ++j) {
                if(R[i][j] > 0){
                    trainingSamples.add(new TrainingSample(i, j, R[i][j]));
                }
            }
        }
    }

    public void run(){
        for(int z = 0; z < iterations; z++){
            Collections.shuffle(trainingSamples);

            for(TrainingSample ts : trainingSamples){
                int i = ts.getUser();
                int j = ts.getItem();
                double r = ts.getRating();
                double prediction = getPrediction(i, j);
                double e = r-prediction;

                for(int y=0; y<k; y++){
                    P[i][y] = a*(e*Q[j][y] - l*P[i][y]);
                }

                for(int y=0; y<k; y++){
                    Q[j][y] = a*(e*P[i][y] - l*Q[j][y]);
                }
            }
        }

    }

}
