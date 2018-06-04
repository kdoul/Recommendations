package gr.unipi.msc.bigdata.recommender.models;

public class TrainingSample {
    private int user, item;
    private double rating;

    public TrainingSample(int user, int item, double rating){
        this.item = item;
        this.user = user;
        this.rating = rating;
    }

    public int getUser() {
        return user;
    }

    public int getItem() {
        return item;
    }

    public double getRating() {
        return rating;
    }
}
