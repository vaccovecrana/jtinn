package io.vacco.jtinn.net;

public class JtPredictionSample {

  public double[] features;
  public double[] labels;

  public static JtPredictionSample of(double[] features, double[] labels) {
    var sample = new JtPredictionSample();
    sample.features = features;
    sample.labels = labels;
    return sample;
  }
}
