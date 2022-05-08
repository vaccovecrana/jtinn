package io.vacco.jtinn.net;

public class JtPredictionSample {

  public float[] features;
  public float[] labels;

  public static JtPredictionSample of(float[] features, float[] labels) {
    JtPredictionSample sample = new JtPredictionSample();
    sample.features = features;
    sample.labels = labels;
    return sample;
  }
}
