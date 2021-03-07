package io.vacco.jtinn.net;

public class JtSgdUpdater implements JtUpdater {

  public double learningRate, annealingRate;

  public JtSgdUpdater init(double learningRate, double annealingRate) {
    this.learningRate = learningRate;
    this.annealingRate = annealingRate;
    return this;
  }

  @Override public void apply(double[] lm1a, JtLayer l) {
    for (int j = 0; j < l.size(); j++) {
      for (int w = 0; w < l.weightSize(); w++) {
        double pda = l instanceof JtOutputLayer ? lm1a[w] : l.actFn.pd(l.a[j]) * lm1a[w];
        l.w[j][w] = l.w[j][w] - (learningRate * l.δ[j] * pda);
      }
      l.b[j] = l.b[j] - l.δ[j];
    }
  }

  @Override public void onEpochEnd(int epoch) {
    this.learningRate = learningRate * annealingRate;
  }
}
