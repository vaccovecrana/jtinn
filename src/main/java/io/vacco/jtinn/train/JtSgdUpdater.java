package io.vacco.jtinn.train;

import io.vacco.jtinn.layer.JtLayer;
import io.vacco.jtinn.layer.JtOutputLayer;
import io.vacco.jtinn.net.JtSchema;

public class JtSgdUpdater implements JtUpdater {

  private static final long serialVersionUID = JtSchema.version;

  public float learningRate, annealingRate;

  public JtSgdUpdater init(float learningRate, float annealingRate) {
    this.learningRate = learningRate;
    this.annealingRate = annealingRate;
    return this;
  }

  @Override public void apply(float[] lm1a, JtLayer l) {
    for (int j = 0; j < l.size(); j++) {
      for (int w = 0; w < l.weightSize(); w++) {
        double pda = l instanceof JtOutputLayer ? lm1a[w] : l.actFn.pd(l.a[j]) * lm1a[w];
        l.w[j][w] = (float) (l.w[j][w] - (learningRate * l.δ[j] * pda));
      }
      l.b[j] = l.b[j] - l.δ[j];
    }
  }

  @Override public void onEpochEnd(int epoch) {
    this.learningRate = learningRate * annealingRate;
  }

}
