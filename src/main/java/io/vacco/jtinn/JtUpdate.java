package io.vacco.jtinn;

import java.io.Serializable;

public class JtUpdate {

  public interface JtUpdater extends Serializable {
    void apply(float[] lm1a, JtLayers.JtLayer l);
    void onEpochEnd(int epoch);
  }

  public static class JtSgdUpdater implements JtUpdater {

    private static final long serialVersionUID = JtUtil.version;

    public float learningRate, annealingRate;

    public JtSgdUpdater init(float learningRate, float annealingRate) {
      this.learningRate = learningRate;
      this.annealingRate = annealingRate;
      return this;
    }

    @Override public void apply(float[] lm1a, JtLayers.JtLayer l) {
      for (int j = 0; j < l.size(); j++) {
        for (int w = 0; w < l.weightSize(); w++) {
          float pda = l instanceof JtLayers.JtOutputLayer ? lm1a[w] : l.actFn.pd(l.a[j]) * lm1a[w];
          l.w[j][w] = l.w[j][w] - (learningRate * l.δ[j] * pda);
        }
        l.b[j] = l.b[j] - l.δ[j];
      }
    }

    @Override public void onEpochEnd(int epoch) {
      this.learningRate = learningRate * annealingRate;
    }
  }

}
