package io.vacco.jtinn.core;

import java.io.Serializable;

public class JtUpdate {

  public interface JtUpdater extends Serializable {
    void apply(JtTensor3 lm1a, JtLayers.JtLayer3 l);
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

    @Override public void apply(JtTensor3 lm1a, JtLayers.JtLayer3 l) {
      for (int j = 0; j < l.size(); j++) {
        for (int w = 0; w < l.weightSize(); w++) {
          float act = l.a.data[j];
          float pda = l instanceof JtLayers.JtOutputLayer3 ? lm1a.data[w] : l.actFn.pd(act) * lm1a.data[w];
          l.w[j][w] -= learningRate * l.δ.data[j] * pda;
        }
        l.b[j] -= learningRate * l.δ.data[j];
      }
    }

    @Override public void onEpochEnd(int epoch) {
      this.learningRate = learningRate * annealingRate;
    }
  }

}