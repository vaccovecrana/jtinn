package io.vacco.jtinn;

import java.io.Serializable;

public class JtNetwork implements Serializable {

  private static final long serialVersionUID = JtUtil.version;

  public JtLayers.JtLayer[] layerSpec;
  public JtUpdate.JtUpdater updater;

  public JtNetwork init(int inputSize, JtInit.JtParamInitializer paramInitializer,
                        JtUpdate.JtUpdater updater, JtLayers.JtLayer... layerSpec) {
    this.layerSpec = layerSpec;
    this.updater = updater;
    layerSpec[0] = layerSpec[0].withWeights(inputSize);
    paramInitializer.apply(layerSpec[0]);

    for (int k = 1; k < layerSpec.length; k++) {
      var lk = layerSpec[k];
      var lkm1 = layerSpec[k - 1];
      lk.withWeights(lkm1.size());
      paramInitializer.apply(lk);
    }

    return this;
  }

  private void activate(float[] in, boolean update, JtLayers.JtLayer l) {
    float z;
    float[] out = update ? l.a : l.ar;

    if (!update) { System.arraycopy(l.a, 0, l.ar, 0, l.a.length); }
    for (int j = 0; j < l.size(); j++) {
      z = 0;
      for (int a = 0; a < l.weightSize(); a++) {
        float aj = in[a];
        float wj = l.w[j][a];
        z = z + aj * wj;
      }
      out[j] = l.actFn.apply(z + l.b[j]);
    }
  }

  private void forward(float[] in, boolean update) {
    activate(in, update, layerSpec[0]);
    for (int i = 1; i < layerSpec.length; i++) {
      activate(update ? layerSpec[i - 1].a : layerSpec[i - 1].ar, update, layerSpec[i]);
    }
  }

  private void bp1(float[] target, JtLayers.JtOutputLayer l) {
    JtUtil.checkSize(target, l.a);
    for (int j = 0; j < l.size(); j++) {
      l.δ[j] = l.errFn.pd(l.a[j], target[j]) * l.actFn.pd(l.a[j]);
    }
  }

  private void bp2(JtLayers.JtLayer l, JtLayers.JtLayer lp1) {
    float d;
    for (int j = 0; j < l.size(); j++) {
      d = 0;
      for (int k = 0; k < lp1.size(); k++) {
        d = d + (lp1.δ[k] * lp1.w[k][j]);
      }
      l.δ[j] = d;
    }
  }

  private void backProp(float[] in, float[] tg) {
    bp1(tg, getOutput());
    for (int l = layerSpec.length - 2; l >= 0; l--) {
      bp2(layerSpec[l], layerSpec[l + 1]);
    }
    for (int l = layerSpec.length - 1; l > 0; l--) {
      updater.apply(layerSpec[l - 1].a, layerSpec[l]);
    }
    updater.apply(in, layerSpec[0]);
  }

  public float totalError(float[] out) {
    var ol = getOutput();
    JtUtil.checkSize(out, ol.a);
    float dt = 0;
    for (int j = 0; j < out.length; j++) {
      dt = dt + ol.errFn.of(ol.a[j], out[j]);
    }
    return dt;
  }

  public float train(float[] in, float[] out) {
    forward(in, true);
    backProp(in, out);
    return totalError(out);
  }

  public float[] estimate(float[] in) {
    forward(in, false);
    return getOutput().ar;
  }

  public JtLayers.JtOutputLayer getOutput() {
    return (JtLayers.JtOutputLayer) layerSpec[layerSpec.length - 1];
  }

}
