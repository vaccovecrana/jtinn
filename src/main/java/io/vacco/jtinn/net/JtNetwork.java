package io.vacco.jtinn.net;

import io.vacco.jtinn.util.JtArrays;

import java.io.Serializable;

public class JtNetwork implements Serializable {

  private static final long serialVersionUID = JtSchema.version;

  public JtLayer[] layerSpec;
  public JtUpdater updater;

  public JtNetwork init(int inputSize, JtParamInitializer paramInitializer,
                        JtUpdater updater, JtLayer... layerSpec) {
    this.layerSpec = layerSpec;
    this.updater = updater;
    layerSpec[0] = layerSpec[0].withWeights(inputSize);
    paramInitializer.apply(layerSpec[0]);

    for (int k = 1; k < layerSpec.length; k++) {
      JtLayer lk = layerSpec[k];
      JtLayer lkm1 = layerSpec[k - 1];
      lk.withWeights(lkm1.size());
      paramInitializer.apply(lk);
    }

    return this;
  }

  private void activate(double[] in, boolean update, JtLayer l) {
    double z;
    double[] out = update ? l.a : l.ar;

    if (!update) { System.arraycopy(l.a, 0, l.ar, 0, l.a.length); }
    for (int j = 0; j < l.size(); j++) {
      z = 0;
      for (int a = 0; a < l.weightSize(); a++) {
        double aj = in[a];
        double wj = l.w[j][a];
        z = z + aj * wj;
      }
      out[j] = l.actFn.apply(z + l.b[j]);
    }
  }

  private void forward(double[] in, boolean update) {
    activate(in, update, layerSpec[0]);
    for (int i = 1; i < layerSpec.length; i++) {
      activate(update ? layerSpec[i - 1].a : layerSpec[i - 1].ar, update, layerSpec[i]);
    }
  }

  private void bp1(double[] target, JtOutputLayer l) {
    JtArrays.checkSize(target, l.a);
    for (int j = 0; j < l.size(); j++) {
      l.δ[j] = l.errFn.pd(l.a[j], target[j]) * l.actFn.pd(l.a[j]);
    }
  }

  private void bp2(JtLayer l, JtLayer lp1) {
    double d;
    for (int j = 0; j < l.size(); j++) {
      d = 0;
      for (int k = 0; k < lp1.size(); k++) {
        d = d + (lp1.δ[k] * lp1.w[k][j]);
      }
      l.δ[j] = d;
    }
  }

  private void backProp(double[] in, double[] tg) {
    bp1(tg, getOutput());
    for (int l = layerSpec.length - 2; l >= 0; l--) {
      bp2(layerSpec[l], layerSpec[l + 1]);
    }
    for (int l = layerSpec.length - 1; l > 0; l--) {
      updater.apply(layerSpec[l - 1].a, layerSpec[l]);
    }
    updater.apply(in, layerSpec[0]);
  }

  public double totalError(double[] out) {
    JtOutputLayer ol = getOutput();
    JtArrays.checkSize(out, ol.a);
    double dt = 0;
    for (int j = 0; j < out.length; j++) {
      dt = dt + ol.errFn.of(ol.a[j], out[j]);
    }
    return dt;
  }

  public double train(double[] in, double[] out) {
    forward(in, true);
    backProp(in, out);
    return totalError(out);
  }

  public double[] estimate(double[] in) {
    forward(in, false);
    return getOutput().ar;
  }

  public JtOutputLayer getOutput() { return (JtOutputLayer) layerSpec[layerSpec.length - 1]; }

}
