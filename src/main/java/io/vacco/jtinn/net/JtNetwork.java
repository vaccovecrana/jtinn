package io.vacco.jtinn.net;

import io.vacco.jtinn.util.JtArrays;

public class JtNetwork {

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

  private void activate(double[] in, JtLayer l) {
    double z;
    for (int j = 0; j < l.size(); j++) {
      z = 0;
      for (int a = 0; a < l.weightSize(); a++) {
        double aj = in[a];
        double wj = l.w[j][a];
        z = z + aj * wj;
      }
      l.a[j] = l.actFn.apply(z + l.b[j]);
    }
  }

  // TODO feed forward may need an option to update (or not) the layer's output values.
  public void forward(double[] in) {
    activate(in, layerSpec[0]);
    for (int i = 1; i < layerSpec.length; i++) {
      activate(layerSpec[i - 1].a, layerSpec[i]);
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

  public void backProp(double[] in, double[] tg) {
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
    forward(in);
    backProp(in, out);
    return totalError(out);
  }

  public double[] estimate(double[] in) {
    forward(in);
    return getOutput().a;
  }

  public JtOutputLayer getOutput() { return (JtOutputLayer) layerSpec[layerSpec.length - 1]; }

}
