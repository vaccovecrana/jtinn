package io.vacco.jtinn;

import java.io.Serializable;

public class JtNetwork3 implements Serializable {

  private static final long serialVersionUID = JtUtil.version;

  public JtLayers.JtLayer3[] layers;
  public JtUpdate.JtUpdater updater;

  public JtNetwork3 init(int c, int h, int w, JtInit.JtParamInitializer paramInitializer,
                         JtUpdate.JtUpdater updater, JtLayers.JtLayer3... layerSpec) {
    this.layers = layerSpec;
    this.updater = updater;
    layerSpec[0].withWeights(JtUtil.product(c, h, w));
    paramInitializer.apply(layerSpec[0]);

    for (int k = 1; k < layerSpec.length; k++) {
      var lk = layerSpec[k];
      var lkm1 = layerSpec[k - 1];
      lk.withWeights(lkm1.size());
      paramInitializer.apply(lk);
    }

    return this;
  }

  private void activate(JtTensor3 in, boolean update, JtLayers.JtLayer3 l) {
    var out = update ? l.a : l.ar;
    if (!update) { out.copyFrom(l.a); }
    int outSize = l.size();
    int inSize = l.weightSize();
    if (inSize >= 8) {
      JtVec.avxFloatMatMul(in.data, l.w, l.b, out.data, inSize, outSize);
    } else {
      float z;
      for (int j = 0; j < outSize; j++) {
        z = l.b[j];
        for (int a = 0; a < inSize; a++) {
          z += in.data[a] * l.w[j][a];
        }
        out.data[j] = z;
      }
    }
    for (int j = 0; j < outSize; j++) {
      out.data[j] = l.actFn.apply(out.data[j]);
    }
  }

  private void forward(JtTensor3 in, boolean update) {
    activate(in, update, layers[0]);
    for (int i = 1; i < layers.length; i++) {
      activate(update ? layers[i - 1].a : layers[i - 1].ar, update, layers[i]);
    }
  }

  private void bp1(JtTensor3 target, JtLayers.JtOutputLayer3 l) {
    JtUtil.checkTensor(target, l.a);
    for (int j = 0; j < l.size(); j++) {
      var act = l.a.data[j];
      l.δ.data[j] = l.errFn.pd(act, target.data[j]) * l.actFn.pd(act);
    }
  }

  private void bp2(JtLayers.JtLayer3 l, JtLayers.JtLayer3 lp1) {
    float d;
    for (int j = 0; j < l.size(); j++) {
      d = 0;
      for (int k = 0; k < lp1.size(); k++) {
        d += lp1.δ.data[k] * lp1.w[k][j];
      }
      l.δ.data[j] = d;
    }
  }

  private void backProp(JtTensor3 in, JtTensor3 tg) {
    bp1(tg, getOutput());
    for (int l = layers.length - 2; l >= 0; l--) {
      bp2(layers[l], layers[l + 1]);
    }
    for (int l = layers.length - 1; l > 0; l--) {
      updater.apply(layers[l - 1].a, layers[l]);
    }
    updater.apply(in, layers[0]);
  }

  public float totalError(JtTensor3 out) {
    var ol = getOutput();
    JtUtil.checkTensor(out, ol.a);
    float dt = 0;
    for (int j = 0; j < out.size(); j++) {
      dt += ol.errFn.of(ol.a.data[j], out.data[j]);
    }
    return dt;
  }

  public float train(JtTensor3 in, JtTensor3 out) {
    forward(in, true);
    backProp(in, out);
    return totalError(out);
  }

  public JtTensor3 estimate(JtTensor3 in) {
    forward(in, false);
    return getOutput().ar;
  }

  public JtLayers.JtOutputLayer3 getOutput() {
    return (JtLayers.JtOutputLayer3) layers[layers.length - 1];
  }

}
