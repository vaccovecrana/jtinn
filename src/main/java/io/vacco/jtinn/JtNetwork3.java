package io.vacco.jtinn;

import java.io.Serializable;

import static io.vacco.jtinn.JtUtil.*;

public class JtNetwork3 implements Serializable {

  private static final long serialVersionUID = JtUtil.version;

  public JtLayers.JtLayer3[] layers;
  public JtUpdate.JtUpdater updater;

  public JtNetwork3 init(int c, int h, int w, JtInit.JtParamInitializer paramInitializer,
                         JtUpdate.JtUpdater updater, JtLayers.JtLayer3 ... layers) {
    this.layers = layers;
    this.updater = updater;
    var currentShape = shape3(c, h, w);
    for (var l : layers) {
      if (l instanceof JtLayers.JtConvLayer3) {
        var cl = (JtLayers.JtConvLayer3) l;
        cl.inChannels = currentShape[0];
        cl.calculateOutputShape(currentShape);
        cl.allocateParams();
        paramInitializer.apply(l);
        cl.flattenWeights();
        currentShape = cl.outputShape;
      } else if (l instanceof JtLayers.JtBatchNormLayer3) {
        var bn = (JtLayers.JtBatchNormLayer3) l;
        bn.calculateOutputShape(currentShape);
        bn.allocateParams();
        paramInitializer.apply(l);
        currentShape = bn.outputShape;
      } else if (l instanceof JtLayers.JtMaxPoolLayer3) {
        var mpl = (JtLayers.JtMaxPoolLayer3) l;
        mpl.calculateOutputShape(currentShape);
        currentShape = mpl.outputShape;
      } else {
        int inSize = JtUtil.product(currentShape[0], currentShape[1], currentShape[2]);
        l.withWeights(inSize);
        paramInitializer.apply(l);
        currentShape = l.outputShape;
      }
    }
    return this;
  }

  private JtTensor3 forward(JtTensor3 in, boolean update) {
    var current = in;
    for (var l : layers) {
      current = l.forward(current, update);
    }
    return current;
  }

  private void bp1(JtTensor3 target, JtLayers.JtOutputLayer3 l) {
    checkTensor(target, l.a);
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
    checkTensor(out, ol.a);
    float dt = 0;
    for (int j = 0; j < out.size(); j++) {
      dt += ol.errFn.of(ol.a.data[j], out.data[j]);
    }
    return dt;
  }

  public float train(JtTensor3 in, JtTensor3 out) {
    for (var l : layers) {
      if (l instanceof JtLayers.JtConvLayer3 || l instanceof JtLayers.JtBatchNormLayer3 || l instanceof JtLayers.JtMaxPoolLayer3) {
        throw new UnsupportedOperationException("Training not supported for convolutional, batch norm, or max pool layers");
      }
    }
    forward(in, true);
    backProp(in, out);
    return totalError(out);
  }

  public JtTensor3 estimate(JtTensor3 in) {
    return forward(in, false);
  }

  public JtLayers.JtOutputLayer3 getOutput() {
    return (JtLayers.JtOutputLayer3) layers[layers.length - 1];
  }

}
