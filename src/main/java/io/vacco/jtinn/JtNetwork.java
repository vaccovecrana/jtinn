package io.vacco.jtinn;

import java.io.Serializable;

import static java.lang.Math.min;
import static java.lang.Math.max;
import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.round;
import static io.vacco.jtinn.JtUtil.clamp;

public class JtNetwork implements Serializable {

  private static final long serialVersionUID = JtUtil.version;

  public JtLayers.JtLayer[] layers;
  public JtUpdate.JtUpdater updater;

  private transient boolean quantized = false;
  private transient float scale_input;

  public JtNetwork init(int inputSize, JtInit.JtParamInitializer paramInitializer,
                        JtUpdate.JtUpdater updater, JtLayers.JtLayer ... layers) {
    this.layers = layers;
    this.updater = updater;
    layers[0] = layers[0].withWeights(inputSize);
    paramInitializer.apply(layers[0]);

    for (int k = 1; k < layers.length; k++) {
      var lk = layers[k];
      var lkm1 = layers[k - 1];
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
    activate(in, update, layers[0]);
    for (int i = 1; i < layers.length; i++) {
      activate(update ? layers[i - 1].a : layers[i - 1].ar, update, layers[i]);
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
    for (int l = layers.length - 2; l >= 0; l--) {
      bp2(layers[l], layers[l + 1]);
    }
    for (int l = layers.length - 1; l > 0; l--) {
      updater.apply(layers[l - 1].a, layers[l]);
    }
    updater.apply(in, layers[0]);
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

  private void forwardQuant(byte[] in_q) {
    activateQuant(in_q, layers[0]);
    for (int i = 1; i < layers.length; i++) {
      activateQuant(layers[i - 1].ar_q, layers[i]);
    }
  }

  private void activateQuant(byte[] in_q, JtLayers.JtLayer l) {
    for (int j = 0; j < l.size(); j++) {
      int z = 0;
      for (int a = 0; a < l.weightSize(); a++) {
        z += (int) in_q[a] * (int) l.w_q[j][a];
      }
      z += l.b_q[j];
      if (l.actFn instanceof JtActivation.JtLeakyRelu) {
        var fn = (JtActivation.JtLeakyRelu) l.actFn;
        long mult = z > 0 ? fn.mult_pos : fn.mult_neg;
        long a_long = ((long) z * mult) >> fn.shift;
        l.ar_q[j] = (byte) clamp(a_long, -128, 127);
      } else if (l.actFn instanceof JtActivation.JtSigmoid) {
        var fn = (JtActivation.JtSigmoid) l.actFn;
        long index_long = (((long) z * fn.mult_index) >> fn.shift) + fn.offset;
        int index = clamp(index_long, 0, fn.table.length - 1);
        l.ar_q[j] = fn.table[index];
      } else {
        throw new IllegalStateException("Unsupported activation for quantization: " + l.actFn.getClass());
      }
    }
  }

  public float[] estimate(float[] in) {
    if (quantized) {
      byte[] in_q = new byte[in.length];
      for (int i = 0; i < in.length; i++) {
        in_q[i] = (byte) clamp(round(in[i] / scale_input), -128, 127);
      }
      forwardQuant(in_q);
      var out = getOutput();
      float[] result = new float[out.size()];
      for (int j = 0; j < result.length; j++) {
        result[j] = out.ar_q[j] * out.scale_a;
      }
      return result;
    } else {
      forward(in, false);
      return getOutput().ar;
    }
  }

  public void quantize(JtTrain.JtSampler calib, int numBatches) {
    float min_input = Float.MAX_VALUE;
    float max_input = -Float.MAX_VALUE;
    for (JtLayers.JtLayer l : layers) {
      l.min_a = Float.MAX_VALUE;
      l.max_a = -Float.MAX_VALUE;
    }
    for (int b = 0; b < numBatches; b++) {
      JtTrain.JtSample[] batch = calib.get();
      for (JtTrain.JtSample s : batch) {
        forward(s.features, false);
        for (float f : s.features) {
          min_input = min(min_input, f);
          max_input = max(max_input, f);
        }
        for (JtLayers.JtLayer l : layers) {
          for (float aa : l.ar) {
            l.min_a = min(l.min_a, aa);
            l.max_a = max(l.max_a, aa);
          }
        }
      }
    }
    scale_input = max(abs(min_input), abs(max_input)) / 127f;
    if (scale_input == 0) scale_input = 1f / 127f;
    layers[0].scale_in = scale_input;
    for (int i = 0; i < layers.length; i++) {
      JtLayers.JtLayer l = layers[i];
      l.scale_a = max(abs(l.min_a), abs(l.max_a)) / 127f;
      if (l.scale_a == 0) l.scale_a = 1f / 127f;
      l.scale_w = 0;
      for (float[] row : l.w) {
        for (float ww : row) {
          l.scale_w = max(l.scale_w, abs(ww));
        }
      }
      l.scale_w /= 127f;
      if (l.scale_w == 0) l.scale_w = 1f / 127f;
      if (i < layers.length - 1) {
        layers[i + 1].scale_in = l.scale_a;
      }
    }
    for (JtLayers.JtLayer l : layers) {
      l.w_q = new byte[l.size()][l.weightSize()];
      for (int j = 0; j < l.size(); j++) {
        for (int k = 0; k < l.weightSize(); k++) {
          l.w_q[j][k] = (byte) clamp(round(l.w[j][k] / l.scale_w), -128, 127);
        }
      }
      l.b_q = new int[l.size()];
      for (int j = 0; j < l.size(); j++) {
        float sbw = l.scale_in * l.scale_w;
        l.b_q[j] = round(l.b[j] / sbw);
      }
      l.ar_q = new byte[l.size()];
      int fixedShift = 20;
      float zClip = 8.0f;
      int tableSize = 256;
      if (l.actFn instanceof JtActivation.JtLeakyRelu) {
        var fn = (JtActivation.JtLeakyRelu) l.actFn;
        float m = l.scale_in * l.scale_w / l.scale_a;
        fn.shift = fixedShift;
        fn.mult_pos = round(m * (1L << fixedShift));
        fn.mult_neg = round(m * fn.α * (1L << fixedShift));
      } else if (l.actFn instanceof JtActivation.JtSigmoid) {
        var fn = (JtActivation.JtSigmoid) l.actFn;
        float step = (2.0f * zClip) / (tableSize - 1.0f);
        float k = (l.scale_in * l.scale_w) / step;
        fn.shift = fixedShift;
        fn.mult_index = round(k * (1L << fixedShift));
        fn.offset = tableSize / 2;
        fn.table = new byte[tableSize];
        for (int i = 0; i < tableSize; i++) {
          float z = (i - fn.offset) * step;
          float sig = 1.0f / (1.0f + (float) exp(-z));
          int aq = round(sig / l.scale_a);
          fn.table[i] = (byte) clamp(aq, -128, 127);
        }
      }
    }
    quantized = true;
  }

  public JtLayers.JtOutputLayer getOutput() {
    return (JtLayers.JtOutputLayer) layers[layers.length - 1];
  }

}
