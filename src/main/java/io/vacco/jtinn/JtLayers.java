package io.vacco.jtinn;

import java.io.Serializable;

import static io.vacco.jtinn.JtUtil.shape3;

public class JtLayers {

  public static class JtLayer3 implements Serializable {

    private static final long serialVersionUID = JtUtil.version;

    /**
     * Layer activations (plus buffer copy for non-updating estimation),
     * biases, and error derivatives.
     */
    public JtTensor3 a, ar, δ;
    public float[] b;

    /** Weights from input layer, if any */
    public float[][] w;

    public JtActivation.JtActivationFn actFn;

    public int[] outputShape;

    public JtLayer3 init(int c, int h, int w, JtActivation.JtActivationFn actFn) {
      this.outputShape = shape3(c, h, w);
      this.a  = new JtTensor3(c, h, w);
      this.ar = new JtTensor3(c, h, w);
      this.δ  = new JtTensor3(c, h, w);
      this.b  = new float[a.size()];
      this.actFn = actFn;
      return this;
    }

    public JtLayer3 withWeights(int weights) {
      this.w = new float[a.size()][weights];
      return this;
    }

    public int size() { return a == null ? 0 : a.size(); }
    public int weightSize() { return w == null ? 0 : w[0].length; }

    public void applyActivation(JtTensor3 input, JtTensor3 output) {
      JtUtil.checkTensor(input, output);
      for (int i = 0; i < input.size(); i++) {
        output.data[i] = actFn.apply(input.data[i]);
      }
    }

    @Override public String toString() {
      return String.format(
        "ly[prm: %s, act: %s]",
        w != null ? ((w.length * w[0].length) + (b.length)) : 0,
        actFn != null ? actFn.getClass().getSimpleName() : "?"
      );
    }
  }

  public static class JtOutputLayer3 extends JtLayer3 {

    private static final long serialVersionUID = JtUtil.version;

    public JtError.JtErrorFn errFn;

    public JtOutputLayer3 init(int c, int h, int w, JtActivation.JtActivationFn actFn) {
      return (JtOutputLayer3) super.init(c, h, w, actFn);
    }

    public JtOutputLayer3 errFn(JtError.JtErrorFn errFn) {
      this.errFn = errFn;
      return this;
    }
  }

}