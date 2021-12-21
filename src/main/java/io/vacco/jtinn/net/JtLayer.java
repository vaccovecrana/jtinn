package io.vacco.jtinn.net;

import io.vacco.jtinn.activation.JtActivationFn;

import java.io.Serializable;

public class JtLayer implements Serializable {

  private static final long serialVersionUID = JtSchema.version;

  /**
   * Layer activations (plus buffer copy for non-updating estimation),
   * biases, and error derivatives.
   */
  public double[] a, ar, b, δ;

  /** Weights from input layer, if any */
  public double[][] w;

  public JtActivationFn actFn;

  public JtLayer init(int size, JtActivationFn actFn) {
    this.ar = new double[size];
    this.a = new double[size];
    this.b = new double[size];
    this.δ = new double[size];
    this.actFn = actFn;
    return this;
  }

  public JtLayer withWeights(int weights) {
    this.w = new double[size()][weights];
    return this;
  }

  public int size() { return a == null ? 0 : a.length; }
  public int weightSize() { return w == null ? 0 : w[0].length; }

  @Override public String toString() {
    return String.format(
        "lyr[prm: %s, act: %s]",
        w != null ? ((w.length * w[0].length) + (b.length)) : 0,
        actFn != null ? actFn.getClass().getSimpleName() : "?"
    );
  }
}
