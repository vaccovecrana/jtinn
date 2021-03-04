package io.vacco.jtinn.net;

import io.vacco.jtinn.activation.JtActivationFn;

public class JtLayer {

  /** Layer weighted sums, activations, biases, and errors */
  public final double[] z, a, b, δ;

  /** Weights from input layer, if any */
  public double[][] w;

  public final JtActivationFn actFn;

  public JtLayer(int size, JtActivationFn actFn) {
    this.z = new double[size];
    this.a = new double[size];
    this.b = new double[size];
    this.δ = new double[size];
    this.actFn = actFn;
  }

  public JtLayer withWeights(int weights) {
    this.w = new double[size()][weights];
    return this;
  }

  public int size() { return z.length; }
  public int weightSize() { return w == null ? 0 : w[0].length; }

  @Override public String toString() {
    return String.format(
        "lyr[prm: %s, act: %s]",
        w != null ? ((w.length * w[0].length) + (b.length)) : 0,
        actFn.getClass().getSimpleName()
    );
  }
}
