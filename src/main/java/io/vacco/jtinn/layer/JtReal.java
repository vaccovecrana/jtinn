package io.vacco.jtinn.layer;

import io.vacco.jtinn.net.JtSchema;

public class JtReal {

  private static final long serialVersionUID = JtSchema.version;

  /**
   * Layer activations (plus buffer copy for non-updating estimation),
   * biases, and error derivatives.
   */
  public float[] a, ar, b, δ;

  /** Weights from input layer, if any */
  public float[][] w;



}
