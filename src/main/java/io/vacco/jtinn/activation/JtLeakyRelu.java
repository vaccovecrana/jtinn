package io.vacco.jtinn.activation;

import io.vacco.jtinn.net.JtSchema;

public class JtLeakyRelu implements JtActivationFn {

  private static final long serialVersionUID = JtSchema.version;

  public float α;

  @Override public float apply(float z) {
    return z > 0 ? z : α * z;
  }

  @Override public float pd(float z) {
    return z > 0 ? 1 : α;
  }

  public JtLeakyRelu withAlpha(float α) {
    this.α = α;
    return this;
  }

}
