package io.vacco.jtinn.activation;

import io.vacco.jtinn.net.JtSchema;

public class JtLeakyRelu implements JtActivationFn {

  private static final long serialVersionUID = JtSchema.version;

  public double α;

  @Override public double apply(double z) {
    return z > 0 ? z : α * z;
  }

  @Override public double pd(double z) {
    return z > 0 ? 1 : α;
  }

  public JtLeakyRelu withAlpha(double α) {
    this.α = α;
    return this;
  }

}
