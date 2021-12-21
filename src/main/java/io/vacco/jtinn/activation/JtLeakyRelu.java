package io.vacco.jtinn.activation;

public class JtLeakyRelu implements JtActivationFn {

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
