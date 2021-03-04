package io.vacco.jtinn.activation;

public class JtLeakyRelu implements JtActivationFn {

  private final double alpha;

  public JtLeakyRelu(double alpha) {
    this.alpha = alpha;
  }

  @Override public double apply(double z) {
    return z > 0 ? z : alpha * z;
  }

  @Override public double pd(double z) {
    return z > 0 ? 1 : alpha;
  }
}
