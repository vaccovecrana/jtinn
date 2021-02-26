package io.vacco.jtinn.activation;

public class JtLeakyRelu implements JtActivationFn {

  private final double alpha;

  public JtLeakyRelu(double alpha) {
    this.alpha = alpha;
  }

  @Override public double apply(double x) {
    return x > 0 ? x : alpha * x;
  }

  @Override public double pd(double x) {
    return x > 0 ? 1 : alpha;
  }
}
