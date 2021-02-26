package io.vacco.jtinn.activation;

public class JtSigmoid implements JtActivationFn {

  @Override public double apply(double x) {
    return 1.0f / (1.0f + Math.exp(-x));
  }

  @Override public double pd(double x) {
    return x * (1.0f - x);
  }
}
