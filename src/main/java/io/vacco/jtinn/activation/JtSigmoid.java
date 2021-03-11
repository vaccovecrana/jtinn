package io.vacco.jtinn.activation;

public class JtSigmoid implements JtActivationFn {

  @Override public double apply(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }

  @Override public double pd(double z) {
    return z * (1.0 - z);
  }
}
