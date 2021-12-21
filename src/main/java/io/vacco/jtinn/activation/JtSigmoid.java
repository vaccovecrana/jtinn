package io.vacco.jtinn.activation;

import io.vacco.jtinn.net.JtSchema;

public class JtSigmoid implements JtActivationFn {

  private static final long serialVersionUID = JtSchema.version;

  @Override public double apply(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }

  @Override public double pd(double z) {
    return z * (1.0 - z);
  }
}
