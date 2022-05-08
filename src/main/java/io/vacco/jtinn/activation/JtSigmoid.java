package io.vacco.jtinn.activation;

import io.vacco.jtinn.net.JtSchema;

public class JtSigmoid implements JtActivationFn {

  private static final long serialVersionUID = JtSchema.version;

  @Override public float apply(float z) {
    return (float) (1.0 / (1.0 + Math.exp(-z)));
  }

  @Override public float pd(float z) {
    return (float) (z * (1.0 - z));
  }
}
