package io.vacco.jtinn.error;

import io.vacco.jtinn.net.JtSchema;

public class JtMeanSquaredError implements JtErrorFn {

  private static final long serialVersionUID = JtSchema.version;

  @Override public float of(float a, float b) {
    return 0.5f * (a - b) * (a - b);
  }

  @Override public float pd(float act, float target) {
    return act - target;
  }

}
