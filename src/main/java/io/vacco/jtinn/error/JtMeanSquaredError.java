package io.vacco.jtinn.error;

import io.vacco.jtinn.net.JtSchema;

public class JtMeanSquaredError implements JtErrorFn {

  private static final long serialVersionUID = JtSchema.version;

  @Override public double of(double a, double b) {
    return 0.5f * (a - b) * (a - b);
  }

  @Override public double pd(double act, double target) {
    return act - target;
  }

}
