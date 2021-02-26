package io.vacco.jtinn.error;

public class JtMeanSquaredError implements JtErrorFn {

  @Override public double of(double a, double b) {
    return 0.5f * (a - b) * (a - b);
  }

  @Override public double pd(double a, double b) {
    return a - b;
  }

}
