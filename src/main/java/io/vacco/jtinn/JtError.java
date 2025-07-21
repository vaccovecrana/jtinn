package io.vacco.jtinn;

import java.io.Serializable;

public class JtError {

  public interface JtErrorFn extends Serializable {
    double of(double a, double b);
    double pd(double act, double target);
  }

  public static class JtMeanSquaredError implements JtErrorFn {
    private static final long serialVersionUID = JtUtil.version;
    @Override public double of(double a, double b) {
      return 0.5f * (a - b) * (a - b);
    }
    @Override public double pd(double act, double target) {
      return act - target;
    }
  }

}
