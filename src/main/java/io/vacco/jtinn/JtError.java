package io.vacco.jtinn;

import java.io.Serializable;

public class JtError {

  public interface JtErrorFn extends Serializable {
    float of(float a, float b);
    float pd(float act, float target);
  }

  public static class JtMeanSquaredError implements JtErrorFn {
    private static final long serialVersionUID = JtUtil.version;
    @Override public float of(float a, float b) {
      return 0.5f * (a - b) * (a - b);
    }
    @Override public float pd(float act, float target) {
      return act - target;
    }
  }

}
