package io.vacco.jtinn;

import java.io.Serializable;

public class JtActivation {

  public interface JtActivationFn extends Serializable {
    float apply(float z);
    float pd(float z);
  }

  public static class JtLeakyRelu implements JtActivationFn {

    private static final long serialVersionUID = JtUtil.version;
    public float α;

    public transient long mult_pos, mult_neg;
    public transient int shift;

    @Override public float apply(float z) {
      return z > 0 ? z : α * z;
    }

    @Override public float pd(float z) {
      return z > 0 ? 1 : α;
    }

    public JtLeakyRelu withAlpha(float α) {
      this.α = α;
      return this;
    }
  }

  public static class JtSigmoid implements JtActivationFn {

    private static final long serialVersionUID = JtUtil.version;

    public transient long mult_index;
    public transient int shift;
    public transient int offset;
    public transient byte[] table;

    @Override public float apply(float z) {
      return 1.0f / (1.0f + (float) Math.exp(-z));
    }

    @Override public float pd(float z) {
      return z * (1.0f - z);
    }
  }

}
