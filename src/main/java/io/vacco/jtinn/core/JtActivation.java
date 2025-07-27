package io.vacco.jtinn.core;

import java.io.Serializable;

public class JtActivation {

  public interface JtActivationFn extends Serializable {
    float apply(float z);
    float pd(float z);
  }

  public static class JtLeakyRelu implements JtActivationFn {
    private static final long serialVersionUID = JtUtil.version;
    public float α;

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

    @Override public float apply(float z) {
      return 1.0f / (1.0f + (float) Math.exp(-z));
    }

    @Override public float pd(float z) {
      return z * (1.0f - z);
    }
  }

  public static class JtRelu implements JtActivationFn {
    private static final long serialVersionUID = JtUtil.version;

    @Override public float apply(float z) {
      return Math.max(0, z);
    }

    @Override public float pd(float z) {
      return z > 0 ? 1 : 0;
    }
  }

}