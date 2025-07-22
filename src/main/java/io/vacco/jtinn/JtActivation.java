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

    public long q8mp, q8mn; // mult pos, mult neg
    public int  q8s;        // shift

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

    public long   q8mi; // mult index
    public int    q8s;  // shift
    public int    q8o;  // offset
    public byte[] q8t;  // table

    @Override public float apply(float z) {
      return 1.0f / (1.0f + (float) Math.exp(-z));
    }

    @Override public float pd(float z) {
      return z * (1.0f - z);
    }
  }

}
