package io.vacco.jtinn;

import java.io.Serializable;

public class JtActivation {

  public interface JtActivationFn extends Serializable {
    double apply(double z);
    double pd(double z);
  }

  public static class JtLeakyRelu implements JtActivationFn {
    private static final long serialVersionUID = JtUtil.version;
    public double α;

    @Override public double apply(double z) {
      return z > 0 ? z : α * z;
    }

    @Override public double pd(double z) {
      return z > 0 ? 1 : α;
    }

    public JtLeakyRelu withAlpha(double α) {
      this.α = α;
      return this;
    }
  }

  public static class JtSigmoid implements JtActivationFn {
    private static final long serialVersionUID = JtUtil.version;

    @Override public double apply(double z) {
      return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override public double pd(double z) {
      return z * (1.0 - z);
    }
  }

}
