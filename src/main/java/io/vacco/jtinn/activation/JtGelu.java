package io.vacco.jtinn.activation;

public class JtGelu implements JtActivationFn {

  public static final double TwoPiSqrt = Math.sqrt(2 / Math.PI);

  @Override public double apply(double z) {
    double z3 = z * z * z;
    return 0.5 * z * (1 + Math.tanh(TwoPiSqrt * (z + (0.044715 * z3))));
  }

  public double sech(double x) {
    return 1.0 / Math.cosh(x);
  }

  @Override public double pd(double z) {
    double z3 = z * z * z;
    double c0 = (0.0356774 * z3) + (0.797885 * z);
    double c1 = (0.0535161 * z3) + (0.398942 * z);
    double c2 = (0.0356774 * z3) + (0.797885 * z);

    double sechC2 = sech(c2);
    double sech2C2 = sechC2 * sechC2;

    return (0.5 + Math.tanh(c0)) + (c1 * sech2C2) + 0.5;
  }
}
