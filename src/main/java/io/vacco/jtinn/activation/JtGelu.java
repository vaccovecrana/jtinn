package io.vacco.jtinn.activation;

public class JtGelu implements JtActivationFn {

  public static final double TwoPiSqrt = Math.sqrt(2 / Math.PI);

  @Override public double apply(double x) {
    double x3 = x * x * x;
    return 0.5 * x * (1 + Math.tanh(TwoPiSqrt * (x + (0.044715 * x3))));
  }

  public double sech(double x) {
    return 1.0 / Math.cosh(x);
  }

  @Override public double pd(double x) {
    double x3 = x * x * x;
    double c0 = (0.0356774 * x3) + (0.797885 * x);
    double c1 = (0.0535161 * x3) + (0.398942 * x);
    double c2 = (0.0356774 * x3) + (0.797885 * x);

    double sechC2 = sech(c2);
    double sech2C2 = sechC2 * sechC2;

    return (0.5 + Math.tanh(c0)) + (c1 * sech2C2) + 0.5;
  }
}
