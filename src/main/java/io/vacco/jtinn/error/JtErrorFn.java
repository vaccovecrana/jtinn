package io.vacco.jtinn.error;

import java.io.Serializable;

public interface JtErrorFn extends Serializable {

  double of(double a, double b);
  double pd(double act, double target);

}
