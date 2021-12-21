package io.vacco.jtinn.activation;

import java.io.Serializable;

public interface JtActivationFn extends Serializable {
  double apply(double z);
  double pd(double z);
}
