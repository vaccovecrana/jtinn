package io.vacco.jtinn.activation;

import java.io.Serializable;

public interface JtActivationFn extends Serializable {
  float apply(float z);
  float pd(float z);
}
