package io.vacco.jtinn.error;

import java.io.Serializable;

public interface JtErrorFn extends Serializable {

  float of(float a, float b);
  float pd(float act, float target);

}
