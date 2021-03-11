package io.vacco.jtinn.net;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.error.JtErrorFn;

public class JtOutputLayer extends JtLayer {

  public JtErrorFn errFn;

  public JtOutputLayer init(int size, JtActivationFn actFn, JtErrorFn errFn) {
    this.init(size, actFn);
    this.errFn = errFn;
    return this;
  }
}
