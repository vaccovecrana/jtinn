package io.vacco.jtinn.net;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.error.JtErrorFn;

public class JtOutputLayer extends JtLayer {

  public final JtErrorFn errFn;

  public JtOutputLayer(int size, JtActivationFn actFn, JtErrorFn errFn) {
    super(size, actFn);
    this.errFn = errFn;
  }
}
