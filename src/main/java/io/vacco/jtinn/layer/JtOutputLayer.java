package io.vacco.jtinn.layer;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.error.JtErrorFn;
import io.vacco.jtinn.net.JtSchema;

public class JtOutputLayer extends JtLayer {

  private static final long serialVersionUID = JtSchema.version;

  public JtErrorFn errFn;

  public JtOutputLayer init(int size, JtActivationFn actFn, JtErrorFn errFn) {
    this.init(size, actFn);
    this.errFn = errFn;
    return this;
  }

}
