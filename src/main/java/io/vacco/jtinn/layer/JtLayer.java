package io.vacco.jtinn.layer;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.net.JtSchema;

import java.io.Serializable;

public class JtLayer implements Serializable {

  public JtLayer init(int size) {
    this.ar = new float[size];
    this.a = new float[size];
    this.b = new float[size];
    this.δ = new float[size];
    return this;
  }

  public JtLayer withWeights(int weights) {
    this.w = new float[size()][weights];
    return this;
  }

  public int size() { return a == null ? 0 : a.length; }
  public int weightSize() { return w == null ? 0 : w[0].length; }

  @Override public String toString() {
    return String.format(
        "lyr[prm: %s, act: %s]",
        w != null ? ((w.length * w[0].length) + (b.length)) : 0,
        actFn != null ? actFn.getClass().getSimpleName() : "?"
    );
  }
}
