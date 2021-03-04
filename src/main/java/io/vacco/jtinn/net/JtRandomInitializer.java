package io.vacco.jtinn.net;

import java.util.Random;

public class JtRandomInitializer implements JtParamInitializer {

  public final long seed;
  private transient final Random r;

  public JtRandomInitializer(long seed) {
    this.seed = seed;
    this.r = new Random(seed);
  }

  @Override public void apply(JtLayer layer) {
    for (int i = 0; i < layer.b.length; i++) {
      layer.b[i] = r.nextDouble() - 0.5;
    }
    if (layer.w != null) {
      for (int k = 0; k < layer.w.length; k++) {
        for (int j = 0; j < layer.w[k].length; j++) {
          layer.w[k][j] = r.nextDouble() - 0.5;
        }
      }
    }
  }
}

