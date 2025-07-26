package io.vacco.jtinn;

import java.util.Random;

public class JtInit {

  public interface JtParamInitializer {
    void apply(JtLayers.JtLayer3 layer);
  }

  public static class JtRandomInitializer implements JtParamInitializer {

    public long seed;
    private transient Random r;

    public JtRandomInitializer init(long seed) {
      this.seed = seed;
      this.r = new Random(seed);
      return this;
    }

    @Override public void apply(JtLayers.JtLayer3 layer) {
      for (int i = 0; i < layer.b.length; i++) {
        layer.b[i] = r.nextFloat() - 0.5f;
      }
      if (layer.w != null) {
        for (int k = 0; k < layer.w.length; k++) {
          for (int j = 0; j < layer.w[k].length; j++) {
            layer.w[k][j] = r.nextFloat() - 0.5f;
          }
        }
      }
    }
  }

}
