package io.vacco.jtinn;

import java.io.*;
import java.util.Random;

public class JtInit {

  public interface JtParamInitializer extends Serializable {
    void apply(JtLayers.JtLayer3 layer);
  }

  public static class JtRandomInitializer implements JtParamInitializer {
    private static final long serialVersionUID = JtUtil.version;

    public long seed;
    private transient Random r;

    public JtRandomInitializer init(long seed) {
      this.seed = seed;
      this.r = new Random(seed);
      return this;
    }

    public void apply(JtLayers.JtLayer3 layer) {
      if (r == null) {
        r = new Random(seed);
      }
      for (int i = 0; i < layer.b.length; i++) {
        layer.b[i] = r.nextFloat() - 0.5f;
      }
      if (layer.w != null) {
        for (int k = 0; k < layer.w.length; k++) {
          for (int j = 0; j < layer.w[k].length; j++) {
            layer.w[k][j] = r.nextFloat() - 0.5f;
          }
        }
      } else if (layer instanceof JtLayers.JtConvLayer3) {
        var cl = (JtLayers.JtConvLayer3) layer;
        for (int oc = 0; oc < cl.outputShape[0]; oc++) {
          for (int ic = 0; ic < cl.inChannels; ic++) {
            for (int kh = 0; kh < cl.kernelSize; kh++) {
              for (int kw = 0; kw < cl.kernelSize; kw++) {
                cl.weights[oc][ic][kh][kw] = r.nextFloat() - 0.5f;
              }
            }
          }
        }
      }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
      in.defaultReadObject();
      r = new Random(seed);
    }
  }

}
