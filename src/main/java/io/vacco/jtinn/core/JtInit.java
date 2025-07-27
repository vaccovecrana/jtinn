package io.vacco.jtinn.core;

import io.vacco.jtinn.pipnet.JtResidualBlock;
import java.io.*;
import java.util.Arrays;
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
      if (layer instanceof JtLayers.JtConvLayer3) {
        JtLayers.JtConvLayer3 cl = (JtLayers.JtConvLayer3) layer;
        for (int oc = 0; oc < cl.outputShape[0]; oc++) {
          for (int ic = 0; ic < cl.inChannels; ic++) {
            for (int kh = 0; kh < cl.kernelSize; kh++) {
              for (int kw = 0; kw < cl.kernelSize; kw++) {
                cl.weights[oc][ic][kh][kw] = r.nextFloat() - 0.5f;
              }
            }
          }
        }
      } else if (layer instanceof JtLayers.JtBatchNormLayer3) {
        JtLayers.JtBatchNormLayer3 bn = (JtLayers.JtBatchNormLayer3) layer;
        Arrays.fill(bn.gamma, 1.0f);
        Arrays.fill(bn.beta, 0.0f);
        Arrays.fill(bn.runningMean, 0.0f);
        Arrays.fill(bn.runningVar, 1.0f);
      } else if (layer instanceof JtResidualBlock) {
        JtResidualBlock rb = (JtResidualBlock) layer;
        apply(rb.conv1);
        apply(rb.bn1);
        apply(rb.conv2);
        apply(rb.bn2);
        if (rb.shortcut != null) {
          apply(rb.shortcut.conv);
          apply(rb.shortcut.bn);
        }
      } else {
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

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
      in.defaultReadObject();
      r = new Random(seed);
    }
  }

}
