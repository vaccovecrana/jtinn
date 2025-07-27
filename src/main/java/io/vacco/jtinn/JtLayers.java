package io.vacco.jtinn;

import java.io.Serializable;
import java.util.Arrays;

import static io.vacco.jtinn.JtUtil.shape3;

public class JtLayers {

  public static class JtLayer3 implements Serializable {
    private static final long serialVersionUID = JtUtil.version;

    /**
     * Layer activations (plus buffer copy for non-updating estimation),
     * biases, and error derivatives.
     */
    public JtTensor3 a, ar, δ;
    public float[] b;

    /** Weights from input layer, if any */
    public float[][] w;

    public JtActivation.JtActivationFn actFn;

    public int[] outputShape;

    public JtLayer3 init(int c, int h, int w, JtActivation.JtActivationFn actFn) {
      this.outputShape = shape3(c, h, w);
      this.a  = new JtTensor3(c, h, w);
      this.ar = new JtTensor3(c, h, w);
      this.δ  = new JtTensor3(c, h, w);
      this.b  = new float[a.size()];
      this.actFn = actFn;
      return this;
    }

    protected void computeLinear(JtTensor3 input, JtTensor3 out) {
      int outSize = size();
      int inSize = input.size();
      if (w == null) {
        throw new IllegalStateException("No weights for layer");
      }
      if (inSize != w[0].length) {
        throw new IllegalArgumentException("Input size mismatch");
      }
      if (inSize >= 8) {
        JtVec.avxFloatMatMul(input.data, w, b, out.data, inSize, outSize);
      } else {
        float z;
        for (int j = 0; j < outSize; j++) {
          z = b[j];
          for (int a = 0; a < inSize; a++) {
            z += input.data[a] * w[j][a];
          }
          out.data[j] = z;
        }
      }
    }

    public JtTensor3 forward(JtTensor3 input, boolean training) {
      var out = training ? a : ar;
      computeLinear(input, out);
      if (actFn != null) {
        applyActivation(out, out);
      }
      return out;
    }

    public JtLayer3 withWeights(int weights) {
      this.w = new float[a.size()][weights];
      return this;
    }

    public int size() { return a == null ? 0 : a.size(); }
    public int weightSize() { return w == null ? 0 : w[0].length; }

    public void applyActivation(JtTensor3 input, JtTensor3 output) {
      JtUtil.checkTensor(input, output);
      for (int i = 0; i < input.size(); i++) {
        output.data[i] = actFn.apply(input.data[i]);
      }
    }

    @Override public String toString() {
      return String.format(
        "ly[prm: %s, act: %s]",
        w != null ? ((w.length * w[0].length) + (b.length)) : 0,
        actFn != null ? actFn.getClass().getSimpleName() : "?"
      );
    }
  }

  public static class JtOutputLayer3 extends JtLayer3 {
    private static final long serialVersionUID = JtUtil.version;
    public JtError.JtErrorFn errFn;

    public JtOutputLayer3 init(int c, int h, int w, JtActivation.JtActivationFn actFn) {
      return (JtOutputLayer3) super.init(c, h, w, actFn);
    }

    public JtOutputLayer3 errFn(JtError.JtErrorFn errFn) {
      this.errFn = errFn;
      return this;
    }
  }

  public static class JtConvLayer3 extends JtLayer3 {
    private static final long serialVersionUID = JtUtil.version;
    public float[][][][] weights;
    public float[][] wFlat;
    public int kernelSize, stride, padding, inChannels;

    public JtConvLayer3 init(int outChannels, int kernelSize, int stride, int padding, JtActivation.JtActivationFn actFn) {
      this.outputShape = shape3(outChannels, 0, 0);
      this.kernelSize = kernelSize;
      this.stride = stride;
      this.padding = padding;
      this.actFn = actFn;
      return this;
    }

    public void calculateOutputShape(int[] inputShape) {
      int inH = inputShape[1], inW = inputShape[2];
      int outH = (inH - kernelSize + 2 * padding) / stride + 1;
      int outW = (inW - kernelSize + 2 * padding) / stride + 1;
      outputShape[1] = outH;
      outputShape[2] = outW;
      a = new JtTensor3(outputShape[0], outH, outW);
      ar = new JtTensor3(outputShape[0], outH, outW);
      δ = new JtTensor3(outputShape[0], outH, outW);
    }

    public void allocateParams() {
      int outC = outputShape[0];
      weights = new float[outC][inChannels][kernelSize][kernelSize];
      b = new float[outC];
      int patchSize = inChannels * kernelSize * kernelSize;
      wFlat = new float[outC][patchSize];
    }

    public void flattenWeights() {
      int k = kernelSize;
      int outC = outputShape[0];
      for (int oc = 0; oc < outC; oc++) {
        for (int ic = 0; ic < inChannels; ic++) {
          for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
              int idx = ((ic * k + kh) * k) + kw;
              wFlat[oc][idx] = weights[oc][ic][kh][kw];
            }
          }
        }
      }
    }

    @Override protected void computeLinear(JtTensor3 input, JtTensor3 out) {
      int inC = input.shape[0], inH = input.shape[1], inW = input.shape[2];
      int outC = out.shape[0], outH = out.shape[1], outW = out.shape[2];
      int k = kernelSize, s = stride, p = padding;
      int patchSize = inC * k * k;
      var patch = new float[patchSize];
      var tempOut = new float[outC];
      for (int oh = 0; oh < outH; oh++) {
        for (int ow = 0; ow < outW; ow++) {
          Arrays.fill(patch, 0);
          for (int ic = 0; ic < inC; ic++) {
            for (int kh = 0; kh < k; kh++) {
              int ih = oh * s + kh - p;
              if (ih < 0 || ih >= inH) continue;
              for (int kw = 0; kw < k; kw++) {
                int iw = ow * s + kw - p;
                if (iw < 0 || iw >= inW) continue;
                int idx = ((ic * k + kh) * k) + kw;
                patch[idx] = input.get(ic, ih, iw);
              }
            }
          }
          JtVec.avxFloatMatMul(patch, wFlat, null, tempOut, patchSize, outC);
          for (int oc = 0; oc < outC; oc++) {
            out.set(oc, oh, ow, tempOut[oc] + b[oc]);
          }
        }
      }
    }
  }

}