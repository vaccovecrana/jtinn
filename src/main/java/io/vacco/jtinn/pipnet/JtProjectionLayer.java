package io.vacco.jtinn.pipnet;

import io.vacco.jtinn.core.*;

public class JtProjectionLayer extends JtLayers.JtLayer3 {

  private static final long serialVersionUID = JtUtil.version;

  public JtLayers.JtConvLayer3 conv;
  public JtLayers.JtBatchNormLayer3 bn;

  public JtProjectionLayer init(int inChannels, int outChannels, int stride) {
    conv = new JtLayers.JtConvLayer3().init(outChannels, 1, stride, 0, null);
    bn = new JtLayers.JtBatchNormLayer3().init(null);
    return this;
  }

  public void calculateOutputShape(int[] inputShape) {
    conv.inChannels = inputShape[0];
    conv.calculateOutputShape(inputShape);
    bn.calculateOutputShape(conv.outputShape);
    this.outputShape = bn.outputShape;
    this.a = new JtTensor3(outputShape[0], outputShape[1], outputShape[2]);
    this.ar = new JtTensor3(outputShape[0], outputShape[1], outputShape[2]);
    this.δ = new JtTensor3(outputShape[0], outputShape[1], outputShape[2]);
  }

  public void allocateParams() {
    conv.allocateParams();
    bn.allocateParams();
  }

  public void flattenWeights() {
    conv.flattenWeights();
  }

  @Override public JtTensor3 forward(JtTensor3 input, boolean training) {
    var x = conv.forward(input, training);
    x = bn.forward(x, training);
    return x;
  }

}
