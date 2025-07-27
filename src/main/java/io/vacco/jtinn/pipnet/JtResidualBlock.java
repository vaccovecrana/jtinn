package io.vacco.jtinn.pipnet;

import io.vacco.jtinn.core.*;

public class JtResidualBlock extends JtLayers.JtLayer3 {

  private static final long serialVersionUID = JtUtil.version;

  public JtLayers.JtConvLayer3 conv1, conv2;
  public JtLayers.JtBatchNormLayer3 bn1, bn2;
  public JtProjectionLayer shortcut;

  public JtResidualBlock init(int inChannels, int outChannels, int stride) {
    this.conv1 = new JtLayers.JtConvLayer3().init(outChannels, 3, stride, 1, null);
    this.bn1 = new JtLayers.JtBatchNormLayer3().init(new JtActivation.JtRelu());
    this.conv2 = new JtLayers.JtConvLayer3().init(outChannels, 3, 1, 1, null);
    this.bn2 = new JtLayers.JtBatchNormLayer3().init(null);
    if (stride != 1 || inChannels != outChannels) {
      this.shortcut = new JtProjectionLayer().init(inChannels, outChannels, stride);
    }
    this.actFn = new JtActivation.JtRelu();
    return this;
  }

  public void calculateOutputShape(int[] inputShape) {
    conv1.inChannels = inputShape[0];
    conv1.calculateOutputShape(inputShape);
    bn1.calculateOutputShape(conv1.outputShape);
    conv2.inChannels = conv1.outputShape[0];
    conv2.calculateOutputShape(conv1.outputShape);
    bn2.calculateOutputShape(conv2.outputShape);
    this.outputShape = bn2.outputShape;
    if (shortcut != null) {
      shortcut.calculateOutputShape(inputShape);
      JtUtil.checkShape(this.outputShape, shortcut.outputShape);
    } else {
      JtUtil.checkShape(this.outputShape, inputShape);
    }
    this.a  = new JtTensor3(outputShape[0], outputShape[1], outputShape[2]);
    this.ar = new JtTensor3(outputShape[0], outputShape[1], outputShape[2]);
    this.δ  = new JtTensor3(outputShape[0], outputShape[1], outputShape[2]);
  }

  public void allocateParams() {
    conv1.allocateParams();
    bn1.allocateParams();
    conv2.allocateParams();
    bn2.allocateParams();
    if (shortcut != null) {
      shortcut.allocateParams();
    }
  }

  public void flattenWeights() {
    conv1.flattenWeights();
    conv2.flattenWeights();
    if (shortcut != null) {
      shortcut.flattenWeights();
    }
  }

  @Override public JtTensor3 forward(JtTensor3 input, boolean training) {
    var x = conv1.forward(input, training);
    x = bn1.forward(x, training);
    x = conv2.forward(x, training);
    x = bn2.forward(x, training);
    var residual = (shortcut == null) ? input : shortcut.forward(input, training);
    x.add(residual);
    applyActivation(x, x);
    return x;
  }

}
