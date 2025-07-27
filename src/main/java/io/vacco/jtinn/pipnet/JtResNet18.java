package io.vacco.jtinn.pipnet;

import io.vacco.jtinn.core.*;
import java.util.*;

public class JtResNet18 extends JtNetwork3 {

  public JtResNet18 init(int inputH, int inputW, JtInit.JtParamInitializer initializer) {
    var relu = new JtActivation.JtRelu();
    var stemConv = new JtLayers.JtConvLayer3().init(64, 7, 2, 3, null);
    var stemBn = new JtLayers.JtBatchNormLayer3().init(relu);
    var stemPool = new JtLayers.JtMaxPoolLayer3().init(3, 2);
    stemPool.padding = 1;

    var layers = new ArrayList<JtLayers.JtLayer3>();
    layers.add(stemConv);
    layers.add(stemBn);
    layers.add(stemPool);

    int[] stageChannels = {64, 128, 256, 512};
    int prevChannels = 64;

    for (int i = 0; i < 4; i++) {
      int outChannels = stageChannels[i];
      int stride = (i == 0) ? 1 : 2;
      var rb1 = new JtResidualBlock().init(prevChannels, outChannels, stride);
      var rb2 = new JtResidualBlock().init(outChannels, outChannels, 1);
      layers.add(rb1);
      layers.add(rb2);
      prevChannels = outChannels;
    }

    super.init(3, inputH, inputW, initializer, null, layers.toArray(new JtLayers.JtLayer3[0]));
    return this;
  }

  public JtTensor3 getFeatureMap(JtTensor3 input) {
    return estimate(input);
  }

}