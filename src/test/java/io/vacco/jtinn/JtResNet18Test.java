package io.vacco.jtinn;

import io.vacco.jtinn.core.*;
import io.vacco.jtinn.pipnet.*;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.util.Arrays;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtResNet18Test {

  // See etc/pipnet-test/pipnet-data.py
  private static final float[] pytorchOutput = new float[512];

  static {
    Arrays.fill(pytorchOutput, 4389934852066194868606121017344.00000000f);
  }

  private static final JtInit.JtRandomInitializer initializer = new JtInit.JtRandomInitializer().init(1234);

  private static void setConvWeights(JtLayers.JtConvLayer3 cl, float weightVal) {
    for (float[][][] oc : cl.weights) {
      for (float[][] ic : oc) {
        for (float[] kh : ic) {
          Arrays.fill(kh, weightVal);
        }
      }
    }
    Arrays.fill(cl.b, 0.0f);
    cl.flattenWeights();
  }

  private static void overrideWeightsToConstant(JtNetwork3 net, float weightVal) {
    for (JtLayers.JtLayer3 l : net.layers) {
      if (l instanceof JtLayers.JtConvLayer3) {
        setConvWeights((JtLayers.JtConvLayer3) l, weightVal);
      } else if (l instanceof JtResidualBlock) {
        JtResidualBlock rb = (JtResidualBlock) l;
        setConvWeights(rb.conv1, weightVal);
        setConvWeights(rb.conv2, weightVal);
        if (rb.shortcut != null) {
          setConvWeights(rb.shortcut.conv, weightVal);
        }
        rb.flattenWeights();
      }
    }
  }

  static {

    it("Can initialize and forward through ResNet-18 backbone", () -> {
      var resNet = new JtResNet18().init(256, 256, initializer);
      overrideWeightsToConstant(resNet, 0.1f);
      var input = new JtTensor3(3, 256, 256);
      input.fill(1.0f);
      var output = resNet.getFeatureMap(input);
      assertArrayEquals(new int[]{512, 8, 8}, output.shape);
      // Optional: Check that values are reasonable (e.g., no NaNs, some positive after ReLUs)
      var hasPositive = false;
      for (float val : output.data) {
        assertFalse(Float.isNaN(val));
        if (val > 0) hasPositive = true;
      }
      assertTrue(hasPositive);
      System.out.println("Output shape: " + java.util.Arrays.toString(output.shape));
    });

    it("Matches PyTorch ResNet-18 output for same input and weights", () -> {
      var resNet = new JtResNet18().init(32, 32, initializer);
      overrideWeightsToConstant(resNet, 0.1f);
      var input = new JtTensor3(3, 32, 32);
      input.fill(1.0f);
      var output = resNet.getFeatureMap(input);
      assertArrayEquals(JtUtil.shape3(512, 1, 1), output.shape);

      System.out.println(JtTestUtil.tensorToString(input));
      System.out.println("-------------------------------");
      System.out.println(JtTestUtil.tensorToString(output));

      var maxDiff = 0.0f;
      for (int i = 0; i < output.data.length; i++) {
        float diff = Math.abs((output.data[i] - pytorchOutput[i]) / pytorchOutput[i]);
        if (diff > maxDiff) maxDiff = diff;
        assertTrue("Output mismatch at index " + i + ": Java=" + output.data[i] + ", PyTorch=" + pytorchOutput[i], diff <= 1e-3f);
      }
      System.out.println("Maximum relative difference: " + maxDiff);
    });
  }

}