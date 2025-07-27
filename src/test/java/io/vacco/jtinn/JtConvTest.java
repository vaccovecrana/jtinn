package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtConvTest {

  static {
    it("Can perform convolution without padding, stride=1", () -> {
      var cl = new JtLayers.JtConvLayer3().init(1, 3, 1, 0, null);
      cl.inChannels = 1;
      cl.calculateOutputShape(new int[]{1, 5, 5});
      cl.allocateParams();
      // Set weights to average filter
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          cl.weights[0][0][kh][kw] = 1.0f / 9.0f;
        }
      }
      cl.b[0] = 0;
      cl.flattenWeights();
      var input = new JtTensor3(1, 5, 5);
      input.fill(1.0f);
      var out = cl.forward(input, false);
      assertEquals(3, out.shape[1]);
      assertEquals(3, out.shape[2]);
      for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 3; w++) {
          assertEquals(1.0, out.get(0, h, w), 0.01);
        }
      }
    });

    it("Can perform convolution with padding=1, stride=2", () -> {
      var cl = new JtLayers.JtConvLayer3().init(1, 3, 2, 1, null);
      cl.inChannels = 1;
      cl.calculateOutputShape(new int[]{1, 5, 5});
      cl.allocateParams();
      // Set weights to all 1 (sum filter)
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          cl.weights[0][0][kh][kw] = 1.0f;
        }
      }
      cl.b[0] = 0;
      cl.flattenWeights();
      var input = new JtTensor3(1, 5, 5);
      for (int i = 0; i < 25; i++) {
        input.data[i] = i + 1;
      }
      var out = cl.forward(input, false);
      assertEquals(3, out.shape[1]);
      assertEquals(3, out.shape[2]);
      assertEquals(16.0, out.get(0, 0, 0), 0.01);
      assertEquals(33.0, out.get(0, 0, 1), 0.01);
      assertEquals(28.0, out.get(0, 0, 2), 0.01);
      assertEquals(69.0, out.get(0, 1, 0), 0.01);
      assertEquals(117.0, out.get(0, 1, 1), 0.01);
      assertEquals(87.0, out.get(0, 1, 2), 0.01);
      assertEquals(76.0, out.get(0, 2, 0), 0.01);
      assertEquals(123.0, out.get(0, 2, 1), 0.01);
      assertEquals(88.0, out.get(0, 2, 2), 0.01);
    });

    it("Can apply activation after convolution", () -> {
      var relu = new JtActivation.JtRelu();
      var cl = new JtLayers.JtConvLayer3().init(1, 1, 1, 0, relu);
      cl.inChannels = 1;
      cl.calculateOutputShape(new int[]{1, 3, 3});
      cl.allocateParams();
      cl.weights[0][0][0][0] = 1.0f;
      cl.b[0] = 0;
      cl.flattenWeights();
      var input = new JtTensor3(1, 3, 3);
      input.data = new float[]{-1, 0, 1, -2, 3, -4, 5, -6, 7};
      var out = cl.forward(input, false);
      float[] expected = new float[]{0, 0, 1, 0, 3, 0, 5, 0, 7};
      assertArrayEquals(expected, out.data, 0.01f);
    });
  }

}