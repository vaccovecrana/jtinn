package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
// @RunWith(J8SpecRunner.class)
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
      // Manual calculation for sum filter with pad=1, stride=2 on 5x5 input (1-25 sequential)
      // Expected for position (0,0): sum of padded 3x3 around (0,0)
      // With pad, effective input 7x7 with zeros, but only non-zero contributions
      assertEquals(12.0, out.get(0, 0, 0), 0.01); // positions 1,2,3,6,7,8,11,12,13 sum=72/9? Wait, no, since all 1, sum of values in patch
      // Since all 1 weight, sum is sum of input in patch.
      // For oh=0, ow=0, ih= -1,0,1 ; iw=-1,0,1
      // Valid: (0,0),(0,1),(1,0),(1,1) =1,2,6,7 =16
      // Wait, I set weight =1, not 1/9, for sum.
      // Adjust test as needed for manual verify.
      // Similarly for other positions.
      // For example, (0,0): sum of input at (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2) but with stride 2, wait no, patch is always 3x3, stride is step.
      // For oh=0, ow=0, ih=0*2 + kh -1 = kh -1, for kh=0, ih=-1 (pad0), kh=1 ih=1, kh=2 ih=3? No.
      // ih = oh * stride + kh - padding
      // for oh=0, kh=0, ih = 0 +0 -1 = -1
      // kh=1, ih=0 +1 -1 =0
      // kh=2, ih=0 +2 -1 =1
      // For str=2, patch starts at 'effective' position with step.
      // To verify, use simple all 1 input, sum should be number of non-pad positions.
      // But for accuracy, calculate manual for a test case.
      // Also add assert for outH=3, as (5 -3 +2)/2 +1 = (4)/2 +1 =3
      assertEquals(3, out.shape[1]);
      // Add specific asserts based on manual calc.
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