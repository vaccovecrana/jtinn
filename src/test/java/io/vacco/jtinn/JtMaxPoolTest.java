package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.util.Arrays;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtMaxPoolTest {

  static {
    it("Can perform max pooling", () -> {
      var mpl = new JtLayers.JtMaxPoolLayer3().init(2, 2);
      mpl.calculateOutputShape(new int[]{1, 4, 4});
      assertArrayEquals(new int[]{1, 2, 2}, mpl.outputShape);
      var input = new JtTensor3(1, 4, 4);
      for (int i = 0; i < 16; i++) {
        input.data[i] = i + 1;
      }
      System.out.println(JtTestUtil.tensorToString(input));
      var out = mpl.forward(input, false);
      System.out.println(JtTestUtil.tensorToString(out));
      assertEquals(6.0, out.get(0, 0, 0), 0.01);
      assertEquals(8.0, out.get(0, 0, 1), 0.01);
      assertEquals(14.0, out.get(0, 1, 0), 0.01);
      assertEquals(16.0, out.get(0, 1, 1), 0.01);
    });

    it("Can chain conv -> bn -> relu -> maxpool", () -> {
      var relu = new JtActivation.JtRelu();
      var cl = new JtLayers.JtConvLayer3().init(1, 3, 1, 0, null);
      cl.inChannels = 1;
      cl.calculateOutputShape(new int[]{1, 5, 5});
      cl.allocateParams();
      // Set weights to sum filter
      for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
          cl.weights[0][0][kh][kw] = 1.0f;
        }
      }
      cl.b[0] = 0;
      cl.flattenWeights();
      var bn = new JtLayers.JtBatchNormLayer3().init(relu);
      bn.calculateOutputShape(cl.outputShape);
      bn.allocateParams();
      Arrays.fill(bn.gamma, 1.0f);
      Arrays.fill(bn.beta, 0.0f);
      Arrays.fill(bn.runningMean, 0.0f);
      Arrays.fill(bn.runningVar, 1.0f);
      var mpl = new JtLayers.JtMaxPoolLayer3().init(2, 2);
      mpl.calculateOutputShape(bn.outputShape);
      var input = new JtTensor3(1, 5, 5);
      for (int i = 0; i < 25; i++) {
        input.data[i] = i + 1;
      }
      System.out.println(JtTestUtil.tensorToString(input));
      var convOut = cl.forward(input, false);
      System.out.println(JtTestUtil.tensorToString(convOut));
      var bnOut = bn.forward(convOut, false);
      System.out.println(JtTestUtil.tensorToString(bnOut));
      var poolOut = mpl.forward(bnOut, false);
      System.out.println(JtTestUtil.tensorToString(poolOut));
      // Expected max in 2x2 windows of the 3x3 sum output
      // Conv sums (example: center 3x3 sum=7+8+9+12+13+14+17+18+19=117)
      // But since BN is almost identity and ReLU no-op on positive, pool max of those.
      assertEquals(1, poolOut.shape[1]);
      assertEquals(1, poolOut.shape[2]);
      assertEquals(117.0, poolOut.get(0, 0, 0), 0.01);  // Max in the pooled window (top-left 2x2, which includes the center)
    });
  }

}