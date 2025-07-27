package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.util.Arrays;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtBatchNormTest {

  static {
    it("Can apply batch normalization without scaling or shifting", () -> {
      var bn = new JtLayers.JtBatchNormLayer3().init(null);
      bn.calculateOutputShape(new int[]{2, 3, 3});
      bn.allocateParams();
      Arrays.fill(bn.gamma, 1.0f);
      Arrays.fill(bn.beta, 0.0f);
      Arrays.fill(bn.runningMean, 0.0f);
      Arrays.fill(bn.runningVar, 1.0f);
      var input = new JtTensor3(2, 3, 3);
      input.fill(1.0f);
      var out = bn.forward(input, false);
      for (float val : out.data) {
        assertEquals(1.0f, val, 1e-4f);
      }
      System.out.println(JtTestUtil.tensorToString(input));
      System.out.println(JtTestUtil.tensorToString(out));
    });

    it("Can apply batch normalization with scaling and shifting", () -> {
      var bn = new JtLayers.JtBatchNormLayer3().init(null);
      bn.calculateOutputShape(new int[]{2, 3, 3});
      bn.allocateParams();
      Arrays.fill(bn.gamma, 2.0f);
      Arrays.fill(bn.beta, 1.0f);
      Arrays.fill(bn.runningMean, 0.0f);
      Arrays.fill(bn.runningVar, 1.0f);
      var input = new JtTensor3(2, 3, 3);
      input.fill(1.0f);
      var out = bn.forward(input, false);
      for (float val : out.data) {
        assertEquals(3.0f, val, 1e-4f);
      }
      System.out.println(JtTestUtil.tensorToString(input));
      System.out.println(JtTestUtil.tensorToString(out));
    });

    it("Can apply per-channel batch normalization", () -> {
      var bn = new JtLayers.JtBatchNormLayer3().init(null);
      bn.calculateOutputShape(new int[]{2, 3, 3});
      bn.allocateParams();
      Arrays.fill(bn.gamma, 1.0f);
      Arrays.fill(bn.beta, 0.0f);
      bn.runningMean[0] = 0.0f;
      bn.runningMean[1] = 1.0f;
      Arrays.fill(bn.runningVar, 1.0f);
      var input = new JtTensor3(2, 3, 3);
      input.fill(1.0f);
      var out = bn.forward(input, false);
      for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 3; w++) {
          assertEquals(1.0f, out.get(0, h, w), 1e-4f);
          assertEquals(0.0f, out.get(1, h, w), 1e-4f);
        }
      }
      System.out.println(JtTestUtil.tensorToString(input));
      System.out.println(JtTestUtil.tensorToString(out));
    });
  }

}
