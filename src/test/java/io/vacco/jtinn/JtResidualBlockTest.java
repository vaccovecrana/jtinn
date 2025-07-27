package io.vacco.jtinn;

import io.vacco.jtinn.core.*;
import io.vacco.jtinn.pipnet.*;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtResidualBlockTest {

  private static final JtInit.JtRandomInitializer initializer = new JtInit.JtRandomInitializer().init(1234);

  static {
    it("Can perform identity residual block", () -> {
      var rb = new JtResidualBlock().init(1, 1, 1);
      rb.calculateOutputShape(new int[]{1, 5, 5});
      rb.allocateParams();
      initializer.apply(rb);
      rb.flattenWeights();
      // Override to center 1, others 0
      for (int oc = 0; oc < 1; oc++) {
        for (int ic = 0; ic < 1; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              rb.conv1.weights[oc][ic][kh][kw] = (kh == 1 && kw == 1) ? 1.0f : 0.0f;
              rb.conv2.weights[oc][ic][kh][kw] = (kh == 1 && kw == 1) ? 1.0f : 0.0f;
            }
          }
        }
      }
      rb.conv1.b[0] = 0;
      rb.conv2.b[0] = 0;
      rb.bn1.gamma[0] = 1.0f;
      rb.bn1.beta[0] = 0.0f;
      rb.bn1.runningMean[0] = 0.0f;
      rb.bn1.runningVar[0] = 1.0f;
      rb.bn2.gamma[0] = 1.0f;
      rb.bn2.beta[0] = 0.0f;
      rb.bn2.runningMean[0] = 0.0f;
      rb.bn2.runningVar[0] = 1.0f;
      rb.conv1.flattenWeights();
      rb.conv2.flattenWeights();
      var input = new JtTensor3(1, 5, 5);
      input.fill(1.0f);
      var out = rb.forward(input, false);
      for (int h = 0; h < 5; h++) {
        for (int w = 0; w < 5; w++) {
          assertEquals(2.0, out.get(0, h, w), 0.01);
        }
      }
      System.out.println(JtTestUtil.tensorToString(input));
      System.out.println(JtTestUtil.tensorToString(out));
    });

    it("Can perform downsampling residual block", () -> {
      var rb = new JtResidualBlock().init(1, 2, 2);
      rb.calculateOutputShape(new int[]{1, 5, 5});
      rb.allocateParams();
      initializer.apply(rb);
      rb.flattenWeights();
      // Set to center 1, others 0 for conv1, conv2; for shortcut 1x1 weight 1
      for (int oc = 0; oc < 2; oc++) {
        for (int ic = 0; ic < 1; ic++) {
          for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
              rb.conv1.weights[oc][ic][kh][kw] = (kh == 1 && kw == 1) ? 1.0f : 0.0f;
              rb.conv2.weights[oc][ic][kh][kw] = (kh == 1 && kw == 1) ? 1.0f : 0.0f;
            }
          }
        }
        for (int ic = 0; ic < 1; ic++) {
          rb.shortcut.conv.weights[oc][ic][0][0] = 1.0f;
        }
        rb.conv1.b[oc] = 0;
        rb.conv2.b[oc] = 0;
        rb.shortcut.conv.b[oc] = 0;
        rb.bn1.gamma[oc] = 1.0f;
        rb.bn1.beta[oc] = 0.0f;
        rb.bn1.runningMean[oc] = 0.0f;
        rb.bn1.runningVar[oc] = 1.0f;
        rb.bn2.gamma[oc] = 1.0f;
        rb.bn2.beta[oc] = 0.0f;
        rb.bn2.runningMean[oc] = 0.0f;
        rb.bn2.runningVar[oc] = 1.0f;
        rb.shortcut.bn.gamma[oc] = 1.0f;
        rb.shortcut.bn.beta[oc] = 0.0f;
        rb.shortcut.bn.runningMean[oc] = 0.0f;
        rb.shortcut.bn.runningVar[oc] = 1.0f;
      }
      rb.conv1.flattenWeights();
      rb.conv2.flattenWeights();
      rb.shortcut.flattenWeights();
      var input = new JtTensor3(1, 5, 5);
      for (int i = 0; i < 25; i++) {
        input.data[i] = i + 1;
      }
      var out = rb.forward(input, false);
      assertEquals(2, out.get(0,0,0), 0.01);
      assertEquals(6, out.get(0,0,1), 0.01);
      assertEquals(10, out.get(0,0,2), 0.01);
      assertEquals(22, out.get(0,1,0), 0.01);
      assertEquals(26, out.get(0,1,1), 0.01);
      assertEquals(30, out.get(0,1,2), 0.01);
      assertEquals(42, out.get(0,2,0), 0.01);
      assertEquals(46, out.get(0,2,1), 0.01);
      assertEquals(50, out.get(0,2,2), 0.01);
      // Check channel 1 same as channel 0
      assertEquals(out.get(0,1,1), out.get(1,1,1), 0.01);
      System.out.println(JtTestUtil.tensorToString(input));
      System.out.println(JtTestUtil.tensorToString(out));
    });

    it("Can chain conv stem + residual block", () -> {
      var cl = new JtLayers.JtConvLayer3().init(64, 7, 2, 3, new JtActivation.JtRelu());
      var mpl = new JtLayers.JtMaxPoolLayer3().init(3, 2);
      var rb1 = new JtResidualBlock().init(64, 64, 1);
      var rb2 = new JtResidualBlock().init(64, 128, 2);
      var net = new JtNetwork3().init(
        3, 64, 64,
        initializer,
        new JtUpdate.JtSgdUpdater().init(0.01f, 0.99f),
        cl, mpl, rb1, rb2
      );
      var input = new JtTensor3(3, 64, 64);
      input.fill(1.0f);
      var out = net.estimate(input);
      assertEquals(128, out.shape[0]);
      assertEquals(8, out.shape[1]);
      assertEquals(8, out.shape[2]);
      System.out.println("Output shape: " + java.util.Arrays.toString(out.shape));
    });
  }

}