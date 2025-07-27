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
public class JtResNet18Test {

  private static final JtInit.JtRandomInitializer initializer = new JtInit.JtRandomInitializer().init(1234);

  static {
    it("Can initialize and forward through ResNet-18 backbone", () -> {
      var resNet = new JtResNet18().init(256, 256, initializer);
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
  }

}