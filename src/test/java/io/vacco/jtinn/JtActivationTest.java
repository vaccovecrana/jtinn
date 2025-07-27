package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtActivationTest {

  static {
    it("Can apply ReLU activation", () -> {
      var relu = new JtActivation.JtRelu();
      assertEquals(0, relu.apply(-1), 0);
      assertEquals(0, relu.apply(0), 0);
      assertEquals(1, relu.apply(1), 0);
      assertEquals(0, relu.pd(-1), 0);
      assertEquals(0, relu.pd(0), 0);
      assertEquals(1, relu.pd(1), 0);

      var input = new JtTensor3(1, 1, 3);
      input.data = new float[]{-1, 0, 1};
      var output = new JtTensor3(1, 1, 3);
      var layer = new JtLayers.JtLayer3().init(1, 1, 3, relu);
      layer.applyActivation(input, output);
      assertArrayEquals(new float[]{0, 0, 1}, output.data, 0);
    });
  }

}