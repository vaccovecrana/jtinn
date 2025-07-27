package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtTensor3Test {

  static {
    it("Can manage tensor data", () -> {
      var t = new JtTensor3(2, 3, 4);
      t.fill(0);
      assertEquals(0, t.get(0, 0, 0), 0.01);
      t.set(1, 2, 3, 42);
      assertEquals(42, t.get(1, 2, 3), 0.01);
      var tc = t.copy();
      assertEquals(42, tc.get(1, 2, 3), 0.01);
      t.set(1, 2, 3, 0);
      assertNotEquals("Not a deep copy", 42, tc.get(1, 2, 3));
      var tr = t.reshape(1, 1, 24);
      if (tr.size() != 24 || tr.shape[0] != 1 || tr.shape[1] != 1 || tr.shape[2] != 24) {
        throw new AssertionError("Reshape failed");
      }
      System.out.println(JtTestUtil.tensorToString(t));
      System.out.println(JtTestUtil.tensorToString(tc));
    });
  }

}