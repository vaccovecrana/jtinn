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
public class JtPipNetTest {

  private static final JtInit.JtRandomInitializer initializer = new JtInit.JtRandomInitializer().init(1234);

  static {
    it("Can forward through head to produce maps", () -> {
      int N = 2, C = 0;
      var pip = new JtPipNet(N, C).init(initializer);
      var features = new JtTensor3(512, 8, 8);
      features.fill(1.0f);
      var head = pip.layers[pip.layers.length - 1];
      var maps = head.forward(features, false);
      assertArrayEquals(new int[]{N + 2 * N + 2 * C * N, 8, 8}, maps.shape);
    });

    it("Can decode landmarks from synthetic maps", () -> {
      int N = 2, C = 1;
      var pip = new JtPipNet(N, C);
      pip.setNeighborIndices(new int[][] { {1}, {0} });
      var maps = new JtTensor3(N + 2 * N + 2 * C * N, 8, 8);
      maps.set(0, 4, 4, 1.0f); // score 0 at center
      maps.set(1, 3, 3, 1.0f); // score 1 at (3,3)
      // offsets 0
      // neighbor offsets: for j=1, k=0 (to 0), set dx=-32, dy=-32 (to avg towards center)
      int neighBase = N + 2 * N;
      maps.set(neighBase + 2 * C * 1 + 0, 3, 3, -32.0f); // dx for neighbor 0 of j=1
      maps.set(neighBase + 2 * C * 1 + 1, 3, 3, -32.0f); // dy
      // for j=0, k=0 to 1, set dx=32, dy=32
      maps.set(neighBase + 0, 4, 4, 32.0f);
      maps.set(neighBase + 1, 4, 4, 32.0f);
      var lms = pip.getLandmarks(maps, 0.0f, 256, 256);
      assertEquals(N, lms.size());
      float[] lm0 = lms.get(0);
      float[] lm1 = lms.get(1);
      assertEquals(112.0f, (lm0[0] + lm1[0]) / 2, 5.0f); // averaged
      assertEquals(112.0f, (lm0[1] + lm1[1]) / 2, 5.0f);
    });

    it("Can detect landmarks on synthetic image", () -> {
      int N = 1, C = 0;
      var pip = new JtPipNet(N, C).init(initializer);
      pip.setNeighborIndices(new int[][]{new int[0]});
      var head = (JtLayers.JtConvLayer3) pip.layers[pip.layers.length - 1];
      for (float[] row : head.wFlat) {
        Arrays.fill(row, 0.0f);
      }
      Arrays.fill(head.b, 0.0f);
      for (int i = 0; i < N; i++) {
        head.b[i] = 1.0f;
      }
      float[] pixels = new float[256 * 256 * 3];
      Arrays.fill(pixels, 127.0f);
      var lms = pip.detectLandmarks(pixels, 512, 512);
      assertEquals(N, lms.size());
      var lm = lms.get(0);
      System.out.println(Arrays.toString(lm));
      assertTrue(lm != null && lm.length == 2);
      assertTrue(lm[0] >= 0 && lm[0] <= 512 && lm[1] >= 0 && lm[1] <= 512);
    });
  }

}
