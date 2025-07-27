package io.vacco.jtinn;

import io.vacco.jtinn.core.JtVec;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtVecTest {
  static {
    it("Can perform AVX float matrix multiplication", () -> {
      var in  = new float[8];
      var w   = new float[8][8];
      var b   = new float[8];
      var out = new float[8];
      for (int i = 0; i < 8; i++) {
        in[i] = i + 1;
        b[i] = 0;
        for (int j = 0; j < 8; j++) {
          w[i][j] = (i == j ? 1.0f : 0.0f);
        }
      }
      JtVec.avxFloatMatMul(in, w, b, out, 8, 8);
      for (int i = 0; i < 8; i++) {
        if (Math.abs(out[i] - (i + 1)) > 1e-6) {
          throw new AssertionError("AVX float matrix multiplication failed");
        }
      }
    });

    it("Can perform SSE int8 matrix multiplication", () -> {
      var in  = new byte[16];
      var w   = new byte[16][16];
      var b   = new float[16];
      var out = new float[16];
      for (int i = 0; i < 16; i++) {
        in[i] = (byte) (i + 1);
        b[i] = 0;
        for (int j = 0; j < 16; j++) {
          w[i][j] = (byte) (i == j ? 1 : 0);
        }
      }
      JtVec.sseInt8MatMul(in, w, b, out, 16, 16, 1.0f);
      for (int i = 0; i < 16; i++) {
        if (Math.abs(out[i] - (i + 1)) > 1e-6) {
          throw new AssertionError("SSE int8 matrix multiplication failed");
        }
      }
    });
  }
}