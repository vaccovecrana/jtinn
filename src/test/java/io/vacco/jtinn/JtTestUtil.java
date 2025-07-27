package io.vacco.jtinn;

import java.util.Arrays;

public class JtTestUtil {

  private static Float[] castFa(float[] in) {
    var out = new Float[in.length];
    for (int i = 0; i < in.length; i++) {
      out[i] = in[i];
    }
    return out;
  }

  public static String asString2f(float[] in) {
    return Arrays.toString(
      Arrays.stream(castFa(in)).map(f -> String.format("%.02f", f)).toArray()
    );
  }

  public static String asString8f(float[] in) {
    return Arrays.toString(
      Arrays.stream(castFa(in)).map(f -> String.format("%.08f", f)).toArray()
    );
  }

  public static String tensorToString(JtTensor3 t, String format) {
    if (t == null || t.data == null || t.shape == null || t.shape.length != 3) {
      return "[Invalid or empty tensor]";
    }
    StringBuilder sb = new StringBuilder();
    for (int c = 0; c < t.shape[0]; c++) {
      sb.append("Channel ").append(c).append(":\n");
      for (int h = 0; h < t.shape[1]; h++) {
        for (int w = 0; w < t.shape[2]; w++) {
          sb.append(String.format(format, t.get(c, h, w))).append(" ");
        }
        sb.append("\n");
      }
    }
    return sb.toString();
  }

  public static String tensorToString(JtTensor3 t) {
    return tensorToString(t, "%08.3f");
  }

}
