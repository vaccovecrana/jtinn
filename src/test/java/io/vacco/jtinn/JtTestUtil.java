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

}
