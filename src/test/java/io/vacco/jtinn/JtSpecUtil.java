package io.vacco.jtinn;

import java.util.Arrays;

public class JtSpecUtil {

  private static Float[] castFa(float[] in) {
    var out = new Float[in.length];
    for (int i = 0; i < in.length; i++) {
      out[i] = in[i];
    }
    return out;
  }

  public static String asString2d(float[] in) {
    return Arrays.toString(
      Arrays.stream(castFa(in)).map(f -> String.format("%.2f", f)).toArray()
    );
  }

  public static String asString14d(float[] in) {
    return Arrays.toString(
      Arrays.stream(castFa(in)).map(f -> String.format("%.14f", f)).toArray()
    );
  }

}
