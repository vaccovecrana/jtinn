package io.vacco.jtinn;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class JtSpecUtil {

  public static String asString2d(float[] in) {
    String[] sa = new String[in.length];
    for (int i = 0; i < in.length; i++) {
      sa[i] = String.format("%.2f", in[i]);
    }
    return Arrays.toString(sa);
  }

  public static String asString14d(float[] in) {
    String[] sa = new String[in.length];
    for (int i = 0; i < in.length; i++) {
      sa[i] = String.format("%.14f", in[i]);
    }
    return Arrays.toString(sa);
  }

}
