package io.vacco.jtinn;

import java.util.Arrays;
import java.util.stream.DoubleStream;

public class JtSpecUtil {

  public static String asString2d(double[] in) {
    return Arrays.toString(
        DoubleStream.of(in).mapToObj(d -> String.format("%.2f", d)).toArray()
    );
  }

  public static String asString14d(double[] in) {
    return Arrays.toString(
        DoubleStream.of(in).mapToObj(d -> String.format("%.14f", d)).toArray()
    );
  }

}
