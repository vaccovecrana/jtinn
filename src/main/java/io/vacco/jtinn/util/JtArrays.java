package io.vacco.jtinn.util;

import static java.lang.String.format;

public class JtArrays {

  public static void checkSize(double[] in0, double[] in1) {
    if (in0 == null || in1 == null) {
      throw new IllegalArgumentException(
          format("Invalid array inputs: [%s], [%s]",
              in0 != null ? in0.length : null,
              in1 != null ? in1.length : null
          )
      );
    }
    if (in0.length != in1.length) {
      throw new IllegalArgumentException(
          format("Invalid input size: [%s], [%s]", in0.length, in1.length)
      );
    }
  }
}
