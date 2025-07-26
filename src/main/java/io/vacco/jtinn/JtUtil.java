package io.vacco.jtinn;

import java.io.*;
import java.util.Arrays;

import static java.lang.String.format;

public class JtUtil {

  public static final long version = 3;

  public static int[] shape3(int c, int h, int w) {
    return new int[] { c, h, w };
  }

  public static void checkTensor(JtTensor3 in0, JtTensor3 in1) {
    if (in0 == null || in1 == null) {
      throw new IllegalArgumentException(
        format("Invalid tensor inputs: [%s], [%s]",
          in0 != null ? in0.size() : null,
          in1 != null ? in1.size() : null
        )
      );
    }
    checkShape(in0.shape, in1.shape);
  }

  public static void checkShape(int[] s0, int[] s1) {
    if (!Arrays.equals(s0, s1)) {
      throw new IllegalArgumentException(
        format("Invalid shape: [%s], [%s]", Arrays.toString(s0), Arrays.toString(s1))
      );
    }
  }

  public static int product(int c, int h, int w) {
    return c * h * w;
  }

  public static void writeNet(JtNetwork3 net, OutputStream out) {
    try {
      var oos = new ObjectOutputStream(out);
      oos.writeObject(net);
      oos.close();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  public static JtNetwork3 readNet(InputStream in) {
    try {
      var ois = new ObjectInputStream(in);
      var net = (JtNetwork3) ois.readObject();
      ois.close();
      return net;
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

}