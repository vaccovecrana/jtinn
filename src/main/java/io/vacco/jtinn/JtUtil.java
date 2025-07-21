package io.vacco.jtinn;

import java.io.*;

import static java.lang.String.format;

public class JtUtil {

  public static final long version = 2;

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

  public static void writeNet(JtNetwork net, OutputStream out) {
    try {
      var oos = new ObjectOutputStream(out);
      oos.writeObject(net);
      oos.close();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  public static JtNetwork readNet(InputStream in) {
    try {
      var ois = new ObjectInputStream(in);
      var net = (JtNetwork) ois.readObject();
      ois.close();
      return net;
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

}
