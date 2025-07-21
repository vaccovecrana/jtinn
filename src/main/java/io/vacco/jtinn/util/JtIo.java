package io.vacco.jtinn.util;

import io.vacco.jtinn.net.JtNetwork;
import java.io.*;

public class JtIo { // TODO still need to determine how to accommodate schema versions.

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
