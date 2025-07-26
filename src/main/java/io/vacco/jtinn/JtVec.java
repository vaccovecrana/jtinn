package io.vacco.jtinn;

import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class JtVec {
  private static void load(String path) {
    try (var is = JtVec.class.getResourceAsStream(path)) {
      if (is == null) {
        throw new RuntimeException(path + " not found in classpath");
      }
      var tempFile = Files.createTempFile("libjtinn", ".jni");
      Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
      System.load(tempFile.toAbsolutePath().toString());
    } catch (Exception e) {
      throw new RuntimeException("Failed to load libjtinn", e);
    }
  }

  static {
    var osName = System.getProperty("os.name").toLowerCase();
    var osArch = System.getProperty("os.arch").toLowerCase();
    if (osName.contains("linux") && osArch.equals("amd64")) {
      load("/io/vacco/jtinn/libjtinn.so");
    } else if (osName.contains("mac os x") && osArch.equals("x86_64")) {
      load("/io/vacco/jtinn/libjtinn.dylib");
    }
  }

  public static native void avxFloatMatMul(float[] in, float[][] w,
                                           float[] b, float[] out,
                                           int inSize, int outSize);

  public static native void sseInt8MatMul(byte[] in, byte[][] w,
                                          float[] b, float[] out,
                                          int inSize, int outSize, float scale);
}
