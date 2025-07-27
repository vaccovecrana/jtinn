package io.vacco.jtinn.core;

import java.io.Serializable;
import java.util.Arrays;

import static io.vacco.jtinn.core.JtUtil.*;

public class JtTensor3 implements Serializable {

  private static final long serialVersionUID = JtUtil.version;

  public float[] data;
  public int[] shape;

  public JtTensor3(int c, int h, int w) {
    this.shape = shape3(c, h, w);
    this.data = new float[product(c, h, w)];
  }

  public JtTensor3(int c, int h, int w, float[] data) {
    this(c, h, w);
    if (data.length != this.data.length) {
      throw new IllegalArgumentException("Data length mismatch");
    }
    System.arraycopy(data, 0, this.data, 0, this.data.length);
  }

  public float get(int c, int h, int w) {
    return data[c * shape[1] * shape[2] + h * shape[2] + w];
  }

  public void set(int c, int h, int w, float val) {
    data[c * shape[1] * shape[2] + h * shape[2] + w] = val;
  }

  public JtTensor3 copy() {
    return new JtTensor3(shape[0], shape[1], shape[2], data);
  }

  public void fill(float val) {
    Arrays.fill(data, val);
  }

  public int size() {
    return data.length;
  }

  public JtTensor3 reshape(int c, int h, int w) {
    if ((c * h * w) != data.length) {
      throw new IllegalArgumentException("Invalid reshape dimensions");
    }
    this.shape = shape3(c, h, w);
    return this;
  }

  public void add(JtTensor3 other) {
    JtUtil.checkTensor(this, other);
    for (int i = 0; i < data.length; i++) {
      data[i] += other.data[i];
    }
  }

}
