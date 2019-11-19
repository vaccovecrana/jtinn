package io.vacco.jtinn;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class XTinn {

  public static double err(double a, double b) {
    return 0.5f * (a - b) * (a - b);
  }

  /** @return partial derivative of error function. */
  public static double pderr(double a, double b) {
    return a - b;
  }

  /** @return total error of target to output. */
  public static double totErr(double[] tg, double[] o, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
      sum += err(tg[i], o[i]);
    return sum;
  }

  /** @return Sigmoid activation. */
  public static double actSigmoid(double a) {
    return 1.0f / (1.0f + Math.exp(-a));
  }

  /** @return partial derivative of activation function. */
  public static double pdact(double a) {
    return a * (1.0f - a);
  }

  /** Performs back propagation. */
  public static void bprop(Tinn t, double[] in, double[] tg, double rate) {
    for (int i = 0; i < t.nhid; i++) {
      double sum = 0.0f;
      // Calculate total error change with respect to output.
      for (int j = 0; j < t.nops; j++) {
        double a = pderr(t.o[j], tg[j]);
        double b = pdact(t.o[j]);
        sum += a * b * t.x[j * t.nhid + i];
        // Correct weights in hidden to output layer.
        t.x[j * t.nhid + i] -= rate * a * b * t.h[i];
      }
      // Correct weights in input to hidden layer.
      for (int j = 0; j < t.nips; j++) {
        t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * in[j];
      }
    }
  }

  /** Performs forward propagation. */
  public static void fprop(Tinn t, double[] in) {
    // Calculate hidden layer neuron values.
    for(int i = 0; i < t.nhid; i++) {
      double sum = 0.0f;
      for(int j = 0; j < t.nips; j++) {
        sum += in[j] * t.w[i * t.nips + j];
      }
      t.h[i] = actSigmoid(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < t.nops; i++) {
      double sum = 0.0f;
      for(int j = 0; j < t.nhid; j++) {
        sum += t.h[j] * t.x[i * t.nhid + j];
      }
      t.o[i] = actSigmoid(sum + t.b[1]);
    }
  }

  /** @return an output prediction given an input. */
  public static double[] xtpredict(Tinn t, double[] in) {
    fprop(t, in);
    return t.o;
  }

  /**
   * Trains a tinn with an input and target output with a learning rate.
   * @param t target network
   * @param in feature array
   * @param tg label array
   * @param rate learning rate
   * @return target to output error.
   */
  public static double xttrain(Tinn t, double[] in, double[] tg, double rate) {
    fprop(t, in);
    bprop(t, in, tg, rate);
    return totErr(tg, t.o, t.nops);
  }

  /** Constructs a tinn with number of inputs, number of hidden neurons, and number of outputs. */
  public static Tinn xtbuild(int nips, int nhid, int nops) {
    Tinn t = new Tinn();
    t.nb = 2; // Tinn only supports one hidden layer so there are two biases.
    t.nw = nhid * (nips + nops);
    t.w = new double[t.nw];
    t.x = new double[nhid * nops];
    t.b = new double[t.nb];
    t.h = new double[nhid];
    t.o = new double[nops];
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    wbrand(t);
    return t;
  }

  /** V1 serialization. */
  public static void xtsave(Tinn t, PrintWriter w0) {
    try (PrintWriter w = w0) {
      w.printf("%d %d %d%n", t.nips, t.nhid, t.nops);
      for (double bias : t.b) { w.printf("%f%n", bias); }
      for (double weight : t.w) { w.printf("%f%n", weight); }
    }
  }

  /** V1 deserialization. */
  public static Tinn xtload(InputStream in) throws IOException {
    try (BufferedReader r = new BufferedReader(new InputStreamReader(in))) {
      String header = r.readLine();
      int[] tData = Arrays.stream(header.split(" ")).mapToInt(Integer::parseInt).toArray();
      Tinn t = xtbuild(tData[0], tData[1], tData[2]);
      for (int i = 0; i < t.nb; i++) { t.b[i] = Double.parseDouble(r.readLine()); }
      for (int i = 0; i < t.nw; i++) { t.w[i] = Double.parseDouble(r.readLine()); }
      return t;
    }
  }

  /** Randomizes a tinn's weights and biases. */
  public static void wbrand(Tinn t) {
    Random r = new Random();
    for(int i = 0; i < t.nw; i++) { t.w[i] = r.nextDouble() - 0.5f; }
    for(int i = 0; i < t.nb; i++) { t.b[i] = r.nextDouble() - 0.5f; }
  }
}
