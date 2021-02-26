package io.vacco.jtinn;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.error.JtErrorFn;
import java.util.Random;

public class XTinn {

  /** @return total error of target to output. */
  public static double totErr(JtErrorFn errorFn, double[] target, double[] out, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
      sum += errorFn.of(target[i], out[i]);
    return sum;
  }

  /** Performs back propagation. */
  public static void bprop(Tinn t, JtErrorFn errorFn, JtActivationFn activationFn,
                           double[] in, double[] tg, double rate) {
    for (int i = 0; i < t.nhid; i++) {
      double sum = 0.0f;
      // Calculate total error change with respect to output.
      for (int j = 0; j < t.nops; j++) {
        double a = errorFn.pd(t.o[j], tg[j]);
        double b = activationFn.pd(t.o[j]);
        int idx = j * t.nhid + i;
        sum += a * b * t.x[idx];
        t.x[idx] -= rate * a * b * t.h[i]; // Correct weights in hidden to output layer.
      }
      // Correct weights in input to hidden layer.
      for (int j = 0; j < t.nips; j++) {
        t.w[i * t.nips + j] -= rate * sum * activationFn.pd(t.h[i]) * in[j];
      }
    }
  }

  /** Performs forward propagation. */
  public static void fprop(Tinn t, JtActivationFn activationFn, double[] in) {
    // Calculate hidden layer neuron values.
    for(int i = 0; i < t.nhid; i++) {
      double sum = 0.0f;
      for(int j = 0; j < t.nips; j++) {
        sum += in[j] * t.w[i * t.nips + j];
      }
      t.h[i] = activationFn.apply(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < t.nops; i++) {
      double sum = 0.0f;
      for(int j = 0; j < t.nhid; j++) {
        sum += t.h[j] * t.x[i * t.nhid + j];
      }
      t.o[i] = activationFn.apply(sum + t.b[1]);
    }
  }

  /** @return an output prediction given an input. */
  public static double[] xtpredict(Tinn t, JtActivationFn activationFn, double[] in) {
    fprop(t, activationFn, in);
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
  public static double xttrain(Tinn t, JtErrorFn errorFn, JtActivationFn activationFn,
                               double[] in, double[] tg, double rate) {
    fprop(t, activationFn, in);
    bprop(t, errorFn, activationFn, in, tg, rate);
    return totErr(errorFn, tg, t.o, t.nops);
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

  /** Randomizes a tinn's weights and biases. */
  public static void wbrand(Tinn t) {
    Random r = new Random();
    for(int i = 0; i < t.nw; i++) { t.w[i] = r.nextDouble() - 0.5f; }
    for(int i = 0; i < t.nb; i++) { t.b[i] = r.nextDouble() - 0.5f; }
  }
}
