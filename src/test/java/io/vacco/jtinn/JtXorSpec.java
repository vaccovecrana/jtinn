package io.vacco.jtinn;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.activation.JtLeakyRelu;
import io.vacco.jtinn.activation.JtSigmoid;
import io.vacco.jtinn.error.JtErrorFn;
import io.vacco.jtinn.error.JtMeanSquaredError;
import io.vacco.jtinn.net.JtLayer;
import io.vacco.jtinn.net.JtNetwork;
import io.vacco.jtinn.net.JtOutputLayer;
import io.vacco.jtinn.net.JtRandomInitializer;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.util.*;
import java.util.stream.DoubleStream;

import static j8spec.J8Spec.*;

class XorSample {
  public double[] in;
  public double[] out;

  private static XorSample of(double in0, double in1, double out) {
    XorSample s = new XorSample();
    s.in = new double[] {in0, in1};
    s.out = new double[] {out};
    return s;
  }

  public static List<XorSample> trainingSet() {
    return new ArrayList<>(Arrays.asList(
        of(0, 0, 0),
        of(0, 1, 1),
        of(1, 0, 1),
        of(1, 1, 0)
    ));
  }
}

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtXorSpec {
  static {
    it("Can train a network to learn the XOR function", () -> {
      List<XorSample> tSet = XorSample.trainingSet();
      JtActivationFn fn = new JtSigmoid();
      JtErrorFn eFn = new JtMeanSquaredError();
      JtNetwork net = new JtNetwork(2, new JtRandomInitializer(1234),
          new JtLayer(4, fn),
          new JtOutputLayer(1, fn, eFn)
      );
      double learningRate = 0.08;
      double[] error = new double[1];

      Collections.shuffle(tSet);
      for (XorSample smp : tSet) {
        error[0] = net.train(smp.in, smp.out, learningRate);
        System.out.printf("Error: %s%n", toString(error));
      }

      for (XorSample smp : tSet) {
        double[] guess = net.estimate(smp.in);
        System.out.printf("Sample: %s => %s%n", toString(smp.in), toString(guess));
      }
    });
  }

  private static String toString(double[] in) {
    return Arrays.toString(
        DoubleStream.of(in).mapToObj(d -> String.format("%.14f", d)).toArray()
    );
  }
}
