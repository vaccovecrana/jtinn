package io.vacco.jtinn;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.activation.JtSigmoid;
import io.vacco.jtinn.error.JtErrorFn;
import io.vacco.jtinn.error.JtMeanSquaredError;
import io.vacco.jtinn.net.*;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static io.vacco.jtinn.JtSpecUtil.asString14d;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtXorSpec {
  static {
    it("Can train a network to learn the XOR function", () -> {
      JtActivationFn fn = new JtSigmoid();
      JtErrorFn eFn = new JtMeanSquaredError();
      JtNetwork net = new JtNetwork().init(2,
          new JtRandomInitializer().init(1234),
          new JtSgdUpdater().init(1, 1),
          new JtLayer().init(4, fn),
          new JtOutputLayer().init(1, fn, eFn)
      );
      double[] err = new double[1];
      XorData xor = new XorData();
      JtTrainer trainer = new JtTrainer(
          net,
          (network, epoch, error) -> {
            err[0] = error;
            System.out.printf("Delta err: %s%n", asString14d(err));
            return epoch == 8192;
          },
          xor
      );

      trainer.start();
      for (JtPredictionSample smp : xor.get()) {
        double[] guess = net.estimate(smp.features);
        System.out.printf("Sample: %s => %s%n", asString14d(smp.features), asString14d(guess));
      }
    });
  }
}
