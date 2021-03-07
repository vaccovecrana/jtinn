package io.vacco.jtinn;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.activation.JtSigmoid;
import io.vacco.jtinn.error.JtErrorFn;
import io.vacco.jtinn.error.JtMeanSquaredError;
import io.vacco.jtinn.net.*;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static io.vacco.jtinn.JtSpecUtil.*;
import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtSemeionSpec {
  static {
    it("Can recognize hand-written digits", () -> {
      JtActivationFn actFn = new JtSigmoid();
      JtErrorFn errFn = new JtMeanSquaredError();
      JtNetwork net = new JtNetwork().init(
          256,
          new JtRandomInitializer().init(1234),
          new JtSgdUpdater().init(1.0, 0.99),
          new JtLayer().init(28, actFn),
          new JtOutputLayer().init(10, actFn, errFn)
      );
      double[] err = new double[1];
      JtPredictionSampleSupplier digits = new SemeionData();
      JtTrainer trainer = new JtTrainer(net,
          (network, epoch, error) -> {
            err[0] = error;
            System.out.printf("Delta err: %s%n", asString14d(err));
            return epoch == 200;
          },
          digits
      );

      trainer.start();
      JtPredictionSample sample = digits.get()[0];
      double[] guess = net.estimate(sample.features);
      System.out.printf("Sample: %s%n", asString2d(sample.features));
      System.out.printf("Guess:  %s%n", asString2d(guess));
      System.out.printf("Actual: %s%n", asString2d(sample.labels));
    });
  }
}
