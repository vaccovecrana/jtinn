package io.vacco.jtinn;

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
      var fn = new JtActivation.JtSigmoid();
      var eFn = new JtError.JtMeanSquaredError();
      var net = new JtNetwork().init(2,
          new JtInit.JtRandomInitializer().init(1234),
          new JtUpdate.JtSgdUpdater().init(1, 1),
          new JtLayers.JtLayer().init(4, fn),
          new JtLayers.JtOutputLayer().init(1, fn, eFn)
      );
      var err = new float[1];
      var xor = new XorData();
      var trainer = new JtTrain.JtTrainer(
          net,
          (network, epoch, error) -> {
            err[0] = error;
            if (epoch % 1000 == 0) {
              System.out.printf("Epoch [%s] Delta err: %s%n", epoch, asString14d(err));
            }
            return epoch == 9000;
          },
          xor
      );

      trainer.start();
      net.quantize(xor, 16);
      for (var smp : xor.get()) {
        var guess = net.estimate(smp.features);
        System.out.printf("Sample: %s => %s%n", asString14d(smp.features), asString14d(guess));
      }
    });
  }
}
