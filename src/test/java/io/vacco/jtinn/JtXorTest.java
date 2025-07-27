package io.vacco.jtinn;

import io.vacco.jtinn.core.*;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import static j8spec.J8Spec.*;
import static io.vacco.jtinn.JtTestUtil.asString8f;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtXorTest {
  static {
    it("Can train a network to learn the XOR function", () -> {
      var fn = new JtActivation.JtSigmoid();
      var eFn = new JtError.JtMeanSquaredError();
      var net = new JtNetwork3().init(
        2, 1, 1,
        new JtInit.JtRandomInitializer().init(1234),
        new JtUpdate.JtSgdUpdater().init(1, 1),
        new JtLayers.JtLayer3().init(4, 1, 1, fn),
        new JtLayers.JtOutputLayer3().init(1, 1, 1, fn).errFn(eFn)
      );
      var err = new float[1];
      var xor = new XorData();
      var trainer = new JtTrain.JtTrainer(
        net,
        (network, epoch, error) -> {
          err[0] = error;
          if (epoch % 1000 == 0) {
            System.out.printf("Epoch [%03d] Delta err: %s%n", epoch, asString8f(err));
          }
          return epoch == 9000;
        },
        xor
      );

      trainer.start();
      for (var smp : xor.get()) {
        var guess = net.estimate(smp.features);
        System.out.printf(
          "Sample: %s => %s%n",
          asString8f(smp.features.data),
          asString8f(guess.data)
        );
      }
    });
  }
}
