package io.vacco.jtinn;

import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.io.*;

import static io.vacco.jtinn.JtSpecUtil.*;
import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtSemeionSpec {

  public static final JtActivation.JtActivationFn reluFn = new JtActivation.JtLeakyRelu().withAlpha(0.08);
  public static final JtUpdate.JtUpdater reluUpd = new JtUpdate.JtSgdUpdater().init(0.025, 0.99);

  public static final JtActivation.JtActivationFn sigmoidFn = new JtActivation.JtSigmoid();
  public static final JtUpdate.JtUpdater sigmoidUpd = new JtUpdate.JtSgdUpdater().init(1.0, 0.99);

  public static final String sigmoid = "sigmoid.ser", leakyRelu = "leaky-relu.ser";

  private static void eval(JtActivation.JtActivationFn actFn, JtUpdate.JtUpdater updFn, String netName) throws Exception {
    System.out.printf("%n================ %s ================%n", actFn.getClass().getCanonicalName());
    var errFn = new JtError.JtMeanSquaredError();
    var net = new JtNetwork().init(
        256,
        new JtInit.JtRandomInitializer().init(1234),
        updFn,
        new JtLayers.JtLayer().init(28, actFn),
        new JtLayers.JtOutputLayer().init(10, actFn, errFn)
    );

    var netFile = new File("./build", netName);

    JtUtil.writeNet(net, new FileOutputStream(netFile));

    var err = new double[1];
    var digits = new SemeionData();
    var trainer = new JtTrain.JtTrainer(net,
        (network, epoch, error) -> {
          err[0] = error;
          if (epoch % 20 == 0) {
            System.out.printf("Epoch [%s] Delta err: %s%n", epoch, asString14d(err));
          }
          return epoch == 200;
        },
        digits
    );

    trainer.start();
    JtUtil.writeNet(net, new FileOutputStream(netFile));

    var net0 = JtUtil.readNet(new FileInputStream(netFile));
    var sample = digits.get()[0];
    var guess = net0.estimate(sample.features);

    System.out.printf("Sample: %s%n", asString2d(sample.features));
    System.out.printf("Guess:  %s%n", asString2d(guess));
    System.out.printf("Actual: %s%n", asString2d(sample.labels));
  }

  static {
    it("Can recognize hand-written digits (Sigmoid)", () -> eval(sigmoidFn, sigmoidUpd, sigmoid));
    it("Can recognize hand-written digits (Leaky ReLU)", () -> eval(reluFn, reluUpd, leakyRelu));
  }
}
