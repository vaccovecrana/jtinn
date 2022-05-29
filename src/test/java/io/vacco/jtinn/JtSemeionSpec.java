package io.vacco.jtinn;

import io.vacco.jtinn.activation.*;
import io.vacco.jtinn.error.*;
import io.vacco.jtinn.layer.JtLayer;
import io.vacco.jtinn.layer.JtOutputLayer;
import io.vacco.jtinn.net.*;
import io.vacco.jtinn.train.*;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;
import java.io.*;

import static io.vacco.jtinn.util.JtIo.*;
import static io.vacco.jtinn.JtSpecUtil.*;
import static j8spec.J8Spec.*;

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class JtSemeionSpec {

  public static final JtActivationFn reluFn = new JtLeakyRelu().withAlpha(0.04f);
  public static final JtUpdater reluUpd = new JtSgdUpdater().init(0.025f, 0.99f);

  public static final JtActivationFn sigmoidFn = new JtSigmoid();
  public static final JtUpdater sigmoidUpd = new JtSgdUpdater().init(1.0f, 0.99f);

  public static final String sigmoid = "sigmoid.ser", leakyRelu = "leaky-relu.ser";

  private static void eval(JtActivationFn actFn, JtUpdater updFn, String netName) throws Exception {
    System.out.printf("%n================ %s ================%n", actFn.getClass().getCanonicalName());
    JtErrorFn errFn = new JtMeanSquaredError();
    JtNetwork net = new JtNetwork().init(
        256,
        new JtRandomInitializer().init(1234),
        updFn,
        new JtLayer().init(28, actFn),
        new JtOutputLayer().init(10, actFn, errFn)
    );

    File netFile = new File("./build", netName);

    writeNet(net, new FileOutputStream(netFile));

    float[] err = new float[1];
    JtPredictionSampleSupplier digits = new SemeionData();
    JtTrainer trainer = new JtTrainer(net,
        (network, epoch, error) -> {
          err[0] = error;
          if (epoch % 20 == 0) {
            System.out.printf("Epoch [%s] Delta err: [%s]%n", epoch, asString14d(err));
          }
          return epoch == 200;
        },
        digits
    );

    trainer.start();
    writeNet(net, new FileOutputStream(netFile));

    JtNetwork net0 = readNet(new FileInputStream(netFile));
    JtPredictionSample sample = digits.get()[0];
    float[] guess = net0.estimate(sample.features);

    System.out.printf("Sample: %s%n", asString2d(sample.features));
    System.out.printf("Guess:  %s%n", asString2d(guess));
    System.out.printf("Actual: %s%n", asString2d(sample.labels));
  }

  static {
    it("Can recognize hand-written digits (Sigmoid)", () -> eval(sigmoidFn, sigmoidUpd, sigmoid));
    it("Can recognize hand-written digits (Leaky ReLU)", () -> eval(reluFn, reluUpd, leakyRelu));
  }
}
