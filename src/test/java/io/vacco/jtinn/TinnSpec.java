package io.vacco.jtinn;

import io.vacco.jtinn.activation.JtActivationFn;
import io.vacco.jtinn.activation.JtGelu;
import io.vacco.jtinn.activation.JtLeakyRelu;
import io.vacco.jtinn.activation.JtSigmoid;
import io.vacco.jtinn.error.JtErrorFn;
import io.vacco.jtinn.error.JtMeanSquaredError;
import io.vacco.jtinn.io.JtSerializer;
import io.vacco.jtinn.io.JtTextSerializer;
import j8spec.annotation.DefinedOrder;
import j8spec.junit.J8SpecRunner;
import org.junit.runner.RunWith;

import java.io.*;
import java.nio.file.Files;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.*;

import static j8spec.J8Spec.*;
import static org.junit.Assert.*;

class DigitSample {
  public double[] features;
  public double[] labels;
}

@DefinedOrder
@RunWith(J8SpecRunner.class)
public class TinnSpec {

  public static Tinn tinn;

  static {
    describe("JTinn", () -> {
      it("Can perform training on a reference data set", () -> {

        File f = new File("./src/test/resources/semeion.data");
        JtErrorFn errorFn = new JtMeanSquaredError();
        JtActivationFn activationFn = new JtSigmoid(); // new JtLeakyRelu(0.002); // new JtGelu();

        try (Stream<String> lines = Files.lines(f.toPath())) {
          // Load the training set.
          List<DigitSample> samples = lines.map(row -> Arrays.stream(row.split(" "))
              .map(Double::parseDouble).mapToDouble(Double::doubleValue)
              .toArray()).map(dArr -> {
            DigitSample sample = new DigitSample();
            sample.features = Arrays.copyOfRange(dArr, 0, 256);
            sample.labels = Arrays.copyOfRange(dArr, 256, 266);
            return sample;
          }).collect(Collectors.toList());

          int nips = 256;
          int nops = 10;

          // Hyper Parameters.
          // Learning rate is annealed and thus not constant.
          // It can be fine tuned along with the number of hidden layers.
          // Feel free to modify the anneal rate.
          // The number of iterations can be changed for stronger training.
          double rate = 1.0f;
          int nhid = 28;
          float anneal = 0.99f;
          int iterations = 128;

          // Train, baby, train.
          tinn = XTinn.xtbuild(nips, nhid, nops);

          for(int i = 0; i < iterations; i++) {
            double error = 0.0f;
            Collections.shuffle(samples);
            for (DigitSample ds : samples) {
              error += XTinn.xttrain(tinn, errorFn, activationFn, ds.features, ds.labels, rate);
            }
            System.out.printf("error %.12f :: learning rate %f\n", error / samples.size(), rate);
            rate *= anneal;
          }

          // Now we do a prediction with the neural network we loaded from disk.
          // Ideally, we would also load a testing set to make the prediction with,
          // but for the sake of brevity here we just reuse the training set from earlier.
          // One data set is picked at random (zero index of input and target arrays is enough
          // as they were both shuffled earlier).
          DigitSample s0 = samples.get(0);

          File tinnF = new File("./build/digits.tinn");
          JtSerializer s = new JtTextSerializer(new PrintWriter(new FileWriter(tinnF)));
          s.write(tinn);
          Tinn lTinn = s.read(new FileInputStream(tinnF));

          double[] predict = XTinn.xtpredict(lTinn, activationFn, s0.features);

          DecimalFormat df = new DecimalFormat("0.00");
          String[] tgtFmt = Arrays.stream(s0.labels).mapToObj(df::format).toArray(String[]::new);
          String[] predfmt = Arrays.stream(predict).mapToObj(df::format).toArray(String[]::new);

          System.out.println("======================");
          System.out.printf("Target:  %s%n", Arrays.toString(tgtFmt));
          System.out.printf("Predict: %s%n", Arrays.toString(predfmt));
          System.out.println("======================");
        } catch (Exception e) { throw new IllegalStateException(e); }
      });

      it("Can serialize/deserialize a trained network", () -> {
        File tmp = File.createTempFile("tinn", "data");
        JtSerializer s = new JtTextSerializer(new PrintWriter(new FileWriter(tmp)));
        s.write(tinn);
        Tinn t0 = s.read(new FileInputStream(tmp));
        assertNotNull(t0);
      });
    });
  }
}
