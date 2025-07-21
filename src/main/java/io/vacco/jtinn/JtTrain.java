package io.vacco.jtinn;

import java.util.function.Supplier;

public class JtTrain {

  public static class JtSample {

    public double[] features;
    public double[] labels;

    public static JtSample of(double[] features, double[] labels) {
      var sample = new JtSample();
      sample.features = features;
      sample.labels = labels;
      return sample;
    }
  }

  public interface JtSampler extends Supplier<JtSample[]> { }

  public interface JtStopCondition {
    boolean evaluate(JtNetwork network, int epoch, double error);
  }

  public static class JtTrainer {

    private final JtNetwork network;
    private final JtStopCondition stopFn;
    private final JtSampler miniBatchSupplier;

    public JtTrainer(JtNetwork network, JtStopCondition stopFn,
                     JtSampler miniBatchSupplier) {
      this.network = network;
      this.stopFn = stopFn;
      this.miniBatchSupplier = miniBatchSupplier;
    }

    public void start() {
      int epoch = 0;
      double batchError = -1;
      while (!stopFn.evaluate(network, epoch, batchError)) {
        epoch++;
        batchError = 0;
        var samples = miniBatchSupplier.get();
        for (var sample : samples) {
          batchError += network.train(sample.features, sample.labels);
        }
        batchError /= samples.length;
        network.updater.onEpochEnd(epoch);
      }
    }
  }

}
