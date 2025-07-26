package io.vacco.jtinn;

import java.util.function.Supplier;

public class JtTrain {

  public static class JtSample {

    public JtTensor3 features;
    public JtTensor3 labels;

    public static JtSample of(JtTensor3 features, JtTensor3 labels) {
      var sample = new JtSample();
      sample.features = features;
      sample.labels = labels;
      return sample;
    }
  }

  public interface JtSampler extends Supplier<JtSample[]> { }

  public interface JtStopCondition {
    boolean evaluate(JtNetwork3 network, int epoch, float error);
  }

  public static class JtTrainer {

    private final JtNetwork3 network;
    private final JtStopCondition stopFn;
    private final JtSampler miniBatchSupplier;

    public JtTrainer(JtNetwork3 network, JtStopCondition stopFn,
                     JtSampler miniBatchSupplier) {
      this.network = network;
      this.stopFn = stopFn;
      this.miniBatchSupplier = miniBatchSupplier;
    }

    public void start() {
      int epoch = 0;
      float batchError = -1;
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