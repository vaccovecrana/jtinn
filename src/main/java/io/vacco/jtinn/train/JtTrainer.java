package io.vacco.jtinn.train;

import io.vacco.jtinn.net.JtNetwork;
import io.vacco.jtinn.net.JtPredictionSample;

public class JtTrainer {

  private final JtNetwork network;
  private final JtStopCondition stopFn;
  private final JtPredictionSampleSupplier miniBatchSupplier;

  public JtTrainer(JtNetwork network, JtStopCondition stopFn,
                   JtPredictionSampleSupplier miniBatchSupplier) {
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
      JtPredictionSample[] samples = miniBatchSupplier.get();
      for (JtPredictionSample sample : samples) {
        batchError += network.train(sample.features, sample.labels);
      }
      batchError /= samples.length;
      network.updater.onEpochEnd(epoch);
    }
  }
}
