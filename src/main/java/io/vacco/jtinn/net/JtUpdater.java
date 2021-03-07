package io.vacco.jtinn.net;

public interface JtUpdater {

  void apply(double[] lm1a, JtLayer l);
  void onEpochEnd(int epoch);

}
