package io.vacco.jtinn.net;

import java.io.Serializable;

public interface JtUpdater extends Serializable {

  void apply(double[] lm1a, JtLayer l);
  void onEpochEnd(int epoch);

}
