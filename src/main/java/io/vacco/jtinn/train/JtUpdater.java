package io.vacco.jtinn.train;

import io.vacco.jtinn.layer.JtLayer;

import java.io.Serializable;

public interface JtUpdater extends Serializable {

  void apply(float[] lm1a, JtLayer l);
  void onEpochEnd(int epoch);

}
