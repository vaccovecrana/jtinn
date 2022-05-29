package io.vacco.jtinn.train;

import io.vacco.jtinn.net.JtNetwork;

public interface JtStopCondition {
  boolean evaluate(JtNetwork network, int epoch, float error);
}
