package io.vacco.jtinn.net;

public interface JtStopCondition {
  boolean evaluate(JtNetwork network, int epoch, float error);
}
