package io.vacco.jtinn.train;

import io.vacco.jtinn.layer.JtLayer;

public interface JtParamInitializer {
  void apply(JtLayer layer);
}
