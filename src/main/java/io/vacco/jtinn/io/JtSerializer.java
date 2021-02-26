package io.vacco.jtinn.io;

import io.vacco.jtinn.Tinn;
import java.io.InputStream;

public interface JtSerializer {
  void write(Tinn t);
  Tinn read(InputStream in);
}
