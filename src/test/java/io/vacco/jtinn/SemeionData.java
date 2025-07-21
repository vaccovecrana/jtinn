package io.vacco.jtinn;

import java.io.File;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class SemeionData implements JtTrain.JtSampleSupplier {

  private static final File f = new File("./src/test/resources/semeion.data");
  private static final List<JtTrain.JtSample> samples = new ArrayList<>();

  static {
    try {
      try (var lines = Files.lines(f.toPath())) {
        samples.addAll(
            lines.map(
                row -> Arrays.stream(
                    row.split(" "))
                    .map(Double::parseDouble)
                    .mapToDouble(Double::doubleValue)
                    .toArray()
            ).map(dArr -> JtTrain.JtSample.of(
                Arrays.copyOfRange(dArr, 0, 256),
                Arrays.copyOfRange(dArr, 256, 266)
            )).collect(Collectors.toList())
        );
      }
    } catch (Exception e) { throw new IllegalStateException(e); }
  }

  private static final JtTrain.JtSample[] buffer = new JtTrain.JtSample[samples.size()];

  @Override
  public JtTrain.JtSample[] get() {
    Collections.shuffle(samples);
    return samples.toArray(buffer);
  }
}
