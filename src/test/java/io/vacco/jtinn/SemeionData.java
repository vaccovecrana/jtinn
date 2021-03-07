package io.vacco.jtinn;

import io.vacco.jtinn.net.JtPredictionSample;
import io.vacco.jtinn.net.JtPredictionSampleSupplier;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class SemeionData implements JtPredictionSampleSupplier {

  private static final File f = new File("./src/test/resources/semeion.data");
  private static final List<JtPredictionSample> samples = new ArrayList<>();

  static {
    try {
      try (Stream<String> lines = Files.lines(f.toPath())) {
        samples.addAll(
            lines.map(
                row -> Arrays.stream(
                    row.split(" "))
                    .map(Double::parseDouble)
                    .mapToDouble(Double::doubleValue)
                    .toArray()
            ).map(dArr -> JtPredictionSample.of(
                Arrays.copyOfRange(dArr, 0, 256),
                Arrays.copyOfRange(dArr, 256, 266)
            )).collect(Collectors.toList())
        );
      }
    } catch (Exception e) { throw new IllegalStateException(e); }
  }

  private static final JtPredictionSample[] buffer = new JtPredictionSample[samples.size()];

  @Override
  public JtPredictionSample[] get() {
    Collections.shuffle(samples);
    return samples.toArray(buffer);
  }
}
