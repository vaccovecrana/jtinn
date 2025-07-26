package io.vacco.jtinn;

import java.io.File;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class SemeionData implements JtTrain.JtSampler {

  private static final File f = new File("./src/test/resources/semeion.data");
  private static final List<JtTrain.JtSample> samples = new ArrayList<>();

  static {
    try {
      try (var lines = Files.lines(f.toPath())) {
        samples.addAll(
          lines.map(row -> {
            var vals = row.split(" ");
            var out = new float[vals.length];
            for (int i = 0; i < vals.length; i++) {
              out[i] = Float.parseFloat(vals[i]);
            }
            return out;
          }).map(fA -> JtTrain.JtSample.of(
            new JtTensor3(256, 1, 1, Arrays.copyOfRange(fA, 0, 256)),
            new JtTensor3(10, 1, 1, Arrays.copyOfRange(fA, 256, 266))
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