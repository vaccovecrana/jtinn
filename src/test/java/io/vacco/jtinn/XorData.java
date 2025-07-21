package io.vacco.jtinn;

import io.vacco.jtinn.net.*;
import java.util.*;

class XorData implements JtPredictionSampleSupplier {

  private static JtPredictionSample of(double in0, double in1, double out) {
    var ft = new double[] { in0, in1 };
    var lb = new double[] { out };
    return JtPredictionSample.of(ft, lb);
  }

  private static final List<JtPredictionSample> trainingList = new ArrayList<>(
      Arrays.asList(
          of(0, 0, 0),
          of(0, 1, 1),
          of(1, 0, 1),
          of(1, 1, 0)
      )
  );

  private static final JtPredictionSample[] buffer = new JtPredictionSample[trainingList.size()];

  @Override
  public JtPredictionSample[] get() {
    Collections.shuffle(trainingList);
    return trainingList.toArray(buffer);
  }

}
