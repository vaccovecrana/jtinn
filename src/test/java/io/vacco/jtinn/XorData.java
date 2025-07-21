package io.vacco.jtinn;

import java.util.*;

class XorData implements JtTrain.JtSampleSupplier {

  private static JtTrain.JtSample of(double in0, double in1, double out) {
    var ft = new double[] { in0, in1 };
    var lb = new double[] { out };
    return JtTrain.JtSample.of(ft, lb);
  }

  private static final List<JtTrain.JtSample> trainingList = new ArrayList<>(
      Arrays.asList(
          of(0, 0, 0),
          of(0, 1, 1),
          of(1, 0, 1),
          of(1, 1, 0)
      )
  );

  private static final JtTrain.JtSample[] buffer = new JtTrain.JtSample[trainingList.size()];

  @Override
  public JtTrain.JtSample[] get() {
    Collections.shuffle(trainingList);
    return trainingList.toArray(buffer);
  }

}
