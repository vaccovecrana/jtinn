package io.vacco.jtinn;

import io.vacco.jtinn.net.JtPredictionSample;
import io.vacco.jtinn.net.JtPredictionSampleSupplier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class XorData implements JtPredictionSampleSupplier {

  private static JtPredictionSample of(double in0, double in1, double out) {
    double[] ft = new double[]{in0, in1};
    double[] lb = new double[]{out};
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
