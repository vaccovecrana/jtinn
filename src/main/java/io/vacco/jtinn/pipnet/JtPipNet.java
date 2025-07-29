package io.vacco.jtinn.pipnet;

import io.vacco.jtinn.core.*;
import java.util.*;

public class JtPipNet extends JtNetwork3 {

  private static final long serialVersionUID = JtUtil.version;

  private static final float[] mean = {0.485f, 0.456f, 0.406f};
  private static final float[] std = {0.229f, 0.224f, 0.225f};

  private final int numLandmarks, numNeighbors;
  private int[][] neighborIndices;
  public  JtLayers.JtConvLayer3 clsLayer, xLayer, yLayer, nbXLayer, nbYLayer;

  public JtPipNet(int numLandmarks, int numNeighbors) {
    this.numLandmarks = numLandmarks;
    this.numNeighbors = numNeighbors;
  }

  public JtPipNet setNeighborIndices(int[][] neighborIndices) {
    if (neighborIndices.length != numLandmarks || (neighborIndices.length > 0 && neighborIndices[0].length != numNeighbors)) {
      throw new IllegalArgumentException("Invalid neighbor indices shape");
    }
    this.neighborIndices = neighborIndices;
    return this;
  }

  public JtPipNet init(JtInit.JtParamInitializer initializer) {
    var backbone = new JtResNet18().init(256, 256, initializer);
    clsLayer = new JtLayers.JtConvLayer3().init(numLandmarks, 1, 1, 0, null);
    xLayer   = new JtLayers.JtConvLayer3().init(numLandmarks, 1, 1, 0, null);
    yLayer   = new JtLayers.JtConvLayer3().init(numLandmarks, 1, 1, 0, null);
    nbXLayer = new JtLayers.JtConvLayer3().init(numLandmarks * numNeighbors, 1, 1, 0, null);
    nbYLayer = new JtLayers.JtConvLayer3().init(numLandmarks * numNeighbors, 1, 1, 0, null);
    var allLayers = new ArrayList<>(Arrays.asList(backbone.layers));
    allLayers.add(clsLayer);
    allLayers.add(xLayer);
    allLayers.add(yLayer);
    allLayers.add(nbXLayer);
    allLayers.add(nbYLayer);
    super.init(3, 256, 256, initializer, null, allLayers.toArray(new JtLayers.JtLayer3[0]));
    return this;
  }

  @Override public JtTensor3 estimate(JtTensor3 input) {
    var x = input;
    for (int i = 0; i < layers.length - 5; i++) {
      x = layers[i].forward(x, false);
    }
    // Compute head outputs
    var clsOut = clsLayer.forward(x, false);
    var xOut   = xLayer.forward(x, false);
    var yOut   = yLayer.forward(x, false);
    var nbXOut = nbXLayer.forward(x, false);
    var nbYOut = nbYLayer.forward(x, false);
    // Concatenate: [cls(N), x(N), y(N), nb_x(N*numNeighbors), nb_y(N*numNeighbors)]
    int totalC = numLandmarks + 2 * numLandmarks + 2 * numNeighbors * numLandmarks;
    var maps = new JtTensor3(totalC, 8, 8);
    int idx = 0;
    for (int c = 0; c < numLandmarks; c++) {
      for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
          maps.set(idx++, h, w, clsOut.get(c, h, w));
        }
      }
    }
    for (int c = 0; c < numLandmarks; c++) {
      for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
          maps.set(idx++, h, w, xOut.get(c, h, w));
        }
      }
    }
    for (int c = 0; c < numLandmarks; c++) {
      for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
          maps.set(idx++, h, w, yOut.get(c, h, w));
        }
      }
    }
    for (int c = 0; c < numLandmarks * numNeighbors; c++) {
      for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
          maps.set(idx++, h, w, nbXOut.get(c, h, w));
        }
      }
    }
    for (int c = 0; c < numLandmarks * numNeighbors; c++) {
      for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
          maps.set(idx++, h, w, nbYOut.get(c, h, w));
        }
      }
    }
    return maps;
  }

  public List<float[]> getLandmarks(JtTensor3 maps, float scoreThreshold, int origW, int origH) {
    if (neighborIndices == null) {
      throw new IllegalStateException("Neighbor indices not set");
    }
    JtUtil.checkShape(maps.shape, new int[]{numLandmarks + 2 * numLandmarks + 2 * numNeighbors * numLandmarks, 8, 8});
    int[] bestY = new int[numLandmarks], bestX = new int[numLandmarks];
    var maxScores = new float[numLandmarks];
    Arrays.fill(maxScores, Float.NEGATIVE_INFINITY);
    for (int i = 0; i < numLandmarks; i++) {
      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          float s = maps.get(i, y, x);
          if (s > maxScores[i]) {
            maxScores[i] = s;
            bestY[i] = y;
            bestX[i] = x;
          }
        }
      }
    }
    float[] initX = new float[numLandmarks], initY = new float[numLandmarks];
    for (int i = 0; i < numLandmarks; i++) {
      if (maxScores[i] <= scoreThreshold) continue;
      initX[i] = bestX[i] * 32f + maps.get(numLandmarks + 2 * i, bestY[i], bestX[i]);
      initY[i] = bestY[i] * 32f + maps.get(numLandmarks + 2 * i + 1, bestY[i], bestX[i]);
    }
    var landmarks = new ArrayList<float[]>();
    int neighBase = numLandmarks + 2 * numLandmarks;
    for (int i = 0; i < numLandmarks; i++) {
      if (maxScores[i] <= scoreThreshold) {
        landmarks.add(null);
        continue;
      }
      float sumX = initX[i], sumY = initY[i];
      int count = 1;
      for (int j = 0; j < numLandmarks; j++) {
        if (j == i || maxScores[j] <= scoreThreshold) continue;
        int k = -1;
        for (int nk = 0; nk < numNeighbors; nk++) {
          if (neighborIndices[j][nk] == i) {
            k = nk;
            break;
          }
        }
        if (k == -1) continue;
        int chBase = neighBase + 2 * numNeighbors * j;
        float dx = maps.get(chBase + 2 * k, bestY[j], bestX[j]);
        float dy = maps.get(chBase + 2 * k + 1, bestY[j], bestX[j]);
        sumX += initX[j] + dx;
        sumY += initY[j] + dy;
        count++;
      }
      float finalX = sumX / count * (origW / 256f);
      float finalY = sumY / count * (origH / 256f);
      landmarks.add(new float[]{finalX, finalY});
    }
    return landmarks;
  }

  public List<float[]> detectLandmarks(float[] rgbPixels, int origW, int origH) {
    if (rgbPixels.length != 256 * 256 * 3) {
      throw new IllegalArgumentException("Invalid pixel array size");
    }
    var input = new JtTensor3(3, 256, 256);
    int idx = 0;
    for (int h = 0; h < 256; h++) {
      for (int w = 0; w < 256; w++) {
        float r = rgbPixels[idx++] / 255f;
        float g = rgbPixels[idx++] / 255f;
        float b = rgbPixels[idx++] / 255f;
        input.set(0, h, w, (r - mean[0]) / std[0]);
        input.set(1, h, w, (g - mean[1]) / std[1]);
        input.set(2, h, w, (b - mean[2]) / std[2]);
      }
    }
    var maps = estimate(input);
    return getLandmarks(maps, 0.1f, origW, origH);
  }

}
