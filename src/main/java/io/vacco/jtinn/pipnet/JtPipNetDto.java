package io.vacco.jtinn.pipnet;

import io.vacco.jtinn.core.JtInit;
import io.vacco.jtinn.core.JtLayers;
import java.util.Arrays;
import java.util.List;

public class JtPipNetDto {

  public Metadata metadata;
  public List<Weight> weights;

  public static class Metadata {
    public int num_landmarks;
    public int num_neighbors;
    public int[][] neighbor_indices;
  }

  public static class Weight {
    public String key;
    public int[] shape;
    public float[] data;
    public Integer count;
  }

  // Sample hardcoded 300W neighbor graph (68x10) - simplified, needs real topology
  private static final int[][] DEFAULT_300W_NEIGHBORS = {
    {1,2,3,4,5,6,7,8,9,10}, // Landmark 0 (e.g., left face boundary)
    {0,2,3,4,5,6,7,8,9,11}, // Landmark 1
    // ... Add 66 more rows for 68 landmarks
    // Example: Left eye corner (27) neighbors right eye, nose, mouth
    {28,29,30,31,36,39,42,45,48,51}, // Landmark 27
    // Full graph should be derived from data/data_300W/meanface.txt
  };

  private static void reshapeConvWeights(float[] data, int[] shape, float[][][][] target) {
    int outC = shape[0], inC = shape[1], kH = shape[2], kW = shape[3];
    int idx = 0;
    for (int oc = 0; oc < outC; oc++)
      for (int ic = 0; ic < inC; ic++)
        for (int kh = 0; kh < kH; kh++)
          for (int kw = 0; kw < kW; kw++)
            target[oc][ic][kh][kw] = data[idx++];
  }

  private static void mapLayerConvWeight(JtPipNet model, String key, float[] data, int[] shape) {
    var parts = key.split("\\.");
    int stage = Integer.parseInt(parts[0].replace("layer", "")) - 1;
    int block = Integer.parseInt(parts[1]);
    var conv = parts[2];
    int layerIdx = 3 + stage * 2 + block;
    var rb = (JtResidualBlock) model.layers[layerIdx];
    float[][][][] target = conv.equals("conv1") ? rb.conv1.weights : rb.conv2.weights;
    if (key.contains("downsample.0")) {
      target = rb.shortcut.conv.weights;
    }
    reshapeConvWeights(data, shape, target);
    if (conv.equals("conv1")) rb.conv1.flattenWeights();
    else if (conv.equals("conv2")) rb.conv2.flattenWeights();
    else rb.shortcut.flattenWeights();
  }

  private static void mapLayerBias(JtPipNet model, String key, float[] data) {
    var parts = key.split("\\.");
    int stage = Integer.parseInt(parts[0].replace("layer", "")) - 1;
    int block = Integer.parseInt(parts[1]);
    var conv = parts[2];
    int layerIdx = 3 + stage * 2 + block;
    var rb = (JtResidualBlock) model.layers[layerIdx];
    float[] target = conv.equals("conv1") ? rb.conv1.b : rb.conv2.b;
    if (key.contains("downsample.0")) {
      target = rb.shortcut.conv.b;
    }
    System.arraycopy(data, 0, target, 0, data.length);
  }

  private static void mapBnParams(JtPipNet model, String key, float[] data) {
    var parts = key.split("\\.");
    boolean isStem = key.startsWith("bn1.");
    int layerIdx;
    JtLayers.JtBatchNormLayer3 bn;

    if (isStem) {
      layerIdx = 1;
      bn = (JtLayers.JtBatchNormLayer3) model.layers[layerIdx];
    } else {
      int stage = Integer.parseInt(parts[0].replace("layer", "")) - 1;
      int block = Integer.parseInt(parts[1]);
      var bnType = parts[2];
      layerIdx = 3 + stage * 2 + block;
      var rb = (JtResidualBlock) model.layers[layerIdx];
      bn = bnType.equals("bn1") ? rb.bn1 : bnType.equals("bn2") ? rb.bn2 : rb.shortcut.bn;
    }

    if (key.endsWith(".weight")) {
      System.arraycopy(data, 0, bn.gamma, 0, data.length);
    } else if (key.endsWith(".bias")) {
      System.arraycopy(data, 0, bn.beta, 0, data.length);
    } else if (key.endsWith(".running_mean")) {
      System.arraycopy(data, 0, bn.runningMean, 0, data.length);
    } else if (key.endsWith(".running_var")) {
      System.arraycopy(data, 0, bn.runningVar, 0, data.length);
    }
  }

  public static JtPipNet fromDto(JtPipNetDto dto) {
    var model =
      new JtPipNet(dto.metadata.num_landmarks, dto.metadata.num_neighbors)
        .init(new JtInit.JtRandomInitializer().init(1234));

    // Validate neighbor indices
    boolean isCyclic = true;
    for (int i = 0; i < dto.metadata.neighbor_indices.length; i++) {
      int[] neighbors = dto.metadata.neighbor_indices[i];
      for (int j = 0; j < neighbors.length; j++) {
        if (neighbors[j] != (i + j + 1) % dto.metadata.num_landmarks) {
          isCyclic = false;
          break;
        }
      }
      if (!isCyclic) break;
    }
    if (isCyclic && dto.metadata.num_landmarks == 68) {
      model.setNeighborIndices(DEFAULT_300W_NEIGHBORS);
    } else {
      model.setNeighborIndices(dto.metadata.neighbor_indices);
    }

    // Map weights
    for (JtPipNetDto.Weight w : dto.weights) {
      String key = w.key;
      float[] data = w.data != null ? w.data : new float[0];
      int[] shape = w.shape;

      // Validate data length
      if (w.data != null) {
        int expectedSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (data.length != expectedSize) {
          throw new IllegalArgumentException("Data length mismatch for key: " + key);
        }
      }

      // Map head weights
      if (key.equals("cls_layer.weight")) {
        reshapeConvWeights(data, shape, model.clsLayer.weights);
        model.clsLayer.flattenWeights();
      } else if (key.equals("x_layer.weight")) {
        reshapeConvWeights(data, shape, model.xLayer.weights);
        model.xLayer.flattenWeights();
      } else if (key.equals("y_layer.weight")) {
        reshapeConvWeights(data, shape, model.yLayer.weights);
        model.yLayer.flattenWeights();
      } else if (key.equals("nb_x_layer.weight")) {
        reshapeConvWeights(data, shape, model.nbXLayer.weights);
        model.nbXLayer.flattenWeights();
      } else if (key.equals("nb_y_layer.weight")) {
        reshapeConvWeights(data, shape, model.nbYLayer.weights);
        model.nbYLayer.flattenWeights();
      } else if (key.equals("cls_layer.bias")) {
        System.arraycopy(data, 0, model.clsLayer.b, 0, data.length);
      } else if (key.equals("x_layer.bias")) {
        System.arraycopy(data, 0, model.xLayer.b, 0, data.length);
      } else if (key.equals("y_layer.bias")) {
        System.arraycopy(data, 0, model.yLayer.b, 0, data.length);
      } else if (key.equals("nb_x_layer.bias")) {
        System.arraycopy(data, 0, model.nbXLayer.b, 0, data.length);
      } else if (key.equals("nb_y_layer.bias")) {
        System.arraycopy(data, 0, model.nbYLayer.b, 0, data.length);
      }
      // Map backbone weights
      else if (key.endsWith(".weight")) {
        if (key.equals("conv1.weight")) {
          reshapeConvWeights(data, shape, ((JtLayers.JtConvLayer3) model.layers[0]).weights);
          ((JtLayers.JtConvLayer3) model.layers[0]).flattenWeights();
        } else if (key.contains("layer")) {
          mapLayerConvWeight(model, key, data, shape);
        }
      } else if (key.endsWith(".bias")) {
        if (key.equals("conv1.bias")) {
          System.arraycopy(data, 0, model.layers[0].b, 0, data.length);
        } else if (key.contains("layer")) {
          mapLayerBias(model, key, data);
        }
      } else if (key.contains("bn")) {
        mapBnParams(model, key, data);
      } else {
        System.err.println("Warning: Unknown weight key: " + key);
      }
    }

    // Validate head layers
    if (model.clsLayer.weights == null || model.xLayer.weights == null ||
      model.yLayer.weights == null || model.nbXLayer.weights == null ||
      model.nbYLayer.weights == null) {
      throw new IllegalStateException("Missing head layer weights");
    }

    return model;
  }

}