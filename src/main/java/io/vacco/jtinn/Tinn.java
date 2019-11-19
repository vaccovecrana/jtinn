package io.vacco.jtinn;

public class Tinn {

  public double [] w; // All the weights.
  public double [] x; // Hidden to output layer weights.
  public double [] b; // Biases.

  public double [] h; // Hidden layer.
  public double [] o; // Output layer.

  public int nb;   // Number of biases - always two - Tinn only supports a single hidden layer.
  public int nw;   // Number of weights.
  public int nips; // Number of inputs.
  public int nhid; // Number of hidden neurons.
  public int nops; // Number of outputs.
}
