/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package meka.classifiers.multilabel;

import java.util.Arrays;
import java.util.Random;

import Jama.Matrix;
import meka.classifiers.multilabel.NN.AbstractNeuralNet;
import meka.core.MLUtils;
import meka.core.MatrixUtils;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @TEMP
 */

/**
 * BPNN.java - Back Propagation Neural Network. This is a standard back-propagated Neural Network
 * with multiple outputs that correspond to multiple labels.<br>
 * If trained 'from scratch' only 1 layer is possible, but if you initialise it (from another
 * method) with pre-trained weight matrices, the number of layers is inferred from that.
 *
 * @author Jesse Read
 * @version March 2013
 */

public class BPNN extends AbstractNeuralNet {

  private static final long serialVersionUID = -4568680054917021671L;

  /** Weight Matrix */
  public Matrix W[] = null;
  protected Random r = null;
  protected Matrix dW_[] = null;

  public BPNN() {
    // different default for now
    this.m_E = 100;
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    double X_[][] = MLUtils.getXfromD(D);
    double Y_[][] = MLUtils.getYfromD(D);
    this.r = new Random(this.m_Seed);

    if (this.W == null) {
      if (this.getDebug()) {
        System.out.println("initialize weights ...");
      }
      int h[] = new int[] { this.m_H }; // TODO: parameterize this
      int d = X_[0].length;
      int L = D.classIndex();
      this.initWeights(d, L, h);
    }
    // else ... probably pre-initialized, continue ...
    else if (this.getDebug()) {
      System.out.println("weights already preset, continue ...");
    }

    this.train(X_, Y_, this.m_E);
  }

  @Override
  public double[] distributionForInstance(final Instance xy) throws Exception {
    double x[] = MLUtils.getxfromInstance(xy);
    return this.popy(x);
  }

  /**
   * Preset Weights - Initialize a BPNN with (pre-trained) weight matrices W (which also determines X
   * dimensions).
   *
   * @param W
   *          pre-trained weight matrix (should include bias weights, assume W[-1]-1 hidden units in
   *          penultimate layer not including bias])
   * @param L
   *          the number of labels (for making the final matrix)
   */
  public void presetWeights(final Matrix W[], final int L) throws Exception {

    this.r = new Random(0);

    this.W = new Matrix[W.length + 1];
    for (int l = 0; l < W.length; l++) {
      this.W[l] = W[l];
    }

    int h = W[1].getRowDimension() - 1;
    this.W[W.length] = MatrixUtils.randomn(h + 1, L, this.r).timesEquals(0.1);

    this.makeMomentumMatrices();
  }

  private void makeMomentumMatrices() {
    this.dW_ = new Matrix[this.W.length]; // weight deltas

    for (int i = 0; i < this.dW_.length; i++) {
      this.dW_[i] = new Matrix(this.W[i].getRowDimension(), this.W[i].getColumnDimension(), 0.0);
    }
  }

  /**
   * InitWeights - Initialize a BPNN of H.length hidden layers with H[0], H[1], etc hidden units in
   * each layer (W will be random, and of the corresponding dimensions).
   *
   * @param d
   *          number of visible units
   * @param L
   *          number of labels (output units)
   * @param H
   *          number of units in hidden layers, H.length = number of hidden layers. CURRENTLY LIMITED
   *          TO 1.
   */
  public void initWeights(final int d, final int L, int H[]) throws Exception {

    int numHidden = H.length;

    if (this.getDebug()) {
      System.out.println("Initializing " + (H.length) + " hidden Layers ...");
      System.out.println("d = " + d);
      System.out.println("L = " + L);
    }

    // We need weights for Z to Y, as well as from X to Z
    Matrix W[] = new Matrix[H.length + 1];
    int h = H[0];
    H = new int[] { d, h, L };

    // Hidden layers
    System.out.println("" + Arrays.toString(H));
    for (int n = 0; n < H.length - 1; n++) {
      W[n] = MatrixUtils.randomn(H[n] + 1, H[n + 1], this.r).timesEquals(0.1);
      if (this.getDebug()) {
        System.out.println("W[" + n + "] = " + (H[n] + 1) + " x " + H[n + 1]);
      }
    }

    // setWeights(W, L);
    this.W = W;
    this.makeMomentumMatrices();

  }

  public double train(final double X_[][], final double Y_[][]) throws Exception {
    return this.train(X_, Y_, this.m_E);
  }

  /**
   * Train - Train for I iterations. I is not necessarily m_E (yet)!
   */
  public double train(final double[][] X_, final double[][] Y_, int I) throws Exception {
    if (this.getDebug()) {
      System.out.println("BPNN train; For " + I + " epochs ...");
    }
    int N = X_.length;
    boolean breakEarly = (I < 0) ? true : false;
    I = Math.abs(I);
    double E_ = Double.MAX_VALUE;
    double E = 0.0;
    for (int e = 0; e < I; e++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      E = this.update(X_, Y_);
      if (breakEarly && E > E_) {
        if (this.getDebug()) {
          System.out.println(" early stopped at epcho " + e + " ... ");
        }
        break; // positive gradient
      }
      E_ = E;
    }
    if (this.getDebug()) {
      System.out.println("Done.");
    }
    return E;
  }

  /**
   * Update - A single training epoch.
   */
  public double update(final double X_[][], final double Y_[][]) throws Exception {
    int N = X_.length;
    double E = 0.0;
    for (int i = 0; i < N; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      E += this.backPropagate(new double[][] { X_[i] }, new double[][] { Y_[i] });
    }
    return E;
  }

  /**
   * Forward Pass - Given input x_, get output y_.
   *
   * @param x_
   *          input
   * @return y_ output
   * @throws InterruptedException
   */
  public double[] popy(final double x_[]) throws InterruptedException {
    return this.popY(new double[][] { x_ })[0];
  }

  /**
   * Forward Pass - Given input X_, get output Y_.
   *
   * @param X_
   *          input
   * @return Y_ output
   * @throws InterruptedException
   */
  public double[][] popY(final double X_[][]) throws InterruptedException {
    Matrix Z[] = this.forwardPass(X_);
    int n = Z.length - 1;
    return Z[n].getArray();
  }

  /**
   * Forward Pass - Given input X_, get output of all layers Z[0]...
   *
   * @param X_
   *          input (no bias included)
   * @return output Z[] = {X,Z1,Z2,...,Y}
   * @throws InterruptedException
   */
  public Matrix[] forwardPass(final double X_[][]) throws InterruptedException {

    int numW = this.W.length; // number of weight matrices
    Matrix Z[] = new Matrix[numW + 1];

    // input activations
    Z[0] = new Matrix(MatrixUtils.addBias(X_));

    // hidden layer(s)
    int i = 1;
    for (i = 1; i < numW; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      if (this.getDebug()) {
        System.out.print("DO: [" + i + "] " + MatrixUtils.getDim(Z[i - 1].getArray()) + " * " + MatrixUtils.getDim(this.W[i - 1].getArray()) + " => ");
      }

      Matrix A_z = Z[i - 1].times(this.W[i - 1]); // A = X * W1 = Z[n-1] * W[n-1]
      Z[i] = MatrixUtils.sigma(A_z);
      Z[i] = MatrixUtils.addBias(Z[i]); // ACTIVATIONS Z[n] = sigma(A) =

      if (this.getDebug()) {
        System.out.println("==: " + MatrixUtils.getDim(A_z.getArray()));
      }
    }

    // output layer
    if (this.getDebug()) {
      System.out.print("DX: [" + i + "] " + MatrixUtils.getDim(Z[i - 1].getArray()) + " * " + MatrixUtils.getDim(this.W[i - 1].getArray()) + " => ");
    }
    Matrix A_y = Z[i - 1].times(this.W[i - 1]); // A = X * W1 = Z[n-1] * W[n-1]
    if (this.getDebug()) {
      System.out.println("==: " + MatrixUtils.getDim(A_y.getArray()));
    }
    Z[numW] = MatrixUtils.sigma(A_y); // ACTIVATIONS Z[n] = sigma(A) =

    return Z;
  }

  /**
   * Back Propagate - Do one round of Back Propagation on batch X_,Y_.
   *
   * @param X_
   *          input
   * @param Y_
   *          teacher values
   */
  public double backPropagate(final double[][] X_, final double[][] Y_) throws Exception {

    int N = X_.length; // batch size
    int L = Y_[0].length; // num. of labels
    int nW = this.W.length; // num. of weight matrices

    Matrix T = new Matrix(Y_); // TARGETS

    /*
     * 1. FORWARD PROPAGATION. Forward-propagate X through the neural net to produce Z_1, Z_2, ..., Y.
     */

    // Matrix X = new Matrix(M.addBias(X_)); // INPUT
    Matrix Z[] = this.forwardPass(X_); // ALL LAYERS

    /*
     * 2. BACKWARD PROPAGATION. Propagate the errors backward through the neural net.
     */

    Matrix dZ[] = new Matrix[nW + 1]; // *new*

    // Error terms (output)
    Matrix E_y = T.minus(Z[nW]); // ERROR

    dZ[nW] = MatrixUtils.dsigma(Z[nW]).arrayTimes(E_y);

    // Error terms (hidden) *NEW*
    for (int i = nW - 1; i > 0; i--) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      Matrix E = dZ[i + 1].times(this.W[i].transpose());
      dZ[i] = MatrixUtils.dsigma(Z[i]).arrayTimes(E);
      dZ[i] = new Matrix(MatrixUtils.removeBias(dZ[i].getArray()));
    }

    // Error terms (hidden)
    // Matrix E_z = dY.times(W[1].transpose());
    // Matrix dZ = M.dsigma(Z[1]).arrayTimes(E_z);
    // dZ = new Matrix(M.removeBias(dZ.getArray()));

    // Weight derivatives
    Matrix dW[] = new Matrix[nW];
    for (int i = 0; i < nW; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      dW[i] = (Z[i].transpose().times(this.m_R).times(dZ[i + 1])).plus(this.dW_[i].times(this.m_M));
    }

    // Weight update
    for (int i = 0; i < nW; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      // W[i] = W[i].plusEquals(dW[i]);
      this.W[i].plusEquals(dW[i]);
    }

    // Update momentum records
    this.dW_ = dW;

    // double SSE = (E_y.transpose().times(E_y)).trace(); // SUM of SQUARE ERROR (faster?)
    double SSE = E_y.normF(); // SQRT of SUM of SQUARE ERROR (not the sqrt is not necessary, thus the following line should also
                              // suffice)
    return SSE;
  }

  public static void main(final String args[]) throws Exception {
    ProblemTransformationMethod.evaluation(new BPNN(), args);
  }
}
