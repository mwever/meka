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

package rbms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import Jama.Matrix;
import meka.core.MatrixUtils;
import weka.core.Utils;

/**
 * RBM.java - Restricted Boltzmann Machine. Using Contrastive Divergence.
 *
 * You have inputs X; You want to output to hidden units Z. To do this, you learn weight matrix W,
 * where Z ~ sigma(W'X).
 *
 * <verbatim> ============== TRAINING (batches of 10) ===== RBM rbm = new RBM(); rbm.setOptions("-E
 * 100 -H 10 -r 0.1 -m 0.8"); // to build 10 hidden units, over 100 epochs, with learning rate 0.1,
 * momentum 0.8 rbm.train(X,10); // train in batches of 10 Z = rbm.getZ(X); // get output
 * ============== UPDATING (one epoch) ========= rbm.update(xnew); ============== TESTING (single
 * input) ======= z = rbm.getz(xnew); </verbatim> Note: should be binary for hidden states, can be
 * probabilities for visible states.
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version April 2013
 */
public class RBM {

  protected double LEARNING_RATE = 0.1; // set between v small and 0.1 in [0,1]
  protected double MOMENTUM = 0.1; // set between 0.1 and 0.9 in [0,1]
  protected double COST = 0.0002 * this.LEARNING_RATE; // set between v small to ~ 0.001 * LEARNING_RATE The rate at which to degrade connection weights to
                                                       // penalize large weights
  protected int m_E = 1000;
  protected int m_H = 10;
  private boolean m_V = false; // cut out of var(10) < 0.0001
  private int batch_size = 0; // @todo implement this option (with CLI option)

  protected Matrix W = null; // the weight matrix
  protected Matrix dW_ = null; // used for momentum

  protected Random m_R = new Random(0); // for random init. of matrices and sampling

  /**
   * RBM - Create an RBM with default options.
   */
  public RBM() {
  }

  /**
   * RBM - Create an RBM with 'options' (using WEKA-style option processing).
   */
  public RBM(final String options[]) throws Exception {
    this.setOptions(options);
  }

  /**
   * Set Options - WEKA-style option processing.
   */
  public void setOptions(final String[] options) throws Exception {

    try {
      this.setH(Integer.parseInt(Utils.getOption('H', options)));
      this.setE(Integer.parseInt(Utils.getOption('E', options)));
      this.setLearningRate(Double.parseDouble(Utils.getOption('r', options)));
      this.setMomentum(Double.parseDouble(Utils.getOption('m', options)));
    } catch (Exception e) {
      System.err.println("Missing option!");
      e.printStackTrace();
      System.exit(1);
    }

    // super.setOptions(options);
  }

  /**
   * GetOptions - WEKA-style option processing.
   */
  public String[] getOptions() throws Exception {
    ArrayList<String> result;
    result = new ArrayList<>(); // Arrays.asList(super.getOptions()));
    result.add("-r");
    result.add(String.valueOf(this.LEARNING_RATE));
    result.add("-m");
    result.add(String.valueOf(this.MOMENTUM));
    result.add("-E");
    result.add(String.valueOf(this.getE()));
    result.add("-H");
    result.add(String.valueOf(this.getH()));
    return result.toArray(new String[result.size()]);
  }

  /**
   * Hidden Activation Probability - returns P(z|x) where p(z[i]==1|x) for each element. A Bias is
   * added (and removed) automatically.
   *
   * @param x_
   *          x (without bias)
   * @return z (without bias)
   */
  public double[] prob_z(final double x_[]) {
    Matrix x = new Matrix(MatrixUtils.addBias(x_), 1);
    double z[] = MatrixUtils.sigma(x.times(this.W).getArray()[0]);
    return MatrixUtils.removeBias(z);
  }

  /**
   * Hidden Activation Probability - returns P(Z|X). A Bias column added (and removed) automatically.
   *
   * @param X_
   *          X (without bias)
   * @return P(Z|X)
   */
  public double[][] prob_Z(final double X_[][]) {
    Matrix X = new Matrix(MatrixUtils.addBias(X_));
    return MatrixUtils.removeBias(this.prob_Z(X).getArray());
  }

  /**
   * Hidden Activation Probability - returns P(Z|X). A bias column is assumed to be included.
   *
   * @param X
   *          X (bias included)
   * @return P(Z|X)
   */
  public Matrix prob_Z(final Matrix X) {
    Matrix P_Z = MatrixUtils.sigma(X.times(this.W)); // (this is the activation function)
    MatrixUtils.fillCol(P_Z.getArray(), 0, 1.0); // fix bias ... set first col to 1.0
    return P_Z;
  }

  /**
   * Hidden Activation Value. A bias column added (and removed) automatically.
   *
   * @param X_
   *          X (without bias)
   * @return 1 if P(Z|X) greater than 0.5
   */
  public double[][] propUp(final double X_[][]) {
    return MatrixUtils.threshold(this.prob_Z(X_), 0.5); // ... or just go down
  }

  /**
   * Sample Hidden Value - returns Z ~ P(Z|X). A bias column is assumed to be included.
   *
   * @param X
   *          X (bias included)
   * @return Z ~ P(Z|X)
   */
  public Matrix sample_Z(final Matrix X) {
    Matrix P_Z = this.prob_Z(X);
    return MatrixUtils.sample(P_Z, this.m_R);
  }

  /**
   * Sample Hidden Value - returns z[i] ~ p(z[i]==1|x) for each i-th element. A bias column added (and
   * removed) automatically.
   *
   * @param x_
   *          x (without bias)
   * @return z ~ P(z|x) (without bias)
   */
  public double[] sample_z(final double x_[]) {
    double p[] = this.prob_z(x_);
    return MatrixUtils.sample(p, this.m_R);
  }

  /**
   * Sample Visible - returns x[j] ~ p(x[j]==1|z) for each j-th element. A bias is added (and removed)
   * automatically.
   *
   * @param z_
   *          z (without bias)
   * @return x ~ P(x|z) (without bias)
   */
  public double[] sample_x(final double z_[]) {
    double p_x[] = this.prob_x(z_);
    return MatrixUtils.sample(p_x, this.m_R);
  }

  /**
   * Sample Visible - returns X ~ P(X|Z). A bias column is assumed to be included.
   *
   * @param Z
   *          Z (bias included)
   * @return X ~ P(X|Z)
   */
  public Matrix sample_X(final Matrix Z) {
    Matrix P_X = this.prob_X(Z);
    return MatrixUtils.sample(P_X, this.m_R);
  }

  /**
   * Visible Activation Probability - returns P(x|z) where p(x[j]==1|z) for each j-th element. A bias
   * is added (and removed) automatically.
   *
   * @param z_
   *          z (without bias)
   * @return x (without bias)
   */
  public double[] prob_x(final double z_[]) {
    Matrix z = new Matrix(MatrixUtils.addBias(z_), 1);
    double x[] = MatrixUtils.sigma(z.times(this.W.transpose()).getArray()[0]);
    return MatrixUtils.removeBias(x);
  }

  /**
   * Visible Activation Probability - returns P(X|Z). A bias column is assumed to be included.
   *
   * @param Z
   *          z (bias included)
   * @return P(X|Z)
   */
  public Matrix prob_X(final Matrix Z) {
    Matrix X = new Matrix(MatrixUtils.sigma(Z.times(this.W.transpose()).getArray())); // (this is the activation function)
    MatrixUtils.fillCol(X.getArray(), 0, 1.0); // fix bias - set first col to 1.0
    return X;
  }

  /**
   * Make W matrix of dimensions d+1 and h+1 (+1 for biases). Initialized from ~N(0,0.2) (seems to
   * work better than ~N(0.0.01)) -- except biases (set to 0)
   *
   * @param d
   *          number of rows (visible units)
   * @param h
   *          number of columns (hidden units)
   * @param r
   *          for getting random rumbers
   * @return W
   */
  public static Matrix makeW(final int d, final int h, final Random r) {
    double W_[][] = MatrixUtils.multiply(MatrixUtils.randn(d + 1, h + 1, r), 0.20); // ~ N(0.0,0.01)
    MatrixUtils.fillRow(W_, 0, 0.0); // set the first row to 0 for bias
    MatrixUtils.fillCol(W_, 0, 0.0); // set the first col to 0 for bias
    return new Matrix(W_);
  }

  protected Matrix makeW(final int d, final int h) {
    return makeW(d, h, this.m_R);
  }

  /**
   * Initialize W, and make _dW (for momentum) of the same dimensions.
   *
   * @param X_
   *          X (only to know d = X_[0].length)
   */
  private void initWeights(final double X_[][]) {

    this.initWeights(X_[0].length, this.m_H);
  }

  /**
   * Initialize W, and make _dW (for momentum) of the same dimensions.
   *
   * @param d
   *          number of visible units
   * @param h
   *          number of hidden units
   */
  private void initWeights(final int d, final int h) {

    this.W = this.makeW(d, h);
    this.dW_ = new Matrix(this.W.getRowDimension(), this.W.getColumnDimension()); // for momentum
  }

  /**
   * Initialize W, and make _dW (for momentum) of the same dimensions.
   *
   * @param d
   *          number of visible units
   */
  public void initWeights(final int d) {

    this.initWeights(d, this.m_H);
  }

  /**
   * Update - Carry out one epoch of CD, update W. We use dW_ to manage momentum. <br>
   * TODO weight decay SHOULD NOT BE APPLIED TO BIASES
   *
   * @param X
   *          X
   * @throws InterruptedException
   */
  public void update(final Matrix X) throws InterruptedException {

    Matrix CD = this.epoch(X);

    Matrix dW = (CD.minusEquals(this.W.times(this.COST))).timesEquals(this.LEARNING_RATE); // with COST
    this.W.plusEquals(dW.plus(this.dW_.timesEquals(this.MOMENTUM))); // with MOMENTUM.
    this.dW_ = dW; // for the next update
  }

  /**
   * Update - Carry out one epoch of CD, update W. <br>
   * TODO combine with above fn.
   *
   * @param X
   *          X
   * @param s
   *          multiply the gradient by this scalar
   * @throws InterruptedException
   */
  public void update(final Matrix X, final double s) throws InterruptedException {

    Matrix CD = this.epoch(X);

    Matrix dW = (CD.minusEquals(this.W.times(this.COST))).timesEquals(this.LEARNING_RATE); // with COST
    dW = dW.times(s); // *scaling factor
    this.W.plusEquals(dW.plus(this.dW_.timesEquals(this.MOMENTUM))); // with MOMENTUM.
    this.dW_ = dW; // for the next update
  }

  /**
   * Update - On raw data (with no bias column)
   *
   * @param X_
   *          raw double[][] data (with no bias column)
   * @throws InterruptedException
   */
  public void update(final double X_[][]) throws InterruptedException {
    Matrix X = new Matrix(MatrixUtils.addBias(X_));
    this.update(X);
  }

  /**
   * Update - On raw data (with no bias column)
   *
   * @param x_
   *          raw double[] data (with no bias column)
   * @throws InterruptedException
   */
  public void update(final double x_[]) throws InterruptedException {
    this.update(new double[][] { x_ });
  }

  /**
   * Update - On raw data (with no bias column)
   *
   * @param x_
   *          raw double[] data (with no bias column)
   * @param s
   *          multiply the gradient by this scalar
   * @throws InterruptedException
   */
  public void update(final double x_[], final double s) throws InterruptedException {
    Matrix X = new Matrix(MatrixUtils.addBias(new double[][] { x_ }));
    this.update(X, s);
  }

  /**
   * Train - Setup and train the RBM on X, over m_E epochs.
   *
   * @param X_
   *          X
   * @return the error (@TODO unnecessary)
   */
  public double train(final double X_[][]) throws Exception {

    this.initWeights(X_);

    Matrix X = new Matrix(MatrixUtils.addBias(X_));

    double _error = Double.MAX_VALUE; // prev error , necessary only when using m_V

    // TRAIN FOR m_E EPOCHS.

    for (int e = 0; e < this.m_E; e++) {

      // BREAK OUT IF THE GRADIENT IS POSITIVE
      if (this.m_V) {
        double err_now = this.calculateError(X); // Retrieve error
        if (_error < err_now) {
          System.out.println("broken out @" + e);
          break;
        }
        _error = err_now;
      }

      /*
       * The update
       */
      this.update(X);
    }

    return _error;
  }

  /**
   * Train - Setup and batch-train the RBM on X. <br>
   * TODO, above function train(X_) could really be trained with train(X_,N), so, should share code
   * with train(X) <br>
   * TODO, divide gradient by the size of the batch! (doing already? .. no)
   *
   * @param X_
   *          X
   * @param batchSize
   *          the batch size
   */
  public double train(double X_[][], final int batchSize) throws Exception {

    this.initWeights(X_);

    X_ = MatrixUtils.addBias(X_);

    int N = X_.length; // N
    if (batchSize == N) {
      return this.train(X_);
    }
    int N_n = (int) Math.ceil(N * 1. / batchSize);// Number of batches

    Matrix X_n[] = new Matrix[N_n];
    for (int n = 0, i = 0; n < N; n += batchSize, i++) {
      // @TODO, could save some small-time memory/speed here
      X_n[i] = new Matrix(Arrays.copyOfRange(X_, n, Math.min(n + batchSize, N)));
    }

    for (int e = 0; e < this.m_E; e++) {

      // @TODO could be random, see function below
      for (Matrix X : X_n) {
        this.update(X, 1. / N_n);
      }
    }

    return 1.0;
  }

  /**
   * Train - Setup and batch-train the RBM on X, with some random sampling involved. <br>
   * TODO should share code with train(X)
   *
   * @param X_
   *          X
   * @param batchSize
   *          the batch size
   * @param r
   *          the randomness
   */
  public double train(double X_[][], final int batchSize, final Random r) throws Exception {

    this.initWeights(X_);

    X_ = MatrixUtils.addBias(X_);
    int N = X_.length; // N
    int N_n = (int) Math.ceil(N * 1. / batchSize);// Number of batches

    // @TODO select the batches randomly at each epoch
    Matrix X_n[] = new Matrix[N_n];
    for (int n = 0, i = 0; n < N; n += batchSize, i++) {
      X_n[i] = new Matrix(Arrays.copyOfRange(X_, n, Math.min(n + batchSize, N)));
    }

    for (int e = 0; e < this.m_E; e++) {
      for (int i = 0; i < N_n; i++) {
        this.update(X_n[r.nextInt(N_n)]);
      }
    }

    return 1.0;
  }

  /**
   * Calculate the Error right now. <br>
   * NOTE: this will take a few miliseconds longer than calculating directly in the epoch() loop
   * (where we have to calculate X_down anyway). <br>
   * TODO rename this function
   *
   * @param X
   *          X
   * @return The error
   */
  public double calculateError(final Matrix X) {

    Matrix Z_up = this.prob_Z(X); // up @TODO replace with getZ(X,W), etc
    Matrix X_down = this.prob_X(Z_up); // go down

    // MSE
    return MatrixUtils.meanSquaredError(X.getArray(), X_down.getArray()); // @note: this can take some milliseconds to calculate
  }

  /**
   * Epoch - Run X through one epcho of CD of the RBM.
   *
   * @param X_0
   *          The input matrix (includes bias column).
   * @return the contrastive divergence (CD) for this epoch.
   *
   *         <verbatim> x_0 = x
   *
   *         for k = 0,...,K-1 z_k = sample up x_k+1 = sample down
   *
   *         e+ = pz|x_0 e- = pz|x_K
   *
   *         CD = e+ - e- </verbatim>
   *
   *         Note: should be binary for hidden states, can be probabilities for visible states.
   * @throws InterruptedException
   */
  public Matrix epoch(final Matrix X_0) throws InterruptedException {
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Thread got interrupted");
    }

    int N = X_0.getArray().length;

    // POSITIVE
    Matrix Z_0 = this.prob_Z(X_0); // sample up
    Matrix E_pos = X_0.transpose().times(Z_0); // positive energy, H_1 * V_1

    // NEGATIVE
    Matrix X_1 = this.prob_X(Z_0); // go down -- can either sample down
    // Matrix X_1 = Mat.threshold(prob_X(Z_0),0.5); // ... or just go down
    Matrix pZ_1 = this.prob_Z(X_1); // go back up again
    Matrix E_neg = X_1.transpose().times(pZ_1); // negative energy, P(Z_1) * X_1

    // CALCULATE ERROR (Optional!)
    // double _Err = Mat.meanSquaredError(X_0.getArray(),X_1.getArray()); // @note: this take some
    // milliseconds to calculate
    // System.out.println(""+_Err);

    // CONTRASTIVE DIVERGENCE
    Matrix CD = ((E_pos.minusEquals(E_neg)).times(1. / N)); // CD = difference between energies

    return CD;

  }

  // SAME AS ABOVE, BUT USES SAMPLING INSTEAD OF RAW PROBABILITIES. DOESN'T SEEM TO WORK AS WELL.
  public Matrix sample_epoch(final Matrix X_0) {

    int N = X_0.getArray().length;

    // POSITIVE
    Matrix Z_0 = this.sample_Z(X_0); // sample up
    Matrix E_pos = X_0.transpose().times(Z_0); // positive energy, H_1 * V_1

    // NEGATIVE
    Matrix X_1 = this.sample_X(Z_0); // go down -- can either sample down
    // Matrix X_1 = Mat.threshold(prob_X(Z_0),0.5); // ... or just go down
    Matrix pZ_1 = this.prob_Z(X_1); // go back up again
    Matrix E_neg = X_1.transpose().times(pZ_1); // negative energy, P(Z_1) * X_1

    // CALCULATE ERROR (Optional!)
    double _Err = MatrixUtils.meanSquaredError(X_0.getArray(), X_1.getArray()); // @note: this take some milliseconds to calculate
    System.out.println("" + _Err);

    // CONTRASTIVE DIVERGENCE
    Matrix CD = ((E_pos.minusEquals(E_neg)).times(1. / N)); // CD = difference between energies

    return CD;
  }

  /*
   * ********************************************************************************* Get / Set
   * Parameters
   **********************************************************************************/

  public void setH(final int h) {
    this.m_H = h;
  }

  public int getH() {
    return this.m_H;
  }

  /**
   * SetE - set the number of epochs (if n is negative, it means max epochs).
   */
  public void setE(final int n) {
    if (n < 0) {
      this.m_V = true;
      this.m_E = -n;
    } else {
      this.m_E = n;
    }
  }

  public int getE() {
    return this.m_E;
  }

  public void setLearningRate(final double r) {
    this.LEARNING_RATE = r;
    this.COST = 0.0002 * this.LEARNING_RATE;
  }

  public double getLearningRate() {
    return this.LEARNING_RATE;
  }

  public void setMomentum(final double m) {
    this.MOMENTUM = m;
  }

  public double getMomentum() {
    return this.MOMENTUM;
  }

  public void setSeed(final int seed) {
    this.m_R = new Random(seed);
  }

  /*
   * ********************************************************************************* Get Weight
   * Matrix(es)
   **********************************************************************************/

  public Matrix[] getWs() {
    return new Matrix[] { this.W };
  }

  public Matrix getW() {
    return this.W;
  }

  /**
   * ToString - return a String representation of the weight Matrix defining this RBM.
   */
  @Override
  public String toString() {
    Matrix W = this.getW();
    return MatrixUtils.toString(W);
  }

  /**
   * Main - do some test routines.
   */
  public static void main(final String argv[]) throws Exception {
  }

}
