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

import java.util.HashMap;

import meka.classifiers.multitarget.MultiTargetClassifier;
import meka.core.MLUtils;
import weka.core.Instance;
import weka.core.Instances;

/**
 * MajorityLabelset.java - The most simplest multi-label classifier.
 * <p>
 * Predicts the most common labelset from the training data for <i>all</i> test instances.
 * </p>
 *
 * @author Jesse Read
 * @version September 2015
 */
public class MajorityLabelset extends AbstractMultiLabelClassifier implements MultiTargetClassifier {

  /** for serialization. */
  private static final long serialVersionUID = -5932291001079843869L;

  protected double prediction[] = null;
  protected HashMap<String, Double> classFreqs = new HashMap<>();

  protected double maxValue = 0.0;

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Majority Labelset Classifier: Always predict the combination of labels which occurs most frequently in the training set.";
  }

  protected void updateCount(final Instance x, final int L) throws InterruptedException {
    String y = MLUtils.toBitString(x, L);
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Thread has been interrupted.");
    }

    if (this.classFreqs.containsKey(y)) {
      double freq = this.classFreqs.get(y) + x.weight();
      this.classFreqs.put(y, freq);
      if (this.maxValue < freq) {
        this.maxValue = freq;
        this.prediction = MLUtils.fromBitString(y);
      }
    } else {
      this.classFreqs.put(y, x.weight());
    }
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    int L = D.classIndex();
    this.prediction = new double[L];

    for (int i = 0; i < D.numInstances(); i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      this.updateCount(D.instance(i), L);
    }

  }

  @Override
  public double[] distributionForInstance(final Instance test) throws Exception {
    return this.prediction;
  }

  public static void main(final String args[]) {
    AbstractMultiLabelClassifier.evaluation(new MajorityLabelset(), args);
  }

}
