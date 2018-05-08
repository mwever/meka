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

import meka.core.A;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * PCC.java - (Bayes Optimal) Probabalistic Classifier Chains. Exactly like CC at build time, but
 * explores all possible paths as inference at test time (hence, 'Bayes optimal'). <br>
 * This version is multi-target capable. <br>
 * See: Dembczynsky et al, <i>Bayes Optimal Multi-label Classification via Probabalistic Classifier
 * Chains</i>, ICML 2010.
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version November 2012
 */
public class PCC extends CC implements TechnicalInformationHandler {
  private static final long serialVersionUID = -7669951968300150007L; // MT Capable

  /**
   * Push - increment y[0] until = K[0], then reset and start with y[0], etc ... Basically a counter.
   *
   * @return True if finished
   */
  private static boolean push(final double y[], final int K[], int j) {
    if (j >= y.length) {
      return true;
    } else if (y[j] < K[j] - 1) {
      y[j]++;
      return false;
    } else {
      y[j] = 0.0;
      return push(y, K, ++j);
    }
  }

  /**
   * GetKs - return [K_1,K_2,...,K_L] where each Y_j \in {1,...,K_j}. In the multi-label case, K[j] =
   * 2 for all j = 1,...,L.
   *
   * @param D
   *          a dataset
   * @return an array of the number of values that each label can take
   * @throws InterruptedException
   */
  private static int[] getKs(final Instances D) throws InterruptedException {
    int L = D.classIndex();
    int K[] = new int[L];
    for (int k = 0; k < L; k++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      K[k] = D.attribute(k).numValues();
    }
    return K;
  }

  /**
   * Return multi-label probabilities. Where p(y_j = y[j]) = confidence[j], then return [p(y_j =
   * 1),...,p(y_L = 1)].
   */
  private static double[] convertConfidenceToProbability(final double y[], final double confidences[]) {
    double p[] = new double[confidences.length];
    for (int j = 0; j < confidences.length; j++) {
      p[j] = confidences[j] * y[j] + (1. - confidences[j]) * Math.abs(y[j] - 1.);
    }
    return p;
  }

  @Override
  public double[] distributionForInstance(final Instance xy) throws Exception {

    int L = xy.classIndex();

    double y[] = new double[L];
    double conf[] = new double[L];
    double w = 0.0;

    /*
     * e.g. K = [3,3,5] we push y_[] from [0,0,0] to [2,2,4] over all necessary iterations.
     */
    int K[] = getKs(xy.dataset());
    if (this.getDebug()) {
      System.out.println("K[] = " + Arrays.toString(K));
    }
    double y_[] = new double[L];

    for (int i = 0; i < 1000000; i++) { // limit to 1m
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      double conf_[] = super.probabilityForInstance(xy, y_);
      double w_ = A.product(conf_);
      // System.out.println(""+i+" "+Arrays.toString(y_)+" "+w_+"
      // "+Arrays.toString(conf_)+"/"+Arrays.toString(convertConfidenceToProbability(y_,conf_)));
      if (w_ > w) {
        if (this.getDebug()) {
          System.out.println("y' = " + Arrays.toString(y_) + ", :" + w_);
        }
        y = Arrays.copyOf(y_, y_.length);
        w = w_;
        conf = conf_;
      }
      if (push(y_, K, 0)) {
        // Done !
        if (this.getDebug()) {
          System.out.println("Tried all " + (i + 1) + " combinations.");
        }
        break;
      }
    }

    // If it's multi-label (binary only), return the probabilistic output (else just the values).
    return (A.max(K) > 2) ? y : convertConfidenceToProbability(conf, y); // return p_y; //y;
  }

  @Override
  public String globalInfo() {
    return "Probabalistic Classifier Chains. " + "For more information see:\n" + this.getTechnicalInformation().toString();
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Krzysztof Dembczynsky and Weiwei Cheng and Eyke Hullermeier");
    result.setValue(Field.TITLE, "Bayes Optimal Multi-label Classification via Probabalistic Classifier Chains");
    result.setValue(Field.BOOKTITLE, "ICML '10: 27th International Conference on Machine Learning");
    result.setValue(Field.YEAR, "2010");

    return result;
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new PCC(), args);
  }

}
