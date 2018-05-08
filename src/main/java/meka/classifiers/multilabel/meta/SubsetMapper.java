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

package meka.classifiers.multilabel.meta;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.ProblemTransformationMethod;
import meka.core.MLUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * Maps the output of a multi-label classifier to a known label combination using the hamming
 * distance. described in <i>Improved Boosting Algorithms Using Confidence-rated Predictions</i> by
 * Schapire, Robert E. and Singer, Yoram
 *
 * @author Jesse Read (jmr30@cs.waikato.ac.nz)
 */
public class SubsetMapper extends ProblemTransformationMethod implements TechnicalInformationHandler {

  /** for serialization. */
  private static final long serialVersionUID = -6587406787943635084L;

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Maps the output of a multi-label classifier to a known label combination using the hamming distance." + "For more information see:\n"
        + this.getTechnicalInformation().toString();
  }

  public SubsetMapper() {
    // default classifier for GUI
    this.m_Classifier = new BR();
  }

  @Override
  protected String defaultClassifierString() {

    // default classifier for CLI
    return "meka.classifiers.multilabel.BR";
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Robert E. Schapire, Yoram Singer ");
    result.setValue(Field.TITLE, "Improved Boosting Algorithms Using Confidence-rated Predictions");
    result.setValue(Field.JOURNAL, "Machine Learning Journal");
    result.setValue(Field.YEAR, "1999");
    result.setValue(Field.VOLUME, "37");
    result.setValue(Field.NUMBER, "3");
    result.setValue(Field.PAGES, "297-336");

    return result;
  }

  protected HashMap<String, Integer> m_Count = new HashMap<>();

  protected double[] nearestSubset(final double d[]) throws Exception {

    String comb = MLUtils.toBitString(doubles2ints(d));

    // If combination exists
    if (this.m_Count.get(comb) != null) {
      return MLUtils.fromBitString(comb);
    }

    int closest_count = 0;
    int min_distance = Integer.MAX_VALUE;
    String nearest = comb;

    for (String current : this.shuffle(this.m_Count.keySet())) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      int distance = hammingDistance(current, comb);
      if (distance == min_distance) {
        int count = this.m_Count.get(current);
        if (count > closest_count) {
          nearest = current;
          closest_count = count;
        }
      }
      if (distance < min_distance) {
        min_distance = distance;
        nearest = current;
        closest_count = this.m_Count.get(nearest);
      }
    }
    return MLUtils.fromBitString(nearest);
  }

  private Collection<String> shuffle(final Set<String> labelSubsets) {
    int seed = 1;
    Vector<String> result = new Vector<>(labelSubsets.size());
    result.addAll(labelSubsets);
    Collections.shuffle(result, new Random(seed));
    return result;
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    for (int i = 0; i < D.numInstances(); i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      this.m_Count.put(MLUtils.toBitString(D.instance(i), D.classIndex()), 0);
    }

    this.m_Classifier.buildClassifier(D);

  }

  @Override
  public double[] distributionForInstance(final Instance TestInstance) throws Exception {

    double r[] = ((ProblemTransformationMethod) this.m_Classifier).distributionForInstance(TestInstance);

    return this.nearestSubset(r);
  }

  private static final int[] doubles2ints(final double d[]) {
    int b[] = new int[d.length];
    for (int i = 0; i < d.length; i++) {
      b[i] = (int) Math.round(d[i]);
    }
    return b;
  }

  private static final int hammingDistance(final String s1, final String s2) {
    int dist = 0;
    for (int i = 0; i < Math.min(s1.length(), s2.length()); i++) {
      dist += Math.abs(MLUtils.char2int(s1.charAt(i)) - MLUtils.char2int(s2.charAt(i)));
    }
    return dist;
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new SubsetMapper(), args);
  }

}
