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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import meka.core.OptionUtils;
import meka.core.SuperLabelUtils;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * RAkEL - RAndom k-labEL subsets: Draws M subsets of size k from the set of labels.
 *
 * This method draws M subsets of size k from the set of labels. and trains PS upon each one, then
 * combines label votes from these PS classifiers to get a label-vector prediction. The original
 * RAkEL by <i>Tsoumakas et al.</i> was a meta method, typically taking LC (aka label powerset) base
 * classifiers. This implementation is based on (extends) PS, making it potentially very fast (due
 * to the pruning mechanism offered by PS).
 *
 * See also <i>RAkEL</i> from the <a href=http://mulan.sourceforge.net>MULAN</a> framework.
 *
 * @see PS
 * @author Jesse Read
 * @version September 2015
 */

public class RAkEL extends RAkELd {

  /** for serialization. */
  private static final long serialVersionUID = -6208337124440497991L;

  /**
   * Description to display in the GUI.
   * 
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Draws M subsets of size k from the set of labels, and trains PS upon each one, then combines label votes from the PS classifiers to get a label-vector prediction.";
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    int L = D.classIndex();
    Random random = new Random(this.m_S);

    if (this.getDebug()) {
      System.out.println("Building " + this.m_M + " models of " + this.m_K + " random subsets:");
    }

    this.m_InstancesTemplates = new Instances[this.m_M];
    this.kMap = new int[this.m_M][this.m_K];
    this.m_Classifiers = AbstractClassifier.makeCopies(this.m_Classifier, this.m_M);
    for (int i = 0; i < this.m_M; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      this.kMap[i] = SuperLabelUtils.get_k_subset(L, this.m_K, random);
      if (this.getDebug()) {
        System.out.println("\tmodel " + (i + 1) + "/" + this.m_M + ": " + Arrays.toString(this.kMap[i]) + ", P=" + this.m_P + ", N=" + this.m_N);
      }
      Instances D_i = SuperLabelUtils.makePartitionDataset(D, this.kMap[i], this.m_P, this.m_N);
      this.m_Classifiers[i].buildClassifier(D_i);
      this.m_InstancesTemplates[i] = new Instances(D_i, 0);
    }
  }

  @Override
  public String kTipText() {
    return "The number of labels k in each subset (must be 0 < k < L for L labels)";
  }

  /**
   * Get the M parameter (the number of subsets).
   */
  public int getM() {
    return this.m_M;
  }

  /**
   * Sets the M parameter (the number of subsets)
   */
  public void setM(final int M) {
    this.m_M = M;
  }

  public String mTipText() {
    return "The number of subsets to draw (which together form an ensemble)";
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;
    TechnicalInformation additional;

    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Grigorios Tsoumakas and Ioannis Katakis and Ioannis Vlahavas");
    result.setValue(Field.TITLE, "Random k-Labelsets for Multi-Label Classification");
    result.setValue(Field.JOURNAL, "IEEE Transactions on Knowledge and Data Engineering");
    result.setValue(Field.VOLUME, "99");
    result.setValue(Field.NUMBER, "1");
    result.setValue(Field.YEAR, "2010");

    additional = new TechnicalInformation(Type.INPROCEEDINGS);
    additional.setValue(Field.AUTHOR, "Jesse Read, Antti Puurula, Albert Bifet");
    additional.setValue(Field.TITLE, "Multi-label Classification with Meta-labels");
    additional.setValue(Field.BOOKTITLE, "International Conference on Data Mining");
    additional.setValue(Field.YEAR, "2014");

    result.add(additional);

    return result;
  }

  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    result.addElement(new Option("\tSets M (default 10): the number of subsets", "M", 1, "-M <num>"));
    OptionUtils.add(result, super.listOptions());
    return OptionUtils.toEnumeration(result);
  }

  @Override
  public void setOptions(final String[] options) throws Exception {
    this.setM(OptionUtils.parse(options, 'M', 10));
    super.setOptions(options);
  }

  @Override
  public String[] getOptions() {
    List<String> result = new ArrayList<>();
    OptionUtils.add(result, 'M', this.getM());
    OptionUtils.add(result, super.getOptions());
    return OptionUtils.toArray(result);
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new RAkEL(), args);
  }

}
