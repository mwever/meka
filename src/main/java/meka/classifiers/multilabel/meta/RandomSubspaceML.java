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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import meka.classifiers.multilabel.ProblemTransformationMethod;
import meka.core.A;
import meka.core.F;
import meka.core.MLUtils;
import meka.core.OptionUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * RandomSubspaceML.java - Subsample the attribute space and instance space randomly for each
 * ensemble member. Basically a generalized version of Random Forests. It is computationally cheaper
 * than EnsembleML for the same number of models. <br>
 * As used with CC in: Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank. <i>Classifier
 * Chains for Multi-label Classification</i>. Machine Learning Journal. Springer. Vol. 85(3), pp
 * 333-359. (May 2011). <br>
 * In earlier versions of Meka this class was called <i>BaggingMLq</i> and used Bagging procedure.
 * Now it uses a simple ensemble cut. <br>
 *
 * @author Jesse Read
 * @version June 2014
 */

public class RandomSubspaceML extends MetaProblemTransformationMethod implements TechnicalInformationHandler {

  /** for serialization. */
  private static final long serialVersionUID = 3608541911971484299L;

  protected int m_AttSizePercent = 50;

  protected int m_IndicesCut[][] = null;
  protected Instances m_InstancesTemplates[] = null;
  protected Instance m_InstanceTemplates[] = null;

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    this.m_InstancesTemplates = new Instances[this.m_NumIterations];
    this.m_InstanceTemplates = new Instance[this.m_NumIterations];

    if (this.getDebug()) {
      System.out.println("-: Models: ");
    }

    this.m_Classifiers = ProblemTransformationMethod.makeCopies((ProblemTransformationMethod) this.m_Classifier, this.m_NumIterations);

    Random r = new Random(this.m_Seed);

    int N_sub = (D.numInstances() * this.m_BagSizePercent / 100);

    int L = D.classIndex();
    int d = D.numAttributes() - L;
    int d_new = d * this.m_AttSizePercent / 100;
    this.m_IndicesCut = new int[this.m_NumIterations][];

    for (int i = 0; i < this.m_NumIterations; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }

      // Downsize the instance space (exactly like in EnsembleML.java)

      if (this.getDebug()) {
        System.out.print("\t" + (i + 1) + ": ");
      }
      D.randomize(r);
      Instances D_cut = new Instances(D, 0, N_sub);
      if (this.getDebug()) {
        System.out.print("N=" + D.numInstances() + " -> N'=" + D_cut.numInstances() + ", ");
      }

      // Downsize attribute space

      D_cut.setClassIndex(-1);
      int indices_a[] = A.make_sequence(L, d + L);
      A.shuffle(indices_a, r);
      indices_a = Arrays.copyOfRange(indices_a, 0, d - d_new);
      Arrays.sort(indices_a);
      this.m_IndicesCut[i] = A.invert(indices_a, D.numAttributes());
      D_cut = F.remove(D_cut, indices_a, false);
      D_cut.setClassIndex(L);
      if (this.getDebug()) {
        System.out.print(" A:=" + (D.numAttributes() - L) + " -> A'=" + (D_cut.numAttributes() - L) + " (" + this.m_IndicesCut[i][L] + ",...,"
            + this.m_IndicesCut[i][this.m_IndicesCut[i].length - 1] + ")");
      }

      // Train multi-label classifier

      if (this.m_Classifiers[i] instanceof Randomizable) {
        ((Randomizable) this.m_Classifiers[i]).setSeed(this.m_Seed + i);
      }
      if (this.getDebug()) {
        System.out.println(".");
      }

      this.m_Classifiers[i].buildClassifier(D_cut);
      this.m_InstanceTemplates[i] = D_cut.instance(1);
      this.m_InstancesTemplates[i] = new Instances(D_cut, 0);
    }
    if (this.getDebug()) {
      System.out.println(":-");
    }
  }

  @Override
  public double[] distributionForInstance(final Instance x) throws Exception {

    int L = x.classIndex();
    double p[] = new double[L];

    for (int i = 0; i < this.m_NumIterations; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      // Use a template Instance from training, and copy values over
      // (this is faster than copying x and cutting it to shape)
      Instance x_ = this.m_InstanceTemplates[i];
      MLUtils.copyValues(x_, x, this.m_IndicesCut[i]);
      x_.setDataset(this.m_InstancesTemplates[i]);

      // TODO, use generic voting scheme somewhere?
      double d[] = ((ProblemTransformationMethod) this.m_Classifiers[i]).distributionForInstance(x_);
      for (int j = 0; j < d.length; j++) {
        p[j] += d[j];
      }
    }

    return p;
  }

  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    result.addElement(
        new Option("\tSize of attribute space, as a percentage of total attribute space size (must be between 1 and 100, default: 50)", "A", 1, "-A <size percentage>"));
    OptionUtils.add(result, super.listOptions());
    return OptionUtils.toEnumeration(result);
  }

  @Override
  public void setOptions(final String[] options) throws Exception {
    this.setAttSizePercent(OptionUtils.parse(options, 'A', 50));
    super.setOptions(options);
  }

  @Override
  public String[] getOptions() {
    List<String> result = new ArrayList<>();
    OptionUtils.add(result, 'A', this.getAttSizePercent());
    OptionUtils.add(result, super.getOptions());
    return OptionUtils.toArray(result);
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new RandomSubspaceML(), args);
  }

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Combining several multi-label classifiers in an ensemble where the attribute space for each model is a random subset of the original space.";
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank");
    result.setValue(Field.TITLE, "Classifier Chains for Multi-label Classification");
    result.setValue(Field.JOURNAL, "Machine Learning Journal");
    result.setValue(Field.YEAR, "2011");
    result.setValue(Field.VOLUME, "85");
    result.setValue(Field.NUMBER, "3");
    result.setValue(Field.PAGES, "333-359");

    return result;
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }

  /**
   * Sets the percentage of attributes to sample from the original set.
   */
  public void setAttSizePercent(final int value) {
    this.m_AttSizePercent = value;
  }

  /**
   * Gets the percentage of attributes to sample from the original set.
   */
  public int getAttSizePercent() {
    return this.m_AttSizePercent;
  }

  public String attSizePercentTipText() {
    return "Size of attribute space, as a percentage of total attribute space size";
  }
}
