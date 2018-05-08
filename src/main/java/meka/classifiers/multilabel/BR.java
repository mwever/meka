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
import java.util.Map;

import meka.core.F;
import meka.core.MLUtils;
import meka.core.MultiLabelDrawable;
/**
 * BR.java - The Binary Relevance Method.
 * The standard baseline Binary Relevance method (BR) -- create a binary problems for each label and learn a model for them individually.
 * See also <i>BR</i> from the <a href=http://mulan.sourceforge.net>MULAN</a> framework
 * @author 	Jesse Read (jmr30@cs.waikato.ac.nz)
 */
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;

public class BR extends ProblemTransformationMethod implements MultiLabelDrawable {

  /** for serialization. */
  private static final long serialVersionUID = -5390512540469007904L;

  protected Classifier m_MultiClassifiers[] = null;
  protected Instances m_InstancesTemplates[] = null;

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "The Binary Relevance Method.\n" + "See also MULAN framework:\n" + "http://mulan.sourceforge.net";
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    int L = D.classIndex();

    if (this.getDebug()) {
      System.out.print("Creating " + L + " models (" + this.m_Classifier.getClass().getName() + "): ");
    }
    this.m_MultiClassifiers = AbstractClassifier.makeCopies(this.m_Classifier, L);
    this.m_InstancesTemplates = new Instances[L];

    for (int j = 0; j < L; j++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }

      // Select only class attribute 'j'
      Instances D_j = F.keepLabels(new Instances(D), L, new int[] { j });
      D_j.setClassIndex(0);

      // Build the classifier for that class
      this.m_MultiClassifiers[j].buildClassifier(D_j);
      if (this.getDebug()) {
        System.out.print(" " + (D_j.classAttribute().name()));
      }

      this.m_InstancesTemplates[j] = new Instances(D_j, 0);
    }
  }

  @Override
  public double[] distributionForInstance(final Instance x) throws Exception {

    int L = x.classIndex();

    double y[] = new double[L];

    for (int j = 0; j < L; j++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      Instance x_j = (Instance) x.copy();
      x_j.setDataset(null);
      x_j = MLUtils.keepAttributesAt(x_j, new int[] { j }, L);
      x_j.setDataset(this.m_InstancesTemplates[j]);
      // y[j] = m_MultiClassifiers[j].classifyInstance(x_j);
      y[j] = this.m_MultiClassifiers[j].distributionForInstance(x_j)[1];
    }

    return y;
  }

  /**
   * Returns the type of graph representing the object.
   *
   * @return the type of graph representing the object (label index as key)
   */
  @Override
  public Map<Integer, Integer> graphType() {
    Map<Integer, Integer> result;
    int i;

    result = new HashMap<>();

    if (this.m_MultiClassifiers != null) {
      for (i = 0; i < this.m_MultiClassifiers.length; i++) {
        if (this.m_MultiClassifiers[i] instanceof Drawable) {
          result.put(i, ((Drawable) this.m_MultiClassifiers[i]).graphType());
        }
      }
    }

    return result;
  }

  /**
   * Returns a string that describes a graph representing the object. The string should be in XMLBIF
   * ver. 0.3 format if the graph is a BayesNet, otherwise it should be in dotty format.
   *
   * @return the graph described by a string (label index as key)
   * @throws Exception
   *           if the graph can't be computed
   */
  @Override
  public Map<Integer, String> graph() throws Exception {
    Map<Integer, String> result;
    int i;

    result = new HashMap<>();

    if (this.m_MultiClassifiers != null) {
      for (i = 0; i < this.m_MultiClassifiers.length; i++) {
        if (this.m_MultiClassifiers[i] instanceof Drawable) {
          result.put(i, ((Drawable) this.m_MultiClassifiers[i]).graph());
        }
      }
    }

    return result;
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new BR(), args);
  }

}
