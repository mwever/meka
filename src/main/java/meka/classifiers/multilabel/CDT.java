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
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import meka.classifiers.multilabel.cc.CNode;
import meka.classifiers.multilabel.cc.Trellis;
import meka.core.A;
import meka.core.OptionUtils;
import meka.core.StatUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * CDT.java - Conditional Dependency Trellis. Like CDN, but with a trellis structure (like CT)
 * rather than a fully connected network.
 *
 * @see CDN
 * @see CT
 * @author Jesse Read
 * @version January 2014
 */
public class CDT extends CDN {

  private static final long serialVersionUID = -1237783546336254364L;

  protected int m_Width = -1;
  protected int m_Density = 1;
  protected String m_DependencyMetric = "None";

  Trellis trel = null;

  protected CNode nodes[] = null;

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    int L = D.classIndex();
    int d = D.numAttributes() - L;
    this.m_R = new Random(this.getSeed());
    int width = this.m_Width;

    if (this.m_Width < 0) {
      width = (int) Math.sqrt(L);
    } else if (this.m_Width == 0) {
      width = L;
    }

    this.nodes = new CNode[L];
    /*
     * Make the Trellis.
     */
    if (this.getDebug()) {
      System.out.println("Make Trellis of width " + this.m_Width);
    }
    int indices[] = A.make_sequence(L);
    A.shuffle(indices, new Random(this.getSeed()));
    this.trel = new Trellis(indices, width, this.m_Density);
    if (this.getDebug()) {
      System.out.println("==>\n" + this.trel.toString());
    }

    /* Rearrange the Trellis */
    if (!this.m_DependencyMetric.equals("None")) {
      this.trel = CT.orderTrellis(this.trel, StatUtils.margDepMatrix(D, this.m_DependencyMetric), this.m_R);
    }

    /*
     * Build Trellis
     */
    if (this.getDebug()) {
      System.out.println("Build Trellis");
    }

    if (this.getDebug()) {
      System.out.println("nodes: " + Arrays.toString(this.trel.indices));
    }

    for (int j = 0; j < L; j++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      int jv = this.trel.indices[j];
      if (this.getDebug()) {
        System.out.println("Build Node h_" + jv + "] : P(y_" + jv + " | x_[1:d], y_" + Arrays.toString(this.trel.getNeighbours(j)) + ")");
      }
      this.nodes[jv] = new CNode(jv, null, this.trel.getNeighbours(j));
      this.nodes[jv].build(D, this.m_Classifier);
    }

  }

  @Override
  public double[] distributionForInstance(final Instance x) throws Exception {

    int L = x.classIndex();
    double y[] = new double[L]; // for sampling
    double y_marg[] = new double[L]; // for collectiing marginal

    int sequence[] = A.make_sequence(L);

    double likelihood[] = new double[L];

    for (int i = 0; i < this.I; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      Collections.shuffle(Arrays.asList(sequence));
      for (int j : sequence) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread has been interrupted.");
        }
        // sample
        y[j] = this.nodes[j].sample(x, y, this.m_R);
        // collect marginals
        if (i > (this.I - this.I_c)) {
          y_marg[j] += y[j];
        }
        // else still burning in
      }
    }
    // finish, calculate marginals
    for (int j = 0; j < L; j++) {
      y_marg[j] /= this.I_c;
    }

    return y_marg;
  }

  /* NOTE: these options in common with CT */

  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    result.addElement(new Option("\t" + this.widthTipText(), "H", 1, "-H <value>"));
    result.addElement(new Option("\t" + this.densityTipText(), "L", 1, "-L <value>"));
    result.addElement(new Option("\t" + this.dependencyMetricTipText(), "X", 1, "-X <value>"));
    OptionUtils.add(result, super.listOptions());
    return OptionUtils.toEnumeration(result);
  }

  @Override
  public void setOptions(final String[] options) throws Exception {
    this.setWidth(OptionUtils.parse(options, 'H', -1));
    this.setDensity(OptionUtils.parse(options, 'L', 1));
    this.setDependencyMetric(OptionUtils.parse(options, 'X', "None"));
    super.setOptions(options);
  }

  @Override
  public String[] getOptions() {
    List<String> result = new ArrayList<>();
    OptionUtils.add(result, 'H', this.getWidth());
    OptionUtils.add(result, 'L', this.getDensity());
    OptionUtils.add(result, 'X', this.getDependencyMetric());
    OptionUtils.add(result, super.getOptions());
    return OptionUtils.toArray(result);
  }

  /**
   * GetDensity - Get the neighbourhood density (number of neighbours for each node).
   */
  public int getDensity() {
    return this.m_Density;
  }

  /**
   * SetDensity - Sets the neighbourhood density (number of neighbours for each node).
   */
  public void setDensity(final int c) {
    this.m_Density = c;
  }

  public String densityTipText() {
    return "Determines the neighbourhood density (the number of neighbours for each node in the trellis).";
  }

  /**
   * GetH - Get the trellis width.
   */
  public int getWidth() {
    return this.m_Width;
  }

  /**
   * SetH - Sets the trellis width.
   */
  public void setWidth(final int h) {
    this.m_Width = h;
  }

  public String widthTipText() {
    return "Determines the width of the trellis (use 0 for chain; use -1 for a square trellis, i.e., width of sqrt(number of labels)).";
  }

  /**
   * GetDependency - Get the type of depependency to use in rearranging the trellis (None by default)
   */
  public String getDependencyMetric() {
    return this.m_DependencyMetric;
  }

  /**
   * SetDependency - Sets the type of depependency to use in rearranging the trellis (None by default)
   */
  public void setDependencyMetric(final String m) {
    this.m_DependencyMetric = m;
  }

  public String dependencyMetricTipText() {
    return "The dependency heuristic to use in rearranging the trellis (None by default).";
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new CDT(), args);
  }

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "A Conditional Dependency Trellis. Like CDN, but with a trellis structure (like CT) rather than a fully connected network." + "For more information see:\n"
        + this.getTechnicalInformation().toString();
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Yuhong Guoand and Suicheng Gu");
    result.setValue(Field.TITLE, "Multi-Label Classification Using Conditional Dependency Networks");
    result.setValue(Field.BOOKTITLE, "IJCAI '11");
    result.setValue(Field.YEAR, "2011");

    result.add(new CT().getTechnicalInformation());

    return result;
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }
}
