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

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * FW.java Four-class pairWise classification. Trains a multi-class base classifier for each pair of
 * labels -- (L*(L-1))/2 in total --, each with four possible class values: {00,01,10,11}
 * representing the possible combinations of relevant (1) /irrelevant (0) for the pair. Uses a
 * voting + threshold scheme at testing time where e.g., 01 from pair jk gives one vote to label k;
 * any label with votes above the threshold is considered relevant.
 *
 * @version October 2012
 * @author Jesse Read (jesse@tsc.uc3m.es)
 */
public class FW extends ProblemTransformationMethod {

  private static final long serialVersionUID = -8259554419725274112L;
  Classifier h[][] = null;
  Attribute classAttribute = null;

  @Override
  public String globalInfo() {
    return "The Fourclass Pairwise (FW) method.\n"
        + "Trains a multi-class base classifier for each pair of labels -- (L*(L-1))/2 in total --, each with four possible class values: {00,01,10,11} representing the possible combinations of relevant (1) /irrelevant (0) for the pair. Uses a voting + threshold scheme at testing time where e.g., 01 from pair jk gives one vote to label k; any label with votes above the threshold is considered relevant.";
  }

  protected Instances convert(Instances D, final int j, final int k) throws InterruptedException {

    int L = D.classIndex();

    D = new Instances(D);

    D.insertAttributeAt(this.classAttribute, 0);
    D.setClassIndex(0);

    for (int i = 0; i < D.numInstances(); i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      String c = (int) Math.round(D.instance(i).value(j + 1)) + "" + (int) Math.round(D.instance(i).value(k + 1));
      D.instance(i).setClassValue(c);
    }

    for (int i = 0; i < L; i++) {
      D.deleteAttributeAt(1);
    }

    this.m_InstancesTemplate = new Instances(D, 0);

    return D;
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    FastVector values = new FastVector(4);
    values.addElement("00");
    values.addElement("10");
    values.addElement("01");
    values.addElement("11");
    this.classAttribute = new Attribute("TheCLass", values);

    int L = D.classIndex();

    this.h = new Classifier[L][L];

    for (int j = 0; j < L; j++) {
      for (int k = j + 1; k < L; k++) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread has been interrupted.");
        }
        if (this.getDebug()) {
          System.out.print(".");
        }
        Instances D_pair = this.convert(D, j, k);
        this.h[j][k] = AbstractClassifier.forName(this.getClassifier().getClass().getName(), ((AbstractClassifier) this.getClassifier()).getOptions());
        this.h[j][k].buildClassifier(D_pair);
      }
      if (this.getDebug()) {
        System.out.println("");
      }
    }

  }

  @Override
  public double[] distributionForInstance(Instance x) throws Exception {

    int L = x.classIndex();

    x = (Instance) x.copy();
    x.setDataset(null);
    for (int i = 1; i < L; i++) {
      x.deleteAttributeAt(1);
    }
    x.setDataset(this.m_InstancesTemplate);

    double r[] = new double[L];

    for (int j = 0; j < L; j++) {
      for (int k = j + 1; k < L; k++) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread has been interrupted.");
        }
        // double d[] = h[j][k].distributionForInstance(x);
        int c = (int) Math.round(this.h[j][k].classifyInstance(x));
        if (c == 1) {
          r[j] += 1.0;
        }
        if (c == 2) {
          r[k] += 1.0;
        }
        if (c == 3) {
          r[j] += 1.0;
          r[k] += 1.0;
        }
      }
    }

    return r;
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new FW(), args);
  }

}
