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

import java.util.Random;

import meka.classifiers.multilabel.ProblemTransformationMethod;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.RevisionUtils;

/**
 * BaggingMLdup.java - A version of BaggingML where Instances are duplicated instead of assigned
 * higher weighs. Duplicates Instances instead of assigning higher weights -- should work for
 * methods that do not handle weights at all.
 *
 * @author Jesse Read (jmr30@cs.waikato.ac.nz)
 */
public class BaggingMLdup extends MetaProblemTransformationMethod {

  /** for serialization. */
  private static final long serialVersionUID = -5606278379913020097L;

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Combining several multi-label classifiers using Bootstrap AGGregatING.\n"
        + "Duplicates Instances instead of assigning higher weights -- should work for methods that do not handle weights at all.";
  }

  @Override
  public void buildClassifier(final Instances train) throws Exception {
    this.testCapabilities(train);

    if (this.getDebug()) {
      System.out.print("-: Models: ");
    }

    // m_Classifiers = (MultilabelClassifier[]) AbstractClassifier.makeCopies(m_Classifier,
    // m_NumIterations);
    this.m_Classifiers = ProblemTransformationMethod.makeCopies((ProblemTransformationMethod) this.m_Classifier, this.m_NumIterations);

    for (int i = 0; i < this.m_NumIterations; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      Random r = new Random(this.m_Seed + i);
      Instances bag = new Instances(train, 0);
      if (this.m_Classifiers[i] instanceof Randomizable) {
        ((Randomizable) this.m_Classifiers[i]).setSeed(this.m_Seed + i);
      }
      if (this.getDebug()) {
        System.out.print("" + i + " ");
      }

      int bag_no = (this.m_BagSizePercent * train.numInstances() / 100);
      // System.out.println(" bag no: "+bag_no);
      while (bag.numInstances() < bag_no) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread has been interrupted.");
        }
        bag.add(train.instance(r.nextInt(train.numInstances())));
      }
      this.m_Classifiers[i].buildClassifier(bag);
    }
    if (this.getDebug()) {
      System.out.println(":-");
    }
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new BaggingMLdup(), args);
  }

}
