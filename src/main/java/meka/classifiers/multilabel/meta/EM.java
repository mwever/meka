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
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.ProblemTransformationMethod;
import meka.classifiers.multilabel.SemisupervisedClassifier;
import meka.core.MLUtils;
import meka.core.OptionUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * EM.java - Expectation Maximization using any multi-label classifier.
 *
 * A specified multi-label classifier is built on the training data. This model is then used to
 * classify the test data. The confidence with which instances are classified is used to reweight
 * them. This data is then used to retrain the classifier. This cycle continues ('EM'-style) for I
 * iterations. The final model is used to officially classifier the test data. <br>
 * Because of the weighting, it is advised to use a classifier which gives good confidence
 * (probabalistic) outputs. <br>
 *
 * @version 2010
 * @author Jesse Read
 */
public class EM extends ProblemTransformationMethod implements SemisupervisedClassifier, TechnicalInformationHandler {

  private static final long serialVersionUID = 2622231824673975335L;
  protected int m_I = 10;
  protected Instances D_ = null;

  public EM() {
    // default classifier for GUI
    this.m_Classifier = new CC();
  }

  @Override
  protected String defaultClassifierString() {
    // default classifier for CLI
    return "meka.classifiers.multilabel.CC";
  }

  @Override
  public void introduceUnlabelledData(final Instances D) {
    this.D_ = D;
  }

  @Override
  public String globalInfo() {
    return ""// "Train a classifier using labelled and unlabelled data (semi-supervised) using an EM-type
             // algorithm. Works best if the classifier can give good probabalistic outputs. " + "A similar
             // procedure was used with LC and Naive Bayes in:\n" + getTechnicalInformation().toString();
        + "A specified multi-label classifier is built on the training data. This model is then used to classify the test data. "
        + "The confidence with which instances are classified is used to reweight them. This data is then used to retrain the classifier. "
        + "This cycle continues ('EM'-style) for I iterations. The final model is used to officially classifier the test data. "
        + "Because of the weighting, it is advised to use a classifier which gives good confidence (probabalistic) outputs. ";
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    if (this.getDebug()) {
      System.out.println("Initial build ...");
    }

    this.m_Classifier.buildClassifier(D);

    Instances DA = MLUtils.combineInstances(D, this.D_);

    if (this.getDebug()) {
      System.out.print("Performing " + this.m_I + " 'EM' Iterations: [");
    }
    for (int i = 0; i < this.m_I; i++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      if (this.getDebug()) {
        System.out.print(".");
      }
      // expectation (classify + update weights)
      this.updateWeights((ProblemTransformationMethod) this.m_Classifier, DA);
      // maximization of parameters (training)
      this.m_Classifier.buildClassifier(DA);
    }
    System.out.println("]");
  }

  protected void updateWeights(final ProblemTransformationMethod h, final Instances D) throws Exception {
    for (Instance x : D) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      double w = 1.0; // weight (product of probability)
      double y[] = h.distributionForInstance(x);
      // convert ML distribution into probability vector, and multiply to w as we go ..
      for (int j = 0; j < y.length; j++) {
        w *= (y[j] < 0.5) ? 1. - y[j] : y[j];
      }
      x.setWeight(w);
    }
  }

  @Override
  public double[] distributionForInstance(final Instance x) throws Exception {
    return this.m_Classifier.distributionForInstance(x);
  }

  public void setIterations(final int i) {
    this.m_I = i;
  }

  public int getIterations() {
    return this.m_I;
  }

  public String iterationsTipText() {
    return "The number of EM iterations to perform.";
  }

  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    result.addElement(new Option("\tThe number of iterations of EM to carry out (default: 10)", "I", 1, "-I <value>"));
    OptionUtils.add(result, super.listOptions());
    return OptionUtils.toEnumeration(result);
  }

  @Override
  public void setOptions(final String[] options) throws Exception {
    this.setIterations(OptionUtils.parse(options, 'I', 10));
    super.setOptions(options);
  }

  @Override
  public String[] getOptions() {
    List<String> result = new ArrayList<>();
    OptionUtils.add(result, 'I', this.getIterations());
    OptionUtils.add(result, super.getOptions());
    return OptionUtils.toArray(result);
  }

  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 9117 $");
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Nigam, Kamal and Mccallum, Andrew K. and Thrun, Sebastian and Mitchell, Tom M.");
    result.setValue(Field.TITLE, "Text classification from Labeled and Unlabeled Documents using EM");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "39");
    result.setValue(Field.NUMBER, "2/3");
    result.setValue(Field.PAGES, "103--134");
    result.setValue(Field.YEAR, "2010");

    return result;
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new EM(), args);
  }

}
