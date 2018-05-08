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

/*
 * FilteredClassifier.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, NZ
 */

package meka.classifiers.multilabel.meta;

import java.util.Random;

import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.classifiers.multilabel.ProblemTransformationMethod;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.WeightedAttributesHandler;
import weka.core.WeightedInstancesHandler;
import weka.filters.AllFilter;
import weka.filters.Filter;

/**
 * Allows the application of a filter in conjunction with a multi-label classifier.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class FilteredClassifier extends weka.classifiers.meta.FilteredClassifier implements MultiLabelClassifier {

  private static final long serialVersionUID = 4466454723202805056L;

  /**
   * Default constructor.
   *
   * Turns off check for modified class attribute.
   */
  public FilteredClassifier() {
    super();
    this.setDoNotCheckForModifiedClassAttribute(true);
    this.m_Classifier = new BR();
    this.m_Filter = new AllFilter();
  }

  /**
   * String describing default classifier.
   */
  @Override
  protected String defaultClassifierString() {
    return BR.class.getName();
  }

  /**
   * Set the base learner.
   *
   * @param newClassifier
   *          the classifier to use.
   */
  @Override
  public void setClassifier(final Classifier newClassifier) {
    if (!(newClassifier instanceof MultiLabelClassifier)) {
      throw new IllegalArgumentException("Classifier must be a " + MultiLabelClassifier.class.getName() + "!");
    }
    super.setClassifier(newClassifier);
  }

  /**
   * TestCapabilities. Make sure the training data is suitable.
   * 
   * @param D
   *          the data
   */
  public void testCapabilities(final Instances D) throws Exception {
    // get the classifier's capabilities, enable all class attributes and do the usual test
    Capabilities cap = this.getCapabilities();
    cap.enableAllClasses();
    // get the capabilities again, test class attributes individually
    int L = D.classIndex();
    for (int j = 0; j < L; j++) {
      Attribute c = D.attribute(j);
      cap.testWithFail(c, true);
    }
  }

  /**
   * Sets up the filter and runs checks.
   *
   * @return filtered data
   */
  protected Instances setUp(Instances data) throws Exception {
    String relName;
    String classAtt;

    relName = data.relationName();
    classAtt = data.classAttribute().name();

    if (this.m_Classifier == null) {
      throw new Exception("No base classifiers have been set!");
    }

    this.getCapabilities().testWithFail(data);

    // get fresh instances object
    data = new Instances(data);

    Attribute classAttribute = (Attribute) data.classAttribute().copy();
    this.m_Filter.setInputFormat(data); // filter capabilities are checked here
    data = Filter.useFilter(data, this.m_Filter);
    if ((!classAttribute.equals(data.classAttribute())) && (!this.m_DoNotCheckForModifiedClassAttribute)) {
      throw new IllegalArgumentException("Cannot proceed: " + this.getFilterSpec() + " has modified the class attribute!");
    }

    data.setRelationName(relName);
    data.setClassIndex(data.attribute(classAtt).index());

    // can classifier handle the data?
    this.testCapabilities(data);

    this.m_FilteredInstances = data.stringFreeStructure();

    return data;
  }

  /**
   * Build the classifier on the filtered data.
   *
   * @param data
   *          the training data
   * @throws Exception
   *           if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

    if (this.m_Classifier == null) {
      throw new Exception("No base classifier has been set!");
    }

    this.getCapabilities().testWithFail(data);

    Random r = (data.numInstances() > 0) ? data.getRandomNumberGenerator(this.getSeed()) : new Random(this.getSeed());
    data = this.setUp(data, r);
    if (!data.allInstanceWeightsIdentical() && !(this.m_Classifier instanceof WeightedInstancesHandler)) {
      data = data.resampleWithWeights(r); // The filter may have assigned weights.
    }
    if (!data.allAttributeWeightsIdentical() && !(this.m_Classifier instanceof WeightedAttributesHandler)) {
      data = this.resampleAttributes(data, false, r);
    }

    if (this.m_Classifier instanceof Randomizable) {
      ((Randomizable) this.m_Classifier).setSeed(r.nextInt());
    }

    this.m_Classifier.buildClassifier(data);
  }

  /**
   * Returns a string representation of the model.
   *
   * @return the model
   */
  @Override
  public String getModel() {
    if (this.m_Classifier instanceof MultiLabelClassifier) {
      return ((MultiLabelClassifier) this.m_Classifier).getModel();
    } else {
      return this.toString();
    }
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new FilteredClassifier(), args);
  }
}
