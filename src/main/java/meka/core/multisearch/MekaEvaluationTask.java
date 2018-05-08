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
 * MekaEvaluationTask.java
 * Copyright (C) 2017 University of Waikato, Hamilton, NZ
 */

package meka.core.multisearch;

import java.io.Serializable;

import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.Result;
import weka.classifiers.Classifier;
import weka.classifiers.meta.multisearch.AbstractEvaluationTask;
import weka.classifiers.meta.multisearch.MultiSearchCapable;
import weka.classifiers.meta.multisearch.Performance;
import weka.core.Instances;
import weka.core.SetupGenerator;
import weka.core.setupgenerator.Point;

/**
 * Meka Evaluation task.
 */
public class MekaEvaluationTask extends AbstractEvaluationTask {

  /** the threshold option. */
  protected String m_TOP;

  /** the verbosity option. */
  protected String m_VOP;

  /**
   * Initializes the task.
   *
   * @param owner
   *          the owning MultiSearch classifier
   * @param train
   *          the training data
   * @param test
   *          the test data, can be null
   * @param generator
   *          the generator to use
   * @param values
   *          the setup values
   * @param folds
   *          the number of cross-validation folds
   * @param eval
   *          the type of evaluation
   * @param classLabel
   *          the class label index (0-based; if applicable)
   */
  public MekaEvaluationTask(final MultiSearchCapable owner, final Instances train, final Instances test, final SetupGenerator generator, final Point<Object> values,
      final int folds, final int eval, final int classLabel) {
    super(owner, train, test, generator, values, folds, eval, classLabel);
    this.m_TOP = "PCut1";
    this.m_VOP = "3";
  }

  /**
   * Returns whether predictions can be discarded (depends on selected measure).
   */
  protected boolean canDiscardPredictions() {
    switch (this.m_Owner.getEvaluation().getSelectedTag().getID()) {
      default:
        return true;
    }
  }

  /**
   * Performs the evaluation.
   *
   * @return false if evaluation fails
   */
  @Override
  protected Boolean doRun() throws Exception {
    Point<Object> evals;
    Result eval;
    MultiLabelClassifier classifier;
    Performance performance;
    boolean completed;

    // setup
    evals = this.m_Generator.evaluate(this.m_Values);
    classifier = (MultiLabelClassifier) this.m_Generator.setup((Serializable) this.m_Owner.getClassifier(), evals);

    // evaluate
    try {
      if (this.m_Test == null) {
        if (this.m_Folds >= 2) {
          eval = Evaluation.cvModel(classifier, this.m_Train, this.m_Folds, this.m_TOP, this.m_VOP);
        } else {
          classifier.buildClassifier(this.m_Train);
          eval = Evaluation.evaluateModel(classifier, this.m_Train, this.m_TOP, this.m_VOP);
        }
      } else {
        classifier.buildClassifier(this.m_Train);
        eval = Evaluation.evaluateModel(classifier, this.m_Test, this.m_TOP, this.m_VOP);
      }
      completed = true;
    } catch (Exception e) {
      eval = null;
      completed = false;
    }

    // store performance
    performance = new Performance(this.m_Values, this.m_Owner.getFactory().newWrapper(eval), this.m_Evaluation, this.m_ClassLabel,
        (Classifier) this.m_Generator.setup((Serializable) this.m_Owner.getClassifier(), evals));
    this.m_Owner.getAlgorithm().addPerformance(performance, this.m_Folds);

    // log
    this.m_Owner.log(performance + ": cached=false");

    return completed;
  }
}
