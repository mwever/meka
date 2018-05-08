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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.MultiLabelDrawable;
import meka.core.OptionUtils;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * CC.java - The Classifier Chains Method. Like BR, but label outputs become new inputs for the next
 * classifiers in the chain. <br>
 * See: Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank. <i>Classifier Chains for
 * Multi-label Classification</i>. Machine Learning Journal. Springer. Vol. 85(3), pp 333-359. (May
 * 2011). <br>
 * See: Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank. <i>Classifier Chains for
 * Multi-label Classification</i>. In Proc. of 20th European Conference on Machine Learning (ECML
 * 2009). Bled, Slovenia, September 2009. <br>
 *
 * Note that the code was reorganised substantially since earlier versions, to accomodate additional
 * functionality needed for e.g., MCC, PCC.
 *
 * @author Jesse Read
 * @version December 2013
 */
public class CC extends ProblemTransformationMethod implements Randomizable, TechnicalInformationHandler, MultiLabelDrawable {

  private static final long serialVersionUID = -4115294965331340629L;

  protected CNode nodes[] = null;

  protected int m_S = this.getDefaultSeed();

  protected Random m_R = null;

  protected int m_Chain[] = null;

  /**
   * Prepare a Chain. One of the following:<br>
   * - Use pre-set chain. If there is none, then <br>
   * - Use default chain (1,2,...,L). Unless a different random seed has been set, then <br>
   * - Use a random chain.
   *
   * @param L
   *          number of labels
   */
  protected void prepareChain(final int L) {

    int chain[] = this.retrieveChain();

    // if has not yet been manually chosen ...
    if (chain == null) {

      // create the standard order (1,2,...,L) ..
      chain = A.make_sequence(L);

      // and shuffle if m_S > 0
      if (this.m_S != 0) {
        this.m_R = new Random(this.m_S);
        A.shuffle(chain, this.m_R);
      }
    }

    // set it
    this.prepareChain(chain);

  }

  /**
   * Prepare a Chain. Set the specified 'chain'. It must contain all indices [0,...,L-1] (but in any
   * order)
   *
   * @param chain
   *          a specified chain
   */
  public void prepareChain(final int chain[]) {
    this.m_Chain = Arrays.copyOf(chain, chain.length);
    if (this.getDebug()) {
      System.out.println("Chain s=" + Arrays.toString(this.m_Chain));
    }
  }

  public int[] retrieveChain() {
    return this.m_Chain;
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    int L = D.classIndex();

    this.prepareChain(L);

    /*
     * make a classifier node for each label, taking the parents of all previous nodes
     */
    if (this.getDebug()) {
      System.out.print(":- Chain (");
    }
    this.nodes = new CNode[L];
    int pa[] = new int[] {};

    for (int j : this.m_Chain) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      if (this.getDebug()) {
        System.out.print(" " + D.attribute(j).name());
      }
      this.nodes[j] = new CNode(j, null, pa);
      this.nodes[j].build(D, this.m_Classifier);
      pa = A.append(pa, j);
    }
    if (this.getDebug()) {
      System.out.println(" ) -:");
    }

    // to store posterior probabilities (confidences)
    this.confidences = new double[L];
  }

  protected double confidences[] = null;

  /**
   * GetConfidences - get the posterior probabilities of the previous prediction (after calling
   * distributionForInstance(x)).
   */
  public double[] getConfidences() {
    return this.confidences;
  }

  @Override
  public double[] distributionForInstance(final Instance x) throws Exception {
    int L = x.classIndex();
    double y[] = new double[L];

    for (int j : this.m_Chain) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      // h_j : x,pa_j -> y_j
      y[j] = this.nodes[j].classify((Instance) x.copy(), y);
    }

    return y;
  }

  /**
   * SampleForInstance. predict y[j] stochastically rather than deterministically (as with
   * distributionForInstance(Instance x)).
   *
   * @param x
   *          test Instance
   * @param r
   *          Random &lt;- TODO probably can use this.m_R instead
   */
  public double[] sampleForInstance(final Instance x, final Random r) throws Exception {
    int L = x.classIndex();
    double y[] = new double[L];

    for (int j : this.m_Chain) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      double p[] = this.nodes[j].distribution(x, y);
      y[j] = A.samplePMF(p, r);
      this.confidences[j] = p[(int) y[j]];
    }

    return y;
  }

  /**
   * GetTransformTemplates - pre-transform the instance x, to make things faster.
   *
   * @return the templates
   */
  public Instance[] getTransformTemplates(final Instance x) throws Exception {
    int L = x.classIndex();
    Instance t_[] = new Instance[L];
    double ypred[] = new double[L];
    for (int j : this.m_Chain) {
      t_[j] = this.nodes[j].transform(x, ypred);
    }
    return t_;
  }

  /**
   * SampleForInstance - given an Instance template for each label, and a Random.
   *
   * @param t_
   *          Instance templates (pre-transformed) using #getTransformTemplates(x)
   */
  public double[] sampleForInstanceFast(final Instance t_[], final Random r) throws Exception {

    int L = t_.length;
    double y[] = new double[L];

    for (int j : this.m_Chain) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      double p[] = this.nodes[j].distribution(t_[j], y); // e.g., [0.4, 0.6]
      y[j] = A.samplePMF(p, r); // e.g., 0
      this.confidences[j] = p[(int) y[j]]; // e.g., 0.4
      this.nodes[j].updateTransform(t_[j], y); // need to update the transform #SampleForInstance(x,r)
    }

    return y;
  }

  /**
   * TransformInstances - this function is DEPRECATED. this function preloads the instances with the
   * correct class labels ... to make the chain much faster, but CNode does not yet have this
   * functionality ... need to do something about this!
   */
  public Instance[] transformInstance(final Instance x) throws Exception {
    return null;
    /*
     * //System.out.println("CHAIN : "+Arrays.toString(this.getChain())); int L = x.classIndex();
     * Instance x_copy[] = new Instance[L]; root.transform(x,x_copy); return x_copy;
     */
  }

  /**
   * ProbabilityForInstance - Force our way down the imposed 'path'. <br>
   * TODO rename distributionForPath ? and simplify like distributionForInstance ? <br>
   * For example p (y=1010|x) = [0.9,0.8,0.1,0.2]. If the product = 1, this is probably the correct
   * path!
   *
   * @param x
   *          test Instance
   * @param path
   *          the path we want to go down
   * @return the probabilities associated with this path: [p(Y_1==path[1]|x),...,p(Y_L==path[L]|x)]
   */
  public double[] probabilityForInstance(final Instance x, final double path[]) throws Exception {
    int L = x.classIndex();
    double p[] = new double[L];

    for (int j : this.m_Chain) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      // h_j : x,pa_j -> y_j
      double d[] = this.nodes[j].distribution((Instance) x.copy(), path); // <-- posterior distribution
      int k = (int) Math.round(path[j]); // <-- value of interest
      p[j] = d[k]; // <-- p(y_j==k) i.e., 'confidence'
      // y[j] = path[j];
    }

    return p;
  }

  /**
   * Rebuild - NOT YET IMPLEMENTED. For efficiency reasons, we may want to rebuild part of the chain.
   * If chain[] = [1,2,3,4] and new_chain[] = [1,2,4,3] we only need to rebuild the final two links.
   *
   * @param new_chain
   *          the new chain
   * @param D
   *          the original training data
   */
  public void rebuildClassifier(final int new_chain[], final Instances D) throws Exception {
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("Thread has been interrupted.");
    }
  }

  public int getDefaultSeed() {
    return 0;
  }

  @Override
  public int getSeed() {
    return this.m_S;
  }

  @Override
  public void setSeed(final int s) {
    this.m_S = s;
  }

  public String seedTipText() {
    return "The seed value for randomizing the data.";
  }

  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    OptionUtils.addOption(result, this.seedTipText(), "" + this.getDefaultSeed(), 'S');
    OptionUtils.add(result, super.listOptions());
    return OptionUtils.toEnumeration(result);
  }

  @Override
  public void setOptions(final String[] options) throws Exception {
    this.setSeed(OptionUtils.parse(options, 'S', this.getDefaultSeed()));
    super.setOptions(options);
  }

  @Override
  public String[] getOptions() {
    List<String> result = new ArrayList<>();
    OptionUtils.add(result, 'S', this.getSeed());
    OptionUtils.add(result, super.getOptions());
    return OptionUtils.toArray(result);
  }

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Classifier Chains. " + "For more information see:\n" + this.getTechnicalInformation().toString();
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;
    TechnicalInformation additional;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank");
    result.setValue(Field.TITLE, "Classifier Chains for Multi-label Classification");
    result.setValue(Field.JOURNAL, "Machine Learning Journal");
    result.setValue(Field.YEAR, "2011");
    result.setValue(Field.VOLUME, "85");
    result.setValue(Field.NUMBER, "3");
    result.setValue(Field.PAGES, "333-359");

    additional = new TechnicalInformation(Type.INPROCEEDINGS);
    additional.setValue(Field.AUTHOR, "Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank");
    additional.setValue(Field.TITLE, "Classifier Chains for Multi-label Classification");
    additional.setValue(Field.BOOKTITLE, "20th European Conference on Machine Learning (ECML 2009). Bled, Slovenia, September 2009");
    additional.setValue(Field.YEAR, "2009");

    result.add(additional);

    return result;
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

    if (this.nodes != null) {
      for (i = 0; i < this.nodes.length; i++) {
        if (this.nodes[i].getClassifier() instanceof Drawable) {
          result.put(i, ((Drawable) this.nodes[i].getClassifier()).graphType());
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

    if (this.nodes != null) {
      for (i = 0; i < this.nodes.length; i++) {
        if (this.nodes[i].getClassifier() instanceof Drawable) {
          result.put(i, ((Drawable) this.nodes[i].getClassifier()).graph());
        }
      }
    }

    return result;
  }

  /**
   * Returns a string representation of the model.
   *
   * @return the model
   */
  @Override
  public String getModel() {
    StringBuilder result;
    int i;

    if (this.nodes == null) {
      return "No model built yet";
    }

    result = new StringBuilder();
    for (i = 0; i < this.nodes.length; i++) {
      if (i > 0) {
        result.append("\n\n");
      }
      result.append(this.getClass().getName() + ": Node #" + (i + 1) + "\n\n");
      result.append(this.nodes[i].getClassifier().toString());
    }

    return result.toString();
  }

  @Override
  public String toString() {
    return Arrays.toString(this.retrieveChain());
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new CC(), args);
  }
}
