package meka.classifiers.multilabel;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import meka.classifiers.multilabel.cc.CNode;
import meka.classifiers.multilabel.cc.Trellis;
import meka.core.OptionUtils;
import meka.core.StatUtils;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * CT - Classifier Trellis. CC in a trellis structure (rather than a cascaded chain). You set the
 * width and type/connectivity/density of the trellis, and optionally change the dependency
 * heuristic which guides the placement of nodes (labels) within the trellis.
 *
 * @author Jesse Read
 * @version September 2015
 */
public class CT extends MCC implements TechnicalInformationHandler {

  private static final long serialVersionUID = -5773951599734753129L;

  protected int m_Width = -1;
  protected int m_Density = 1;
  protected String m_DependencyMetric = "Ibf";

  Trellis trel = null;

  private String info = "";

  @Override
  public String toString() {
    return this.info;
  }

  @Override
  public String globalInfo() {
    return "CC in a trellis structure (rather than a cascaded chain). You set the width and type/connectivity of the trellis, and optionally change the payoff function which guides the placement of nodes (labels) within the trellis.";
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {

    int L = D.classIndex();
    int d = D.numAttributes() - L;
    this.m_R = new Random(this.getSeed());
    int width = this.m_Width;

    if (this.m_Width < 0) {
      // If no width specified for the trellis, use sqrt(L)
      width = (int) Math.sqrt(L);
      if (this.getDebug()) {
        System.out.println("Setting width to " + width);
      }
    } else if (this.m_Width == 0) {
      // 0-width is not possible, use it to indicate a width of L
      width = L;
      if (this.getDebug()) {
        System.out.println("Setting width to " + width);
      }
    }

    /*
     * Make the Trellis. Start with a random structure (unless -S 0 specified, see CC.java).
     */
    if (this.getDebug()) {
      System.out.println("Make Trellis");
    }

    this.prepareChain(L);
    int indices[] = this.retrieveChain();

    this.trel = new Trellis(indices, width, this.m_Density);

    long start = System.currentTimeMillis();

    /*
     * If specified, try and reorder the nodes in the trellis (i.e., get a superior structure)
     */
    if (this.m_Is > 0) {
      double I[][] = StatUtils.margDepMatrix(D, this.m_DependencyMetric);

      /*
       * Get dependency Matrix
       */
      if (this.getDebug()) {
        System.out.println("Got " + this.m_DependencyMetric + "-type Matrix in " + ((System.currentTimeMillis() - start) / 1000.0) + "s");
      }

      // ORDER THE TRELLIS ACCORDING TO THE DEPENDENCY MATRIX
      this.trel = orderTrellis(this.trel, I, this.m_R);
    }

    this.info = String.valueOf((System.currentTimeMillis() - start) / 1000.0);

    if (this.getDebug()) {
      System.out.println("\nTrellis built in: " + this.info + "s");
    }

    /*
     * Build Trellis
     */
    if (this.getDebug()) {
      System.out.println("Build Trellis");
    }

    this.nodes = new CNode[L];
    for (int jv : this.trel.indices) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      if (this.getDebug()) {
        System.out.print(" -> " + jv);
        // System.out.println("Build Node h_"+jv+"] : P(y_"+jv+" | x_[1:d],
        // y_"+Arrays.toString(trel.trellis[jv])+")");
      }
      this.nodes[jv] = new CNode(jv, null, this.trel.trellis[jv]);
      this.nodes[jv].build(D, this.m_Classifier);
    }
    if (this.getDebug()) {
      System.out.println();
    }

    // So we can use the MCC.java and CC.java framework
    this.confidences = new double[L];
    this.m_Chain = this.trel.indices;
  }

  /**
   * OrderTrellis - order the trellis according to marginal label dependencies.
   *
   * @param trel
   *          a randomly initialised trellis
   * @param I
   *          a matrix of marginal pairwise dependencies
   * @param rand
   *          a random seed
   * @return the modified trellis TODO: move to Trellis.java ?
   * @throws InterruptedException
   */
  public static Trellis orderTrellis(Trellis trel, final double I[][], final Random rand) throws InterruptedException {

    int L = I.length;
    int Y[] = new int[L];

    /*
     * Make list of indices
     */
    ArrayList<Integer> list = new ArrayList<>();
    for (int i : trel.indices) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      list.add(new Integer(i));
    }

    /*
     * Take first index, and proceed
     */
    Y[0] = list.remove(rand.nextInt(L));
    // if (getDebug())
    // System.out.print(" "+String.format("%4d", Y[0]));
    // @todo: update(I,j_0) to make faster
    for (int j = 1; j < L; j++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }

      // if (getDebug() && j % m_Width == 0)
      // System.out.println();

      double max_w = -1.;
      int j_ = -1;
      for (int j_prop : list) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread has been interrupted.");
        }
        double w = trel.weight(Y, j, j_prop, I);
        if (w >= max_w) {
          max_w = w;
          j_ = j_prop;
        }
      }
      list.remove(new Integer(j_));

      // if (getDebug()) {
      // System.out.print(" "+String.format("%4d", j_));
      // }

      Y[j] = j_;
      // @todo: update(I,j_), because it will be a parent now
    }
    // if (getDebug())
    // System.out.println();

    trel = new Trellis(Y, trel.WIDTH, trel.TYPE);
    return trel;
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
    return "Determines the neighbourhood density (the number of neighbours for each node in the trellis). Default = 1, BR = 0.";
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
   * GetDependency - Get the type of depependency to use in rearranging the trellis
   */
  public String getDependencyMetric() {
    return this.m_DependencyMetric;
  }

  /**
   * SetDependency - Sets the type of depependency to use in rearranging the trellis
   */
  public void setDependencyMetric(final String m) {
    this.m_DependencyMetric = m;
  }

  public String dependencyMetricTipText() {
    return "The dependency heuristic to use in rearranging the trellis (applicable if chain iterations > 0), default: Ibf (Mutual Information, fast binary version for multi-label data)";
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Jesse Read, Luca Martino, David Luengo, Pablo Olmos");
    result.setValue(Field.TITLE, "Scalable multi-output label prediction: From classifier chains to classifier trellises");
    result.setValue(Field.JOURNAL, "Pattern Recognition");
    result.setValue(Field.URL, "http://www.sciencedirect.com/science/article/pii/S0031320315000084");
    result.setValue(Field.YEAR, "2015");

    return result;
  }

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
    this.setDependencyMetric(OptionUtils.parse(options, 'X', "Ibf"));
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

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new CT(), args);
  }
}
