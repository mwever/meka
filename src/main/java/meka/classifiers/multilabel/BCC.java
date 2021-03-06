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
import java.util.List;
import java.util.Random;
import java.util.Vector;

import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import meka.core.MatrixUtils;
import meka.core.OptionUtils;
import meka.core.StatUtils;
import mst.Edge;
import mst.EdgeWeightedGraph;
import mst.KruskalMST;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

/**
 * BCC.java - Bayesian Classifier Chains. Probably would be more aptly called Bayesian Classifier
 * Tree. Creates a maximum spanning tree based on marginal label dependence; then employs a CC
 * classifier. The original paper used Naive Bayes as a base classifier, hence the name. <br>
 * See Zaragoza et al. "Bayesian Classifier Chains for Multi-dimensional Classification. IJCAI 2011.
 * <br>
 *
 * @author Jesse Read
 * @version June 2013
 */
public class BCC extends CC {

  private static final long serialVersionUID = 585507197229071545L;

  /**
   * Description to display in the GUI.
   *
   * @return the description
   */
  @Override
  public String globalInfo() {
    return "Bayesian Classifier Chains (BCC).\n" + "Creates a maximum spanning tree based on marginal label dependence. Then employs CC.\n" + "For more information see:\n"
        + this.getTechnicalInformation().toString();
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.INPROCEEDINGS);
    result.setValue(Field.AUTHOR, "Julio H. Zaragoza et al.");
    result.setValue(Field.TITLE, "Bayesian Chain Classifiers for Multidimensional Classification");
    result.setValue(Field.BOOKTITLE, "IJCAI'11: International Joint Conference on Artificial Intelligence.");
    result.setValue(Field.YEAR, "2011");

    return result;
  }

  @Override
  public void buildClassifier(final Instances D) throws Exception {
    this.testCapabilities(D);

    this.m_R = new Random(this.getSeed());
    int L = D.classIndex();
    int d = D.numAttributes() - L;

    /*
     * Measure [un]conditional label dependencies (frequencies).
     */
    if (this.getDebug()) {
      System.out.println("Get unconditional dependencies ...");
    }
    double CD[][] = null;
    if (this.m_DependencyType.equals("L")) {
      // New Option
      if (this.getDebug()) {
        System.out.println("The 'LEAD' method for finding conditional dependence.");
      }
      CD = StatUtils.LEAD(D, this.getClassifier(), this.m_R);
    } else {
      // Old/default Option
      if (this.getDebug()) {
        System.out.println("The Frequency method for finding marginal dependence.");
      }
      CD = StatUtils.margDepMatrix(D, this.m_DependencyType);
    }

    if (this.getDebug()) {
      System.out.println(MatrixUtils.toString(CD));
    }

    /*
     * Make a fully connected graph, each edge represents the dependence measured between the pair of
     * labels.
     */
    CD = MatrixUtils.multiply(CD, -1); // because we want a *maximum* spanning tree
    if (this.getDebug()) {
      System.out.println("Make a graph ...");
    }
    EdgeWeightedGraph G = new EdgeWeightedGraph(L);
    for (int i = 0; i < L; i++) {
      for (int j = i + 1; j < L; j++) {
        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("Thread has been interrupted.");
        }
        Edge e = new Edge(i, j, CD[i][j]);
        G.addEdge(e);
      }
    }

    /*
     * Run an off-the-shelf MST algorithm to get a MST.
     */
    if (this.getDebug()) {
      System.out.println("Get an MST ...");
    }
    KruskalMST mst = new KruskalMST(G);

    /*
     * Define graph connections based on the MST.
     */
    int paM[][] = new int[L][L];
    for (Edge e : mst.edges()) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      int j = e.either();
      int k = e.other(j);
      paM[j][k] = 1;
      paM[k][j] = 1;
      // StdOut.println(e);
    }
    if (this.getDebug()) {
      System.out.println(MatrixUtils.toString(paM));
    }

    /*
     * Turn the DAG into a Tree with the m_Seed-th node as root
     */
    int root = this.getSeed();
    if (this.getDebug()) {
      System.out.println("Make a Tree from Root " + root);
    }
    int paL[][] = new int[L][0];
    int visted[] = new int[L];
    Arrays.fill(visted, -1);
    visted[root] = 0;
    this.treeify(root, paM, paL, visted);
    if (this.getDebug()) {
      for (int i = 0; i < L; i++) {
        System.out.println("pa_" + i + " = " + Arrays.toString(paL[i]));
      }
    }
    this.m_Chain = Utils.sort(visted);
    if (this.getDebug()) {
      System.out.println("sequence: " + Arrays.toString(this.m_Chain));
    }
    /*
     * Bulid a classifier 'tree' based on the Tree
     */
    if (this.getDebug()) {
      System.out.println("Build Classifier Tree ...");
    }
    this.nodes = new CNode[L];
    for (int j : this.m_Chain) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      if (this.getDebug()) {
        System.out.println("\t node h_" + j + " : P(y_" + j + " | x_[1:" + d + "], y_" + Arrays.toString(paL[j]) + ")");
      }
      this.nodes[j] = new CNode(j, null, paL[j]);
      this.nodes[j].build(D, this.m_Classifier);
    }

    if (this.getDebug()) {
      System.out.println(" * DONE * ");
    }

    /*
     * Notes ... paL[j] = new int[]{}; // <-- BR !! paL[j] = MLUtils.gen_indices(j); // <-- CC !!
     */
  }

  /**
   * Treeify - make a tree given the structure defined in paM[][], using the root-th node as root.
   * 
   * @throws InterruptedException
   */
  private void treeify(final int root, final int paM[][], final int paL[][], final int visited[]) throws InterruptedException {
    int children[] = new int[] {};
    for (int j = 0; j < paM[root].length; j++) {
      if (Thread.currentThread().isInterrupted()) {
        throw new InterruptedException("Thread has been interrupted.");
      }
      if (paM[root][j] == 1) {
        if (visited[j] < 0) {
          children = A.append(children, j);
          paL[j] = A.append(paL[j], root);
          visited[j] = visited[Utils.maxIndex(visited)] + 1;
        }
        // set as visited
        // paM[root][j] = 0;
      }
    }
    // go through again
    for (int child : children) {
      this.treeify(child, paM, paL, visited);
    }
  }

  /*
   * TODO: Make a generic abstract -dependency_user- class that has this option, and extend it here
   */

  String m_DependencyType = "Ibf";

  public void setDependencyType(final String value) {
    this.m_DependencyType = value;
  }

  public String getDependencyType() {
    return this.m_DependencyType;
  }

  public String dependencyTypeTipText() {
    return "XXX";
  }

  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    result.addElement(new Option("\tThe way to measure dependencies.\n\tdefault: " + this.m_DependencyType + " (frequencies only)", "X", 1, "-X <value>"));
    OptionUtils.add(result, super.listOptions());
    return OptionUtils.toEnumeration(result);
  }

  @Override
  public void setOptions(final String[] options) throws Exception {
    this.setDependencyType(OptionUtils.parse(options, 'X', "Ibf"));
    super.setOptions(options);
  }

  @Override
  public String[] getOptions() {
    List<String> result = new ArrayList<>();
    OptionUtils.add(result, 'X', this.getDependencyType());
    OptionUtils.add(result, super.getOptions());
    return OptionUtils.toArray(result);
  }

  public static void main(final String args[]) {
    ProblemTransformationMethod.evaluation(new BCC(), args);
  }
}
