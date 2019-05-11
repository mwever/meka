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

import meka.core.A;
import meka.core.CCUtils;
import meka.core.OptionUtils;
import meka.core.Result;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * MCC.java - CC with Monte Carlo optimisation.
 *
 * Note inference is now a bit slower than reported in the paper, <br>
 * Jesse Read, Luca Martino, David Luengo. <i>Efficient Monte Carlo Optimization for
 * Multi-dimensional Classifier Chains</i>. http://arxiv.org/abs/1211.2190. 2012 <br>
 * There we used a faster implementation, full of ugly hacks, but it got broken when I updated
 * CC.java.<br>
 * This version extends CC, and thus is a bit cleaner, but for some reason inference is quite slower
 * than expected with high m_Iy.
 *
 * TODO Option for hold-out set, instead of training and testing on training data (internally).
 *
 * @see meka.classifiers.multilabel.CC
 * @author Jesse Read
 * @version March 2015
 */
public class MCC extends CC implements TechnicalInformationHandler, MultiTargetCapable {

	private static final long serialVersionUID = 5085402586815030939L;
	protected int m_Is = 0;
	protected int m_Iy = 10;
	protected String m_Payoff = "Exact match";

	/**
	 * Payoff - Return a default score of h evaluated on D.
	 *
	 * @param h
	 *          a classifier
	 * @param D
	 *          a dataset
	 */
	public double payoff(final CC h, final Instances D) throws Exception {
		Result r = Evaluation.testClassifier(h, D);
		// assume multi-label for now
		r.setInfo("Type", "ML");
		r.setInfo("Threshold", "0.5");
		r.setInfo("Verbosity", "7");
		r.output = Result.getStats(r, "7");
		return (Double) r.getMeasurement(this.m_Payoff);
	}

	@Override
	public void buildClassifier(final Instances D) throws Exception {
		this.testCapabilities(D);

		// Variables

		int L = D.classIndex();
		int N = D.numInstances();
		int d = D.numAttributes() - L;
		this.m_R = new Random(this.m_S);

		this.prepareChain(L);
		int s[] = this.retrieveChain();

		if (this.getDebug()) {
			System.out.println("s_[0] = " + Arrays.toString(s));
		}

		// If we want to optimize the chain space ...
		if (this.m_Is > 0) {

			// Make CC
			CC h = CCUtils.buildCC(s, D, this.m_Classifier);
			if (Thread.currentThread().isInterrupted()) {
				throw new InterruptedException("Thread has been interrupted.");
			}

			if (this.getDebug()) {
				System.out.println("Optimising s ... (" + this.m_Is + " iterations):");
			}

			double w = this.payoff(h, new Instances(D));
			if (this.getDebug()) {
				System.out.println("h_{t=" + 0 + "} := " + Arrays.toString(s)); // +"; w = "+w);
			}

			for (int t = 0; t < this.m_Is; t++) {
				if (Thread.currentThread().isInterrupted()) {
					throw new InterruptedException("Thread has been interrupted.");
				}

				// propose a chain s' by swapping two elements in s
				int s_[] = Arrays.copyOf(A.swap(s, this.m_R), s.length);

				// build h'
				CC h_ = CCUtils.buildCC(s_, D, this.m_Classifier);

				// rate h'
				double w_ = this.payoff(h_, new Instances(D));

				// accept h' over h ?
				if (w_ > w) {
					w = w_;
					s = s_;
					h = h_;
					if (this.getDebug()) {
						System.out.println("h_{t=" + (t + 1) + "} := " + Arrays.toString(s)); // +"; w = "+w);
						// if (getDebug()) System.out.print("& "+Utils.doubleToString(likelihood(h_,new
						// Instances(D),1),8,2));
						// if (getDebug()) System.out.print("& "+Utils.doubleToString(likelihood(h_,new
						// Instances(D),2),8,2));
						// if (getDebug()) System.out.println("& "+Utils.doubleToString(likelihood(h_,new
						// Instances(D),5),8,2));
					}
				}
			}
		}
		if (this.getDebug()) {
			System.out.println("---");
		}

		this.prepareChain(s);
		super.buildClassifier(D);
	}

	@Override
	public double[] distributionForInstance(final Instance x) throws Exception {

		// T = 0
		double y[] = super.distributionForInstance(x);

		// T > 0
		if (this.m_Iy > 0) {
			// double yT[] = CCUtils.RandomSearchaa(this,x,m_Iy,m_R,y0);

			double w = A.product(this.probabilityForInstance(x, y)); // p(y|x)

			Instance t_[] = this.getTransformTemplates(x);

			// System.out.println("----");
			// System.out.println("p0("+Arrays.toString(y)+") = "+Arrays.toString(h.getConfidences())+", w="+w);
			for (int t = 0; t < this.m_Iy; t++) {
				if (Thread.currentThread().isInterrupted()) {
					throw new InterruptedException("Thread has been interrupted.");
				}
				double y_[] = this.sampleForInstanceFast(t_, this.m_R); // propose y' by sampling i.i.d.
				// double y_[] = this.sampleForInstance(x,m_R); // propose y' by sampling i.i.d.
				// double p_[] = h.getConfidences(); //
				double w_ = A.product(this.getConfidences()); // rate y' as w' --- TODO allow for command-line option
				// System.out.println("p("+Arrays.toString(y_)+") = "+Arrays.toString(p_)+", w="+w_);
				if (w_ > w) { // accept ?
					if (this.getDebug()) {
						System.out.println("y' = " + Arrays.toString(y_) + ", :" + w_);
					}
					w = w_;
					// y = y_;
					y = Arrays.copyOf(y_, y_.length);
					// System.out.println("* ACCEPT *");
				}
			}
		}

		return y;
	}

	@Override
	public Enumeration listOptions() {
		Vector result = new Vector();
		result.addElement(new Option("\t" + this.chainIterationsTipText() + "\n\tdefault: 0", "Is", 1, "-Is <value>"));
		result.addElement(new Option("\t" + this.inferenceIterationsTipText() + "\n\tdefault: 10", "Iy", 1, "-Iy <value>"));
		result.addElement(new Option("\t" + this.payoffTipText() + "\n\tdefault: Exact match", "P", 1, "-P <value>"));
		OptionUtils.add(result, super.listOptions());
		return OptionUtils.toEnumeration(result);
	}

	@Override
	public void setOptions(final String[] options) throws Exception {
		this.setChainIterations(OptionUtils.parse(options, "Is", 0));
		this.setInferenceIterations(OptionUtils.parse(options, "Iy", 10));
		this.setPayoff(OptionUtils.parse(options, 'P', "Exact match"));
		super.setOptions(options);
	}

	@Override
	public String[] getOptions() {
		List<String> result = new ArrayList<>();
		OptionUtils.add(result, "Is", this.getChainIterations());
		OptionUtils.add(result, "Iy", this.getInferenceIterations());
		OptionUtils.add(result, 'P', this.getPayoff());
		OptionUtils.add(result, super.getOptions());
		return OptionUtils.toArray(result);
	}

	/** Set the inference iterations */
	public void setInferenceIterations(final int iy) {
		this.m_Iy = iy;
	}

	/** Get the inference iterations */
	public int getInferenceIterations() {
		return this.m_Iy;
	}

	public String inferenceIterationsTipText() {
		return "The number of iterations to search the output space at test time.";
	}

	/** Set the iterations of s (chain order) */
	public void setChainIterations(final int is) {
		this.m_Is = is;
	}

	/** Get the iterations of s (chain order) */
	public int getChainIterations() {
		return this.m_Is;
	}

	public String chainIterationsTipText() {
		return "The number of iterations to search the chain space at train time.";
	}

	/** Set the payoff function */
	public void setPayoff(final String p) {
		this.m_Payoff = p;
	}

	/** Get the payoff function */
	public String getPayoff() {
		return this.m_Payoff;
	}

	public String payoffTipText() {
		return "Sets the payoff function. Any of those listed in regular evaluation output will do (e.g., 'Exact match').";
	}

	@Override
	public String globalInfo() {
		return "Classifier Chains with Monte Carlo optimization. " + "For more information see:\n" + this.getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;
		TechnicalInformation additional;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "Jesse Read and Luca Martino and David Luengo");
		result.setValue(Field.TITLE, "Efficient Monte Carlo Optimization for Multi-label Classifier Chains");
		result.setValue(Field.BOOKTITLE, "ICASSP'13: International Conference on Acoustics, Speech, and Signal Processing");
		result.setValue(Field.YEAR, "2013");

		additional = new TechnicalInformation(Type.ARTICLE);
		additional.setValue(Field.AUTHOR, "Jesse Read and Luca Martino and David Luengo");
		additional.setValue(Field.TITLE, "Efficient Monte Carlo Optimization for Multi-dimensional Classifier Chains");
		additional.setValue(Field.JOURNAL, "Elsevier Pattern Recognition");
		additional.setValue(Field.YEAR, "2013");

		result.add(additional);
		return result;
	}

	public static void main(final String args[]) {
		ProblemTransformationMethod.evaluation(new MCC(), args);
	}

}
