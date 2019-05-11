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

package meka.core;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Metrics.java - Evaluation Metrics.
 * <p>L_ are loss/error measures (less is better)</p>
 * <p>P_ are payoff/accuracy measures (higher is better).</p>
 * For more on the evaluation and threshold selection implemented here, see
 * <br> Jesse Read, <i>Scalable Multi-label Classification</i>. PhD Thesis, University of Waikato, Hamilton, New Zealand (2010).
 * @author 		Jesse Read (jesse@tsc.uc3m.es)
 * @version 	Feb 2013
 */
public abstract class Metrics {

	/**
	 * Helper function for missing values in the labels. Transforms a double array to an
	 * int array.
	 *
	 * @return the new array
	 */
	public static int[] toIntArray(final double[] doubles) {
		int[] res = new int[doubles.length];
		for (int i = 0; i < doubles.length; i++) {
			res[i] = (int) doubles[i];
		}
		return res;
	}

	/**
	 * Helper function for missing values in the labels. Simply checks if all
	 * real labels are missing.
	 *
	 * @return If all labels are missing
	 */
	public static boolean allMissing(final int[] real) {
		for (int i = 0; i < real.length; i++) {
			if (real[i] != -1) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Helper function for missing values in the labels. Simply returns number of
	 * real missing labels.
	 *
	 * @return Number of missing labels.
	 */
	public static int numberOfMissingLabels(final int[] real) {
		int missing = 0;
		for (int i = 0; i < real.length; i++) {
			if (real[i] == -1) {
				missing++;
			}
		}
		return missing;
	}

	/**
	 * Helper function for missing values in the labels and missing predictions
	 * (i.e., from abstaining classifiers). Aligns the predictions
	 * with the real labels, discarding labels and predictions that are missing.
	 *
	 * @param real The real values from the data
	 * @param pred The predicted values from the classifiers
	 * @return Aligned predicted and real labels.
	 */
	public static int[][] align(final int[] real, final int[] pred) {

		int missing = numberOfMissingLabels(real);

		int[] _real = new int[real.length - missing];
		int[] _pred = new int[real.length - missing];

		int offset = 0;
		for (int i = 0; i < real.length; i++) {
			if (real[i] == -1 || pred[i] == -1) {
				offset++;
				continue;
			}
			_real[i - offset] = real[i];
			_pred[i - offset] = pred[i];
		}

		int[][] res = new int[2][0];
		res[0] = _real;
		res[1] = _pred;

		return res;
	}

	/**
	 * Helper function for missing values in the labels and missing predictions
	 * (i.e., from abstaining classifiers). Aligns the predictions
	 * with the real labels, discarding labels and predictions that are missing.
	 *
	 * @param real The real values from the data
	 * @param pred The predicted values from the classifiers
	 * @return Aligned predicted and real labels.
	 */
	public static double[][] align(final int[] real, final double[] pred) {

		int missing = numberOfMissingLabels(real);

		double[] _real = new double[real.length - missing];
		double[] _pred = new double[real.length - missing];

		int offset = 0;
		for (int i = 0; i < real.length; i++) {
			if (real[i] == -1 || pred[i] == -1.0 || Double.isNaN(pred[i])) {
				offset++;
				continue;
			}
			_real[i - offset] = real[i];
			_pred[i - offset] = pred[i];
		}

		double[][] res = new double[2][0];
		res[0] = _real;
		res[1] = _pred;

		return res;
	}

	/** Exact Match, i.e., 1 - [0/1 Loss]. */
	public static double P_ExactMatch(final int Y[][], final int Ypred[][]) {
		// works with missing
		return 1. - L_ZeroOne(Y, Ypred);
	}

	/** 0/1 Loss. */
	public static double L_ZeroOne(final int y[], final int ypred[]) {
		// works with missing
		int[][] aligned = align(y, ypred);

		int[] yAligned = aligned[0];
		int[] ypredAligned = aligned[1];

		int L = yAligned.length;
		for (int j = 0; j < L; j++) {
			if (yAligned[j] != ypredAligned[j]) {
				return 1.;
			}
		}
		return 0.;
	}

	/** 0/1 Loss. */
	public static double L_ZeroOne(final int Y[][], final int Ypred[][]) {
		// works with missing
		int allMissings = 0;

		int N = Y.length;

		double loss = 0.0;

		for (int i = 0; i < N; i++) {
			if (allMissing(Y[i])) {
				allMissings++;
				continue;
			}

			double curLoss = L_ZeroOne(Y[i], Ypred[i]);

			if (Double.isNaN(curLoss)) {
				allMissings++;
				continue;
			}

			loss += curLoss;
		}
		return loss / (N - allMissings);
	}

	/** Hamming loss. */
	public static double L_Hamming(final int y[], final int ypred[]) {
		// works with missing

		int[][] aligned = align(y, ypred);

		int[] yAligned = aligned[0];
		int[] ypredAligned = aligned[1];

		int L = yAligned.length;

		if (L == 0) {
			return Double.NaN;
		}

		double loss = 0.0;
		for (int j = 0; j < L; j++) {
			if (yAligned[j] != ypredAligned[j]) {
				loss++;
			}
		}
		return loss / L;
	}

	/** Hamming loss. */
	public static double L_Hamming(final int Y[][], final int Ypred[][]) {
		// works with missing

		int N = Y.length;

		int allMissings = 0;

		double loss = 0.0;

		for (int i = 0; i < N; i++) {
			if (allMissing(Y[i])) {
				allMissings++;
				continue;
			}

			double curLoss = L_Hamming(Y[i], Ypred[i]);

			if (Double.isNaN(curLoss)) {
				allMissings++;
				continue;
			}

			loss += curLoss;
		}
		return loss / (N - allMissings);
	}

	/** Hamming score aka label accuracy. */
	public static double P_Hamming(final int Y[][], final int Ypred[][]) {
		// works with missing
		return 1. - L_Hamming(Y, Ypred);
	}

	/** Hamming score aka label accuracy. */
	public static double P_Hamming(final int Y[][], final int Ypred[][], final int j) {
		// works with missing
		int y_j[] = MatrixUtils.getCol(Y, j);
		int ypred_j[] = MatrixUtils.getCol(Ypred, j);

		int[][] aligned = align(y_j, ypred_j);

		int[] y_jAligned = aligned[0];
		int[] ypred_jAligned = aligned[1];

		return 1. - L_Hamming(y_jAligned, ypred_jAligned);
	}

	/** Harmonic Accuracy. Multi-label only. */
	public static double P_Harmonic(final int y[], final int ypred[]) {
		// works with missing
		int[][] aligned = align(y, ypred);

		int[] yAligned = aligned[0];
		int[] ypredAligned = aligned[1];

		int L = yAligned.length;
		double acc[] = new double[2];
		double N[] = new double[2];

		for (int j = 0; j < L; j++) {
			N[yAligned[j]]++;
			if (yAligned[j] == ypredAligned[j]) {
				acc[yAligned[j]]++;
			}
		}
		for (int v = 0; v < 2; v++) {
			acc[v] = acc[v] / N[v];
		}
		return 2. / ((1. / acc[0]) + (1. / acc[1]));
	}

	/** Harmonic Accuracy -- for the j-th label. Multi-label only. */
	public static double P_Harmonic(final int Y[][], final int Ypred[][], final int j) {
		// works with missing
		int y_j[] = MatrixUtils.getCol(Y, j);
		int ypred_j[] = MatrixUtils.getCol(Ypred, j);
		return P_Harmonic(y_j, ypred_j);
	}

	/** Harmonic Accuracy -- average over all labels. Multi-label only. */
	public static double P_Harmonic(final int Y[][], final int Ypred[][]) {
		// works with missing
		int allMissings = 0;

		int N = Y.length;
		double loss = 0.0;
		for (int i = 0; i < N; i++) {
			if (allMissing(Y[i])) {
				allMissings++;
				continue;
			}
			double curLoss = P_Harmonic(Y[i], Ypred[i]);

			if (Double.isNaN(curLoss)) {
				allMissings++;
				continue;
			}

			loss += curLoss;

		}
		return loss / (N - allMissings);
	}

	/** Jaccard Index -- often simply called multi-label 'accuracy'. Multi-label only. */
	public static double P_Accuracy(final int y[], final int ypred[]) {
		// works with missing
		int[][] aligned = align(y, ypred);

		int[] yAligned = aligned[0];
		int[] ypredAligned = aligned[1];

		int L = yAligned.length;
		int set_union = 0;
		int set_inter = 0;
		for (int j = 0; j < L; j++) {
			if (yAligned[j] == 1 || ypredAligned[j] == 1) {
				set_union++;
			}
			if (yAligned[j] == 1 && ypredAligned[j] == 1) {
				set_inter++;
			}
		}
		// = intersection / union; (or, if both sets are empty, then = 1.)
		return (set_union > 0) ? (double) set_inter / (double) set_union : 1.0;
	}

	/** Jaccard Index -- often simply called multi-label 'accuracy'. Multi-label only. */
	public static double P_Accuracy(final int Y[][], final int Ypred[][]) {
		// works with missing
		int allMissings = 0;

		int N = Y.length;
		double accuracy = 0.0;

		for (int i = 0; i < Y.length; i++) {
			if (allMissing(Y[i])) {
				allMissings++;
				continue;
			}
			accuracy += P_Accuracy(Y[i], Ypred[i]);
		}
		return accuracy / (N - allMissings);
	}

	/** Jaccard Index -- often simply called multi-label 'accuracy'. Multi-label only. */
	public static double P_JaccardIndex(final int Y[][], final int Ypred[][]) {
		// works with missing
		return P_Accuracy(Y, Ypred);
	}

	/** Jaccard Distance -- the loss version of Jaccard Index */
	public static double L_JaccardDist(final int Y[][], final int Ypred[][]) {
		// works with missing
		return 1. - P_Accuracy(Y, Ypred);
	}

	/**
	 * L_LogLoss - the log loss between real-valued confidence rpred and true prediction y.
	 * @param	y		label
	 * @param	rpred	prediction (confidence)
	 * @param	C		limit (maximum loss of log(C))
	 * @return 	Log loss
	 */
	public static double L_LogLoss(final double y, final double rpred, final double C) {
		if (y == -1) {
			return 0.0;
		}

		// base 2 ?
		double ans = Math.min(Utils.eq(y, rpred) ? 0.0 : -((y * Math.log(rpred)) + ((1.0 - y) * Math.log(1.0 - rpred))), C);
		return (Double.isNaN(ans) ? 0.0 : ans);
	}

	/**
	 * L_LogLoss - the log loss between real-valued confidences Rpred and true predictions
	 * Y with a maximum penalty based on the number of labels L [Important Note: Earlier
	 * versions of Meka only normalised by N, and not N*L as here].
	 */
	public static double L_LogLossL(final int Y[][], final double Rpred[][]) {

		int N = Y.length;
		int L = Y[0].length;

		int missing = 0;

		for (int i = 0; i < Y.length; i++) {
			if (allMissing(Y[i])) {
				N--;
			}

			for (int j = 0; j < Y[i].length; j++) {
				if (Y[i][j] == -1) {
					missing++;
				}
			}
		}

		if (N == 0) {
			return Double.NaN;
		}

		return L_LogLoss(Y, Rpred, Math.log(L)) / (((double) N * (double) L) - missing);
	}

	/**
	 * L_LogLoss - the log loss between real-valued confidences Rpred and true predictions
	 * Y with a maximum penalty based on the number of examples D [Important Note: Earlier
	 * versions of Meka only normalised by N, and not N*L as here].
	 */
	public static double L_LogLossD(final int Y[][], final double Rpred[][]) {
		int N = Y.length;
		int L = Y[0].length;

		int missing = 0;

		for (int i = 0; i < Y.length; i++) {
			if (allMissing(Y[i])) {
				N--;
			}

			for (int j = 0; j < Y[i].length; j++) {
				if (Y[i][j] == -1) {
					missing++;
				}
			}
		}

		if (N == 0) {
			return Double.NaN;
		}

		return L_LogLoss(Y, Rpred, Math.log(N)) / (((double) N * (double) L) - missing);
	}

	/**
	 * L_LogLoss - the log loss between real-valued confidences Rpred and true predictions Y with a maximum penalty C [Important Note: Earlier versions of Meka only normalised by N, and not N*L as here].
	 */
	public static double L_LogLoss(final int Y[][], final double Rpred[][], final double C) {
		double loss = 0.0;
		for (int i = 0; i < Y.length; i++) {
			if (allMissing(Y[i])) {
				continue;
			}

			for (int j = 0; j < Y[i].length; j++) {
				if (Y[i][j] == -1) {
					continue;
				}

				loss += L_LogLoss(Y[i][j], Rpred[i][j], C);
			}
		}
		return loss;
	}

	/**
	 * P_TruePositives - 1 and supposed to be 1 (the intersection, i.e., logical AND).
	 */
	public static double P_TruePositives(final int y[], final int ypred[]) {
		// works with missing
		// return Utils.sum(A.AND(y,ypred));
		int s = 0;
		for (int j = 0; j < y.length; j++) {
			if (ypred[j] == 1 && y[j] == 1) {
				s++;
			}
		}
		return s;
	}

	/**
	 * P_FalsePositives - 1 but supposed to be 0 (the length of y \ ypred).
	 */
	public static double P_FalsePositives(final int y[], final int ypred[]) {
		// works with missing
		int s = 0;
		for (int j = 0; j < y.length; j++) {
			if (ypred[j] == 1 && y[j] == 0) {
				s++;
			}
		}
		return s;
	}

	/**
	 * P_TrueNegatives - 0 and supposed to be 0.
	 */
	public static double P_TrueNegatives(final int y[], final int ypred[]) {
		// works with missing
		int s = 0;
		for (int j = 0; j < y.length; j++) {
			if (ypred[j] == 0 && y[j] == 0) {
				s++;
			}
		}
		return s;
	}

	/**
	 * P_FalseNegatives - 0 but supposed to be 1 (the length of ypred \ y).
	 */
	public static double P_FalseNegatives(final int y[], final int ypred[]) {
		// works with missing
		int s = 0;
		for (int j = 0; j < y.length; j++) {
			if (ypred[j] == 0 && y[j] == 1) {
				s++;
			}
		}
		return s;
	}

	/**
	 * P_Precision - (retrieved AND relevant) / retrieved
	 */
	public static double P_Precision(final int y[], final int ypred[]) {
		// works with missing
		if (allMissing(y)) {
			return Double.NaN;
		}
		double tp = P_TruePositives(y, ypred);
		double fp = P_FalsePositives(y, ypred);
		if (tp == 0.0 && fp == 0.0) {
			return 0.0;
		}
		return tp / (tp + fp);
	}

	/**
	 * P_Recall - (retrieved AND relevant) / relevant
	 */
	public static double P_Recall(final int y[], final int ypred[]) {
		// works with missing
		if (allMissing(y)) {
			return Double.NaN;
		}
		double tp = P_TruePositives(y, ypred);
		double fn = P_FalseNegatives(y, ypred);
		if (tp == 0.0 && fn == 0.0) {
			return 0.0;
		}
		return tp / (tp + fn);
	}

	/**
	 * F1 - the F1 measure for two sets.
	 */
	public static double F1(final int s1[], final int s2[]) {
		// works with missing
		double p = P_Precision(s1, s2);
		double r = P_Recall(s1, s2);
		if (Double.isNaN(r) || Double.isNaN(p)) {
			return Double.NaN;
		}
		if (p == 0.0 && r == 0.0) {
			return 0.0;
		}
		return 2. * p * r / (p + r);
	}

	/*
	  public static double P_Recall(int Y[][], int YPred[][]) {
	  return P_Recall(flatten(Y),flatten(YPRed));
	  }
	*/

	/**
	 * P_Precision - (retrieved AND relevant) / retrieved
	 */
	public static double P_PrecisionMacro(final int Y[][], final int Ypred[][]) {
		// works with missing
		int L = Y[0].length;
		double m = 0.0;
		int missing = 0;
		for (int j = 0; j < L; j++) {
			int[] y_j = MatrixUtils.getCol(Y, j);
			int[] p_j = MatrixUtils.getCol(Ypred, j);

			if (allMissing(y_j)) {
				missing++;
				continue;
			}

			int[][] aligned = align(y_j, p_j);

			int[] y_jAligned = aligned[0];
			int[] p_jAligned = aligned[1];

			double curPrec = P_Precision(y_jAligned, p_jAligned);

			if (Double.isNaN(curPrec)) {
				continue;
			}

			m += curPrec;

		}

		L -= missing;

		if (L == 0) {
			return Double.NaN;
		}

		return m / L;
	}

	/**
	 * P_Recall - (retrieved AND relevant) / relevant
	 */
	public static double P_RecallMacro(final int Y[][], final int Ypred[][]) {
		// works with missing
		int L = Y[0].length;
		double m = 0.0;

		int missing = 0;

		for (int j = 0; j < L; j++) {
			int[] y_j = MatrixUtils.getCol(Y, j);
			int[] p_j = MatrixUtils.getCol(Ypred, j);

			if (allMissing(y_j)) {
				missing++;
				continue;
			}

			int[][] aligned = align(y_j, p_j);

			int[] y_jAligned = aligned[0];
			int[] p_jAligned = aligned[1];

			double curRecall = P_Recall(y_jAligned, p_jAligned);

			if (Double.isNaN(curRecall)) {
				continue;
			}

			m += curRecall;
		}

		L -= missing;

		if (L == 0) {
			return Double.NaN;
		}

		return m / L;
	}

	/**
	 * P_Precision - (retrieved AND relevant) / retrieved
	 */
	public static double P_PrecisionMicro(final int Y[][], final int Ypred[][]) {
		// works with missing
		return P_Precision(MatrixUtils.flatten(Y), MatrixUtils.flatten(Ypred));
	}

	/**
	 * P_Recall - (retrieved AND relevant) / relevant
	 */
	public static double P_RecallMicro(final int Y[][], final int Ypred[][]) {
		// works with missing
		return P_Recall(MatrixUtils.flatten(Y), MatrixUtils.flatten(Ypred));
	}

	/**
	 * P_Precision - (retrieved AND relevant) / retrieved
	 */
	public static double P_Precision(final int Y[][], final int Ypred[][], final int j) {
		// works with missing
		return P_Precision(MatrixUtils.getCol(Y, j), MatrixUtils.getCol(Ypred, j));
		// int retrieved = M.sum(M.sum(Ypred));
		// int correct = M.sum(M.sum(M.multiply(Y,Ypred)));
		// return (double)correct / (double)predicted;
	}

	/**
	 * P_Recall - (retrieved AND relevant) / relevant
	 */
	public static double P_Recall(final int Y[][], final int Ypred[][], final int j) {
		// works with missing
		return P_Recall(MatrixUtils.getCol(Y, j), MatrixUtils.getCol(Ypred, j));
		// int relevant = M.sum(M.sum(Y));
		// int correct = M.sum(M.sum(M.multiply(Y,Ypred)));
		// return (double)correct / (double)relevant;
	}

	/**
	 * P_FmicroAvg - Micro Averaged F-measure (F1, as if all labels in the dataset formed a single vector)
	 */
	public static double P_FmicroAvg(final int Y[][], final int Ypred[][]) {
		// works with missing
		return F1(MatrixUtils.flatten(Y), MatrixUtils.flatten(Ypred));
		// double precision = P_Precision(M.flatten(Y),M.flatten(Ypred));
		// double recall = P_Recall(M.flatten(Y),M.flatten(Ypred));
		// return (2.0 * precision * recall) / (precision + recall);
	}

	/**
	 * F-Measure Macro Averaged by L - The 'standard' macro average.
	 */
	public static double P_FmacroAvgL(final int Y[][], final int Ypred[][]) {
		// works with missing
		int L = Y[0].length;
		double TP[] = new double[L];
		double FP[] = new double[L];
		double FN[] = new double[L];
		double F[] = new double[L];

		int missing = 0;

		for (int j = 0; j < L; j++) {
			if (allMissing(MatrixUtils.getCol(Y, j))) {
				missing++;
				continue;
			}

			int y_j[] = MatrixUtils.getCol(Y, j);
			int ypred_j[] = MatrixUtils.getCol(Ypred, j);

			TP[j] = P_TruePositives(y_j, ypred_j);
			FP[j] = P_FalsePositives(y_j, ypred_j);
			FN[j] = P_FalseNegatives(y_j, ypred_j);
			if (TP[j] <= 0) {
				F[j] = 0.0;
			} else {
				double prec = TP[j] / (TP[j] + FP[j]);
				double recall = TP[j] / (TP[j] + FN[j]);
				F[j] = 2 * ((prec * recall) / (prec + recall));
			}
		}

		L -= missing;

		if (L == 0) {
			return Double.NaN;
		}

		return A.sum(F) / L;

	}

	/**
	 * F-Measure  Averaged by D - The F-measure macro averaged by example.
	 * The Jaccard index is also averaged this way.
	 */
	public static double P_FmacroAvgD(final int Y[][], final int Ypred[][]) {
		// works with missing
		int N = Y.length;

		double F1_macro_D = 0.0;
		int missing = 0;

		for (int i = 0; i < N; i++) {
			if (allMissing(Y[i])) {
				missing++;
				continue;
			}
			F1_macro_D += F1(Y[i], Ypred[i]);
		}

		N -= missing;

		if (N == 0) {
			return Double.NaN;
		}

		return F1_macro_D / N;
	}

	/**
	 * OneError -
	 */
	public static double L_OneError(final int Y[][], final double Rpred[][]) {
		// works with missing
		int N = Y.length;
		int one_error = 0;

		int missing = 0;

		for (int i = 0; i < N; i++) {
			if (allMissing(Y[i])) {
				missing++;
				continue;
			}
			if (Y[i][Utils.maxIndex(Rpred[i])] == 0) {
				one_error++;
			}
		}

		N -= missing;

		if (N == 0) {
			return Double.NaN;
		}

		return (double) one_error / (double) N;
	}

	public static double P_AveragePrecision(final int Y[][], final double Rpred[][]) {
		// works with missing
		int N = Y.length;

		double loss = 0.0;
		for (int i = 0; i < Y.length; i++) {
			if (allMissing(Y[i])) {
				N--;
				continue;
			}

			loss += P_AveragePrecision(Y[i], Rpred[i]);
		}
		return loss / N;
	}

	/**
	 * Converts confidences in prediction array to ranking array, and continues
	 * with
	 * {@link #P_AveragePrecision(int[], int[])
	 * P_AveragePrecision(using ranking array)}.
	 *
	 *
	 *
	 * @param y The real label values of an instance.
	 * @param rpred The predicted confidences for the labels.
	 * @return the calculated average precision for an instance.
	 */
	public static double P_AveragePrecision(final int y[], final double rpred[]) {

		double[][] aligned = align(y, rpred);

		double[] alignedy = aligned[0];
		double[] alignedp = aligned[1];

		int r[] = MLUtils.predictionsToRanking(alignedp);

		return P_AveragePrecision(toIntArray(alignedy), r);

	}

	/**
	 * Average Precision - computes for each relevant label the percentage
	 * of relevant labels among all labels that are ranked before it.
	 * @param	y	0/1 labels         [0,   0,   1   ] (true labels)
	 * @param	r	ranking position   [1,   2,   0   ]
	 * @return	Average Precision
	 */
	public static double P_AveragePrecision(final int y[], final int r[]) {
		// works with missing
		double avg_prec = 0;

		int L = y.length;

		List<Integer> ones = new ArrayList<Integer>();
		for (int j = 0; j < L; j++) {
			if (y[j] == 1) {
				ones.add(j);
			}
		}

		if (ones.size() <= 0) {
			return 1.0;
		}

		for (int j : ones) {

			// 's' = the percentage of relevant labels ranked before 'j'
			double s = 0.0;
			for (int k : ones) {
				if (r[k] <= r[j]) {
					s++;
				}
			}
			// 's' divided by the position of 'j'
			avg_prec += (s / (1 + r[j]));

		}
		avg_prec /= ones.size();
		return avg_prec;
	}
	//////////////////////////////////////////////////////////////////////////

	public static double L_RankLoss(final int Y[][], final double Rpred[][]) {
		// works with missing
		int N = Y.length;
		double loss = 0.0;
		for (int i = 0; i < Y.length; i++) {
			if (allMissing(Y[i])) {
				N--;
				continue;
			}

			loss += L_RankLoss(Y[i], Rpred[i]);
		}
		return loss / N;
	}

	public static double L_RankLoss(int y[], double rpred[]) {
		// works with missing

		double[][] aligned = align(y, rpred);

		y = toIntArray(aligned[0]);
		rpred = aligned[1];

		int r[] = Utils.sort(rpred);
		return L_RankLoss(y, r);
	}

	/**
	 * Rank Loss - the average fraction of labels which are not correctly ordered.
	 * Thanks to Noureddine Yacine NAIR BENREKIA for providing bug fix for this.
	 * @param	y	0/1 labels         [0,   0,   1   ]
	 * @param	r	ranking position   [1,   2,   0   ]
	 * @return	Ranking Loss
	 */
	public static double L_RankLoss(final int y[], final int r[]) {

		int L = y.length;
		ArrayList<Integer> tI = new ArrayList<Integer>();
		ArrayList<Integer> fI = new ArrayList<Integer>();
		for (int j = 0; j < L; j++) {
			if (y[j] == 1) {
				tI.add(j);
			} else {
				fI.add(j);
			}
		}

		if (!tI.isEmpty() && !fI.isEmpty()) {
			int c = 0;
			for (int k : tI) {
				for (int l : fI) {
					if (position(k, r) < position(l, r)) {
						c++;
					}
				}
			}
			return (double) c / (double) (tI.size() * fI.size());
		} else {
			return 0.0;
		}
	}

	private static int position(final int index, final int r[]) {
		int i = 0;
		while (r[i] != index) {
			i++;
		}
		return i;
	}

	/**
	 * Helper function, returns macro AUROC (roc = true) or macro RPC (roc = false)
	 */
	private static double getMacro(final int Y[][], final double P[][], final boolean roc) {

		// works with missing
		int L = Y[0].length;
		double AUC[] = new double[L];

		int missing = 0;

		for (int j = 0; j < L; j++) {
			if (allMissing(MatrixUtils.getCol(Y, j))) {
				missing++;
				continue;
			}
			ThresholdCurve curve = new ThresholdCurve();

			double[][] aligned = align(MatrixUtils.getCol(Y, j), MatrixUtils.getCol(P, j));

			Instances result = curve.getCurve(MLUtils.toWekaPredictions(toIntArray(aligned[0]), aligned[1]));
			if (roc) {
				AUC[j] = ThresholdCurve.getROCArea(result);
			} else {
				AUC[j] = ThresholdCurve.getPRCArea(result);
			}
			// System.out.println(Arrays.toString(MatrixUtils.getCol(Y, j)));
			// System.out.println(Arrays.toString(MatrixUtils.getCol(P, j)));
			//
			// System.out.println(AUC[j]);
		}

		L -= missing;

		if (L == 0) {
			return Double.NaN;
		}

		return Utils.mean(AUC);

	}

	/** Calculate AUPRC: Area Under the Precision-Recall curve. */
	public static double P_macroAUPRC(final int Y[][], final double P[][]) {
		return getMacro(Y, P, false);
	}

	/** Calculate AUROC: Area Under the ROC curve. */
	public static double P_macroAUROC(final int Y[][], final double P[][]) {
		return getMacro(Y, P, true);
	}

	/** Get Data for Plotting PR and ROC curves. */
	public static Instances curveDataMicroAveraged(final int Y[][], final double P[][]) {
		// works with missing

		int y[] = MatrixUtils.flatten(Y);
		double p[] = MatrixUtils.flatten(P);

		double[][] aligned = align(y, p);

		y = toIntArray(aligned[0]);
		p = aligned[1];

		ThresholdCurve curve = new ThresholdCurve();
		return curve.getCurve(MLUtils.toWekaPredictions(y, p));
	}

	/** Get Data for Plotting PR and ROC curves. */
	public static Instances curveDataMacroAveraged(final int Y[][], final double P[][]) {

		// Note: 'Threshold' contains the probability threshold that gives rise to the previous performance values.

		Instances curveData[] = curveData(Y, P);

		int L = curveData.length;

		int noNullIndex = -1;

		for (int i = 0; i < curveData.length; i++) {
			if (curveData[i] == null) {
				L--;
			} else {
				if (noNullIndex == -1) {
					// checking for the first curveData that is not null (=does not consist of
					// only missing values or 0s)
					noNullIndex = i;
				}

			}

		}

		Instances avgCurve = new Instances(curveData[noNullIndex], 0);
		int D = avgCurve.numAttributes();

		for (double t = 0.0; t < 1.; t += 0.01) {
			Instance x = (Instance) curveData[noNullIndex].instance(0).copy();
			// System.out.println("x1\n"+x);
			boolean firstloop = true;
			for (int j = 0; j < L; j++) {

				// if there are only missing values in a column, curveData[j] is null

				if (curveData[j] == null) {
					continue;
				}

				int i = ThresholdCurve.getThresholdInstance(curveData[j], t);
				if (firstloop) {
					// reset
					for (int a = 0; a < D; a++) {
						x.setValue(a, curveData[j].instance(i).value(a) * 1. / L);
					}
					firstloop = false;
				} else {
					// add
					for (int a = 0; a < D; a++) {
						double v = x.value(a);
						x.setValue(a, v + curveData[j].instance(i).value(a) * 1. / L);
					}
				}
			}
			// System.out.println("x2\n"+x);
			avgCurve.add(x);
		}

		/*
		  System.out.println(avgCurve);
		  System.exit(1);
		
		  // Average everything
		  for (int i = 0; i < avgCurve.numInstances(); i++) {
		  for(int j = 0; j < L; j++) {
		  for (int a = 0; a < D; a++) {
		  double o = avgCurve.instance(i).value(a);
		  avgCurve.instance(i).setValue(a, o / L);
		  }
		  }
		  }
		*/
		return avgCurve;
	}

	/** Get Data for Plotting PR and ROC curves. */
	public static Instances curveData(int y[], double p[]) {
		// works with missing
		double[][] aligned = align(y, p);

		y = toIntArray(aligned[0]);
		p = aligned[1];

		ThresholdCurve curve = new ThresholdCurve();
		return curve.getCurve(MLUtils.toWekaPredictions(y, p));
	}

	/** Get Data for Plotting PR and ROC curves. */
	public static Instances[] curveData(final int Y[][], final double P[][]) {
		// works with missing
		int L = Y[0].length;
		Instances curveData[] = new Instances[L];
		for (int j = 0; j < L; j++) {
			Instances cd = curveData(MatrixUtils.getCol(Y, j), MatrixUtils.getCol(P, j));
			curveData[j] = cd;
		}
		return curveData;
	}

	/** Levenshtein Distance. Multi-target compatible */
	public static double L_LevenshteinDistance(final int Y[][], final int P[][]) {
		double loss = 0.;
		int N = Y.length;

		int missing = 0;

		for (int i = 0; i < N; i++) {

			if (allMissing(Y[i])) {
				missing++;
				continue;
			}

			loss += L_LevenshteinDistance(Y[i], P[i]);
		}

		N -= missing;

		if (N == 0) {
			return Double.NaN;
		}

		return loss / N;
	}

	/** Levenshtein Distance divided by the number of labels. Multi-target compatible */
	public static double L_LevenshteinDistance(final int y[], final int p[]) {
		int L = y.length - numberOfMissingLabels(y);

		int[][] aligned = align(y, p);

		int[] alignedy = aligned[0];
		int[] alignedp = aligned[1];

		return (getLevenshteinDistance(alignedy, alignedp) / (double) L);
	}

	/*
	 * Levenshtein Distance.
	 * Given true labels y, and pRedicted labels r.
	 * based on http://www.merriampark.com/ldjava.htm
	 */
	private static int getLevenshteinDistance(int y[], int r[]) {

		int n = y.length;
		int m = r.length;

		if (n == 0) {
			return m;
		} else if (m == 0) {
			return n;
		}

		if (n > m) {
			final int[] tmp = y;
			y = r;
			r = tmp;
			n = m;
			m = r.length;
		}

		int p[] = new int[n + 1];
		int d[] = new int[n + 1];
		int _d[];

		int i;
		int j;

		int r_j;

		int cost;

		for (i = 0; i <= n; i++) {
			p[i] = i;
		}

		for (j = 1; j <= m; j++) {
			r_j = r[j - 1];
			d[0] = j;

			for (i = 1; i <= n; i++) {
				cost = y[i - 1] == r_j ? 0 : 1;
				d[i] = Math.min(Math.min(d[i - 1] + 1, p[i] + 1), p[i - 1] + cost);
			}

			_d = p;
			p = d;
			d = _d;
		}

		return p[n];
	}

	/** Log Likelihood */
	public double P_LogLikelihood(final int y[], final double p[]) {
		int L = y.length;

		double l = 0.0; // likelihood
		for (int j = 0; j < L; j++) {
			// independence assumption
			l += Math.log(Math.pow(p[j], y[j]) * Math.pow(1. - p[j], 1 - y[j])); // multi-label only
		}
		return l;
	}

	/** MSE */
	public double L_MSE(final int y[], final double p[]) {
		int L = y.length;
		double l[] = new double[L]; // likelihood
		for (int j = 0; j < L; j++) {
			// independence assumption
			l[j] = Math.pow(p[j] - y[j], 2); // MSE
		}
		return A.product(l);
	}

	/** MAE */
	public double L_MAE(final int y[], final double p[]) {
		int L = y.length;
		double l[] = new double[L]; // likelihood
		for (int j = 0; j < L; j++) {
			// independence assumption
			l[j] = Math.abs(p[j] - y[j]); // MAE
		}
		return A.product(l);
	}

	/** Product
	public double P_Product(int Y[][], double P[][]) {
	
	int N = Y.length;
	
	double s = 1.;
	
	for(int i = 0; i < N; i++) {
	s *= L_MAE(Y[i],P[i]);
	}
	
	return s;
	}*/

	/** Log Sum
	public double P_LogSum(int Y[][], double P[][]) {
	
	int N = Y.length;
	
	double s = 0.;
	
	for(int i = 0; i < N; i++) {
	s += Math.log(A.product(L_MAE(Y[i],P[i])));
	}
	
	return s;
	}*/

	/** Avg Sum
	public double P_Avg_Sum(int Y[][], double P[][]) {
	
	int N = Y.length;
	
	double s = 0.;
	
	for(int i = 0; i < N; i++) {
	s += L_MAE(Y[i],P[i]);
	}
	
	return s / N;
	}*/

	/**
	 * Do some tests.
	 */
	public static void main(final String args[]) {
		int Y[][] = new int[][] {
				// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
				{ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
				// {1,2},
				// {3,4,5},
				// {6},
				// {7}
		};
		double P[][] = new double[][] {
				// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
				{ 0, 0.7, 0.8, 0.9, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0.6, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0.8, 0, 0 }, { 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, };
		int Ypred[][] = new int[][] {
				// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
				{ 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
				// {1,2,3,9},
				// {3,4},
				// {6,12},
				// {1}
		};
		System.out.println("0.533333333... = " + P_FmacroAvgD(Y, Ypred));
		System.out.println("LD = " + L_LevenshteinDistance(Y, Ypred));
		System.out.println("MA = \n" + curveDataMacroAveraged(Y, P));
		// System.out.println("\nMi = \n"+curveDataMicroAveraged(Y,P));
	}
}
