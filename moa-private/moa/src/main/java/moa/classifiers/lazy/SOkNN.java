/*
 *    kNN.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
package moa.classifiers.lazy;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import meka.core.A;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.evaluation.BasicRegressionPerformanceEvaluator;

import java.util.*;

/**
 * k Nearest Neighbor.<p>
 * <p>
 * Valid options are:<p>
 * <p>
 * -k number of neighbours <br> -m max instances <br>
 *
 * @author Jesse Read (jesse@tsc.uc3m.es)
 * @version 03.2012
 */
public class SOkNN extends AbstractClassifier implements MultiClassClassifier, Regressor {

    private static final long serialVersionUID = 1L;

    public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

    // For checking regression with mean value or median value
    public FlagOption medianOption = new FlagOption("median", 'm', "median or mean");

    public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
            "LinearNN", "KDTree"},
            new String[]{"Brute force search algorithm for nearest neighbour search. ",
                    "KDTree search algorithm for nearest neighbour search"
            }, 0);
    public FlagOption selfOptimisingOption = new FlagOption("SelfOptimisingOption", 's', "Trigger self optimising option");

    int C = 0;

    //  YS          Evaluators for self optimising
    protected BasicRegressionPerformanceEvaluator[] evaluators;
    //  YS          Array to store predictions
    protected double[] previousPredictions;

    @Override
    public String getPurposeString() {
        return "kNN: special.";
    }

    protected Instances window;

    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.window = new Instances(context, 0); //new StringReader(context.toString())
            this.window.setClassIndex(context.classIndex());
        } catch (Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void resetLearningImpl() {

        this.window = null;

        //  Reset the evaluators
        this.evaluators = new BasicRegressionPerformanceEvaluator[this.limitOption.getValue()];
        for (int i = 0; i < this.evaluators.length; i++) {
            this.evaluators[i] = new BasicRegressionPerformanceEvaluator();
            this.evaluators[i].reset();
        }

        //  YS     Initiate previous predictions
        this.previousPredictions = new double[this.evaluators.length];
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (inst.classValue() > C)
            C = (int) inst.classValue();
        if (this.window == null) {
            this.window = new Instances(inst.dataset());
        }
        //  YS     Update the evaluators
        InstanceExample example = new InstanceExample(inst);
        for (int i = 0; i < this.window.size(); i++) this.evaluators[i].addResult(example, new double[]{this.previousPredictions[i]});


        if (this.limitOption.getValue() <= this.window.numInstances()) {
            this.window.delete(0);
        }
        this.window.add(inst);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double v[] = new double[C + 1];
        try {
            NearestNeighbourSearch search;
            if (this.nearestNeighbourSearchOption.getChosenIndex() == 0) {
                search = new LinearNNSearch(this.window);
            } else {
                search = new KDTree();
                search.setInstances(this.window);
            }
            if (this.window.numInstances() > 0) {

                if (this.selfOptimisingOption.isSet()) {
                    //  YS    Array to store the performances
                    double[] performances = new double[this.window.size()];
                    this.previousPredictions = new double[this.window.size()];
                    //  YS    Calculate and store all the distances
                    double[] distances = new double[this.window.size()];
                    for (int i = 0; i < distances.length; i++) distances[i] = getDistance(inst, this.window.get(i));
                    int[] indices = sortedIndices(distances);
                    for (int i = 0; i < indices.length; i++) {
                        if (i == 0) this.previousPredictions[i] = this.window.get(indices[i]).classValue();
                        else this.previousPredictions[i] = (this.previousPredictions[i - 1] * i + this.window.get(indices[i]).classValue()) / (i + 1);
                    }
                    for (int i = 0; i < performances.length; i++) performances[i] = this.evaluators[i].getSquareError();

                    int k = indexOfSmallestValue(performances);

                    return new double[]{this.previousPredictions[k]};
                }


                Instances neighbours = search.kNearestNeighbours(inst, Math.min(kOption.getValue(), this.window.numInstances()));
                //================== Regression ====================
                if (inst.classAttribute().isNumeric()) {
                    double[] result = new double[1];
                    // For storing the sum of class values of all the k nearest neighbours
                    double sum = 0;
                    // For storing the number of the nearest neighbours
                    int num = neighbours.numInstances();
                    //================== Median ====================
                    if (medianOption.isSet()) {
                        // For storing every neighbour's class value
                        double[] classValues = new double[num];

                        for (int i = 0; i < num; i++) {
                            classValues[i] = neighbours.instance(i).classValue();
                        }
                        // Sort the class values
                        Arrays.sort(classValues);
                        // Assign the median value into result
                        if (classValues.length % 2 == 1) {
                            result[0] = classValues[num / 2];
                        } else {
                            result[0] = (classValues[num / 2 - 1] + classValues[num / 2]) / 2;
                        }
                        return result;
                        //=============== End of Median ============
                    } else {
                        //================== Mean ==================
                        for (int i = 0; i < num; i++) {
                            sum += neighbours.instance(i).classValue();
                        }
                        // Calculate the mean of all k nearest neighbours' class values
                        result[0] = sum / num;
                        return result;
                        //=============== End of Mean ==============
                    }
                    //============= End of Regression ==============
                } else {
                    for (int i = 0; i < neighbours.numInstances(); i++) {
                        v[(int) neighbours.instance(i).classValue()]++;
                    }
                }
            }
        } catch (Exception e) {
            //System.err.println("Error: kNN search failed.");
            //e.printStackTrace();
            //System.exit(1);
            return new double[inst.numClasses()];
        }
        return v;
    }

    // YS      Calculate Distance Between 2 Instances
    private double getDistance(Instance inst1, Instance inst2) {
        if (inst1.numAttributes() != inst2.numAttributes()) return 0;

        double sumOfSquare = 0;
        for (int i = 0; i < inst1.numAttributes() - 1; i++) {
            sumOfSquare += Math.pow((inst1.value(i) - inst2.value(i)), 2);
        }

        return Math.sqrt(sumOfSquare);
    }

    // YS      Gain The Index of The Smallest Value in An Array
    private int indexOfSmallestValue(double[] values) {

        int smallest = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] < values[smallest]) smallest = i;
        }
        return smallest;
    }

    // YS      Gain the Indices of The K Smallest Values in An Array
    private int[] indicesOfKSmallestValues(double[] values, int k) {
        int[] smallest = new int[k];
        for (int i = 0; i < k; i++) {
            smallest[i] = indexOfSmallestValue(values);
            values[smallest[i]] = Double.MAX_VALUE;
        }
        return smallest;
    }


    private int[] sortedIndices(double[] arr) {
        ArrayList<double[]> arrayList = new ArrayList();

        for (int i = 0; i < arr.length; i++) arrayList.add(new double[]{arr[i], i});
        Arrays.sort(arr);

        int[] s = new int[arr.length];
        for (int i = 0; i < arr.length; i++)
            for (double[] d : arrayList)
                if (arr[i] == d[0])
                    s[i] = (int) d[1];




//        int[] s = new int[arr.length];
//        TreeMap pairs = new TreeMap();
//        for (int i = 0; i < arr.length; i++) if (!pairs.containsKey(arr[i]))pairs.put(arr[i], i);
//        for (int i = 0; i < s.length; i++) {
//            Map.Entry entry = (Map.Entry)pairs.entrySet().iterator().next();
//            s[i] = (int) entry.getValue();
//            pairs.remove(entry.getKey());
//        }
        return s;
    }


    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public boolean isRandomizable() {
        return false;
    }
}