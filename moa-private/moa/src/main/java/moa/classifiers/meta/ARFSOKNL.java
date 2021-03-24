/*
 *    ARFSOKNL.java
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */
package moa.classifiers.meta;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
//import com.sun.codemodel.internal.JForEach;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.IntervalsPrediction;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;
import meka.core.A;
import meka.core.F;
import moa.AbstractMOAObject;
import moa.classifiers.Regressor;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.trees.ARFFIMTDDforSOKNL;
import moa.core.*;
import moa.evaluation.BasicRegressionPerformanceEvaluator;
import moa.options.ClassOption;
import org.apache.commons.math3.distribution.NormalDistribution;


import java.beans.beancontext.BeanContextChild;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Implementation of ARFSOKNL, an extension of AdaptiveRandomForest for classification.
 *
 * <p>See details in:<br> Heitor Murilo Gomes, Jean Paul Barddal, Luis Eduardo Boiko Ferreira, Albert Bifet.
 * Adaptive random forests for data stream regression.
 * In European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2018.
 * https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-183.pdf</p>
 */
public class ARFSOKNL extends AbstractClassifier implements Regressor{

    @Override
    public String getPurposeString() {
        return "Adaptive Random Forest Regressor algorithm for evolving data streams from Gomes et al.";
    }

    private static final long serialVersionUID = 1L;


    /**
     * YS      Occupied Characters for Options: l s o m a x p u q i c k n t
     **/

    public ClassOption treeLearnerOption = new ClassOption("treeLearner", 'l',
            "Random Forest Tree.", ARFFIMTDDforSOKNL.class,
            "ARFFIMTDDforSOKNL -s VarianceReductionSplitCriterion -g 50 -c 0.01");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of trees.", 100, 1, Integer.MAX_VALUE);

    //  YS   random seed
    public IntOption randomSeedOption = new IntOption("randomSeed", 'r',"The random seed",1);

    public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o',
            "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
            new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
                    "Percentage (M * (m / 100))"},
            new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 3);

    public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
            "Number of features allowed considered for each split. Negative values corresponds to M - m", 60, Integer.MIN_VALUE, Integer.MAX_VALUE);

    public FloatOption lambdaOption = new FloatOption("lambda", 'a',
            "The lambda parameter for bagging.", 6.0, 1.0, Float.MAX_VALUE);

    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-3");

    public ClassOption warningDetectionMethodOption = new ClassOption("warningDetectionMethod", 'p',
            "Change detector for warnings (start training bkg learner)", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-2");

    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'u',
            "Should use drift detection? If disabled then bkg learner is also disabled");

    public FlagOption disableBackgroundLearnerOption = new FlagOption("disableBackgroundLearner", 'q',
            "Should use bkg learner? If disabled then reset tree immediately.");

    // YS      New Features
    public FlagOption predictionIntervalsOption = new FlagOption("PredictionIntervals", 'i', "Prediction Intervals");

    public FlagOption selfOptimisingKNL = new FlagOption("selfOptimisingKNL", 'f',"Trigger the self optimising KNL ");

    public FloatOption confidenceLevelOption = new FloatOption("ConfidenceLever", 'c', "For users to specify the confidence level.", .95, 0, 1);

    private double squareError;

    private double[] lowerByTrees;
    private double[] upperByTrees;

    public FlagOption kClosestLeavesOption = new FlagOption("KNearestLeaves", 'n', "Use k nearest leaves to predict");

    public IntOption kLeavesOption = new IntOption("NumberOfLeaves", 'k', "Specify how many leaves are being used", 5, 1, 100);

    public FlagOption treesIntervalOption = new FlagOption("TreeIntervals", 't', "Using ARF-Reg for the Prediction Intervals");
    //===============================================================

    protected static final int FEATURES_M = 0;
    protected static final int FEATURES_SQRT = 1;
    protected static final int FEATURES_SQRT_INV = 2;
    protected static final int FEATURES_PERCENT = 3;

    protected ARFSOKNL.ARFFIMTDDforSOKNLBaseLearner[] ensemble;
    protected long instancesSeen;
    protected int subspaceSize;
    protected BasicRegressionPerformanceEvaluator evaluator;

    protected BasicRegressionPerformanceEvaluator[] selfOptimisingEvaluators;

    protected double[] previousPrediction;

    @Override
    public void resetLearningImpl() {
        // Reset attributes
        this.ensemble = null;
        this.subspaceSize = 0;
        this.instancesSeen = 0;
        this.evaluator = new BasicRegressionPerformanceEvaluator();

        // YS     Reset variables
        this.squareError = 0;
        this.lowerByTrees = new double[1];
        this.upperByTrees = new double[1];

        this.classifierRandom = new Random(randomSeedOption.getValue());

        // YS     Initiate the selfOptimisingEvaluators
        this.previousPrediction = new double[this.ensembleSizeOption.getValue()];

        this.selfOptimisingEvaluators = new BasicRegressionPerformanceEvaluator[this.ensembleSizeOption.getValue()];
        for (int i = 0; i < this.selfOptimisingEvaluators.length; i++) {
            this.selfOptimisingEvaluators[i] = new BasicRegressionPerformanceEvaluator();
            this.selfOptimisingEvaluators[i].reset();
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        ++this.instancesSeen;
        if (this.ensemble == null)
            initEnsemble(instance);

        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            InstanceExample example = new InstanceExample(instance);
            this.ensemble[i].evaluator.addResult(example, vote.getArrayRef());
            int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
            if (k > 0) {
                this.ensemble[i].trainOnInstance(instance, k, this.instancesSeen);
            }
        }

        // YS            Update the evaluator array
        InstanceExample example = new InstanceExample(instance);
        for (int i = 0; i < this.previousPrediction.length; i++) {
            this.selfOptimisingEvaluators[i].addResult(example,new double[]{this.previousPrediction[i]});
        }
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();
        if (this.ensemble == null)
            initEnsemble(testInstance);
        double accounted = 0;

        DoubleVector predictions = new DoubleVector();
        DoubleVector ages = new DoubleVector();
        DoubleVector performance = new DoubleVector();

        // YS         For self optimising KNL
        if (this.selfOptimisingKNL.isSet()) {
            // YS       Initiate an array of basic learner evaluators for self-optimising kNL
            InstanceExample example = new InstanceExample(instance);
            int n = selfOptimisingEvaluators.length;
            double[] performances = new double[n];
//            double[] predics = new double[n];

            ArrayList<ARFFIMTDDforSOKNL.LeafNode> candidates = new ArrayList<>();

            // YS       Reset previous predictions to 0
            this.previousPrediction = new double[n];

            for (ARFSOKNL.ARFFIMTDDforSOKNLBaseLearner a : this.ensemble)
                if (a.classifier != null)
                    candidates.add((ARFFIMTDDforSOKNL.LeafNode) a.classifier.getLeafForInstance(instance, a.classifier.getTreeRoot()));

            double[] distances = new double[candidates.size()];

            for (int i = 0; i < candidates.size(); i++) {
                if (candidates.get(i) != null) {
                    double[] centroid = new double[instance.numAttributes() - 1];
                    for (int j = 0; j < centroid.length; j++)
                        if(candidates.get(i).sumsForAllAttrs != null) centroid[j] = candidates.get(i).sumsForAllAttrs[j] / candidates.get(i).learntInstances;

                    distances[i] += getDistanceFromCentroid(instance, centroid) / candidates.get(i).learntInstances;
//                        treePredictions[i] += instance.classValue()/candidates.get(i).leafInstances.size();

                }
            }

            for (int i = 0; i < n; i++){
                double[] temporaryPrediction = {getKNLPrediction(instance, i+1, candidates, distances)};
                this.selfOptimisingEvaluators[i].addResult(example, temporaryPrediction);
                performances[i] =  this.selfOptimisingEvaluators[i].getSquareError();
                this.previousPrediction[i] = temporaryPrediction[0];
            }

            return new double[]{this.previousPrediction[indexOfSmallestValue(performances)]};
        }

        // YS       K Nearest Leaves Predictions
//        if (this.kClosestLeavesOption.isSet()) {
//            double prediction = getKNLPrediction(instance, kLeavesOption.getValue());
//
//            return new double[]{prediction};
//        }

        // Original ARF-Reg
        for (int i = 0; i < this.ensemble.length; ++i) {
            double currentPrediction = this.ensemble[i].getVotesForInstance(testInstance)[0];

            ages.addToValue(i, this.instancesSeen - this.ensemble[i].createdOn);
            performance.addToValue(i, this.ensemble[i].evaluator.getSquareError());
            predictions.addToValue(i, currentPrediction);
            ++accounted;
        }
        double predicted = predictions.sumOfValues() / accounted;
//        if(predicted >= 1500) {
//            System.out.println(this.instancesSeen + " = " + predictions);
//            System.out.println(this.instancesSeen + " = " + ages);
//            System.out.println(this.instancesSeen + " = " + performance);
//            System.out.println();
//        }

        return new double[]{predictions.sumOfValues() / accounted};
    }
    // YS      Extracted method for KNL
    private double getKNLPrediction(Instance instance, int k, ArrayList<ARFFIMTDDforSOKNL.LeafNode> candidates, double[] distances) {

//        double[] treePredictions = new double[candidates.size()];


        double prediction = 0;

//            Arrays.sort(treePredictions);
        if (candidates.size() > 0) {
//                this.lowerByTrees[0] = treePredictions[0];
//                this.upperByTrees[0] = treePredictions[treePredictions.length - 1];

            int[] indices = indicesOfKSmallestValues(distances, Math.min(k, candidates.size()));

            for (Integer i : indices)
                if (candidates.get(i) != null && candidates.get(i).sumsForAllAttrs != null)
                    prediction += candidates.get(i).sumsForAllAttrs[instance.numAttributes()-1] / (candidates.get(i).learntInstances * k);


        }
        return prediction;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    protected void initEnsemble(Instance instance) {
        // Init the ensemble.
        int ensembleSize = this.ensembleSizeOption.getValue();
        this.ensemble = new ARFSOKNL.ARFFIMTDDforSOKNLBaseLearner[ensembleSize];

        // TODO: this should be an option with default = BasicClassificationPerformanceEvaluator
        BasicRegressionPerformanceEvaluator regressionEvaluator = new BasicRegressionPerformanceEvaluator();

        this.subspaceSize = this.mFeaturesPerTreeSizeOption.getValue();

        // The size of m depends on:
        // 1) mFeaturesPerTreeSizeOption
        // 2) mFeaturesModeOption
        int n = instance.numAttributes() - 1; // Ignore class label ( -1 )

        switch (this.mFeaturesModeOption.getChosenIndex()) {
            case AdaptiveRandomForest.FEATURES_SQRT:
                this.subspaceSize = (int) Math.round(Math.sqrt(n)) + 1;
                break;
            case AdaptiveRandomForest.FEATURES_SQRT_INV:
                this.subspaceSize = n - (int) Math.round(Math.sqrt(n) + 1);
                break;
            case AdaptiveRandomForest.FEATURES_PERCENT:
                // If subspaceSize is negative, then first find out the actual percent, i.e., 100% - m.
                double percent = this.subspaceSize < 0 ? (100 + this.subspaceSize) / 100.0 : this.subspaceSize / 100.0;
                this.subspaceSize = (int) Math.round(n * percent);
                break;
        }
        // Notice that if the selected mFeaturesModeOption was
        //  AdaptiveRandomForest.FEATURES_M then nothing is performed in the
        //  previous switch-case, still it is necessary to check (and adjusted)
        //  for when a negative value was used.

        // m is negative, use size(features) + -m
        if (this.subspaceSize < 0)
            this.subspaceSize = n + this.subspaceSize;
        // Other sanity checks to avoid runtime errors.
        //  m <= 0 (m can be negative if this.subspace was negative and
        //  abs(m) > n), then use m = 1
        if (this.subspaceSize <= 0)
            this.subspaceSize = 1;
        // m > n, then it should use n
        if (this.subspaceSize > n)
            this.subspaceSize = n;

        ARFFIMTDDforSOKNL treeLearner = (ARFFIMTDDforSOKNL) getPreparedClassOption(this.treeLearnerOption);
        treeLearner.resetLearning();

        for (int i = 0; i < ensembleSize; ++i) {
            treeLearner.subspaceSizeOption.setValue(this.subspaceSize);
            this.ensemble[i] = new ARFSOKNL.ARFFIMTDDforSOKNLBaseLearner(
                    i,
                    (ARFFIMTDDforSOKNL) treeLearner.copy(),
                    (BasicRegressionPerformanceEvaluator) regressionEvaluator.copy(),
                    this.instancesSeen,
                    !this.disableBackgroundLearnerOption.isSet(),
                    !this.disableDriftDetectionOption.isSet(),
                    driftDetectionMethodOption,
                    warningDetectionMethodOption,
                    false,
                    classifierRandom);
        }
    }


    // YS     Override getPredictionForInstance Method
    @Override
    public Prediction getPredictionForInstance(Instance inst) {
        Prediction prediction;
        if (predictionIntervalsOption.isSet()) {
            prediction = new IntervalsPrediction(3);
            prediction.setVotes(0, getVotesForInstance(inst));
            if (treesIntervalOption.isSet()) {
                prediction.setVotes(1, lowerByTrees);
                prediction.setVotes(2, upperByTrees);
            } else {
                prediction.setVotes(1, getLowerPredictionIntervalsForInstance(inst));
                prediction.setVotes(2, getUpperPredictionIntervalsForInstance(inst));
            }
        } else {
            prediction = new MultiLabelPrediction(1);
            prediction.setVotes(getVotesForInstance(inst));
        }
        return prediction;
    }


    // YS      Get Interval Bounds
    private double[] getLowerPredictionIntervalsForInstance(Instance inst) {
        return new double[]{getVotesForInstance(inst)[0] - calculateIntervalLength(inst)};
    }

    private double[] getUpperPredictionIntervalsForInstance(Instance inst) {
        return new double[]{getVotesForInstance(inst)[0] + calculateIntervalLength(inst)};
    }

    // YS      Calculate Interval Length
    private double calculateIntervalLength(Instance inst) {
        if (!inst.classAttribute().isNumeric()) return 0;

        double prediction = getVotesForInstance(inst)[0];

        NormalDistribution normalDistribution = new NormalDistribution();

        squareError += (inst.classValue() - prediction) * (inst.classValue() - prediction);

        return normalDistribution.inverseCumulativeProbability(0.5 + this.confidenceLevelOption.getValue() / 2) * Math.sqrt(squareError / (instancesSeen + 1));
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

    private double getDistanceFromCentroid(Instance inst, double[] centroid) {
        double sumOfSquare = 0;
        if (inst.numAttributes() - 1 != centroid.length) return 0;
        else {
            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                sumOfSquare += Math.pow((inst.value(i) - centroid[i]), 2);
            }
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


    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }


    protected final class ARFFIMTDDforSOKNLBaseLearner extends AbstractMOAObject {
        public int indexOriginal;
        public long createdOn;
        public long lastDriftOn;
        public long lastWarningOn;
        public ARFFIMTDDforSOKNL classifier;
        public boolean isBackgroundLearner;


        // The drift and warning object parameters.
        protected ClassOption driftOption;
        protected ClassOption warningOption;

        // Drift and warning detection
        protected ChangeDetector driftDetectionMethod;
        protected ChangeDetector warningDetectionMethod;

        public boolean useBkgLearner;
        public boolean useDriftDetector;

        // Bkg learner
        protected ARFSOKNL.ARFFIMTDDforSOKNLBaseLearner bkgLearner;
        // Statistics
        public BasicRegressionPerformanceEvaluator evaluator;
        protected int numberOfDriftsDetected;
        protected int numberOfWarningsDetected;

        private void init(int indexOriginal, ARFFIMTDDforSOKNL instantiatedClassifier, BasicRegressionPerformanceEvaluator evaluatorInstantiated,
                          long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, boolean isBackgroundLearner, Random random) {
            this.indexOriginal = indexOriginal;
            this.createdOn = instancesSeen;
            this.lastDriftOn = 0;
            this.lastWarningOn = 0;

            this.classifier = instantiatedClassifier;
            this.evaluator = evaluatorInstantiated;
            this.useBkgLearner = useBkgLearner;
            this.useDriftDetector = useDriftDetector;

            this.numberOfDriftsDetected = 0;
            this.numberOfWarningsDetected = 0;
            this.isBackgroundLearner = isBackgroundLearner;

            if (this.useDriftDetector) {
                this.driftOption = driftOption;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }

            // Init Drift Detector for Warning detection.
            if (this.useBkgLearner) {
                this.warningOption = warningOption;
                this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
            }

            // YS   Pass the random seed to ARFFIMTDDforSOKNL
            this.classifier.classifierRandom = random;
        }

        public ARFFIMTDDforSOKNLBaseLearner(int indexOriginal, ARFFIMTDDforSOKNL instantiatedClassifier, BasicRegressionPerformanceEvaluator evaluatorInstantiated,
                                    long instancesSeen, boolean useBkgLearner, boolean useDriftDetector, ClassOption driftOption, ClassOption warningOption, boolean isBackgroundLearner, Random random) {
            init(indexOriginal, instantiatedClassifier, evaluatorInstantiated, instancesSeen, useBkgLearner, useDriftDetector, driftOption, warningOption, isBackgroundLearner, random);
        }

        public void reset() {
            if (this.useBkgLearner && this.bkgLearner != null) {
                this.classifier = this.bkgLearner.classifier;

                this.driftDetectionMethod = this.bkgLearner.driftDetectionMethod;
                this.warningDetectionMethod = this.bkgLearner.warningDetectionMethod;

                this.evaluator = this.bkgLearner.evaluator;
                this.createdOn = this.bkgLearner.createdOn;
                this.bkgLearner = null;
            } else {
                this.classifier.resetLearning();
                this.createdOn = instancesSeen;
                this.driftDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.driftOption)).copy();
            }
            this.evaluator.reset();

        }

        public void trainOnInstance(Instance instance, double weight, long instancesSeen) {
            Instance weightedInstance = (Instance) instance.copy();
            weightedInstance.setWeight(instance.weight() * weight);
            this.classifier.trainOnInstance(weightedInstance);

            if (this.bkgLearner != null)
                this.bkgLearner.classifier.trainOnInstance(instance);

            // Should it use a drift detector? Also, is it a backgroundLearner? If so, then do not "incept" another one.
            if (this.useDriftDetector && !this.isBackgroundLearner) {
//                boolean correctlyClassifies = this.classifier.correctlyClassifies(instance);
                double prediction = this.classifier.getVotesForInstance(instance)[0];
                // Check for warning only if useBkgLearner is active
                if (this.useBkgLearner) {
                    // Update the warning detection method
                    this.warningDetectionMethod.input(prediction);
                    // Check if there was a change
                    if (this.warningDetectionMethod.getChange()) {
                        this.lastWarningOn = instancesSeen;
                        this.numberOfWarningsDetected++;
                        // Create a new bkgTree classifier
                        ARFFIMTDDforSOKNL bkgClassifier = (ARFFIMTDDforSOKNL) this.classifier.copy();
                        bkgClassifier.resetLearning();

                        // Resets the evaluator
                        BasicRegressionPerformanceEvaluator bkgEvaluator = (BasicRegressionPerformanceEvaluator) this.evaluator.copy();
                        bkgEvaluator.reset();

                        // Create a new bkgLearner object
                        this.bkgLearner = new ARFSOKNL.ARFFIMTDDforSOKNLBaseLearner(indexOriginal, bkgClassifier, bkgEvaluator, instancesSeen,
                                this.useBkgLearner, this.useDriftDetector, this.driftOption, this.warningOption, true, this.classifier.classifierRandom);

                        // Update the warning detection object for the current object
                        // (this effectively resets changes made to the object while it was still a bkg learner).
                        this.warningDetectionMethod = ((ChangeDetector) getPreparedClassOption(this.warningOption)).copy();
                    }
                }

                /*********** drift detection ***********/

                // Update the DRIFT detection method
                this.driftDetectionMethod.input(prediction);
                // Check if there was a change
                if (this.driftDetectionMethod.getChange()) {
                    this.lastDriftOn = instancesSeen;
                    this.numberOfDriftsDetected++;
                    this.reset();
                }
            }
        }

        public double[] getVotesForInstance(Instance instance) {
//            DoubleVector vote = new DoubleVector(this.classifier.getVotesForInstance(instance));
//            return vote.getArrayRef();
            return this.classifier.getVotesForInstance(instance);
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {
        }

    }
}
