package moa.classifiers;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import moa.classifiers.core.driftdetection.AbstractChangeDetector;
import moa.core.Measurement;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class AdaptativeXGBoost extends AbstractClassifier implements MultiClassClassifier {
    private static final long serialVersionUID = 1L;

    public IntOption nEstimators = new IntOption("numberOfBoosters", 'n', "Number of Boosters", 5, 1,
            Integer.MAX_VALUE);

    public FloatOption learningRate = new FloatOption("learningRate", 'l', "Learning Rate", 0.3, 0, Float.MAX_VALUE);

    public IntOption maxDepth = new IntOption("maxDepth", 'p', "Max Depth", 6, 1, Integer.MAX_VALUE);

    public IntOption maxWindowSize = new IntOption("maxWindowSize", 'm', "Max Window size", 1000, 1, Integer.MAX_VALUE);

    public IntOption minWindowSize = new IntOption("minWindowSize", 'w', "Min Window size", -1, -1, Integer.MAX_VALUE);

    public FlagOption detectDrift = new FlagOption("detectDrift", 'd', "Detect Drift Flag");

    public StringOption updateStrategy = new StringOption("updateStrategy", 'u', "Uptade Strategy", "replace");

    protected ArrayList<Instance> window;

    protected List<Booster> ensemble;

    protected Integer numberOfAttr;

    protected Integer windowSize;

    protected Integer dynamicWindowSize;

    protected Float initMargin;

    protected Integer modelIndex;

    protected Integer samplesSeen;

    protected Map<String, Object> boostingParams;

    private String PUSH_STRATEGY = "push";

    private String REPLACE_STRATEGY = "replace";

    private AbstractChangeDetector driftDetector;

    private Boolean hasFilled = false;

    /* Constructor */
    @Override
    public void resetLearningImpl() {
        window = new ArrayList<Instance>();
        this.ensemble = new ArrayList<Booster>();
        if (this.updateStrategy.getValue() == this.REPLACE_STRATEGY) {
            for (int i = 0; i < this.nEstimators.getValue(); i++) {
                this.ensemble.add(null);
            }
        }
        this.boostingParams = new HashMap<String, Object>() {
            {
                put("eta", learningRate.getValue());
                put("max_depth", maxDepth.getValue());
                put("objective", "binary:logistic");
                put("silent", true);
            }
        };

        if (this.driftDetector == null) {
            this.driftDetector = new ADWINChangeDetector();
        }
        this.resetWindowSize();
        this.initMargin = 0.0f;
        this.modelIndex = 0;
        this.samplesSeen = 0;
    }

    /* Predict method */
    @Override
    public double[] getVotesForInstance(Instance instance) {
        ArrayList<Instance> data = new ArrayList<Instance>();
        data.add(instance);

        double[] votes = new double[instance.numClasses()];

        Arrays.fill(votes, 0.0);

        float[] instancesArray = flattenInstances(data);
        // float[] labels = this.arrayToFloat(getLabels(data));

        int trees_in_ensemble = 0;

        if (this.updateStrategy.getValue() == this.REPLACE_STRATEGY) {
            trees_in_ensemble = this.modelIndex;
            if (this.hasFilled) {
                trees_in_ensemble = this.ensemble.size();
            }
        } else {
            trees_in_ensemble = this.ensemble.size();
        }


        if (trees_in_ensemble <= 0) {
            return votes;
        }

        try {
            DMatrix d_test = new DMatrix(instancesArray, data.size(), this.numberOfAttr.intValue());

            for (int i = 0; i < trees_in_ensemble; i++) {
                float[][] margins = this.ensemble.get(i).predict(d_test, true);
                d_test.setBaseMargin(margins);
            }

            float predicted = this.ensemble.get(trees_in_ensemble - 1).predict(d_test, true)[0][0];

            if (predicted > 0.5f) {
                predicted = 1f;
            } else {
                predicted = 0f;
            }

            int indexPred = (int) predicted;
            votes[indexPred] = 1.0;



        } catch (XGBoostError e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


        return votes;
    }

    /* partial fit function */
    @Override
    public void trainOnInstanceImpl(Instance instance) {

        // System.out.println("Training...");

        if (instance.numClasses() > 2) {
            System.out.println("This code does not support multi class classification");
            return;
        }

        window.add(instance);

        this.numberOfAttr = instance.numAttributes() - 1; // remove class attr

        while (window.size() >= this.windowSize) {
            ArrayList<Instance> miniBatch = new ArrayList<Instance>(window.subList(0, this.windowSize));
            this.trainOnMiniBatch(miniBatch);
            window.subList(0, this.windowSize).clear();
            this.adjustWindowSize();
            if (this.detectDrift.isSet()) {


                double[] votes = this.getVotesForInstance(instance);

                double prediction = this.getLabelPredicted(votes);

                double error = Math.abs(prediction - instance.classValue());

                this.driftDetector.input(error);

                if (this.driftDetector.getChange()) {
                    System.out.println("Drift detected");
                    this.resetWindowSize();
                    if (this.updateStrategy.getValue() == this.REPLACE_STRATEGY) {
                        this.modelIndex = 0;
                    }
                }

                /* To do */
            }
        }
    }

    private double getLabelPredicted(double[] votes) {
        //System.out.println(Arrays.toString(votes));
        int index = 0;
        while (votes[index] < 1) {
            index++;
        }

        return (double) index;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = null;

        return measurements;
    }

    public boolean isRandomizable() {
        return true;
    }

    private void trainOnMiniBatch(ArrayList<Instance> data) {
        if (this.updateStrategy.getValue() == this.REPLACE_STRATEGY) {
            Booster booster = null;
            try {
                booster = this.trainBooster(data, this.modelIndex);
            } catch (XGBoostError e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            this.ensemble.set(this.modelIndex, booster);
            this.samplesSeen = this.samplesSeen + data.size();
            this.updateModelId();
        } else {
            Booster booster = null;
            try {
                booster = this.trainBooster(data, this.ensemble.size());
            } catch (XGBoostError e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            if (this.ensemble.size() == this.nEstimators.getValue()) {
                this.ensemble.remove(0);
            }
            this.ensemble.add(booster);
            this.samplesSeen = this.samplesSeen + data.size();

        }
    }

    private Booster trainBooster(ArrayList<Instance> data, Integer len) throws XGBoostError {
        /* To be implemented */

        float[] instancesArray = flattenInstances(data);
        float[] labels = this.arrayToFloat(getLabels(data));
        DMatrix d_mini_batch_train = new DMatrix(instancesArray, data.size(), this.numberOfAttr.intValue());

        d_mini_batch_train.setLabel(labels);

        float[][] margins = new float[data.size()][1]; // binary classifier

        for (float[] row : margins)
            Arrays.fill(row, this.initMargin);



        for (int j = 0; j < len; j++) {
            float[][] predicts = this.ensemble.get(j).predict(d_mini_batch_train, true);

            for (int i = 0; i < predicts.length; i++) {
                margins[i] = this.sumElementWise(margins[i], predicts[i]);

            }

        }

        d_mini_batch_train.setBaseMargin(margins);

        Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
            private static final long serialVersionUID = 1L;
        };

        return XGBoost.train(d_mini_batch_train, this.boostingParams, 10, watches, null, null);
    }

    public float[] sumElementWise(float[] a1, float[] a2) {
        float[] a3 = new float[a1.length];
        for (int i = 0; i < a1.length; i++) {
            a3[i] = a1[i] + a2[i];
        }

        return a3;
    }

    private float[] flattenInstances(ArrayList<Instance> data) {
        float[] instArray = {};
        for (int i = 0; i < data.size(); i++) {
            Instance inst = (Instance) data.get(i);
            double[] instanceArray = inst.toDoubleArray();
            instanceArray = Arrays.copyOfRange(instanceArray, 0, inst.classIndex());

            instArray = this.concatArrays(instArray, this.arrayToFloat(instanceArray));

        }

        return instArray;

    }

    private double[] getLabels(ArrayList<Instance> data) {
        double[] labels = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = data.get(i).classValue();
        }
        return labels;
    }

    private float[] arrayToFloat(double[] array) {
        float[] newArray = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            newArray[i] = (float) array[i];
        }
        return newArray;
    }

    private float[] concatArrays(float[] a1, float[] a2) {
        float[] result = new float[a1.length + a2.length];

        System.arraycopy(a1, 0, result, 0, a1.length);
        System.arraycopy(a2, 0, result, a1.length, a2.length);

        return result;
    }

    private void updateModelId() {
        this.modelIndex++;
        if (this.modelIndex == this.nEstimators.getValue()) {
            this.modelIndex = 0;
            this.hasFilled = true;
        }
    }

    private void resetWindowSize() {
        if (this.minWindowSize.getValue() != -1) {
            this.dynamicWindowSize = this.minWindowSize.getValue();
        } else {
            this.dynamicWindowSize = this.maxWindowSize.getValue();
        }
        this.windowSize = this.dynamicWindowSize;
    }

    private void adjustWindowSize() {
        if (this.dynamicWindowSize < this.maxWindowSize.getValue()) {
            this.dynamicWindowSize = this.dynamicWindowSize * 2;
            if (this.dynamicWindowSize > this.maxWindowSize.getValue()) {
                this.windowSize = this.maxWindowSize.getValue();
            } else {
                this.windowSize = this.dynamicWindowSize;
            }
        }
    }

}