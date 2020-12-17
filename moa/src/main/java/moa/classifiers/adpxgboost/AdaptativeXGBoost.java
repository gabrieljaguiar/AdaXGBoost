package moa.classifier.adpxgboost;

import java.util.List;
import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;

import moa.core.Measurement;

//import ml.dmlc.xgboost4j.java.DMatrix;
//import ml.dmlc.xgboost4j.java.Booster;
//import ml.dmlc.xgboost4j.java.XGBoost;

class AdaptativeXGBoost extends AbstractClassifier {

    protected Instances window;

    // protected XGBoost[] ensemble;

    public IntOption windowSize = new IntOption("windowSize", 's', "Window size", 1000, 1, Integer.MAX_VALUE);

    @Override
    public void resetLearningImpl() {
    }

    /* Predict method */
    @Override
    public double[] getVotesForInstance(Instance instance) {

        double[] votes = new double[instance.numClasses()];

        Random random = new Random();

        votes = random.doubles(instance.numClasses(), 1, 1000).toArray();

        return votes;
    }

    /* partial fit function */
    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (window == null)
            window = new Instances(instance.dataset(), 0);

        window.add(instance);

        if (window.size() == windowSize.getValue()) {
            window.delete();
        }
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        Measurement[] measurements = null;

        return measurements;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}