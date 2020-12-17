package moa.classifier.adpxgboost;

import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;

class AdaptativeXGBoost extends AbstractClassifier {

    protected Instances window;

    protected XGBoost[] ensemble;

    /* Predict method */
    @Override
    public double[] getVotesForInstance(Instance instance) {

        double[] votes = new double[instance.numClasses()];

        if (model) {
            List<IIndividual>[] rules = algorithm.getSolutions();

            for (int i = 0; i < instance.numClasses(); i++) {
                for (IIndividual rule : rules[i]) {
                    if ((Boolean) ((SyntaxTreeRuleIndividual) rule).getPhenotype().covers(instance))
                        votes[i] += ((SimpleValueFitness) rule.getFitness()).getValue();
                }
            }
        }

        return votes;
    }

    /* partial fit function */
    @Override
    public void trainOnInstanceImpl(Instance instance) {

        if (window == null)
            window = new Instances(instance.dataset(), 0);

        window.add(instance);

        if (window.size() == windowSize.getValue()) {
            algorithm.addChunkData(window);
            algorithm.prepare();
            algorithm.execute();
            model = true;

            window.delete();
        }
    }
}