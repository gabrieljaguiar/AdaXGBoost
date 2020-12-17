package moa.classifiers;

import java.util.List;
import java.util.Random;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Measurement;
import moa.options.ClassOption;

//import ml.dmlc.xgboost4j.java.DMatrix;
//import ml.dmlc.xgboost4j.java.Booster;
//import ml.dmlc.xgboost4j.java.XGBoost;

public class AdaptativeXGBoost extends AbstractClassifier implements MultiClassClassifier{
	private static final long serialVersionUID = 1L;
	
    protected Instances window;

    // protected XGBoost[] ensemble;

    public IntOption windowSize = new IntOption("windowSize", 's', "Window size", 1000, 1, Integer.MAX_VALUE);
    
    public IntOption gracePeriodOption = new IntOption("gracePeriod", 'g',
			"The number of instances to observe between model changes.",
			1000, 0, Integer.MAX_VALUE);

	public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
			"Only allow binary splits.");

	public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
			'c', "Split criterion to use.", SplitCriterion.class,
			"InfoGainSplitCriterion");
    
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

        System.out.println("Training...");

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

    
    public boolean isRandomizable() {
        return true;
    }
}