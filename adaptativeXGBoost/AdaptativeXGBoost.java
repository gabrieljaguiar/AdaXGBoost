package moa.classifiers;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Measurement;
import moa.options.ClassOption;

import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;

public class AdaptativeXGBoost extends AbstractClassifier {
	private static final long serialVersionUID = 1L;
	
    
    public IntOption nEstimators = new IntOption("Number of Estimators", 'n', "Number of Estimators", 5, 1, Integer.MAX_VALUE);

    public FloatOption learningRate = new FloatOption("Learning Rate", 'l', 'Learning Rate', 0.3, 0, Float.MAX_VALUES);

    public IntOption maxDepth = new IntOption("Max Depth", 'p', "Max Depth", 6, 1, Integer.MAX_VALUE);

    public IntOption maxWindowSize = new IntOption("Max windowSize", 'm', "Max Window size", 1000, 1, Integer.MAX_VALUE);

    public IntOption minWindowSize = new IntOption("Min WindowSize", 'w', "Min Window size", null, 1, Integer.MAX_VALUE);

    public FlagOption detectDrift = new FlagOption("Detect Drift", 'd', "Detect Drift Flag");

    public StringOption uptadeStrategy = new StringOption ("Update Strategy", 'u', "Uptade Strategy", "replace");
    


    protected Instances window;

    protected List<XGBoost> ensemble;

    protected Integer windowSize;

    protected Integer dynamicWindowSize;
    
    protected Float initMargin;

    protected Integer modelIndex; 

    protected Integer samplesSeen;

    private PUSH_STRATEGY = "push";

    private REPLACE_STRATEGY = "replace";

    

    /* Constructor */
    @Override
    public void resetLearningImpl() {
        this.ensemble = new ArrayList<XGBoost>();
        if (this.uptadeStrategy.getValue() == this.REPLACE_STRATEGY){
            for (int i = 0; i < this.nEstimators; i++){
                this.ensemble.add(null);
            }
        } 
        this.resetWindowSize();
        this.initMargin = 0.0;
        this.modelIndex = 0;
        this.samplesSeen = 0;
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

        window.add(instance);

        while (window.size() >= this.windowSize.getValue()) {
            this.trainOnMiniBatch(window) // Needs to be improved. Select just 1 to WindowSize Instances;
            int i = 0;
            while (i < this.windowSize.getValue()){
                window.delete(0);
                i++;
            }
            this.adjustWindowSize();
            if (this.detectDrift.getValue()){
                System.out.println("Drift");
                /* To do*/
            }
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

    private trainOnMiniBatch(Instances data){
        if (this.uptadeStrategy.getValue() == this.REPLACE_STRATEGY){
            XGBoost booster = this.trainBooster(data, this.modelIndex);
            this.ensemble.set(this.modelIndex, booster);
            this.samplesSeen = this.samplesSeen + data.getSize();
            this.updateModelId();
        }else{
            XGBoost booster = this.trainBooster(data, this.ensemble.size());
            if (this.ensemble.size() == this.nEstimators.getValue){
                this.ensemble.remove(0);
            }
            this.ensemble.add(booster);
            this.samplesSeen = this.samplesSeen + data.getSize();

        }
    }

    private trainBooster(Intances data, Integer len){
        /* To be implemented */
        return True;
    }

    private updateModelId(){
        this.modelIndex++;
        if (this.modelIndex == this.nEstimators.getValue()){
            this.modelIndex = 0;
        }
    }



    private resetWindowSize(){
        if (this.minWindowSize){
            this.dynamicWindowSize = this.minWindowSize.getValue();
        }else{
            this.dynamicWindowSize = this.maxWindowSize.getValue();
        }
        this.windowSize = this.dynamicWindowSize;
    }

    private adjustWindowSize(){
        if(this.dynamicWindowSize < this.maxWindowSize.getValue()){
            this.dynamicWindowSize = this.dynamicWindowSize * 2;
            if (this.dynamicWindowSize > this.maxWindowSize.getValue()){
                this.windowSize = this.maxWindowSize.getValue();
            }else{
                this.windowSize = this.dynamicWindowSize;
            }
        }
    }


    /* Main just for tests and debugging, remove in final version */
    public static void main(String[] args) {
        
    }
}