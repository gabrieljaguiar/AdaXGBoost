package moa.classifiers;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import org.jfree.util.ArrayUtils;

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
import scala.Array;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class AdaptativeXGBoost extends AbstractClassifier {
	private static final long serialVersionUID = 1L;
	
    
    public IntOption nEstimators = new IntOption("Number of Estimators", 'n', "Number of Estimators", 5, 1, Integer.MAX_VALUE);

    public FloatOption learningRate = new FloatOption("Learning Rate", 'l', "Learning Rate", 0.3, 0, Float.MAX_VALUE);

    public IntOption maxDepth = new IntOption("Max Depth", 'p', "Max Depth", 6, 1, Integer.MAX_VALUE);

    public IntOption maxWindowSize = new IntOption("Max windowSize", 'm', "Max Window size", 1000, 1, Integer.MAX_VALUE);

    public IntOption minWindowSize = new IntOption("Min WindowSize", 'w', "Min Window size", -1, 1, Integer.MAX_VALUE);

    public FlagOption detectDrift = new FlagOption("Detect Drift", 'd', "Detect Drift Flag");

    public StringOption uptadeStrategy = new StringOption ("Update Strategy", 'u', "Uptade Strategy", "replace");
    


    protected List<Instance> window;

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

    

    /* Constructor */
    @Override
    public void resetLearningImpl() {
        window = new ArrayList<Instance>();
        this.ensemble = new ArrayList<Booster>();
        if (this.uptadeStrategy.getValue() == this.REPLACE_STRATEGY){
            for (int i = 0; i < this.nEstimators.getValue(); i++){
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
        this.resetWindowSize();
        this.initMargin = 0.0f;
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
        
        this.numberOfAttr = instance.numAttributes();

        while (window.size() >= this.windowSize) {
            this.trainOnMiniBatch(window) // Needs to be improved. Select just 1 to WindowSize Instances;
            int i = 0;
            while (i < this.windowSize){
                window.remove(0);
                i++;
            }
            this.adjustWindowSize();
            if (this.detectDrift.isSet()){
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

    private void trainOnMiniBatch(List data){
        if (this.uptadeStrategy.getValue() == this.REPLACE_STRATEGY){
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
        }else{
            Booster booster = null;
			try {
				booster = this.trainBooster(data, this.ensemble.size());
			} catch (XGBoostError e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            if (this.ensemble.size() == this.nEstimators.getValue()){
                this.ensemble.remove(0);
            }
            this.ensemble.add(booster);
            this.samplesSeen = this.samplesSeen + data.size();

        }
    }

    private Booster trainBooster(List data, Integer len) throws XGBoostError{
        /* To be implemented */
    	
    	float[] instancesArray = flattenInstances(data);
    	//int[] labels = getLabels(data);
    	DMatrix d_mini_batch_train = new DMatrix(instancesArray,
                data.size(),
                this.numberOfAttr.intValue());
    	
    	float[] margins = new float[data.size()];
    	
    	Arrays.fill(margins, this.initMargin);
    	
    	/*for (int j=0; j< this.ensemble.size(); j++) {
    		this.sumElementWise(margins, this.ensemble.get(j).predict(d_mini_batch_train, true));
    	}*/
    	
    	
    	
    	
        return XGBoost.train(null, this.boostingParams, randomSeed, null, null, null);
    }
    
    private float[] sumElementWise(float[] a1, float[] a2) {
    	float [] a3 = new float[a1.length];
    	for (int i=0; i<a1.length;i++) {
    		a3[i] = a1[i] + a2[i];
    	}
    	
    	return a3;
    }

    private float[] flattenInstances(List data){
        float [] instArray = {};
        for (int i=0; i< data.size();i++){
            Instance inst = (Instance) data.get(i);
            instArray = this.concatArrays(instArray, this.arrayToFloat(inst.toDoubleArray()));
        }
        
        return instArray;

    }
    
    /*private getLabels() {
    	
    }*/
    
    private float[] arrayToFloat(double[] array) {
    	float [] newArray = new float[array.length];
    	for (int i=0; i<array.length;i++) {
    		newArray[i] = (float) array[i];
    	}
    	return newArray;
    }
    
    private float[] concatArrays (float[] a1, float[] a2) {
    	float[] result = new float[a1.length + a2.length];

    	System.arraycopy(a1, 0, result, 0, a1.length);
    	System.arraycopy(a2, 0, result, a1.length, a2.length);
    	
    	return result;
    }

    private void updateModelId(){
        this.modelIndex++;
        if (this.modelIndex == this.nEstimators.getValue()){
            this.modelIndex = 0;
        }
    }



    private void resetWindowSize(){
        if (this.minWindowSize.getValue()){
            this.dynamicWindowSize = this.minWindowSize.getValue();
        }else{
            this.dynamicWindowSize = this.maxWindowSize.getValue();
        }
        this.windowSize = this.dynamicWindowSize;
    }

    private void adjustWindowSize(){
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
    	
    	float[] mat = {0f,1f,2f,1f,3f,4f,5f,1f,6f,7f,8f,0f,9f,10f,11f,1f};
    	try {
	    	DMatrix d_mini_batch_train = new DMatrix(mat,
	                4,
	                4);
	    	float[] labels = d_mini_batch_train.getLabel();
	    	System.out.println(Arrays.toString(labels));
    	} catch(Exception e) {
    		e.printStackTrace();
    	}
    	/*0  0   1   2
    	1  3   4   5
    	2  6   7   8
    	3  9  10  11
    	
    	0  1
    	1  1
    	2  0
    	3  1*/


    }
}