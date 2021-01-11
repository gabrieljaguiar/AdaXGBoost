package edu.gjonas;

import com.yahoo.labs.samoa.instances.Instance;
import moa.core.InstanceExample;
import moa.streams.generators.RandomRBFGenerator;
import moa.classifiers.AdaptativeXGBoost;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.core.TimingUtils;

public class adaXGBoost {
    public void run(int maximumNumberInstances){
        System.out.println(maximumNumberInstances);
        RandomRBFGenerator stream = new RandomRBFGenerator();
        stream.prepareForUse();

        AdaptativeXGBoost learner = new AdaptativeXGBoost();
        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

        int numberInstances = 0;

        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();

        WindowClassificationPerformanceEvaluator evaluator = new WindowClassificationPerformanceEvaluator();

        while (stream.hasMoreInstances() && numberInstances < maximumNumberInstances) {
            InstanceExample example = stream.nextInstance();
            Instance trainInst = example.getData();

            evaluator.addResult(example, learner.getVotesForInstance(trainInst));
            // test-then-train instance by instance
            learner.trainOnInstance(trainInst);

            numberInstances++;
        }

        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);

        System.out.println(numberInstances + " instances processed with " + evaluator.getFractionCorrectlyClassified()*100 + "% accuracy in " + time + " seconds.");

    }

    public static void main(String[] args) {
        adaXGBoost exp = new adaXGBoost();
        exp.run(100000);

    }
}
