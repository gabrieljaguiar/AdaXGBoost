package edu.gjonas;

import moa.streams.generators.RandomRBFGenerator;
import moa.classifiers.AdaptativeXGBoost;

public class adaXGBoost {
    public void run(int maximumNumberInstances){
        System.out.println(maximumNumberInstances);
        RandomRBFGenerator stream = new RandomRBFGenerator();
        stream.prepareForUse();

        AdaptativeXGBoost learner = new AdaptativeXGBoost();
        learner.setModelContext(stream.getHeader());
        learner.prepareForUse();

    }

    public static void main(String[] args) {
        adaXGBoost exp = new adaXGBoost();
        exp.run(100);

    }
}
