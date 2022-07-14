package org.example;

import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Observable;
import com.hazelcast.jet.Util;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.pipeline.BatchStage;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.Sources;
import com.hazelcast.jet.python.PythonServiceConfig;

import java.nio.file.Path;
import java.util.concurrent.ThreadLocalRandom;

import static com.hazelcast.instance.impl.HazelcastInstanceFactory.shutdownAll;
import static com.hazelcast.jet.python.PythonTransforms.mapUsingPython;

/**
 * This example shows you how to invoke a Python function to process the
 * data in the Jet pipeline. The function gets a batch of items in a list
 * and must return a list of result items.
 * <p>
 * The provided code uses the NumPy library to transform the input list by
 * taking the square root of each element. It uses {@code
 * src/main/resources/python} in this project as the Python project
 * directory. There are two files there: {@code requirements.txt} that
 * declares NumPy as a dependency and {@code take_sqrt.py} that defines the
 * {@code transform_list} function that Jet will call with the pipeline
 * data.
 */
public class Python {

    private static final String RESULTS = "py_results";

    private static Pipeline buildPipeline(String baseDir) {
        Pipeline p = Pipeline.create();
        BatchStage<String> lines = p.readFrom(Sources.files("src/main/pythonmodels/archive/user1.txt"))
                //.withoutTimestamps()
                .apply(mapUsingPython(new PythonServiceConfig()
                        .setBaseDir(baseDir)
                        .setHandlerModule("take_sqrt")))
                .setLocalParallelism(2) // controls how many Python processes will be used
                .writeTo(Sinks.observable(RESULTS));
        return p;
    }

    public static void main(String[] args) {
        Path baseDir = Util.getFilePathOfClasspathResource("pythonmodels");
        Pipeline p = buildPipeline(baseDir.toString());

        JetInstance jet = Jet.bootstrappedInstance();
        try {
            Observable<String> observable = jet.getObservable(RESULTS);
            observable.addObserver(System.out::println);
            JobConfig config = new JobConfig().setName("python-model-mapping");
            jet.newJobIfAbsent(p, config).join();
        } finally {
            shutdownAll();
        }
    }
}
