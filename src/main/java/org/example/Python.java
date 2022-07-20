package org.example;

import com.hazelcast.jet.Jet;
import com.hazelcast.jet.JetInstance;
import com.hazelcast.jet.Observable;
import com.hazelcast.jet.Util;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.pipeline.*;
import com.hazelcast.jet.pipeline.file.FileSources;
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

    private static final String RESULTS = "recResults.csv";

    public static void main(String[] args) {
        Pipeline p = Pipeline.create();
        BatchSource<String> source = FileSources.files("src/main/python/archive/movie.txt")
                .build();
        BatchStage<String> newStage = p.readFrom(source)
                .apply(mapUsingPython(new PythonServiceConfig()
                        .setBaseDir("python/")
                        .setHandlerModule("manyFromOneRec")))
                .setLocalParallelism(1)
                .writeTo(Sinks.files(RESULTS));
        JobConfig cfg = new JobConfig().setName("python-function");
        Jet.bootstrappedInstance().newJob(p, cfg);
    }
}