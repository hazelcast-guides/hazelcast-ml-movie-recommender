package com.hazelcast;

import com.hazelcast.config.Config;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.jet.Job;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.pipeline.BatchSource;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.SinkStage;
import com.hazelcast.jet.pipeline.Sinks;
import com.hazelcast.jet.pipeline.file.FileFormat;
import com.hazelcast.jet.pipeline.file.FileSources;
import com.hazelcast.jet.python.PythonServiceConfig;

import java.io.File;
import java.util.logging.Logger;

import static com.hazelcast.jet.python.PythonTransforms.mapUsingPythonBatch;

/**
 * Create a pipeline to run a Python movie recommendation function
 * and apply it to a .txt file containing movie name(s).
 * Output will be a file containing several recommendations
 * based off of similar cast members and common genres.
 */
public class RecPipelineRunner implements Runnable {

    private static final Logger LOG = Logger.getLogger(RecPipelineRunner.class.getName());
    private static final String RESULTS_DIR = "data/output/";

    private final String INPUT_FILE_PATH_ABS;

    private void nameOutputFile() {
        // Rename file to grokable name
        File oldOutputFile = new File(RESULTS_DIR + "/0");
        File newOutputFile = new File(RESULTS_DIR + "/recommendations.txt");

        oldOutputFile.renameTo(newOutputFile);
    }

    public RecPipelineRunner(String relativeInputFilepath) {
        File fileRef = new File(relativeInputFilepath);
        INPUT_FILE_PATH_ABS = fileRef.getAbsolutePath();

        LOG.info("Created RecPipelineRunner");
        LOG.info("input file path (relative) = " + relativeInputFilepath);
        LOG.info("input file path (absolute) = " + INPUT_FILE_PATH_ABS);
    }

    public static void main(String[] args) {
        RecPipelineRunner thisApp = new RecPipelineRunner("data/input/");
        thisApp.run();
    }

    @Override
    public void run() {
        // Create the Hazelcast runtime instance
        Config config = new Config();
        config.getJetConfig().setEnabled(true).setResourceUploadEnabled(true);
        HazelcastInstance hazelcast = Hazelcast.newHazelcastInstance(config);

        // Create the Hazelcast Jet pipeline
        Pipeline p = Pipeline.create();
        BatchSource<String> source = FileSources.files(INPUT_FILE_PATH_ABS)
                .format(FileFormat.lines())
                .build();
        SinkStage newStage = p.readFrom(source)
                .map(String::toLowerCase)
                .apply(mapUsingPythonBatch(new PythonServiceConfig()
                        .setBaseDir("src/main/python/")
                        .setHandlerModule("manyFromOneRec")
                        .setHandlerFunction("do_recommender")))
                .setLocalParallelism(1)
                .writeTo(Sinks.files(RESULTS_DIR));

        // Configure and create the Hazelcast Jet job using the Hazelcast instance
        JobConfig cfg = new JobConfig().setName("movie-recommendation");

        Job newJob = hazelcast.getJet().newJob(p, cfg);

        // Join to the new job to block this thread until parallel execution
        // of the job is over
        newJob.join();

        // rename the output file into something more reasonable
        nameOutputFile();

        // Shutdown the Hazelcast instance
        hazelcast.shutdown();
    }
}