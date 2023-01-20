package hazelcast.platform.solutions.recommender;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.datamodel.Tuple2;
import com.hazelcast.jet.pipeline.*;
import com.hazelcast.nio.serialization.genericrecord.GenericRecord;
import com.hazelcast.nio.serialization.genericrecord.GenericRecordBuilder;

import java.util.ArrayList;
import java.util.Map;

public class RecommenderPipeline {
    public static void main(String[] args) {
        Pipeline pipeline = DummyRecommenderPipeline.createPipeline();
        JobConfig jobConfig = new JobConfig();
        jobConfig.setName("dummy recommender");
        HazelcastInstance hz = Hazelcast.bootstrappedInstance();

        hz.getJet().newJob(pipeline, jobConfig);
    }

    private static final String REQUEST_MAP_NAME = "recommendation_request";
    private static final String RESPONSE_MAP_NAME = "recommendation_response";

    private static Pipeline createPipeline() {
        Pipeline pipeline = Pipeline.create();

        StreamStage<Map.Entry<String, String>> requests =
                pipeline.readFrom(Sources.<String, String>mapJournal(REQUEST_MAP_NAME, JournalInitialPosition.START_FROM_CURRENT))
                        .withIngestionTimestamps()
                        .setName("requests");




        recommendations.writeTo(Sinks.map(RESPONSE_MAP_NAME));

        return pipeline;
    }

}