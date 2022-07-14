package org.example;

import com.hazelcast.config.Config;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.jet.pipeline.*;

public class DataStream {
    public static void main(String[] args) {

        Pipeline p = Pipeline.create();
        p.setPreserveOrder(true);
        p.readFrom(Sources.files("python/archive/title-ratings-data.csv"))
                .filter(x=9)
                .setName("filter out AstroTurf")
                .writeTo("src/main/archive/title-ratings-data-sink-1.csv");

        BatchSource<String> leftSource = Sources.items("the", "quick", "brown", "fox");
        BatchSource<String> rightSource = Sources.items("jumps", "over", "the", "lazy", "dog");

        BatchStage<String> left = p.readFrom(leftSource);
        BatchStage<String> right = p.readFrom(rightSource);

        left.merge(right)
                .writeTo(Sinks.logger());

        HazelcastInstance hz = Hazelcast.bootstrappedInstance();

        hz.getJet().newJob(p);
    }
}
