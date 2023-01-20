#!/bin/bash
docker compose exec cli hz-cli submit \
    -c=hazelcast.platform.solutions.recommender.DummyRecommenderPipeline \
    -t=dev@hz   \
    -n=dummy-recommender \
    /project/recommender-pipeline/target/recommender-pipeline-1.0-SNAPSHOT.jar
