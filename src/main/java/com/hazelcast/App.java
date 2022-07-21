package com.hazelcast;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.jet.pipeline.Pipeline;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        System.out.println("Starting...");
        System.exit(1);
        HazelcastInstance hzInstance = HazelcastClient.newHazelcastClient();
        hzInstance.shutdown();
    }
}
