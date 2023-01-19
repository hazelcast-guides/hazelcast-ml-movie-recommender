## Hazelcast Movie Recommendation Pipeline Demo
___
Create a pipeline to run a Python movie recommendation function 
and apply it to a .txt file containing movie name(s). 
Output will be a file containing several recommendations
based off of similar cast members and common genres.

Please edit data/input/input.txt with the text of your movie.
(Non-English title films - use the original name. If the language uses logograms,
use romanized title.)

If there is a pre-existing recommendations.txt file in the output folder, running
new input will overwrite the file.

Developed using Hazelcast 5.1.2 and Python 3.7.9 as a virtual environment.


## Prerequisites

Before running this project, clone https://github.com/hazelcast-guides/spring-hazelcast-pipeline-dispatcher and 
install it locally by running `mvn clean install`. Verify that the version built matches the one needed by 
this project. See `hazelcast.pipeline.dispatcher.version` in `pom.xml`.  If it does not then check out the 
required version `git checkout n.n.n` and build it locally with `mvn clean install`.

## Additional Notes 
- Data from [Movinder's set](github.com/Movinder/movielens-imdb-exploration).
