---
NOTES ON HOW THE DATASET combined-labeled-tweets.txt WAS CREATED
Nate Chambers
---

All "positive" and "negative" tweets from Sanders data

All "positive", "negative", "neutral" tweets from Sentiment140.com data
    - changed neutral to objective

All "negative" tweets from ICWSM_Chen data
    - total of 201
    - both movies and persons included

130 "positive" tweets from ICWSM_Chen movies
130 "positive" tweets from ICWSM_Chen persons

400 "neutral" tweets from Sanders (changed label to "objective")
200 "objective" tweets from ICWSM_Chen movies
200 "objective" tweets from ICWSM_Chen persons


IMPORTANT
The movies/person data was target-specific sentiment. Many tweets contain sentiment, but it is not directed at the movie or person in the tweet. Those are labeled as objective. A general sentiment classifier will get many of these wrong.
