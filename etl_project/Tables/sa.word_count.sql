CREATE OR REPLACE TABLE sa.word_count
select
	status,
	word_count,
	sentiment_score,
	cluster
from crono$csv(FileName='@@root\..\data\word_count.csv') materialize into stg.tmp tmp