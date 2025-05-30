CREATE OR REPLACE TABLE sa.freq_count
select
	status,
	word,
	[count]
from crono$csv(FileName='@@root\..\data\freq_count.csv') materialize into stg.tmp tmp