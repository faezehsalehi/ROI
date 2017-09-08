set hive.groupby.orderby.position.alias=true;
SET mapreduce.map.memory.mb=6144;

INSERT OVERWRITE LOCAL DIRECTORY '/home/faezeh.salehi/data'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
SELECT advertiser_app,
  country,
  model,
  campaign_type,
  cb_campaign,
  l7_retention,
  SUM(d7_iap_revenue) AS d7_iap_revenue,
  SUM(d7_iap_count) AS d7_iap_count,
  SUM(pia_install) AS pia_install,
  SUM(d7_iap_revenue) / SUM(pia_install) AS d7_iap_revenue_per_install,
  SUM(d7_iap_count) / SUM(pia_install) AS d7_iap_count_per_install
FROM post_install.cohort_aggr_v2
WHERE install_dt >= '2017-04-15'and install_dt <= '2017-06-15'
GROUP BY 1, 2, 3, 4, 5, 6
;
