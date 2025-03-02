SELECT min(i_item_id),
       min(s_state),
       min(ss_quantity),
       min(ss_list_price),
       min(ss_coupon_amt),
       min(ss_sales_price),
       min(ss_item_sk),
       min(ss_ticket_number)
FROM store_sales,
     customer_demographics,
     date_dim,
     store,
     item
WHERE ss_sold_date_sk = d_date_sk
  AND ss_item_sk = i_item_sk
  AND ss_store_sk = s_store_sk
  AND ss_cdemo_sk = cd_demo_sk
  AND cd_gender = '@param0'
  AND cd_marital_status = '@param1'
  AND cd_education_status = '@param2'
  AND d_year = @param3
  AND s_state = '@param4'
  AND i_category = '@param5' ;