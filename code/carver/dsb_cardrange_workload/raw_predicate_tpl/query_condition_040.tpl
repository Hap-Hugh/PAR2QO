d_date BETWEEN (CAST ('@param0' AS date) - interval '30 day') AND (CAST ('@param0' AS date) + interval '30 day')
i_category = '@param1' AND i_manager_id BETWEEN @param2 AND @param3
cs_wholesale_cost BETWEEN @param4 AND @param5
cr_reason_sk = @param6 
