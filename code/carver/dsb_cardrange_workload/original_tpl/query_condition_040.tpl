d_date > (CAST ('@param0' AS date) - interval '30 day')
d_date < (CAST ('@param0' AS date) + interval '30 day')
i_category = '@param1'
i_manager_id > @param2
i_manager_id < @param3
cs_wholesale_cost > @param4
cs_wholesale_cost < @param5
cr_reason_sk = @param6 
