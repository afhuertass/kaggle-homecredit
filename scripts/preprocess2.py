import pandas as pd
import numpy as np
import featuretools as ft 

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb


df_train = pd.read_csv("../data/test/application_test.csv")
df_test = pd.read_csv("../data/test/application_test.csv")

df_train.head()
df_bureau = pd.read_csv("../data/bureau.csv")
df_bureau_bal = pd.read_csv("../data/bureau_balance.csv")
df_pos = pd.read_csv("../data/POS_CASH_balance.csv")
df_card = pd.read_csv("../data/credit_card_balance.csv")
df_prev = pd.read_csv("../data/previous_application.csv")
df_instal = pd.read_csv("../data/installments_payments.csv")

variable_types_train = {
    
    "NAME_CONTRACT_TYPE": ft.variable_types.Categorical , 
    "CODE_GENDER" : ft.variable_types.Categorical  , 
     "FLAG_OWN_CAR" : ft.variable_types.Categorical , 
    "FLAG_OWN_REALTY" : ft.variable_types.Categorical , 
     "FLAG_OWN_REALTY" : ft.variable_types.Categorical , 
    "CNT_CHILDREN" : ft.variable_types.Numeric , 
    "AMT_INCOME_TOTAL" : ft.variable_types.Numeric , 
     "AMT_INCOME_TOTAL" : ft.variable_types.Numeric , 
    "AMT_CREDIT" : ft.variable_types.Numeric , 
    "AMT_ANNUITY" : ft.variable_types.Numeric , 
    "AMT_GOODS_PRICE" : ft.variable_types.Numeric , 
    "NAME_TYPE_SUITE" : ft.variable_types.Categorical , 
    "NAME_INCOME_TYPE" : ft.variable_types.Categorical , 
    "NAME_EDUCATION_TYPE" : ft.variable_types.Categorical , 
    "NAME_FAMILY_STATUS": ft.variable_types.Categorical  , 
    "NAME_HOUSING_TYPE": ft.variable_types.Categorical , 
    "REGION_POPULATION_RELATIVE" : ft.variable_types.Numeric , 
    "DAYS_BIRTH" : ft.variable_types.Numeric , 
    "DAYS_EMPLOYED" : ft.variable_types.Numeric ,
    "DAYS_REGISTRATION" : ft.variable_types.Numeric ,
    "DAYS_ID_PUBLISH" : ft.variable_types.Numeric , 
    "OWN_CAR_AGE" : ft.variable_types.Numeric , 
    "FLAG_MOBIL" : ft.variable_types.Categorical , 
    "FLAG_EMP_PHONE" :ft.variable_types.Categorical , 
    "FLAG_WORK_PHONE": ft.variable_types.Categorical ,
    "FLAG_CONT_MOBILE" : ft.variable_types.Categorical , 
    "FLAG_PHONE" :  ft.variable_types.Categorical , 
    "FLAG_EMAIL" : ft.variable_types.Categorical , 
    "OCCUPATION_TYPE" : ft.variable_types.Categorical , 
    "CNT_FAM_MEMBERS" : ft.variable_types.Numeric , 
    "REGION_RATING_CLIENT" : ft.variable_types.Ordinal , 
    "REGION_RATING_CLIENT_W_CITY" : ft.variable_types.Ordinal , 
    "WEEKDAY_APPR_PROCESS_START": ft.variable_types.Categorical , 
    "HOUR_APPR_PROCESS_START" : ft.variable_types.Categorical , 
    "REG_REGION_NOT_LIVE_REGION" : ft.variable_types.Categorical , 
    "REG_REGION_NOT_WORK_REGION" : ft.variable_types.Categorical , 
    "LIVE_REGION_NOT_WORK_REGION" : ft.variable_types.Categorical , 
    "REG_CITY_NOT_LIVE_CITY" : ft.variable_types.Categorical , 
    "REG_CITY_NOT_WORK_CITY" : ft.variable_types.Categorical , 
    "LIVE_CITY_NOT_WORK_CITY" : ft.variable_types.Categorical , 
    "ORGANIZATION_TYPE" : ft.variable_types.Categorical , 
    "EXT_SOURCE_1" : ft.variable_types.Numeric , 
    "EXT_SOURCE_2" : ft.variable_types.Numeric , 
    "EXT_SOURCE_3" : ft.variable_types.Numeric , 
    "APARTMENTS_AVG" : ft.variable_types.Numeric , 
    "BASEMENTAREA_AVG" : ft.variable_types.Numeric ,  
    "YEARS_BEGINEXPLUATATION_AVG" : ft.variable_types.Numeric ,  
    "YEARS_BUILD_AVG" : ft.variable_types.Numeric ,  
    "COMMONAREA_AVG" : ft.variable_types.Numeric ,  
    "ELEVATORS_AVG" : ft.variable_types.Numeric ,  
    "ENTRANCES_AVG" : ft.variable_types.Numeric ,  
    "FLOORSMAX_AVG" : ft.variable_types.Numeric ,  
    "FLOORSMIN_AVG" : ft.variable_types.Numeric ,  
    "LANDAREA_AVG" : ft.variable_types.Numeric ,  
    "LIVINGAPARTMENTS_AVG" : ft.variable_types.Numeric ,  
    "LIVINGAREA_AVG" : ft.variable_types.Numeric ,  
    "NONLIVINGAPARTMENTS_AVG" : ft.variable_types.Numeric ,  
    "NONLIVINGAREA_AVG" : ft.variable_types.Numeric ,  
    "APARTMENTS_MODE" : ft.variable_types.Numeric ,  
    "BASEMENTAREA_MODE" : ft.variable_types.Numeric ,  
    "YEARS_BEGINEXPLUATATION_MODE" : ft.variable_types.Numeric ,  
    "YEARS_BUILD_MODE" : ft.variable_types.Numeric ,  
    "COMMONAREA_MODE" : ft.variable_types.Numeric ,  
    "ELEVATORS_MODE" : ft.variable_types.Numeric ,  
    "ENTRANCES_MODE" : ft.variable_types.Numeric ,  
    "FLOORSMAX_MODE" : ft.variable_types.Numeric ,  
    "FLOORSMIN_MODE" : ft.variable_types.Numeric ,  
    "LANDAREA_MODE" : ft.variable_types.Numeric ,  
    "LIVINGAPARTMENTS_MODE" : ft.variable_types.Numeric ,  
    "LIVINGAREA_MODE" : ft.variable_types.Numeric ,
    "NONLIVINGAPARTMENTS_MODE" : ft.variable_types.Numeric ,  
    "NONLIVINGAREA_MODE" : ft.variable_types.Numeric ,  
    "APARTMENTS_MEDI" : ft.variable_types.Numeric ,  
    "BASEMENTAREA_MEDI" : ft.variable_types.Numeric ,
    "YEARS_BEGINEXPLUATATION_MEDI" : ft.variable_types.Numeric ,  
    "YEARS_BUILD_MEDI" : ft.variable_types.Numeric ,  
    "COMMONAREA_MEDI" : ft.variable_types.Numeric ,  
    "ELEVATORS_MEDI" : ft.variable_types.Numeric ,
    "ENTRANCES_MEDI" : ft.variable_types.Numeric ,  
    "FLOORSMAX_MEDI" : ft.variable_types.Numeric ,
    "FLOORSMIN_MEDI" : ft.variable_types.Numeric ,  
    "LANDAREA_MEDI" : ft.variable_types.Numeric ,
    "LIVINGAPARTMENTS_MEDI" : ft.variable_types.Numeric ,  
    "LIVINGAREA_MEDI" : ft.variable_types.Numeric ,
    "NONLIVINGAPARTMENTS_MEDI" : ft.variable_types.Numeric ,
    "NONLIVINGAREA_MEDI" : ft.variable_types.Numeric ,  
    "FONDKAPREMONT_MODE" : ft.variable_types.Categorical ,
    "HOUSETYPE_MODE" : ft.variable_types.Categorical ,  
    "TOTALAREA_MODE" : ft.variable_types.Numeric ,
    "WALLSMATERIAL_MODE" : ft.variable_types.Categorical ,
    "EMERGENCYSTATE_MODE" : ft.variable_types.Categorical ,  
    "OBS_30_CNT_SOCIAL_CIRCLE" : ft.variable_types.Numeric ,
    "DEF_30_CNT_SOCIAL_CIRCLE" : ft.variable_types.Numeric ,
    "OBS_60_CNT_SOCIAL_CIRCLE" : ft.variable_types.Numeric ,  
    "DEF_60_CNT_SOCIAL_CIRCLE" : ft.variable_types.Numeric ,
    "DAYS_LAST_PHONE_CHANGE" : ft.variable_types.Numeric ,
    "FLAG_DOCUMENT_2" : ft.variable_types.Categorical ,  
    "FLAG_DOCUMENT_3" : ft.variable_types.Categorical ,  
    "FLAG_DOCUMENT_4" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_5" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_6" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_7" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_8" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_9" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_10" : ft.variable_types.Categorical ,
    "FLAG_DOCUMENT_11" : ft.variable_types.Categorical ,  
    "FLAG_DOCUMENT_12" : ft.variable_types.Categorical ,  
    "FLAG_DOCUMENT_13" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_14" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_15" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_16" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_17" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_18" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_19" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_20" : ft.variable_types.Categorical , 
    "FLAG_DOCUMENT_21" : ft.variable_types.Categorical , 
    "AMT_REQ_CREDIT_BUREAU_HOUR" : ft.variable_types.Numeric , 
    
    "AMT_REQ_CREDIT_BUREAU_DAY" : ft.variable_types.Numeric , 
    "AMT_REQ_CREDIT_BUREAU_WEEK" : ft.variable_types.Numeric , 
    "AMT_REQ_CREDIT_BUREAU_MON" : ft.variable_types.Numeric , 
    "AMT_REQ_CREDIT_BUREAU_QRT" : ft.variable_types.Numeric , 
    "AMT_REQ_CREDIT_BUREAU_YEAR" : ft.variable_types.Numeric , 
}

variable_types_bureau  = {
     "CREDIT_ACTIVE" : ft.variable_types.Categorical , 
    "CREDIT_CURRENCY" : ft.variable_types.Categorical , 
    "DAYS_CREDIT" : ft.variable_types.Numeric , 
    "CREDIT_DAY_OVERDUE" : ft.variable_types.Numeric , 
    "DAYS_CREDIT_ENDDATE": ft.variable_types.Numeric ,
    "DAYS_ENDDATE_FACT": ft.variable_types.Numeric ,
    "AMT_CREDIT_MAX_OVERDUE": ft.variable_types.Numeric ,
    "CNT_CREDIT_PROLONG": ft.variable_types.Numeric ,
    "CNT_CREDIT_PROLONG": ft.variable_types.Numeric ,
    "AMT_CREDIT_SUM": ft.variable_types.Numeric ,
    "AMT_CREDIT_SUM_DEBT": ft.variable_types.Numeric ,
    "AMT_CREDIT_SUM_LIMIT": ft.variable_types.Numeric ,
    "AMT_CREDIT_SUM_OVERDUE": ft.variable_types.Numeric ,
    "CREDIT_TYPE": ft.variable_types.Categorical ,
    "DAYS_CREDIT_UPDATE": ft.variable_types.Numeric ,
    "AMT_ANNUITY": ft.variable_types.Numeric ,
}
variable_types_bureaubal = {
    "MONTHS_BALANCE" : ft.variable_types.Numeric , 
    "STATUS": ft.variable_types.Categorical
    
}
variable_types_pos = {
    "MONTHS_BALANCE" : ft.variable_types.Numeric , 
    "CNT_INSTALMENT" : ft.variable_types.Numeric , 
    "CNT_INSTALMENT_FUTURE" :  ft.variable_types.Numeric , 
    "NAME_CONTRACT_STATUS" : ft.variable_types.Categorical , 
    "SK_DPD" : ft.variable_types.Numeric , 
    "SK_DPD_DEF" : ft.variable_types.Numeric 

}
variable_types_card = {
    
    "MONTHS_BALANCE" : ft.variable_types.Numeric , 
    "AMT_BALANCE" : ft.variable_types.Numeric , 
    "AMT_CREDIT_LIMIT_ACTUAL" : ft.variable_types.Numeric , 
    "AMT_DRAWINGS_ATM_CURRENT" : ft.variable_types.Numeric , 
    "AMT_DRAWINGS_CURRENT" : ft.variable_types.Numeric , 
    "AMT_DRAWINGS_OTHER_CURRENT" : ft.variable_types.Numeric ,  
     "AMT_DRAWINGS_POS_CURRENT" : ft.variable_types.Numeric ,  
     "AMT_INST_MIN_REGULARITY" : ft.variable_types.Numeric ,  
     "AMT_PAYMENT_CURRENT" : ft.variable_types.Numeric ,  
     "AMT_PAYMENT_TOTAL_CURRENT" : ft.variable_types.Numeric ,  
     "AMT_RECEIVABLE_PRINCIPAL" : ft.variable_types.Numeric ,  
    "AMT_RECIVABLE" : ft.variable_types.Numeric ,  
    "AMT_TOTAL_RECEIVABLE" : ft.variable_types.Numeric ,  
    "CNT_DRAWINGS_ATM_CURRENT" : ft.variable_types.Numeric ,  
    "CNT_DRAWINGS_CURRENT" : ft.variable_types.Numeric ,  
    "CNT_DRAWINGS_OTHER_CURRENT" : ft.variable_types.Numeric ,  
      "CNT_DRAWINGS_POS_CURRENT" : ft.variable_types.Numeric , 
    "CNT_INSTALMENT_MATURE_CUM" :  ft.variable_types.Numeric , 
    "SK_DPD" : ft.variable_types.Numeric , 
    "SK_DPD_DEF" : ft.variable_types.Numeric , 
     
}
variable_types_prev = {
    "SK_ID_PREV" : ft.variable_types.Index , 
    "NAME_CONTRACT_TYPE" : ft.variable_types.Categorical , 
    "AMT_ANNUITY" : ft.variable_types.Numeric , 
    "AMT_APPLICATION" : ft.variable_types.Numeric , 
    "AMT_CREDIT" : ft.variable_types.Numeric , 
     "AMT_DOWN_PAYMENT" : ft.variable_types.Numeric , 
     "AMT_GOODS_PRICE" : ft.variable_types.Numeric , 
     "WEEKDAY_APPR_PROCESS_START" : ft.variable_types.Categorical , 
     "HOUR_APPR_PROCESS_START" : ft.variable_types.Categorical , 
     "FLAG_LAST_APPL_PER_CONTRACT" : ft.variable_types.Categorical , 
     "NFLAG_LAST_APPL_IN_DAY" : ft.variable_types.Categorical , 
     "RATE_DOWN_PAYMENT" : ft.variable_types.Numeric , 
     "RATE_INTEREST_PRIMARY" : ft.variable_types.Numeric , 
    "RATE_INTEREST_PRIVILEGED" : ft.variable_types.Numeric , 
     "NAME_CASH_LOAN_PURPOSE" : ft.variable_types.Categorical , 
    "NAME_CONTRACT_STATUS" : ft.variable_types.Categorical , 
     "DAYS_DECISION" : ft.variable_types.Numeric , 
     "NAME_PAYMENT_TYPE" : ft.variable_types.Categorical , 
    "CODE_REJECT_REASON" : ft.variable_types.Categorical , 
     "NAME_TYPE_SUITE" : ft.variable_types.Categorical , 
     "NAME_CLIENT_TYPE" : ft.variable_types.Categorical , 
    "NAME_GOODS_CATEGORY" : ft.variable_types.Categorical , 
     "NAME_PORTFOLIO" : ft.variable_types.Categorical , 
     "NAME_PRODUCT_TYPE" : ft.variable_types.Categorical , 
    "CHANNEL_TYPE" : ft.variable_types.Categorical , 
     "SELLERPLACE_AREA" : ft.variable_types.Categorical, 
     "NAME_SELLER_INDUSTRY" : ft.variable_types.Categorical , 
    "CNT_PAYMENT" : ft.variable_types.Numeric , 
     "NAME_YIELD_GROUP" : ft.variable_types.Categorical , 
     "PRODUCT_COMBINATION" : ft.variable_types.Categorical , 
     "DAYS_FIRST_DRAWING" : ft.variable_types.Numeric , 
     "DAYS_FIRST_DUE" : ft.variable_types.Numeric , 
     "DAYS_LAST_DUE_1ST_VERSION" : ft.variable_types.Numeric , 
     "DAYS_LAST_DUE" : ft.variable_types.Numeric , 
     "DAYS_TERMINATION" : ft.variable_types.Numeric , 
     "NFLAG_INSURED_ON_APPROVAL" : ft.variable_types.Categorical , 

}
variable_types_instal = {
    
    "NUM_INSTALMENT_VERSION" :  ft.variable_types.Numeric ,
    "DAYS_INSTALMENT" :  ft.variable_types.Numeric ,
    "DAYS_ENTRY_PAYMENT" :  ft.variable_types.Numeric ,
    "AMT_INSTALMENT" :  ft.variable_types.Numeric ,
    "AMT_PAYMENT" :  ft.variable_types.Numeric ,
}


entity_train = ft.EntitySet( id = "train" )
entity_train = entity_train.entity_from_dataframe(
entity_id = "bureau" , dataframe = df_bureau  , index ="SK_ID_BUREAU" , variable_types = variable_types_bureau
)

entity_train = entity_train.entity_from_dataframe(
entity_id = "bureau_bal" , dataframe = df_bureau_bal , variable_types = variable_types_bureaubal , index = "bureau_bal_index" , make_index = True)


print("Adding relationships")
r_1 = ft.Relationship( entity_train["bureau"]["SK_ID_BUREAU"] , entity_train["bureau_bal"]["SK_ID_BUREAU"] )

entity_train = entity_train.add_relationship(  r_1 ) 

features, feature_names = ft.dfs(entityset = entity_train, target_entity = 'bureau', 
                                 agg_primitives = ['mean', 'max', 'last' , 'std' ], max_depth = 3 )

print( features.head() )
print( features.shape )

features.to_csv("../data/new_bureau.csv" )