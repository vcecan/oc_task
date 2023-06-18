import pandas as pd
import pickle
import lightgbm
import numpy as np
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))

pages_path = os.path.join(current_dir, 'pages')
sys.path.append(pages_path)

# from page1 import age,gender,score,credit_limit,income,bnr40,offer_crab,offer_delfin,offer_pinguin,produs,other_credits,comission

def predict(age,gender,score,credit_limit,income,bnr40,offer_crab,offer_delfin,offer_pinguin,produs,other_credits,comission):
    model_path = 'model/'
    with open(model_path + 'credit_risk_model.pkl', 'rb') as file:
        model = pickle.load(file)

    columns = ['Crab', 'Dolphin', 'Penguin', 'F', 'M', '0','5', '7','9','Age_23', 'Age_24_29', 'Age_30_34', 'Age_35_40', 'Age_41_46', 'Age_47_51', 'Age_52_57',
                     'Age_58_62', 'Age_63_68', 'Age_69_74', 'CL_491_925', 'CL_925_1350', 'CL_2625_3050',
                     'CL_3900_4325', 'CL_3050_3475', 'CL_5600_6025', 'CL_1775_2200', 'CL_1350_1775', 'CL_2200_2625',
                     'CL_3475_3900', 'CL_8575_9000', 'CL_4325_4750', 'CL_4750_5175', 'CL_6875_7300', 'CL_5175_5600',
                     'CL_6025_6450', 'CL_6450_6875', 'CL_7300_7725', 'CL_8150_8575', 'CL_7725_8150', 'score_558_638',
                     'score_638_718', 'score_398_478', 'score_n1_78', 'score_718_798', 'score_78_158', 'score_158_238',
                     'score_238_318', 'score_318_398', 'ANAFIncome_2620_2845', 'ANAFIncome_3231_3432', 'ANAFIncome_4890_5285',
                     'ANAFIncome_3024_3231', 'ANAFIncome_n0_1790', 'ANAFIncome_1790_2190', 'ANAFIncome_6397_7354',
                     'ANAFIncome_4322_4594', 'ANAFIncome_3857_4081', 'ANAFIncome_2396_2620', 'ANAFIncome_9838_122794', 'ANAFIncome_4081_4322', 'ANAFIncome_4594_4890', 'ANAFIncome_5765_6397', 'ANAFIncome_7354_9838', 'ANAFIncome_2190_2396', 'ANAFIncome_5285_5765',
                     'ANAFIncome_3630_3857', 'ANAFIncome_3432_3630', 'ANAFIncome_2845_3024', 'TotalLoanPayments_n01_148', 'TotalLoanPayments_621_746', 'TotalLoanPayments_2821_3744', 'TotalLoanPayments_148_277', 'TotalLoanPayments_1670_1920', 'TotalLoanPayments_1282_1455',
                     'TotalLoanPayments_506_621', 'TotalLoanPayments_277_388', 'TotalLoanPayments_388_506', 'TotalLoanPayments_2277_2821', 'TotalLoanPayments_996_1130', 'TotalLoanPayments_1130_1282', 'TotalLoanPayments_1455_1670', 'TotalLoanPayments_859_996', 'TotalLoanPayments_1920_2277',
                     'TotalLoanPayments_3744_103920', 'TotalLoanPayments_746_859', 'BNR40Available_n108_0', 'BNR40Available_816_944', 'BNR40Available_68_218', 'BNR40Available_598_703', 'BNR40Available_n704_n356', 'BNR40Available_n356_n108', 'BNR40Available_n101462_1768', 'BNR40Available_n1768_1128',
                     'BNR40Available_1264_1522', 'BNR40Available_404_501', 'BNR40Available_2023_35720', 'BNR40Available_501_598', 'BNR40Available_n1128_n704', 'BNR40Available_1522_2023', 'BNR40Available_703_816', 'BNR40Available_307_404', 'BNR40Available_218_307', 'BNR40Available_1080_1264', 'BNR40Available_944_1080',
                     'BNR40Available_0_68', 'OfferCrab_800_1000', 'OfferCrab_200_400', 'OfferCrab_0_200', 'OfferCrab_n1000_0', 'OfferCrab_500_700', 'OfferCrab_700_800', 'OfferCrab_400_500', 'OfferCrab_1000_1500', 'OfferDolphin_0_1400', 'OfferDolphin_3200_4000', 'OfferDolphin_2400_3000', 'OfferDolphin_4920_6000', 'OfferDolphin_6000_9000',
                     'OfferDolphin_3100_3200', 'OfferDolphin_1400_2400', 'OfferDolphin_4000_4920', 'OfferDolphin_3000_3100', 'OfferPenguin_100700', 'OfferPenguin_1300_1700', 'OfferPenguin_n100_100', 'OfferPenguin_2600_3200', 'OfferPenguin_4100_4500', 'OfferPenguin_1700_2100', 'OfferPenguin_3200_4100', 'OfferPenguin_2100_2600', 'OfferPenguin_700_1000','OfferPenguin_1000_1300']

    new_df=pd.DataFrame(columns=columns)



    #for column in new_df.columns:
    #     column_name_parts = column.split('_')
    #     if len(column_name_parts) == 3 and column_name_parts[0] == 'OfferPenguin':
    #         range_start, range_end = column_name_parts[1],column_name_parts[2]
    #         if range_start < offer_pinguin <= range_end:
    #             new_df[column] = 1




    new_df.loc[0] = [0] * len(new_df.columns)

    if gender==1:
        new_df['M']=1
    elif gender == 0:
        new_df["F"]=1

    if produs=="Crab":
        new_df['Crab']=1
    elif produs=="Delfin":
        new_df['Dolphin']=1
    elif produs == "Pinguin":
        new_df['Penguin']=1

    if comission=='0':
        new_df['0']=1
    elif comission=='5':
        new_df['5']=1
    elif comission=='7':
        new_df['7']=1
    elif comission=='9':
        new_df['9']=1


    if age <= 23:
        new_df['Age_23'] = 1
    elif 23 < age <= 29:
        new_df['Age_24_29'] = 1
    elif 29 < age < 35:
        new_df['Age_30_34'] = 1
    elif 35 <= age <= 40:
        new_df['Age_35_40'] = 1
    elif 40 < age < 47:
        new_df['Age_41_46'] = 1
    elif 47 <= age < 52:
        new_df['Age_47_51'] = 1
    elif 51 < age < 58:
        new_df['Age_52_57'] = 1
    elif 57 < age <=62 :
        new_df['Age_58_62'] = 1
    elif 62 < age <= 68:
        new_df['Age_63_68'] = 1
    elif 68 < age :
        new_df['Age_69_74'] = 1


    if 491 < credit_limit <= 925:
        new_df['CL_491_925'] = 1
    elif 925 < credit_limit <= 1350:
        new_df['CL_925_1350'] = 1
    elif 2625 < credit_limit <= 3050:
        new_df['CL_2625_3050'] = 1
    elif 3900 < credit_limit <= 4325:
        new_df['CL_3900_4325'] = 1
    elif 3050 < credit_limit <= 3475:
        new_df['CL_3050_3475'] = 1
    elif 5600 < credit_limit <= 6025:
        new_df['CL_5600_6025'] = 1
    elif 1775 < credit_limit <= 2200:
        new_df['CL_1775_2200'] = 1
    elif 1350 < credit_limit <= 1775:
        new_df['CL_1350_1775'] = 1
    elif 2200 < credit_limit <= 2625:
        new_df['CL_2200_2625'] = 1
    elif 3475 < credit_limit <= 3900:
        new_df['CL_3475_3900'] = 1
    elif 8575 < credit_limit <= 9000:
        new_df['CL_8575_9000'] = 1
    elif 4325 < credit_limit <= 4750:
        new_df['CL_4325_4750'] = 1
    elif 4750 < credit_limit <= 5175:
        new_df['CL_4750_5175'] = 1
    elif 6875 < credit_limit <= 7300:
        new_df['CL_6875_7300'] = 1
    elif 5175 < credit_limit <= 5600:
        new_df['CL_5175_5600'] = 1
    elif 6025 < credit_limit <= 6450:
        new_df['CL_6025_6450'] = 1
    elif 6450 < credit_limit <= 6875:
        new_df['CL_6450_6875'] = 1
    elif 7300 < credit_limit <= 7725:
        new_df['CL_7300_7725'] = 1
    elif 8150 < credit_limit <= 8575:
        new_df['CL_8150_8575'] = 1
    elif 7725 < credit_limit:
        new_df['CL_7725_8150'] = 1

    if 558 < score <= 638:
        new_df['score_558_638'] = 1
    elif 638 < score <= 718:
        new_df['score_638_718'] = 1
    elif 398 < score <= 478:
        new_df['score_398_478'] = 1
    elif -1 < score <= 78:
        new_df['score_n1_78'] = 1
    elif 718 < score <= 860:
        new_df['score_718_798'] = 1
    elif 78 < score <= 158:
        new_df['score_78_158'] = 1
    elif 158 < score <= 238:
        new_df['score_158_238'] = 1
    elif 238 < score <= 318:
        new_df['score_238_318'] = 1
    elif 318 < score <= 398:
        new_df['score_318_398'] = 1


    # Income conditionals
    if 2620 < income <= 2845:
        new_df['ANAFIncome_2620_2845'] = 1
    elif 3231 < income <= 3432:
        new_df['ANAFIncome_3231_3432'] = 1
    elif 4890 < income <= 5285:
        new_df['ANAFIncome_4890_5285'] = 1
    elif 3024 < income <= 3231:
        new_df['ANAFIncome_3024_3231'] = 1
    elif -1 < income <= 1790:
        new_df['ANAFIncome_n0_1790'] = 1
    elif 1790 < income <= 2190:
        new_df['ANAFIncome_1790_2190'] = 1
    elif 6397 < income <= 7354:
        new_df['ANAFIncome_6397_7354'] = 1
    elif 4322 < income <= 4594:
        new_df['ANAFIncome_4322_4594'] = 1
    elif 3857 < income <= 4081:
        new_df['ANAFIncome_3857_4081'] = 1
    elif 2396 < income <= 2620:
        new_df['ANAFIncome_2396_2620'] = 1
    elif 9838 < income :
        new_df['ANAFIncome_9838_122794'] = 1
    elif 4081 < income <= 4322:
        new_df['ANAFIncome_4081_4322'] = 1
    elif 4594 < income <= 4890:
        new_df['ANAFIncome_4594_4890'] = 1
    elif 5765 < income <= 6397:
        new_df['ANAFIncome_5765_6397'] = 1
    elif 7354 < income <= 9838:
        new_df['ANAFIncome_7354_9838'] = 1
    elif 2190 < income <= 2396:
        new_df['ANAFIncome_2190_2396'] = 1
    elif 5285 < income <= 5765:
        new_df['ANAFIncome_5285_5765'] = 1
    elif 3630 < income <= 3857:
        new_df['ANAFIncome_3630_3857'] = 1
    elif 3432 < income <= 3630:
        new_df['ANAFIncome_3432_3630'] = 1
    elif 2845 < income <= 3024:
        new_df['ANAFIncome_2845_3024'] = 1


    # Other Credits conditionals
    if -1 < other_credits <= 148:
        new_df['TotalLoanPayments_n01_148'] = 1
    elif 621 < other_credits <= 746:
        new_df['TotalLoanPayments_621_746'] = 1
    elif 2821 < other_credits <= 3744:
        new_df['TotalLoanPayments_2821_3744'] = 1
    elif 148 < other_credits <= 277:
        new_df['TotalLoanPayments_148_277'] = 1
    elif 1670 < other_credits <= 1920:
        new_df['TotalLoanPayments_1670_1920'] = 1
    elif 1282 < other_credits <= 1455:
        new_df['TotalLoanPayments_1282_1455'] = 1
    elif 506 < other_credits <= 621:
        new_df['TotalLoanPayments_506_621'] = 1
    elif 277 < other_credits <= 388:
        new_df['TotalLoanPayments_277_388'] = 1
    elif 388 < other_credits <= 506:
        new_df['TotalLoanPayments_388_506'] = 1
    elif 2277 < other_credits <= 2821:
        new_df['TotalLoanPayments_2277_2821'] = 1
    elif 996 < other_credits <= 1130:
        new_df['TotalLoanPayments_996_1130'] = 1
    elif 1130 < other_credits <= 1282:
        new_df['TotalLoanPayments_1130_1282'] = 1
    elif 1455 < other_credits <= 1670:
        new_df['TotalLoanPayments_1455_1670'] = 1
    elif 859 < other_credits <= 996:
        new_df['TotalLoanPayments_859_996'] = 1
    elif 1920 < other_credits <= 2277:
        new_df['TotalLoanPayments_1920_2277'] = 1
    elif 3744 < other_credits <= 103920:
        new_df['TotalLoanPayments_3744_103920'] = 1
    elif 746 < other_credits <= 859:
        new_df['TotalLoanPayments_746_859'] = 1


    # BNR40 conditionals
    if -108 <= bnr40 < 0:
        new_df['BNR40Available_n108_0'] = 1
    elif 816 < bnr40 <= 944:
        new_df['BNR40Available_816_944'] = 1
    elif 68 < bnr40 <= 218:
        new_df['BNR40Available_68_218'] = 1
    elif 598 < bnr40 <= 703:
        new_df['BNR40Available_598_703'] = 1
    elif -704 < bnr40 <= -356:
        new_df['BNR40Available_n704_n356'] = 1
    elif -356 < bnr40 <= -108:
        new_df['BNR40Available_n356_n108'] = 1
    elif -101462 < bnr40 <= 1768:
        new_df['BNR40Available_n101462_1768'] = 1
    elif -1768 < bnr40 <= 1128:
        new_df['BNR40Available_n1768_1128'] = 1
    elif 1264 < bnr40 <= 1522:
        new_df['BNR40Available_1264_1522'] = 1
    elif 404 < bnr40 <= 501:
        new_df['BNR40Available_404_501'] = 1
    elif 2023 < bnr40 <= 35720:
        new_df['BNR40Available_2023_35720'] = 1
    elif 501 < bnr40 <= 598:
        new_df['BNR40Available_501_598'] = 1
    elif -1128 < bnr40 <= -704:
        new_df['BNR40Available_n1128_n704'] = 1
    elif 1522 < bnr40 <= 2023:
        new_df['BNR40Available_1522_2023'] = 1
    elif 703 < bnr40 <= 816:
        new_df['BNR40Available_703_816'] = 1
    elif 307 < bnr40 <= 404:
        new_df['BNR40Available_307_404'] = 1
    elif 218 < bnr40 <= 307:
        new_df['BNR40Available_218_307'] = 1
    elif 1080 < bnr40 <= 1264:
        new_df['BNR40Available_1080_1264'] = 1
    elif 944 < bnr40 <= 1080:
        new_df['BNR40Available_944_1080'] = 1
    elif 0 < bnr40 <= 68:
        new_df['BNR40Available_0_68'] = 1



    # OfferCrab conditionals
    if 800 < offer_crab <= 1000:
        new_df['OfferCrab_800_1000'] = 1
    elif 200 < offer_crab <= 400:
        new_df['OfferCrab_200_400'] = 1
    elif 0 < offer_crab <= 200:
        new_df['OfferCrab_0_200'] = 1
    elif -1000 < offer_crab <= 0:
        new_df['OfferCrab_n1000_0'] = 1
    elif 500 < offer_crab <= 700:
        new_df['OfferCrab_500_700'] = 1
    elif 700 < offer_crab <= 800:
        new_df['OfferCrab_700_800'] = 1
    elif 400 < offer_crab <= 500:
        new_df['OfferCrab_400_500'] = 1
    elif 1000 < offer_crab <= 1500:
        new_df['OfferCrab_1000_1500'] = 1



    # OfferDolphin conditionals
    if 0 < offer_delfin <= 1400:
        new_df['OfferDolphin_0_1400'] = 1
    elif 3200 < offer_delfin <= 4000:
        new_df['OfferDolphin_3200_4000'] = 1
    elif 2400 < offer_delfin <= 3000:
        new_df['OfferDolphin_2400_3000'] = 1
    elif 4920 < offer_delfin <= 6000:
        new_df['OfferDolphin_4920_6000'] = 1
    elif 6000 < offer_delfin <= 9000:
        new_df['OfferDolphin_6000_9000'] = 1
    elif 3100 < offer_delfin <= 3200:
        new_df['OfferDolphin_3100_3200'] = 1
    elif 1400 < offer_delfin <= 2400:
        new_df['OfferDolphin_1400_2400'] = 1
    elif 4000 < offer_delfin <= 4920:
        new_df['OfferDolphin_4000_4920'] = 1
    elif 3000 < offer_delfin <= 3100:
        new_df['OfferDolphin_3000_3100'] = 1


    # OfferPenguin conditionals
    if offer_pinguin == 100700:
        new_df['OfferPenguin_100700'] = 1
    elif 1300 < offer_pinguin <= 1700:
        new_df['OfferPenguin_1300_1700'] = 1
    elif -100 < offer_pinguin <= 100:
        new_df['OfferPenguin_n100_100'] = 1
    elif 2600 < offer_pinguin <= 3200:
        new_df['OfferPenguin_2600_3200'] = 1
    elif 4100 < offer_pinguin <= 4500:
        new_df['OfferPenguin_4100_4500'] = 1
    elif 1700 < offer_pinguin <= 2100:
        new_df['OfferPenguin_1700_2100'] = 1
    elif 3200 < offer_pinguin <= 4100:
        new_df['OfferPenguin_3200_4100'] = 1
    elif 2100 < offer_pinguin <= 2600:
        new_df['OfferPenguin_2100_2600'] = 1
    elif 700 < offer_pinguin <= 1000:
        new_df['OfferPenguin_700_1000'] = 1
    elif 1000 < offer_pinguin <= 1300:
        new_df['OfferPenguin_1000_1300'] = 1





    row = np.array(new_df.iloc[0]).reshape(1, -1)
    result=model.predict(row)
    return result[0]
# print(predict(age,gender,score,credit_limit,income,bnr40,offer_crab,offer_delfin,offer_pinguin,produs,other_credits,comission))
