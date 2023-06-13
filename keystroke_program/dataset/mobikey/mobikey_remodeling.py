
import pandas as pd
import numpy as np
from openpyxl import Workbook


#ejemplo_mobekey=6 usuarios
#ejemplo2_mobekey=5 usuarios
#ejemplo3_mobekey= 4 usuarios
#ejemplo4_mobekey = 5 usuarios
#ejemplo5_mobekey = 3 usuarios
#ejemplo6_mobekey = 1 usuarios
#ejemplo7_mobekey =  4 usuarios
#ejemplo8_mobekey = 5 usuarios
#ejemplo9_mobekey =  4 usuarios
#ejemplo10_mobekey =  4 usuarios
#ejemplo11_mobekey =  6 usuarios
#ejemplo12_mobekey =  4 usuarios
#ejemplo13_mobekey =  3 usuarios

#Total= 51 usuarios

dataset_base = pd.read_excel("C:\\Users\\ADRIAN\\PycharmProjects\\Keystrokes_TFG\\mobikey\\excels_base_mobikey"
                             "\\ejemplo_mobekey.xlsx")

df = pd.DataFrame(dataset_base)

# 3 passwords join
password_join = "kicsikutyatarkaKktsf2!2014.tie5Roanl"

password1 = "kicsikutyatarka"
password2 = "Kktsf2!2014"
password3 = ".tie5Roanl"

columns = df.columns[4:13]

columns_dataframe1 = ["UserId"]
columns_dataframe2 = ["UserId"]
columns_dataframe3 = ["UserId"]


for i in range(len(password1)):
    for c in range(len(columns)):
        columna_nueva = str(columns[c]) + password1[i]
        columns_dataframe1.append(columna_nueva)

for i in range(len(password2)):
    for c in range(len(columns)):
        columna_nueva = str(columns[c]) + password2[i]
        columns_dataframe2.append(columna_nueva)

for i in range(len(password3)):
    for c in range(len(columns)):
        columna_nueva = str(columns[c]) + password3[i]
        columns_dataframe3.append(columna_nueva)


dataframe_final1 = pd.DataFrame(columns=columns_dataframe1)
dataframe_final2 = pd.DataFrame(columns=columns_dataframe2)
dataframe_final3 = pd.DataFrame(columns=columns_dataframe3)


# joining mobekey_datasets
excels = ["mobekey_example", "mobekey_example2", "mobekey_example3", "mobekey_example4", "mobekey_example5"
          , "mobekey_exmple6", "mobekey_example7", "mobekey_example8", "mobekey_example9", "mobekey_example10"
          , "mobekey_ejemplo11", "mobekey_ejemplo12", "mobekey_ejemplo13"]

for t in range(len(excels)):

    dataset = pd.read_excel("C:\\Users\\ADRIAN\\PycharmProjects\\Keystrokes_TFG\\mobikey\\"
                            "excels_base_mobikey\\"+excels[t]+".xlsx")

    df = pd.DataFrame(dataset)

    password_beginning = []
    invalid_rows = []

    for i in range(len(df)):
        if df.loc[i][' Key'] not in password_join:
            invalid_rows.append(i)

    for j in range(len(invalid_rows)):
        df.drop(invalid_rows[j], axis=0)

    dataframe1 = pd.DataFrame(columns=columns_dataframe1)
    dataframe2 = pd.DataFrame(columns=columns_dataframe2)
    dataframe3 = pd.DataFrame(columns=columns_dataframe3)

    counter1 = 0
    counter2 = 0
    counter3 = 0

    for i in range(len(df)):
        if df.loc[i][" Key"] == " k" and df.loc[i+1][' Key'] == " i":
            values = [df.loc[i]["UserId"]]
            for j in range(15):
                fila = i + j
                for c in range(len(columns)):
                    values.append(df.loc[fila, columns[c]])
            dataframe1.loc[counter1] = values
            counter1 += 1

        if df.loc[i][' Key'] == " K" and df.loc[i+1][' Key'] == " k":
            values = [df.loc[i]["UserId"]]
            for j in range(11):
                fila = i + j
                for c in range(len(columns)):
                    values.append(df.loc[fila, columns[c]])
            dataframe2.loc[counter1] = values
            counter2 += 1

        if df.loc[i][' Key'] == " ." and df.loc[i+1][' Key'] == " t":
            values = [df.loc[i]["UserId"]]
            for j in range(10):
                fila = i + j
                for c in range(len(columns)):
                    values.append(df.loc[fila, columns[c]])
            dataframe3.loc[counter1] = values
            counter3 += 1

    dataframe_final1 = pd.concat([dataframe_final1, dataframe1])
    dataframe_final2 = pd.concat([dataframe_final2, dataframe2])
    dataframe_final3 = pd.concat([dataframe_final3, dataframe3])


dataframe_final1.to_excel("Mobikey_dataframe1.xlsx", index=False)
dataframe_final2.to_excel("Mobikey_dataframe2.xlsx", index=False)
dataframe_final3.to_excel("Mobikey_dataframe3.xlsx", index=False)


# Obtaining new Mobekey temporal dataset
mobekey_dataframe1 = pd.read_excel\
                     ("C:\\Users\\ADRIAN\\PycharmProjects\\Keystrokes_TFG"
                      "\\dataset\\mobikey\\Mobikey_dataframe1.xlsx")

df = pd.DataFrame(mobekey_dataframe1)
columns = df.columns.values

# new dataset to create
mobekey_temporaldataframe1 = pd.DataFrame()

# copying the correct columns
mobekey_temporaldataframe1[columns[0]] = df[columns[0]]

for i in range(1, len(df.columns), 9):

    mobekey_temporaldataframe1[columns[i]] = df[columns[i]]
    mobekey_temporaldataframe1[columns[i + 1]] = df[columns[i + 1]]


mobekey_temporaldataframe1.to_excel("C:\\Users\\ADRIAN\\PycharmProjects\\Keystrokes_TFG"
                                        "\\dataset\\mobikey\\Mobikey_dataframe1_temporal.xlsx", index=False)
