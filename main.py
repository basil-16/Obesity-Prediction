import os

import streamlit as st
import pandas as pd
import pickle

def file_preprocessing(df):
  df = df.drop(["id", "SMOKE", "SCC", "FAVC"], axis=1)
  df = df[~df["CAEC"].isin(["Always", "No"])]
  df["MTRANS"] = df["MTRANS"].replace(["Walking", "Motorbike", "Bike"], "walk_two_wheelers")
  df = df[df["Age"] <= 40]
  df["BMI"] = df["Weight"]/df["Height"]**2
  df = df.drop(["Height", "Weight"], axis=1)
  return df

def predict_data(df, model):
    try:
        output = model.predict(df)
        output = pd.DataFrame({"enc_NObeyesdad":output})
    except Exception as e:
        st.error(e)
    else:
        return output

def create_df(predicted_data, test_data):
    try:
        encodings=pd.read_csv("encodings.csv")
        df = predicted_data.merge(encodings[["enc_NObeyesdad", "act_NObeyesdad"]], on="enc_NObeyesdad", how="left")
        df = pd.concat([test_data, df], axis=1)
        df = df.rename(columns={"enc_NObeyesdad":"status_encoding", "act_NObeyesdad":"status"})
    except Exception as e:
        st.error(e)
    else:
        st.download_button("Download results", df.to_csv(), "final_output.csv", "text/csv")


def main():

    st.set_page_config(page_title="Obesity Prediction")
    df = st.file_uploader(label="Please upload your file here")

    required_cols = ['Age', 'Gender', 'family_history_with_overweight', 'FCVC', 'NCP',
                    'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC', 'MTRANS', 'BMI']

    model = pickle.load(open("xgboost.pkl", "rb"))
    if df:
        flag=True
        try:
            data=pd.read_csv(df)
        except Exception as e:
            st.error(e)
        else:
            missing_cols = [i for i in required_cols if i not in data.columns]
            if missing_cols:
                flag = False

        if not flag:
            st.warning(f"Found missing columns:{missing_cols}")

        elif st.button("Pre-process data"):
            data=file_preprocessing(data)


        else:
            predicted_data = predict_data(data, model)
            create_df(predicted_data, data)
    else:
        st.info("Please upload a file")


main()

