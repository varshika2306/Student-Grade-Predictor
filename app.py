import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide"
)


# =====================================
# LOAD MODEL + DATASET
# =====================================

model = joblib.load(
    "model/linear_regression_model.pkl"
)

df = pd.read_csv(
    "student_data.csv"
    
    
)


# Model equation values

coefficient = 0.75860137
intercept = 8.772737067


# =====================================
# FUNCTIONS
# =====================================


def performance_status(score):

    percentage = (score / 20) * 100

    if percentage >= 85:
        return "🌟 Excellent"

    elif percentage >= 70:
        return "🟢 Good"

    elif percentage >= 50:
        return "🟡 Developing"

    else:
        return "🔴 Needs Attention"



def risk_analysis(study, absences, failures):

    risk = 0
    reasons = []


    if study <= 2:
        risk += 1
        reasons.append(
            "Low study commitment"
        )


    if absences > 15:
        risk += 1
        reasons.append(
            "High absenteeism"
        )


    if failures > 0:
        risk += 1
        reasons.append(
            "Previous academic difficulty"
        )


    if risk == 0:
        return (
            "🟢 Low Risk",
            [
            "Healthy academic indicators"
            ]
        )


    elif risk == 1:

        return (
            "🟡 Medium Risk",
            reasons
        )


    else:

        return (
            "🔴 High Risk",
            reasons
        )



# =====================================
# HEADER
# =====================================


st.title(
    "🎓 EduPredict AI"
)


st.subheader(
    "Student Performance Intelligence Dashboard"
)


st.write(
"""
An AI-powered academic analytics system that predicts
student final grades using Machine Learning and provides
personalized performance insights.
"""
)


st.divider()



# =====================================
# SIDEBAR
# =====================================


st.sidebar.header(
    "👤 Student Profile"
)


study_time = st.sidebar.slider(
    "📚 Study Time Level",
    1,
    4,
    2
)


absences = st.sidebar.slider(
    "📅 Number of Absences",
    0,
    50,
    5
)


failures = st.sidebar.slider(
    "⚠ Previous Failures",
    0,
    3,
    0
)



# =====================================
# MODEL PREDICTION
# =====================================


input_data = pd.DataFrame(
    {
        "studytime":[study_time]
    }
)


prediction = model.predict(
    input_data
)[0]


prediction = round(
    prediction,
    2
)



performance = performance_status(
    prediction
)


risk, reasons = risk_analysis(
    study_time,
    absences,
    failures
)



# =====================================
# KPI DASHBOARD
# =====================================


st.subheader(
    "📌 Student Performance Overview"
)


col1,col2,col3 = st.columns(3)


with col1:

    st.metric(
        "Predicted Grade",
        f"{prediction}/20"
    )


with col2:

    st.metric(
        "Performance Level",
        performance
    )


with col3:

    st.metric(
        "Academic Risk",
        risk
    )



# =====================================
# PERFORMANCE SCORE
# =====================================


st.divider()


st.subheader(
    "📊 Performance Score"
)


score = int(
    (prediction/20)*100
)


st.progress(
    score
)


st.write(
f"Current Performance Index: **{score}%**"
)



# =====================================
# RISK ANALYSIS
# =====================================


st.subheader(
    "🧠 AI Risk Analysis"
)


for r in reasons:

    st.warning(r)



# =====================================
# WHAT IF SIMULATOR
# =====================================


st.divider()


st.subheader(
    "🔮 Improvement Simulator"
)


future_level = st.slider(
    "Increase Study Time Level",
    1,
    4,
    study_time
)



future_prediction = model.predict(
    pd.DataFrame(
        {
            "studytime":[future_level]
        }
    )
)[0]


future_prediction = round(
    future_prediction,
    2
)


improvement = round(
    future_prediction - prediction,
    2
)



col1,col2 = st.columns(2)


with col1:

    st.info(
        f"""
Current Prediction

{prediction}/20
"""
    )


with col2:

    st.success(
        f"""
Future Prediction

{future_prediction}/20

Improvement:
+{improvement}
"""
    )



# =====================================
# STUDY TIME IMPACT GRAPH
# =====================================


st.divider()


st.subheader(
    "📈 Study Time Impact"
)


levels = [1,2,3,4]

grades=[]


for level in levels:

    value = model.predict(
        pd.DataFrame(
            {
            "studytime":[level]
            }
        )
    )[0]

    grades.append(
        round(value,2)
    )



fig,ax = plt.subplots()


ax.plot(
    levels,
    grades,
    marker="o"
)


ax.set_xlabel(
    "Study Time Level"
)

ax.set_ylabel(
    "Predicted Grade"
)


ax.set_title(
    "Study Time vs Grade Prediction"
)


st.pyplot(fig)



# =====================================
# DATASET INTELLIGENCE
# =====================================


st.divider()


st.subheader(
    "📊 Student Dataset Intelligence"
)



col1,col2,col3,col4 = st.columns(4)


with col1:

    st.metric(
        "Students",
        len(df)
    )


with col2:

    st.metric(
        "Average Grade",
        round(df["G3"].mean(),2)
    )


with col3:

    st.metric(
        "Highest Grade",
        df["G3"].max()
    )


with col4:

    st.metric(
        "Lowest Grade",
        df["G3"].min()
    )



# Grade Distribution


st.subheader(
    "📚 Grade Distribution"
)


fig,ax = plt.subplots()


ax.hist(
    df["G3"],
    bins=10
)


ax.set_xlabel(
    "Final Grade"
)

ax.set_ylabel(
    "Students"
)


st.pyplot(fig)



# Study Analysis


st.subheader(
    "📖 Average Grade by Study Level"
)



study_analysis = (
    df.groupby("studytime")["G3"]
    .mean()
)



st.bar_chart(
    study_analysis
)



# Attendance


st.subheader(
    "📅 Attendance Impact"
)



attendance = pd.cut(
    df["absences"],
    bins=[
        -1,
        5,
        15,
        100
    ],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)



attendance_analysis = (
    df.groupby(
        attendance,
        observed=False
    )["G3"]
    .mean()
)



st.bar_chart(
    attendance_analysis
)



# =====================================
# MODEL EXPLANATION
# =====================================


st.divider()


st.subheader(
    "🤖 Machine Learning Explanation"
)



st.write(
f"""
### Linear Regression Model

Equation:

**Grade = {intercept:.2f} + ({coefficient:.2f} × Study Time Level)**


Interpretation:

For every increase of one study time level,
the model predicts approximately
**{coefficient:.2f} additional marks**.

Algorithm:
Linear Regression

Dataset:
UCI Student Performance Dataset

Framework:
Scikit-Learn + Streamlit
"""
)



st.caption(
"EduPredict AI | Machine Learning Academic Analytics System"
)