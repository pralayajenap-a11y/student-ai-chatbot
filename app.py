
import os
import pandas as pd
import streamlit as st

from model import predict_performance, get_performance_category

st.set_page_config(
    page_title="AI Student Management System",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "students.csv")

COLUMNS = [
    "Roll_No", "Name", "Branch", "Semester",
    "Attendance", "Study_Hours", "Previous_Marks"
]


def load_students():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_csv(CSV_FILE)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = None

    return df[COLUMNS]


def save_students(df):
    df.to_csv(CSV_FILE, index=False)


st.title("🎓 AI-Based Student Management System")
st.write("Manage student records and predict academic performance.")

students = load_students()

menu = st.sidebar.selectbox(
    "Choose Menu",
    ["Dashboard", "Add Student", "View Students", "Predict Performance"]
)

if menu == "Dashboard":
    st.header("📊 Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(students))

    if not students.empty:
        attendance = pd.to_numeric(students["Attendance"], errors="coerce")
        marks = pd.to_numeric(students["Previous_Marks"], errors="coerce")

        col2.metric(
            "Average Attendance",
            f"{attendance.mean():.1f}%" if attendance.notna().any() else "N/A"
        )
        col3.metric(
            "Average Marks",
            f"{marks.mean():.1f}" if marks.notna().any() else "N/A"
        )

        st.subheader("Student Records")
        st.dataframe(students, use_container_width=True)

        chart = students[["Name", "Previous_Marks"]].copy()
        chart["Previous_Marks"] = pd.to_numeric(
            chart["Previous_Marks"], errors="coerce"
        )
        chart = chart.dropna().set_index("Name")

        if not chart.empty:
            st.subheader("Marks Chart")
            st.bar_chart(chart)
    else:
        st.info("No student records available.")

elif menu == "Add Student":
    st.header("➕ Add New Student")

    with st.form("student_form"):
        roll_no = st.number_input("Roll Number", min_value=1, step=1)
        name = st.text_input("Student Name")
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "ME"])
        semester = st.selectbox("Semester", [1, 2, 3, 4, 5, 6, 7, 8])
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0
        )
        study_hours = st.number_input(
            "Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0
        )
        previous_marks = st.number_input(
            "Previous Marks (%)", min_value=0.0, max_value=100.0, value=60.0
        )

        submitted = st.form_submit_button("Add Student")

    if submitted:
        if not name.strip():
            st.error("Please enter the student name.")
        elif (pd.to_numeric(students["Roll_No"], errors="coerce") == roll_no).any():
            st.error("Roll Number already exists.")
        else:
            new_student = pd.DataFrame([{
                "Roll_No": int(roll_no),
                "Name": name.strip(),
                "Branch": branch,
                "Semester": semester,
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Previous_Marks": previous_marks
            }])

            save_students(pd.concat([students, new_student], ignore_index=True))
            st.success("Student added successfully!")
            st.rerun()

elif menu == "View Students":
    st.header("👨‍🎓 Student Records")

    if students.empty:
        st.info("No student records available.")
    else:
        search = st.text_input("Search by Name or Roll Number")
        filtered = students.copy()

        if search.strip():
            filtered = filtered[
                filtered["Name"].astype(str).str.contains(
                    search, case=False, na=False
                )
                | filtered["Roll_No"].astype(str).str.contains(
                    search, case=False, na=False
                )
            ]

        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="student_records.csv",
            mime="text/csv"
        )

        st.subheader("Delete Student")
        roll_numbers = filtered["Roll_No"].dropna().tolist()

        if roll_numbers:
            selected_roll = st.selectbox("Select Roll Number", roll_numbers)

            if st.button("Delete Selected Student"):
                updated = students[
                    students["Roll_No"].astype(str) != str(selected_roll)
                ]
                save_students(updated)
                st.success("Student deleted successfully!")
                st.rerun()

elif menu == "Predict Performance":
    st.header("🤖 Student Performance Prediction")

    st.warning(
        "Educational demonstration only. The model uses synthetic training "
        "data and is not validated for real academic decisions."
    )

    with st.form("prediction_form"):
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0
        )
        study_hours = st.number_input(
            "Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0
        )
        previous_marks = st.number_input(
            "Previous Marks (%)", min_value=0.0, max_value=100.0, value=60.0
        )
        submitted = st.form_submit_button("Predict Performance")

    if submitted:
        try:
            prediction = predict_performance(
                attendance, study_hours, previous_marks
            )
            category = get_performance_category(prediction)

            col1, col2 = st.columns(2)
            col1.metric("Estimated Marks", f"{prediction:.2f}%")
            col2.metric("Performance Category", category)
            st.progress(int(prediction))

        except FileNotFoundError:
            st.error("Model not found. Run train_model.py first.")

st.sidebar.markdown("---")
st.sidebar.caption("AI Student Management System")