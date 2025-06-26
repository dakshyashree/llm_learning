import pandas as pd

# Load your DataFrame
df = pd.read_csv("./data/salaries_2023.csv").fillna(0)

# 1. Compute mean base salary per grade
grade_means = df.groupby("Grade")["Base_Salary"].mean()

# 2. Find which grade has the highest average base salary
best_grade = grade_means.idxmax()
best_grade_salary = grade_means.max()

# 3. Compute average pay by gender
female_avg = df[df["Gender"] == "F"]["Base_Salary"].mean()
male_avg = df[df["Gender"] == "M"]["Base_Salary"].mean()
x = df.groupby('Division').size().reset_index(name='Count')

# 4. Print results
print(f"Grade with highest average base salary: {best_grade} (${best_grade_salary:,.2f})")
print(f"Average Female Base Salary: ${female_avg:,.2f}")
print(f"Average Male   Base Salary: ${male_avg:,.2f}")
print(x)