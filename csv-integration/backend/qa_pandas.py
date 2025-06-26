import pandas as pd

# 1. Load your CSV (adjust path as needed)
df = pd.read_csv(r"C:\Users\daksh\Downloads\sample_employee_records.csv")

# 2. Select only those whose Name ends with " Anderson"
anderson_names = df[df["Name"].str.endswith(" Anderson")]["Name"].tolist()

# 3. Print the list and the total count
print("Employees with last name 'Anderson':")
for name in anderson_names:
    print("-", name)
print(f"\nTotal count: {len(anderson_names)}")
