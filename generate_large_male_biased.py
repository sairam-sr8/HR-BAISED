import pandas as pd
import numpy as np

n = 1500
np.random.seed(42)
genders = np.random.choice(['Male', 'Female'], n)
age_ranges = np.random.choice(['20-29','30-39','40-49','50-59'], n)
ethnicities = np.random.choice(['Asian','White','Black','Hispanic'], n)
departments = np.random.choice(['IT','Finance','HR','Marketing'], n)
ed_levels = np.random.choice(['Bachelor','Master','PhD'], n)
years_exp = np.random.randint(1, 31, n)
perf_scores = np.random.choice([60,70,80,85,90,95], n)
training = np.random.choice(['Yes','No'], n)
support_div = np.random.choice(['Yes','No'], n)
workplace_bias = np.random.choice(['Yes','No'], n)
projects = np.random.randint(1, 15, n)
overtime = np.random.randint(0, 30, n)
promoted = []
for g in genders:
    if g == 'Male':
        promoted.append('Yes' if np.random.rand() < 0.9 else 'No')
    else:
        promoted.append('Yes' if np.random.rand() < 0.1 else 'No')
data = pd.DataFrame({
    'Gender': genders,
    'Age Range': age_ranges,
    'Ethnicity': ethnicities,
    'Department': departments,
    'Education Level': ed_levels,
    'Years of Experience': years_exp,
    'Performance Score': perf_scores,
    'Training Participation': training,
    'Support for Diversity Initiatives': support_div,
    'Experienced Workplace Bias': workplace_bias,
    'Projects Handled': projects,
    'Overtime Hours': overtime,
    'Promotion Status': promoted
})
data.to_csv('large_male_biased_1500.csv', index=False)
print('Generated large_male_biased_1500.csv')
