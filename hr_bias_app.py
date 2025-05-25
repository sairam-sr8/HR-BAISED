import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt

def run_bias_detection(df):
    # Normalize columns: strip, lowercase, replace underscores/spaces
    def norm_col(col):
        return col.strip().lower().replace('_', '').replace(' ', '')
    import streamlit as st
    # Mapping from normalized to original
    norm_to_orig = {norm_col(col): col for col in df.columns}
    # Define expected columns (normalized)
    expected_categorical = ['Gender', 'Age Range', 'Ethnicity', 'Department', 'Education Level']
    expected_numerical = ['Years of Experience', 'Performance Score', 'Training Participation',
                          'Support for Diversity Initiatives', 'Experienced Workplace Bias',
                          'Projects Handled', 'Overtime Hours']
    expected_all = expected_categorical + expected_numerical + ['Promotion Status']
    norm_expected = [norm_col(col) for col in expected_all]
    # Rename columns in df to standard names if they match normalized
    for std, norm in zip(expected_all, norm_expected):
        if norm in norm_to_orig and std not in df.columns:
            df.rename(columns={norm_to_orig[norm]: std}, inplace=True)
    # Map word-based scores to numbers for Performance Score
    if 'Performance Score' in df.columns:
        perf_map = {'poor': 60, 'average': 70, 'good': 80, 'excellent': 90}
        df['Performance Score'] = df['Performance Score'].apply(lambda x: perf_map.get(str(x).strip().lower(), x) if isinstance(x, str) else x)
        # If still not numeric, treat as categorical
        try:
            df['Performance Score'] = pd.to_numeric(df['Performance Score'])
        except Exception:
            st.warning('Performance Score still contains non-numeric values after mapping. It will be treated as categorical.')
    # Convert all Yes/No columns to 1/0
    for col in df.columns:
        if df[col].dtype == object:
            unique_vals = set(df[col].astype(str).str.lower().str.strip().unique())
            if unique_vals <= {'yes', 'no'}:
                
                df[col] = df[col].astype(str).str.lower().str.strip().map({'yes': 1, 'no': 0})
    # Convert Promotion Status to numeric if needed
    if df['Promotion Status'].dtype == object:
        unique_vals = set(df['Promotion Status'].astype(str).str.lower().str.strip().unique())
        if unique_vals <= {'yes', 'no'}:
            st.info('Converting "Promotion Status" from Yes/No to 1/0 automatically.')
            df['Promotion Status'] = df['Promotion Status'].astype(str).str.lower().str.strip().map({'yes': 1, 'no': 0})
        else:
            st.warning(f'"Promotion Status" column contains unexpected values: {unique_vals}. Please use Yes/No or 1/0.')
    # Identify categorical and numeric columns robustly
    X = df.drop(columns=['Promotion Status'])
    y = df['Promotion Status']
    # Treat all non-numeric columns as categorical
    categorical = [col for col in X.columns if not np.issubdtype(df[col].dtype, np.number)]
    numerical = [col for col in X.columns if np.issubdtype(df[col].dtype, np.number)]
    # Warn about columns that were expected numeric but are not
    expected_num_missing = [col for col in expected_numerical if col in X.columns and col not in numerical]
    if expected_num_missing:
        st.warning(f"These columns were expected numeric but are treated as categorical due to non-numeric values: {expected_num_missing}")
    missing = [col for col in expected_categorical + expected_numerical if col not in X.columns]
    if missing:
        st.warning(f"Missing columns in uploaded data: {missing}. The model will use only available columns.")
    if not categorical:
        raise ValueError("No categorical columns found in the uploaded file. At least one is required.")
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ], remainder='passthrough')
    # Robust stratification: ensure both Gender and Promotion Status are present in both train/test
    stratify_cols = None
    if 'Gender' in X.columns and 'Promotion Status' in df.columns:
        stratify_cols = df[['Gender', 'Promotion Status']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_cols)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    # For sensitive feature, fallback to first available categorical if 'Gender' missing
    sensitive_feature = categorical[0] if 'Gender' not in categorical and categorical else 'Gender'
    sensitive_train = X_train[sensitive_feature]
    sensitive_test = X_test[sensitive_feature]
    base_model = LogisticRegression(solver='liblinear', random_state=42)
    mitigator = ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity()
    )
    mitigator.fit(X_train_enc, y_train, sensitive_features=sensitive_train)
    y_pred = mitigator.predict(X_test_enc)
    accuracy = np.round((y_test == y_pred).mean(), 3)
    metric_frame = MetricFrame(
        metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )
    dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
    eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)
    verdict = verdict_text(dp_diff, eo_diff, metric_frame)
    return metric_frame, dp_diff, eo_diff, verdict

def verdict_text(dp_diff, eo_diff, metric_frame, dp_thresh=0.1, eo_thresh=0.1):
    verdict = []
    # Determine which group is favored (higher selection rate)
    selection_rates = metric_frame.by_group['selection_rate']
    if len(selection_rates) >= 2:
        favored_group = selection_rates.idxmax()
        unfavored_group = selection_rates.idxmin()
        favored_rate = selection_rates.max()
        unfavored_rate = selection_rates.min()
        is_biased = (abs(eo_diff) > eo_thresh) or (abs(dp_diff) > dp_thresh)
        if is_biased:
            verdict.append(f'🚨 This dataset is BIASED. {favored_group} are favored for promotion.')
        else:
            verdict.append('✅ This dataset is NOT biased. No group is favored for promotion.')
        # Optionally, keep detailed metrics below
        if abs(eo_diff) > eo_thresh:
            verdict.append(f'⚠️ Bias detected under equalized odds fairness. {favored_group} are favored for promotion.')
        else:
            verdict.append('✅ No major bias under equalized odds fairness.')
        if abs(dp_diff) > dp_thresh:
            verdict.append(f'⚠️ Bias detected under demographic parity fairness. {favored_group} are favored for promotion.')
        else:
            verdict.append('✅ No major bias under demographic parity fairness.')
    else:
        # Fallback if only one group present
        if (abs(eo_diff) > eo_thresh) or (abs(dp_diff) > dp_thresh):
            verdict.append('🚨 This dataset is BIASED (insufficient group diversity to specify favored group).')
        else:
            verdict.append('✅ This dataset is NOT biased. No group is favored for promotion.')
        if abs(eo_diff) > eo_thresh:
            verdict.append('⚠️ Bias detected under equalized odds fairness.')
        else:
            verdict.append('✅ No major bias under equalized odds fairness.')
        if abs(dp_diff) > dp_thresh:
            verdict.append('⚠️ Bias detected under demographic parity fairness.')
        else:
            verdict.append('✅ No major bias under demographic parity fairness.')
    return verdict

def main():
    st.title('HR Bias Detection and Mitigation App')
    st.markdown('''
### Required CSV Columns & Meanings
| Column Name | Description |
|---|---|
| Gender | Gender of the employee (e.g., Male, Female) |
| Age Range | Age group (e.g., 20-29, 30-39, etc.) |
| Ethnicity | Ethnic background (e.g., Asian, Hispanic, etc.) |
| Department | Department name (e.g., IT, HR, Finance, etc.) |
| Education Level | Highest education achieved (e.g., Bachelor, Master, PhD) |
| Years of Experience | Total years of professional experience |
| Performance Score | Performance evaluation score |
| Training Participation | Yes/No, attended training programs |
| Support for Diversity Initiatives | Yes/No, supports diversity initiatives |
| Experienced Workplace Bias | Yes/No, experienced bias at work |
| Projects Handled | Number of projects handled |
| Overtime Hours | Number of overtime hours worked |
| Promotion Status | Yes/No (target: was promoted?) |

**Sample CSV header:**  
Gender,Age Range,Ethnicity,Department,Education Level,Years of Experience,Performance Score,Training Participation,Support for Diversity Initiatives,Experienced Workplace Bias,Projects Handled,Overtime Hours,Promotion Status

**Sample row:**  
Female,30-39,Asian,IT,Master,7,88,Yes,Yes,No,5,10,Yes

**Tip:** The app will automatically detect columns even if their headers use underscores, spaces, or different capitalization (e.g., "gender", "Gender", "Gender_", "GENDER" are all valid).
''')
    st.write('Upload your HR dataset or enter an individual record to check for bias in promotion outcomes.')
    upload = st.file_uploader('Upload CSV Dataset', type=['csv'])
    data = None
    if upload:
        df = pd.read_csv(upload)
        st.write('Preview:', df.head())
        st.write('Detected columns:', list(df.columns))
        # Flexible check for 'Promotion Status' column
        norm_cols = {col.strip().lower(): col for col in df.columns}
        if 'promotion status' not in norm_cols:
            st.error('Dataset must include a "Promotion Status" column for bias analysis. Detected columns: ' + ", ".join(df.columns))
            return
        # Rename to standard if needed
        if 'Promotion Status' not in df.columns:
            df.rename(columns={norm_cols['promotion status']: 'Promotion Status'}, inplace=True)
        if st.button('Run Bias Detection'):
            metric_frame, dp_diff, eo_diff, verdict = run_bias_detection(df)
            st.subheader('Group-wise Metrics')
            st.dataframe(metric_frame.by_group)
            st.subheader('Fairness Metrics')
            st.write(f'Demographic Parity Difference: {dp_diff:.3f}')
            st.write(f'Equalized Odds Difference: {eo_diff:.3f}')
            st.markdown('---')
            st.markdown('### :mag: Bias Detection Verdict')
            for v in verdict:
                if 'No major bias' in v:
                    st.success(f'✅ {v}')
                else:
                    st.error(f'⚠️ {v}')
            st.markdown('---')
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Group-wise Selection Rate')
                fig, ax = plt.subplots()
                metric_frame.by_group['selection_rate'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
                ax.set_ylabel('Selection Rate')
                st.pyplot(fig)
            with col2:
                st.subheader('Group-wise Accuracy')
                fig2, ax2 = plt.subplots()
                metric_frame.by_group['accuracy'].plot(kind='bar', ax=ax2, color=['#2ca02c', '#d62728'])
                ax2.set_ylabel('Accuracy')
                st.pyplot(fig2)
            st.markdown('---')
            # Pie chart for promotion distribution
            st.subheader('Promotion Status Distribution')
            pie_labels = ['Not Promoted', 'Promoted'] if set(df['Promotion Status'].unique()) <= {0, 1} else df['Promotion Status'].unique()
            fig3, ax3 = plt.subplots()
            df['Promotion Status'].value_counts().plot.pie(autopct='%1.1f%%', labels=pie_labels, ax=ax3, colors=['#ff9999','#66b3ff'])
            ax3.set_ylabel('')
            st.pyplot(fig3)
    
        

if __name__ == '__main__':
    main()
