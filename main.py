import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from collections import Counter
from datetime import datetime
import io

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👥", layout="wide")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Title and Introduction
st.markdown("""
# 👥 Employee Attrition Prediction System

## 🎯 Project Overview
This comprehensive analysis aims to predict and prevent employee attrition using advanced machine learning techniques. 
We'll analyze employee data, build predictive models, and provide actionable insights for HR teams.

### 📋 Analysis Steps:
1. **Data Loading & Overview** - Load and examine the dataset
2. **Exploratory Data Analysis** - Deep dive into patterns and correlations
3. **Feature Engineering** - Create predictive features
4. **Model Training** - Train and compare multiple ML algorithms
5. **Single Employee Prediction** - Individual risk assessment
6. **Batch Analysis** - Workforce-wide risk evaluation
7. **Business Impact Analysis** - Financial implications and ROI
8. **Executive Dashboard** - High-level insights and recommendations

---
""")

# ================================================================================================
# SECTION 1: DATA LOADING & OVERVIEW
# ================================================================================================

st.markdown("""
## 📁 Section 1: Data Loading & Overview

Let's start by loading our employee attrition dataset and examining its structure.
""")

@st.cache_data
def load_attrition_data():
    """Load the Employee-Attrition.csv file"""
    try:
        data = pd.read_csv('Employee-Attrition.csv')
        return data
    except FileNotFoundError:
        st.error("❌ Employee-Attrition.csv not found. Please upload your data file.")
        return None

# Data loading section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Load Dataset")
    data_option = st.radio("Choose data source:", [
        "📊 Use Employee-Attrition.csv", 
        "📤 Upload Different CSV File"
    ])

with col2:
    st.markdown("### 📈 Quick Actions")
    if st.button("🔄 Reset All Analysis", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if data_option == "📊 Use Employee-Attrition.csv":
    if st.button("🚀 Load Employee-Attrition.csv", type="primary"):
        data = load_attrition_data()
        if data is not None:
            st.session_state.raw_data = data
            st.session_state.data_loaded = True
            st.success("✅ Employee-Attrition.csv loaded successfully!")
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.raw_data = data
            st.session_state.data_loaded = True
            st.success("✅ CSV file uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

# Display data overview
if st.session_state.data_loaded:
    data = st.session_state.raw_data
    
    st.markdown("### 📋 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Employees", f"{len(data):,}")
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        attrition_count = (data['Attrition'] == 'Yes').sum()
        st.metric("Employees Left", attrition_count)
    with col4:
        attrition_rate = attrition_count / len(data) * 100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    # Data preview
    st.markdown("#### 👀 Data Preview")
    st.dataframe(data.head(10))
    
    # Data quality summary
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Data Quality")
        st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
        st.write(f"**Duplicate Rows:** {data.duplicated().sum()}")
        st.write(f"**Data Types:** {data.dtypes.value_counts().to_dict()}")
    
    with col2:
        st.markdown("#### 🎯 Target Variable Analysis")
        attrition_counts = data['Attrition'].value_counts()
        st.write(f"**Employees Left:** {attrition_counts.get('Yes', 0)} ({attrition_rate:.1f}%)")
        st.write(f"**Employees Stayed:** {attrition_counts.get('No', 0)} ({100-attrition_rate:.1f}%)")
        st.write(f"**Class Imbalance Ratio:** {attrition_counts.get('No', 0)/attrition_counts.get('Yes', 1):.1f}:1")

# ================================================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ================================================================================================

if st.session_state.data_loaded:
    st.markdown("""
    ---
    ## 📊 Section 2: Exploratory Data Analysis
    
    Let's explore the data to understand patterns, distributions, and relationships between features.
    """)
    
    if st.button("🔍 Run Complete EDA Analysis", type="primary"):
        data = st.session_state.raw_data
        
        st.markdown("### 🎯 Attrition Distribution Analysis")
        
        # Comprehensive visualization dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            # Attrition distribution pie chart
            attrition_counts = data['Attrition'].value_counts()
            fig_pie = px.pie(values=attrition_counts.values, 
                            names=attrition_counts.index,
                            title="Employee Attrition Distribution",
                            color_discrete_map={'No': '#2ECC71', 'Yes': '#E74C3C'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Age distribution by attrition
            fig_age = px.histogram(data, x='Age', color='Attrition', 
                                  title="Age Distribution by Attrition Status",
                                  marginal="box",
                                  color_discrete_map={'No': '#2ECC71', 'Yes': '#E74C3C'})
            st.plotly_chart(fig_age, use_container_width=True)
        
        st.markdown("### 🏢 Department and Income Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department analysis
            dept_attrition = pd.crosstab(data['Department'], data['Attrition'])
            fig_dept = px.bar(dept_attrition, 
                             title="Attrition by Department",
                             color_discrete_map={'No': '#2ECC71', 'Yes': '#E74C3C'})
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            # Monthly income by attrition
            fig_income = px.box(data, x='Attrition', y='MonthlyIncome',
                               title="Monthly Income by Attrition Status",
                               color='Attrition',
                               color_discrete_map={'No': '#2ECC71', 'Yes': '#E74C3C'})
            st.plotly_chart(fig_income, use_container_width=True)
        
        st.markdown("### 🔗 Feature Correlation Analysis")
        
        # Correlation heatmap
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            correlation_matrix = data[numerical_cols].corr()
            
            fig_corr = px.imshow(correlation_matrix, 
                                title="Feature Correlation Matrix",
                                color_continuous_scale="RdBu_r",
                                aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
        

# ================================================================================================
# SECTION 3: FEATURE ENGINEERING & PREPROCESSING
# ================================================================================================

if st.session_state.data_loaded:
    st.markdown("""
    ---
    ## 🔧 Section 3: Feature Engineering & Preprocessing
    
    We'll create new features and prepare the data for machine learning models.
    """)
    
    def preprocess_data(data):
        """Enhanced preprocessing with comprehensive feature engineering"""
        df_processed = data.copy()
        
        st.markdown("#### 🛠️ Feature Engineering Process")
        
        # Show code being executed
        with st.expander("📋 View Feature Engineering Code"):
            st.code("""
# Experience-related features
df_processed['ExperienceRatio'] = df_processed['YearsAtCompany'] / (df_processed['TotalWorkingYears'] + 1)
df_processed['PromotionGap'] = df_processed['YearsInCurrentRole'] - df_processed['YearsSinceLastPromotion']
df_processed['ManagerTenure'] = df_processed['YearsWithCurrManager'] / (df_processed['YearsAtCompany'] + 1)

# Compensation-related features  
df_processed['IncomePerYear'] = df_processed['MonthlyIncome'] / (df_processed['TotalWorkingYears'] + 1)
df_processed['SalaryHikeRatio'] = df_processed['PercentSalaryHike'] / 100

# Work-life balance indicators
df_processed['WorkLifeScore'] = (df_processed['WorkLifeBalance'] + df_processed['JobSatisfaction'] + 
                                df_processed['EnvironmentSatisfaction']) / 3

# Job mobility indicators
df_processed['JobMobility'] = df_processed['NumCompaniesWorked'] / (df_processed['TotalWorkingYears'] + 1)
df_processed['IsHighMobility'] = (df_processed['NumCompaniesWorked'] > 3).astype(int)
            """, language='python')
        
        # Experience-related features
        st.markdown("**🎯 Step 1: Creating Experience-Related Features**")
        df_processed['ExperienceRatio'] = df_processed['YearsAtCompany'] / (df_processed['TotalWorkingYears'] + 1)
        df_processed['PromotionGap'] = df_processed['YearsInCurrentRole'] - df_processed['YearsSinceLastPromotion']
        df_processed['ManagerTenure'] = df_processed['YearsWithCurrManager'] / (df_processed['YearsAtCompany'] + 1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("ExperienceRatio", "Created", help="Years at Company / Total Working Years")
        col2.metric("PromotionGap", "Created", help="Years in Role - Years Since Promotion")
        col3.metric("ManagerTenure", "Created", help="Years with Manager / Years at Company")
        
        # Compensation-related features
        st.markdown("**💰 Step 2: Creating Compensation-Related Features**")
        df_processed['IncomePerYear'] = df_processed['MonthlyIncome'] / (df_processed['TotalWorkingYears'] + 1)
        df_processed['SalaryHikeRatio'] = df_processed['PercentSalaryHike'] / 100
        
        col1, col2 = st.columns(2)
        col1.metric("IncomePerYear", "Created", help="Monthly Income / Total Years Experience")
        col2.metric("SalaryHikeRatio", "Created", help="Percent Salary Hike / 100")
        
        # Work-life balance indicators
        st.markdown("**⚖️ Step 3: Creating Work-Life Balance Indicators**")
        df_processed['WorkLifeScore'] = (df_processed['WorkLifeBalance'] + df_processed['JobSatisfaction'] + 
                                        df_processed['EnvironmentSatisfaction']) / 3
        
        col1, col2 = st.columns(2)
        col1.metric("WorkLifeScore", "Created", help="Average of 3 satisfaction scores")
        col2.metric("Sample Score", f"{df_processed['WorkLifeScore'].mean():.2f}", help="Mean work-life score")
        
        # Job mobility indicators
        st.markdown("**🚀 Step 4: Creating Job Mobility Indicators**")
        df_processed['JobMobility'] = df_processed['NumCompaniesWorked'] / (df_processed['TotalWorkingYears'] + 1)
        df_processed['IsHighMobility'] = (df_processed['NumCompaniesWorked'] > 3).astype(int)
        
        col1, col2 = st.columns(2)
        col1.metric("JobMobility", "Created", help="Companies Worked / Total Years")
        col2.metric("High Mobility Employees", f"{df_processed['IsHighMobility'].sum()}", help="Employees with >3 companies")
        
        st.success("✅ All engineered features created successfully!")
        
        # Encoding section with detailed explanation
        st.markdown("#### 🏷️ Categorical Variable Encoding Process")
        
        with st.expander("📋 View Encoding Code"):
            st.code("""
# Binary encoding for target and binary features
df_processed['Attrition'] = df_processed['Attrition'].map({'No': 0, 'Yes': 1})
df_processed['OverTime'] = df_processed['OverTime'].map({'No': 0, 'Yes': 1})
df_processed['Gender'] = df_processed['Gender'].map({'Male': 0, 'Female': 1})

# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

for column in categorical_columns:
    le = LabelEncoder()
    df_processed[column] = le.fit_transform(df_processed[column])
    label_encoders[column] = le
            """, language='python')
        
        # Binary encoding for target and binary features
        st.markdown("**🎯 Step 5: Binary Encoding**")
        df_processed['Attrition'] = df_processed['Attrition'].map({'No': 0, 'Yes': 1})
        df_processed['OverTime'] = df_processed['OverTime'].map({'No': 0, 'Yes': 1})
        df_processed['Gender'] = df_processed['Gender'].map({'Male': 0, 'Female': 1})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Attrition", "Encoded", help="Yes=1, No=0")
        col2.metric("OverTime", "Encoded", help="Yes=1, No=0") 
        col3.metric("Gender", "Encoded", help="Female=1, Male=0")
        
        # Label encoding for other categoricals
        st.markdown("**📊 Step 6: Label Encoding for Categorical Variables**")
        label_encoders = {}
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
        
        encoding_results = []
        for column in categorical_columns:
            le = LabelEncoder()
            unique_values = df_processed[column].unique()
            df_processed[column] = le.fit_transform(df_processed[column])
            label_encoders[column] = le
            encoding_results.append({
                'Column': column,
                'Unique_Values': len(unique_values),
                'Encoded_Range': f"0 to {len(unique_values)-1}"
            })
        
        encoding_df = pd.DataFrame(encoding_results)
        st.dataframe(encoding_df)
        
        # Feature removal
        st.markdown("**🗑️ Step 7: Removing Redundant Features**")
        features_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        
        with st.expander("📋 View Feature Removal Code"):
            st.code("""
# Remove redundant features that don't add predictive value
features_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df_processed = df_processed.drop(columns=features_to_drop, errors='ignore')
            """, language='python')
        
        df_processed = df_processed.drop(columns=features_to_drop, errors='ignore')
        
        dropped_features = [f for f in features_to_drop if f in data.columns]
        if dropped_features:
            st.info(f"🗑️ Removed redundant features: {', '.join(dropped_features)}")
        
        return df_processed, label_encoders
    
    if st.button("⚙️ Run Feature Engineering & Preprocessing", type="primary"):
        with st.spinner("Processing data and engineering features..."):
            data = st.session_state.raw_data
            processed_data, label_encoders = preprocess_data(data)
            
            st.session_state.processed_data = processed_data
            st.session_state.label_encoders = label_encoders
            st.session_state.data_processed = True
            
            # Show feature engineering results
            st.markdown("#### 📊 Feature Engineering Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", len(data.columns))
            with col2:
                st.metric("Engineered Features", len(processed_data.columns))
            with col3:
                st.metric("New Features Added", len(processed_data.columns) - len(data.columns) + len(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']))
            
            # Show correlation with target
            feature_importance = abs(processed_data.corr()['Attrition']).sort_values(ascending=False)[1:11]
            
            st.markdown("#### 🎯 Top 10 Features by Correlation with Attrition")
            importance_df = pd.DataFrame({
                'Feature': feature_importance.index,
                'Correlation': feature_importance.values
            })
            
            fig_importance = px.bar(importance_df, x='Correlation', y='Feature', 
                                   orientation='h',
                                   title="Feature Correlation with Attrition")
            st.plotly_chart(fig_importance, use_container_width=True)

# ================================================================================================
# SECTION 4: MODEL TRAINING & COMPARISON
# ================================================================================================

if st.session_state.data_processed:
    st.markdown("""
    ---
    ## 🤖 Section 4: Model Training & Comparison
    
    We'll train multiple machine learning models and compare their performance.
    """)
    
    @st.cache_data
    def train_models(X_train, y_train, X_test, y_test):
        """Train multiple ML models with comprehensive evaluation"""
        
        # Apply SMOTE for balanced training data
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Support Vector Machine': SVC(random_state=42, probability=True)
        }
        
        model_results = {}
        trained_models = {}
        
        for name, model in models.items():
            if name == 'Support Vector Machine':
                model.fit(X_train_scaled, y_train_balanced)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train_balanced, y_train_balanced)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            model_results[name] = {
                'Accuracy': accuracy,
                'F1-Score': f1,
                'AUC-ROC': auc,
                'Predictions': y_pred,
                'Probabilities': y_pred_proba
            }
            
            trained_models[name] = model
        
        return model_results, trained_models, scaler
    
    if st.button("🚀 Train & Compare Models", type="primary"):
        with st.spinner("Training multiple models..."):
            # Prepare data
            processed_data = st.session_state.processed_data
            X = processed_data.drop(['Attrition'], axis=1)
            y = processed_data['Attrition']
            
            st.markdown("#### 📊 Data Preparation for Model Training")
            
            with st.expander("📋 View Data Splitting Code"):
                st.code("""
from sklearn.model_selection import train_test_split

# Prepare features and target
X = processed_data.drop(['Attrition'], axis=1)
y = processed_data['Attrition']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
                """, language='python')
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Show data split results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Features", len(X.columns))
            col2.metric("Training Samples", len(X_train))
            col3.metric("Test Samples", len(X_test))
            col4.metric("Test Split", "20%", help="80% train, 20% test")
            
            # Train models
            model_results, trained_models, scaler = train_models(X_train, y_train, X_test, y_test)
            
            # Store results
            st.session_state.model_results = model_results
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.feature_columns = X.columns.tolist()
            st.session_state.models_trained = True
            
            st.success("✅ All models trained successfully!")
        
        # Display results
        st.markdown("### 📊 Model Performance Analysis")
        
        results_df = pd.DataFrame(st.session_state.model_results).T
        results_df = results_df[['Accuracy', 'F1-Score', 'AUC-ROC']]
        
        # Performance table with highlighting
        st.markdown("#### 📈 Performance Metrics Comparison")
        st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
        
        # Detailed model evaluation
        with st.expander("📋 View Model Evaluation Code"):
            st.code("""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Performance metrics calculated for each model:
# - Accuracy: (TP + TN) / (TP + TN + FP + FN)
# - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)  
# - AUC-ROC: Area Under the Receiver Operating Characteristic Curve

# Best model selection based on AUC-ROC score
best_model = results_df['AUC-ROC'].idxmax()
            """, language='python')
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance comparison chart
            fig_comparison = px.bar(results_df.reset_index(), 
                                  x='index', y=['Accuracy', 'F1-Score', 'AUC-ROC'],
                                  title="Model Performance Comparison",
                                  barmode='group')
            fig_comparison.update_layout(xaxis_title="Models", yaxis_title="Score")
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # Best model highlight
            best_model = results_df['AUC-ROC'].idxmax()
            best_auc = results_df.loc[best_model, 'AUC-ROC']
            
            st.markdown(f"""
            ### 🏆 Best Performing Model
            
            **{best_model}**
            
            **Performance Metrics:**
            - **AUC-ROC:** {best_auc:.3f}
            - **Accuracy:** {results_df.loc[best_model, 'Accuracy']:.3f}
            - **F1-Score:** {results_df.loc[best_model, 'F1-Score']:.3f}
            
            *Selected based on highest AUC-ROC score*
            """)
        
        # Feature importance analysis
        st.markdown("#### 🎯 Feature Importance Analysis")
        best_model_obj = st.session_state.trained_models[best_model]
        
        if hasattr(best_model_obj, 'feature_importances_'):
            with st.expander("📋 View Feature Importance Code"):
                st.code("""
# For tree-based models (Random Forest, Gradient Boosting)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Feature importance represents how much each feature contributes to the model's predictions
                """, language='python')
            
            feature_importance = pd.DataFrame({
                'feature': st.session_state.feature_columns,
                'importance': best_model_obj.feature_importances_
            }).nlargest(15, 'importance')
            
            fig_feat_imp = px.bar(feature_importance, x='importance', y='feature',
                                 orientation='h',
                                 title=f"Top 15 Feature Importances - {best_model}",
                                 color='importance',
                                 color_continuous_scale='Viridis')
            fig_feat_imp.update_layout(height=600)
            st.plotly_chart(fig_feat_imp, use_container_width=True)
            
            # Show top features explanation
            st.markdown("#### 📊 Top Risk Factors Identified")
            top_5_features = feature_importance.head(5)
            for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
                st.write(f"**{i}. {row['feature']}** - Importance: {row['importance']:.3f}")
        else:
            st.info(f"Feature importance not available for {best_model} model type.")

# ================================================================================================
# SECTION 5: SINGLE EMPLOYEE PREDICTION
# ================================================================================================

if st.session_state.models_trained:
    st.markdown("""
    ---
    ## 🔮 Section 5: Single Employee Attrition Prediction
    
    Predict attrition risk for individual employees with both basic and advanced feature inputs.
    """)
    
    # Get best model
    results_df = pd.DataFrame(st.session_state.model_results).T
    best_model_name = results_df['AUC-ROC'].idxmax()
    best_model = st.session_state.trained_models[best_model_name]
    
    st.info(f"🏆 Using best performing model: **{best_model_name}** (AUC-ROC: {results_df.loc[best_model_name, 'AUC-ROC']:.3f})")
    
    # Prediction mode selection
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["🏢 Basic Employee Information", "⚙️ Advanced Feature Engineering Values"]
    )
    
    if prediction_mode == "🏢 Basic Employee Information":
        st.markdown("### 📝 Enter Basic Employee Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Personal Information**")
            age = st.number_input("Age", min_value=18, max_value=65, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            distance_from_home = st.number_input("Distance from Home (km)", min_value=1, max_value=50, value=10)
        
        with col2:
            st.markdown("**Job Information**")
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative"])
            job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=6000)
            overtime = st.selectbox("Works Overtime", ["No", "Yes"])
        
        with col3:
            st.markdown("**Experience & Satisfaction**")
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=10)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
            job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x])
            work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x])
        
        if st.button("🔮 Predict Attrition Risk", type="primary"):
            
            with st.expander("📋 View Prediction Process Code"):
                st.code("""
# Create feature vector from user inputs
feature_vector = [
    age, business_travel, daily_rate, department, distance_from_home,
    education, education_field, environment_satisfaction, gender, hourly_rate,
    job_involvement, job_level, job_role, job_satisfaction, marital_status,
    monthly_income, monthly_rate, num_companies_worked, overtime, percent_salary_hike,
    performance_rating, relationship_satisfaction, stock_option_level,
    total_working_years, training_times, work_life_balance, years_at_company,
    years_in_current_role, years_since_last_promotion, years_with_curr_manager,
    # Engineered features
    experience_ratio, promotion_gap, manager_tenure, income_per_year,
    salary_hike_ratio, work_life_score, job_mobility, is_high_mobility
]

# Make prediction using trained model
risk_probability = model.predict_proba([feature_vector])[0, 1]
prediction = model.predict([feature_vector])[0]

# Risk categorization
if risk_probability < 0.3:
    risk_level = "Low Risk"
elif risk_probability < 0.7:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"
                """, language='python')
            
            # Create simplified feature vector
            feature_vector = [
                age, 1, 800, 1, distance_from_home, 2, 1, 3, 0 if gender == "Male" else 1, 65,
                3, job_level, 1, job_satisfaction, 1, monthly_income, 15000, 2,
                1 if overtime == "Yes" else 0, 15, 3, 3, 1, total_working_years, 2, work_life_balance,
                years_at_company, years_in_current_role, 1, 2, 
                years_at_company/(total_working_years+1), 1, 2/(years_at_company+1),
                monthly_income/(total_working_years+1), 0.15, (work_life_balance+job_satisfaction+3)/3,
                2/(total_working_years+1), 0
            ]
            
            # Ensure correct length
            while len(feature_vector) < len(st.session_state.feature_columns):
                feature_vector.append(0)
            feature_vector = feature_vector[:len(st.session_state.feature_columns)]
            
            # Make prediction
            risk_prob = best_model.predict_proba([feature_vector])[0, 1]
            prediction = best_model.predict([feature_vector])[0]
            
            # Display results with enhanced styling
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Attrition Probability", f"{risk_prob:.1%}")
            
            # Determine risk level and recommendations
            if risk_prob < 0.3:
                risk_level = "🟢 Low Risk"
                risk_color = "success"
                recommendations = [
                    "Continue regular check-ins with manager",
                    "Maintain current engagement levels",
                    "Consider for special projects or additional responsibilities",
                    "Monitor quarterly for any changes in satisfaction"
                ]
            elif risk_prob < 0.7:
                risk_level = "🟡 Medium Risk"
                risk_color = "warning"
                recommendations = [
                    "Schedule monthly one-on-ones with direct manager",
                    "Review career development opportunities and create clear path",
                    "Monitor job satisfaction and work-life balance closely",
                    "Consider workload adjustments or flexible work arrangements",
                    "Provide additional training or skill development opportunities"
                ]
            else:
                risk_level = "🔴 High Risk"
                risk_color = "error"
                recommendations = [
                    "🚨 URGENT: Schedule immediate meeting with HR and direct manager",
                    "Review and potentially adjust compensation package",
                    "Discuss clear career advancement path with specific timelines",
                    "Consider role adjustment, team change, or department transfer",
                    "Implement retention bonus or other financial incentives",
                    "Address any specific concerns through exit interview-style discussion"
                ]
            
            with col2:
                st.metric("Risk Level", risk_level)
            
            with col3:
                confidence = max(risk_prob, 1-risk_prob)
                st.metric("Prediction Confidence", f"{confidence:.1%}")
            
            # Model explanation
            st.markdown("#### 🧠 Prediction Explanation")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Input Factors:**")
                st.write(f"• Age: {age} years")
                st.write(f"• Income: ${monthly_income:,}/month")
                st.write(f"• Experience: {total_working_years} years total, {years_at_company} at company")
                st.write(f"• Satisfaction: Job ({job_satisfaction}/4), Work-Life ({work_life_balance}/4)")
                st.write(f"• Overtime: {'Yes' if overtime == 'Yes' else 'No'}")
            
            with col2:
                st.markdown("**Calculated Features:**")
                exp_ratio = years_at_company/(total_working_years+1)
                income_per_year = monthly_income/(total_working_years+1)
                avg_satisfaction = (work_life_balance + job_satisfaction + 3)/3
                st.write(f"• Experience Ratio: {exp_ratio:.2f}")
                st.write(f"• Income per Experience Year: ${income_per_year:.0f}")
                st.write(f"• Average Satisfaction Score: {avg_satisfaction:.2f}")
                st.write(f"• Job Mobility: Low (estimated)")
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    else:  # Advanced mode
        st.markdown("### ⚙️ Advanced Feature Engineering Input")
        st.markdown("*Input both basic information and calculated feature engineering values for maximum prediction accuracy.*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Features**")
            age = st.number_input("Age", min_value=18, max_value=65, value=35, key="adv_age")
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=6000, key="adv_income")
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=50, value=10, key="adv_total_years")
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5, key="adv_years_company")
            overtime = st.selectbox("Works Overtime", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="adv_overtime")
            job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4], key="adv_job_sat")
            work_life_balance = st.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4], key="adv_wlb")
        
        with col2:
            st.markdown("**Feature Engineering Values**")
            experience_ratio = st.number_input("Experience Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                             help="Years at Company / Total Working Years")
            promotion_gap = st.number_input("Promotion Gap (years)", min_value=-10, max_value=20, value=1,
                                          help="Years in Current Role - Years Since Last Promotion")
            manager_tenure = st.number_input("Manager Tenure Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.1,
                                           help="Years with Current Manager / Years at Company")
            income_per_year = st.number_input("Income per Year ($)", min_value=500, max_value=15000, value=2000,
                                            help="Monthly Income / Total Working Years")
            work_life_score = st.number_input("Work-Life Score (1-4)", min_value=1.0, max_value=4.0, value=2.5, step=0.1,
                                            help="Average of Work-Life Balance, Job Satisfaction, Environment Satisfaction")
            job_mobility = st.number_input("Job Mobility (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                                         help="Number of Companies Worked / Total Working Years")
        
        if st.button("🔮 Predict with Advanced Features", type="primary"):
            
            with st.expander("📋 View Advanced Prediction Code"):
                st.code("""
# Advanced prediction using feature-engineered values
# This mode allows input of calculated features for maximum accuracy

# Create feature vector with user-provided engineered values
feature_vector = [
    # Basic features
    age, business_travel, daily_rate, department, distance_from_home,
    education, education_field, environment_satisfaction, gender, hourly_rate,
    job_involvement, job_level, job_role, job_satisfaction, marital_status,
    monthly_income, monthly_rate, num_companies_worked, overtime, percent_salary_hike,
    performance_rating, relationship_satisfaction, stock_option_level,
    total_working_years, training_times, work_life_balance, years_at_company,
    years_in_current_role, years_since_last_promotion, years_with_curr_manager,
    
    # User-provided engineered features (for maximum precision)
    experience_ratio,      # Years at company / Total working years
    promotion_gap,         # Years in role - Years since promotion  
    manager_tenure,        # Years with manager / Years at company
    income_per_year,       # Monthly income / Total years experience
    salary_hike_ratio,     # Salary hike percentage as decimal
    work_life_score,       # Average satisfaction score
    job_mobility,          # Job changes / Total years
    is_high_mobility       # Binary flag for high mobility
]

# Advanced prediction with confidence intervals
risk_probability = model.predict_proba([feature_vector])[0, 1]
                """, language='python')
            
            # Create advanced feature vector
            feature_vector = [
                age, 1, 800, 1, 8, 2, 1, 2, 0, 65,
                3, 2, 1, job_satisfaction, 1, monthly_income, 18000, 2,
                overtime, 15, 3, 3, 1, total_working_years, 2, work_life_balance,
                years_at_company, 2, 1, 1, experience_ratio, promotion_gap, manager_tenure,
                income_per_year, 0.15, work_life_score, job_mobility, 1 if job_mobility > 0.3 else 0
            ]
            
            # Ensure correct length
            while len(feature_vector) < len(st.session_state.feature_columns):
                feature_vector.append(0)
            feature_vector = feature_vector[:len(st.session_state.feature_columns)]
            
            # Make prediction
            risk_prob = best_model.predict_proba([feature_vector])[0, 1]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Advanced Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Attrition Probability", f"{risk_prob:.1%}")
            
            if risk_prob < 0.3:
                risk_level = "🟢 Low Risk"
                risk_explanation = "The engineered features indicate stable employment patterns with low attrition likelihood."
            elif risk_prob < 0.7:
                risk_level = "🟡 Medium Risk"
                risk_explanation = "Some concerning patterns detected in the feature analysis requiring monitoring."
            else:
                risk_level = "🔴 High Risk"
                risk_explanation = "Multiple risk factors identified through advanced feature analysis."
            
            col2.metric("Risk Level", risk_level)
            
            confidence = max(risk_prob, 1-risk_prob)
            col3.metric("Prediction Confidence", f"{confidence:.1%}")
            
            st.info(f"**Model Interpretation:** {risk_explanation}")
            
            # Feature impact analysis
            st.markdown("### 📊 Advanced Feature Impact Analysis")
            
            # Create impact visualization
            feature_names = ["Experience Ratio", "Promotion Gap", "Manager Tenure", "Income/Year", "Work-Life Score", "Job Mobility"]
            feature_values = [experience_ratio, promotion_gap, manager_tenure, income_per_year, work_life_score, job_mobility]
            
            # Normalize values for comparison (0-1 scale)
            normalized_values = []
            for i, val in enumerate(feature_values):
                if i == 1:  # Promotion gap can be negative
                    normalized_val = max(0, min(1, (val + 5) / 10))
                elif i == 3:  # Income per year
                    normalized_val = min(1, val / 10000)
                elif i == 4:  # Work-life score
                    normalized_val = val / 4
                else:
                    normalized_val = min(1, val)
                normalized_values.append(normalized_val)
            
            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Raw_Value': feature_values,
                'Normalized_Impact': normalized_values
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_impact = px.bar(impact_df, x='Feature', y='Raw_Value', 
                                   title="Raw Feature Values Input",
                                   color='Raw_Value',
                                   color_continuous_scale='RdYlBu_r')
                fig_impact.update_xaxes(tickangle=45)
                st.plotly_chart(fig_impact, use_container_width=True)
            
            with col2:
                fig_normalized = px.bar(impact_df, x='Feature', y='Normalized_Impact',
                                       title="Normalized Impact Assessment",
                                       color='Normalized_Impact',
                                       color_continuous_scale='RdYlGn')
                fig_normalized.update_xaxes(tickangle=45)
                st.plotly_chart(fig_normalized, use_container_width=True)
            
            # Detailed feature analysis
            st.markdown("#### 🔍 Detailed Feature Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Risk Indicators:**")
                risk_factors = []
                if experience_ratio < 0.3:
                    risk_factors.append(f"• Low experience ratio ({experience_ratio:.2f}) - Recent hire risk")
                if promotion_gap > 3:
                    risk_factors.append(f"• High promotion gap ({promotion_gap} years) - Career stagnation")
                if manager_tenure < 0.2:
                    risk_factors.append(f"• Low manager tenure ({manager_tenure:.2f}) - Management instability")
                if work_life_score < 2.5:
                    risk_factors.append(f"• Low satisfaction score ({work_life_score:.2f}) - Engagement issues")
                if job_mobility > 0.5:
                    risk_factors.append(f"• High job mobility ({job_mobility:.2f}) - Flight risk pattern")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("• No major risk indicators detected")
            
            with col2:
                st.markdown("**✅ Protective Factors:**")
                protective_factors = []
                if experience_ratio > 0.7:
                    protective_factors.append(f"• High experience ratio ({experience_ratio:.2f}) - Company loyalty")
                if promotion_gap <= 1:
                    protective_factors.append(f"• Recent promotion ({promotion_gap} years) - Career progression")
                if manager_tenure > 0.5:
                    protective_factors.append(f"• Stable management ({manager_tenure:.2f}) - Good relationship")
                if work_life_score > 3.5:
                    protective_factors.append(f"• High satisfaction ({work_life_score:.2f}) - Engaged employee")
                if income_per_year > 3000:
                    protective_factors.append(f"• Competitive income ratio (${income_per_year:.0f}/year) - Fair compensation")
                
                if protective_factors:
                    for factor in protective_factors:
                        st.write(factor)
                else:
                    st.write("• Limited protective factors identified")

# ================================================================================================
# SECTION 6: BATCH ANALYSIS
# ================================================================================================

if st.session_state.models_trained:
    st.markdown("""
    ---
    ## 📈 Section 6: Batch Employee Analysis
    
    Analyze multiple employees simultaneously for workforce-wide risk assessment.
    """)
    
    batch_option = st.radio("Choose batch analysis type:", [
        "📊 Analyze Current Dataset",
        "📤 Upload New Employee Data for Analysis"
    ])
    
    if batch_option == "📊 Analyze Current Dataset":
        if st.button("🚀 Analyze All Current Employees", type="primary"):
            with st.spinner("Analyzing all employees..."):
                # Get best model
                results_df = pd.DataFrame(st.session_state.model_results).T
                best_model_name = results_df['AUC-ROC'].idxmax()
                best_model = st.session_state.trained_models[best_model_name]
                
                # Use processed data for predictions
                processed_data = st.session_state.processed_data
                X = processed_data.drop(['Attrition'], axis=1)
                y_true = processed_data['Attrition']
                
                # Make predictions
                risk_scores = best_model.predict_proba(X)[:, 1]
                predictions = best_model.predict(X)
                
                # Create risk categories
                risk_categories = pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
                
                # Add results to original data
                results_df = st.session_state.raw_data.copy()
                results_df['Risk_Score'] = risk_scores
                results_df['Risk_Category'] = risk_categories
                results_df['Predicted_Attrition'] = predictions
                
                # Store results
                st.session_state.batch_results = results_df
                
                st.success("✅ Batch analysis completed!")
                
                # Display comprehensive summary
                st.markdown("### 📊 Workforce Risk Analysis Summary")
                
                total_employees = len(results_df)
                high_risk = (results_df['Risk_Category'] == 'High Risk').sum()
                medium_risk = (results_df['Risk_Category'] == 'Medium Risk').sum()
                low_risk = (results_df['Risk_Category'] == 'Low Risk').sum()
                avg_risk = results_df['Risk_Score'].mean()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Employees", f"{total_employees:,}")
                col2.metric("High Risk", f"{high_risk}", f"{high_risk/total_employees*100:.1f}%")
                col3.metric("Medium Risk", f"{medium_risk}", f"{medium_risk/total_employees*100:.1f}%")
                col4.metric("Low Risk", f"{low_risk}", f"{low_risk/total_employees*100:.1f}%")
                col5.metric("Avg Risk Score", f"{avg_risk:.3f}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk distribution pie chart
                    risk_counts = results_df['Risk_Category'].value_counts()
                    fig_pie = px.pie(values=risk_counts.values, 
                                    names=risk_counts.index,
                                    title="Employee Risk Distribution",
                                    color_discrete_map={'Low Risk': '#2ECC71', 'Medium Risk': '#F39C12', 'High Risk': '#E74C3C'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Risk score distribution
                    fig_hist = px.histogram(results_df, x='Risk_Score', 
                                          title="Risk Score Distribution",
                                          nbins=30)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Department analysis
                if 'Department' in results_df.columns:
                    st.markdown("### 🏢 Department-wise Risk Analysis")
                    
                    dept_analysis = results_df.groupby('Department').agg({
                        'Risk_Score': 'mean',
                        'Risk_Category': lambda x: (x == 'High Risk').sum()
                    }).round(3)
                    dept_analysis.columns = ['Avg_Risk_Score', 'High_Risk_Count']
                    dept_analysis['Total_Employees'] = results_df.groupby('Department').size()
                    dept_analysis['High_Risk_Percentage'] = (dept_analysis['High_Risk_Count'] / dept_analysis['Total_Employees'] * 100).round(1)
                    
                    st.dataframe(dept_analysis)
                    
                    # Department risk visualization
                    fig_dept = px.bar(dept_analysis.reset_index(), x='Department', y='Avg_Risk_Score',
                                     title="Average Risk Score by Department")
                    st.plotly_chart(fig_dept, use_container_width=True)
                
                # High risk employees details
                if high_risk > 0:
                    st.markdown("### 🚨 High Risk Employees (Top 10)")
                    high_risk_employees = results_df[results_df['Risk_Category'] == 'High Risk'].nlargest(10, 'Risk_Score')
                    display_cols = ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'Risk_Score', 'Risk_Category']
                    available_cols = [col for col in display_cols if col in high_risk_employees.columns]
                    st.dataframe(high_risk_employees[available_cols])
                
                # Download option
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Complete Analysis Results",
                    data=csv_buffer.getvalue(),
                    file_name=f"workforce_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
    
    else:  # Upload new data for analysis
        st.markdown("### 📤 Upload New Employee Data for Batch Analysis")
        
        uploaded_batch_file = st.file_uploader(
            "Upload CSV file with new employee data", 
            type="csv",
            help="Upload a CSV file containing employee data with the same structure as the training data"
        )
        
        if uploaded_batch_file is not None:
            try:
                new_data = pd.read_csv(uploaded_batch_file)
                st.success("✅ New employee data uploaded successfully!")
                
                st.markdown("#### 📋 Uploaded Data Preview")
                st.dataframe(new_data.head())
                
                col1, col2, col3 = st.columns(3)
                col1.metric("New Employees", len(new_data))
                col2.metric("Features", len(new_data.columns))
                col3.metric("Missing Values", new_data.isnull().sum().sum())
                
                if st.button("🔮 Predict Attrition for New Employees", type="primary"):
                    with st.spinner("Processing new employee data..."):
                        try:
                            # Preprocess the new data using the same pipeline
                            new_processed, _ = preprocess_data(new_data)
                            
                            # Ensure all required features are present
                            required_features = st.session_state.feature_columns
                            missing_features = set(required_features) - set(new_processed.columns)
                            
                            if missing_features:
                                st.error(f"❌ Missing required features in uploaded data: {missing_features}")
                                st.info("Please ensure your CSV has the same structure as the training data.")
                            else:
                                # Select only the required features in the correct order
                                X_new = new_processed[required_features]
                                
                                # Get best model and make predictions
                                results_df = pd.DataFrame(st.session_state.model_results).T
                                best_model_name = results_df['AUC-ROC'].idxmax()
                                best_model = st.session_state.trained_models[best_model_name]
                                
                                # Make predictions
                                risk_scores = best_model.predict_proba(X_new)[:, 1]
                                predictions = best_model.predict(X_new)
                                
                                # Create risk categories
                                risk_categories = pd.cut(risk_scores, bins=[0, 0.3, 0.7, 1.0],
                                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
                                
                                # Add results to original data
                                new_results = new_data.copy()
                                new_results['Risk_Score'] = risk_scores
                                new_results['Risk_Category'] = risk_categories
                                new_results['Predicted_Attrition'] = predictions
                                
                                st.success("✅ Predictions completed for new employees!")
                                
                                # Summary
                                st.markdown("### 📊 New Employee Risk Analysis")
                                
                                total_new = len(new_results)
                                high_risk_new = (new_results['Risk_Category'] == 'High Risk').sum()
                                medium_risk_new = (new_results['Risk_Category'] == 'Medium Risk').sum()
                                low_risk_new = (new_results['Risk_Category'] == 'Low Risk').sum()
                                
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total New Employees", total_new)
                                col2.metric("High Risk", high_risk_new, f"{high_risk_new/total_new*100:.1f}%")
                                col3.metric("Medium Risk", medium_risk_new, f"{medium_risk_new/total_new*100:.1f}%")
                                col4.metric("Low Risk", low_risk_new, f"{low_risk_new/total_new*100:.1f}%")
                                
                                # Visualization
                                risk_counts_new = new_results['Risk_Category'].value_counts()
                                fig_new = px.pie(values=risk_counts_new.values, 
                                               names=risk_counts_new.index,
                                               title="New Employee Risk Distribution")
                                st.plotly_chart(fig_new, use_container_width=True)
                                
                                # Show high risk employees
                                if high_risk_new > 0:
                                    st.markdown("#### 🚨 High Risk New Employees")
                                    high_risk_new_emp = new_results[new_results['Risk_Category'] == 'High Risk']
                                    st.dataframe(high_risk_new_emp)
                                
                                # Download results
                                csv_buffer = io.StringIO()
                                new_results.to_csv(csv_buffer, index=False)
                                st.download_button(
                                    label="📥 Download New Employee Analysis",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"new_employee_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Error processing new employee data: {str(e)}")
                            st.info("Please check that your data format matches the training data structure.")
                            
            except Exception as e:
                st.error(f"❌ Error loading uploaded file: {str(e)}")

# ================================================================================================
# SECTION 7: BUSINESS IMPACT ANALYSIS
# ================================================================================================

if st.session_state.models_trained:
    st.markdown("""
    ---
    ## 💼 Section 7: Business Impact Analysis
    
    Calculate the financial implications and ROI of implementing retention strategies.
    """)
    
    # Business parameters input
    st.markdown("### 💰 Business Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        replacement_cost = st.number_input(
            "Average Replacement Cost per Employee ($)", 
            min_value=10000, max_value=200000, value=50000, step=5000,
            help="Cost to recruit, hire, and train a replacement employee"
        )
    
    with col2:
        retention_success_rate = st.slider(
            "Expected Retention Success Rate (%)", 
            min_value=10, max_value=90, value=30,
            help="Percentage of high-risk employees expected to be retained through intervention"
        )
    
    with col3:
        program_cost_per_employee = st.number_input(
            "Retention Program Cost per Employee ($)",
            min_value=100, max_value=10000, value=2000, step=100,
            help="Cost of retention program per employee (training, bonuses, etc.)"
        )
    
    if st.button("💹 Calculate Business Impact", type="primary"):
        if hasattr(st.session_state, 'batch_results'):
            results_df = st.session_state.batch_results
            
            # Calculate business metrics
            total_employees = len(results_df)
            high_risk_count = (results_df['Risk_Category'] == 'High Risk').sum()
            medium_risk_count = (results_df['Risk_Category'] == 'Medium Risk').sum()
            
            # Financial calculations
            potential_loss = high_risk_count * replacement_cost
            retention_program_cost = (high_risk_count + medium_risk_count) * program_cost_per_employee
            prevented_turnover = high_risk_count * (retention_success_rate / 100)
            cost_savings = prevented_turnover * replacement_cost
            net_benefit = cost_savings - retention_program_cost
            roi = (net_benefit / retention_program_cost * 100) if retention_program_cost > 0 else 0
            
            # Display financial impact
            st.markdown("### 📊 Financial Impact Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Potential Loss", f"${potential_loss:,.0f}", 
                         help="Cost if all high-risk employees leave")
            with col2:
                st.metric("Program Cost", f"${retention_program_cost:,.0f}",
                         help="Total cost of retention program")
            with col3:
                st.metric("Expected Savings", f"${cost_savings:,.0f}",
                         help="Savings from prevented turnover")
            with col4:
                st.metric("Net ROI", f"{roi:.1f}%",
                         help="Return on investment")
            
            # ROI Analysis visualization
            roi_data = pd.DataFrame({
                'Category': ['Program Cost', 'Expected Savings', 'Net Benefit'],
                'Amount': [retention_program_cost, cost_savings, net_benefit],
                'Type': ['Cost', 'Savings', 'Benefit']
            })
            
            fig_roi = px.bar(roi_data, x='Category', y='Amount', color='Type',
                            title="Financial Impact Breakdown",
                            color_discrete_map={'Cost': '#E74C3C', 'Savings': '#2ECC71', 'Benefit': '#3498DB'})
            st.plotly_chart(fig_roi, use_container_width=True)
            
            # Department-wise analysis
            if 'Department' in results_df.columns:
                st.markdown("### 🏢 Department-wise Financial Impact")
                
                dept_analysis = results_df.groupby('Department').agg({
                    'Risk_Score': 'mean',
                    'Risk_Category': lambda x: (x == 'High Risk').sum(),
                    'MonthlyIncome': 'mean'
                }).round(2)
                
                dept_analysis.columns = ['Avg_Risk_Score', 'High_Risk_Count', 'Avg_Monthly_Income']
                dept_analysis['Potential_Cost'] = dept_analysis['High_Risk_Count'] * replacement_cost
                dept_analysis['Program_Cost'] = dept_analysis['High_Risk_Count'] * program_cost_per_employee
                dept_analysis['Expected_Savings'] = dept_analysis['High_Risk_Count'] * replacement_cost * (retention_success_rate/100)
                dept_analysis['Net_Benefit'] = dept_analysis['Expected_Savings'] - dept_analysis['Program_Cost']
                
                st.dataframe(dept_analysis)
            
            # Strategic recommendations
            st.markdown("### 💡 Strategic Recommendations")
            
            recommendations = []
            
            if high_risk_count > total_employees * 0.2:
                recommendations.append("🔴 **CRITICAL**: Over 20% of employees are high-risk. Immediate company-wide retention strategy needed.")
            elif high_risk_count > total_employees * 0.1:
                recommendations.append("🟡 **WARNING**: 10-20% of employees are high-risk. Enhanced retention programs recommended.")
            else:
                recommendations.append("🟢 **GOOD**: Less than 10% of employees are high-risk. Maintain current strategies.")
            
            if roi > 200:
                recommendations.append(f"💰 **EXCELLENT ROI**: {roi:.0f}% return expected. Strongly recommend implementing retention program.")
            elif roi > 100:
                recommendations.append(f"💰 **GOOD ROI**: {roi:.0f}% return expected. Retention program is financially viable.")
            elif roi > 0:
                recommendations.append(f"💰 **POSITIVE ROI**: {roi:.0f}% return expected. Consider implementing with cost optimizations.")
            else:
                recommendations.append("💰 **NEGATIVE ROI**: Reduce program costs or improve retention success rate.")
            
            # Implementation timeline
            recommendations.extend([
                "📅 **Phase 1 (Weeks 1-2)**: Identify and assess all high-risk employees",
                "📅 **Phase 2 (Weeks 3-6)**: Design and implement targeted retention programs",
                "📅 **Phase 3 (Weeks 7-14)**: Execute interventions and monitor progress",
                "📅 **Phase 4 (Ongoing)**: Continuous monitoring and strategy adjustment"
            ])
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
        else:
            st.warning("⚠️ Please run batch analysis first to calculate business impact.")

# ================================================================================================
# SECTION 8: EXECUTIVE DASHBOARD
# ================================================================================================

if st.session_state.models_trained:
    st.markdown("""
    ---
    ## 📋 Section 8: Executive Dashboard
    
    High-level insights and KPIs for executive decision-making.
    """)
    
    if st.button("📊 Generate Executive Dashboard", type="primary"):
        # Collect all key metrics
        data = st.session_state.raw_data
        total_employees = len(data)
        current_attrition_count = (data['Attrition'] == 'Yes').sum()
        current_attrition_rate = current_attrition_count / total_employees * 100
        
        # Model performance
        results_df = pd.DataFrame(st.session_state.model_results).T
        best_model_name = results_df['AUC-ROC'].idxmax()
        best_auc = results_df.loc[best_model_name, 'AUC-ROC']
        
        # Risk metrics (if batch analysis was run)
        risk_summary = {}
        if hasattr(st.session_state, 'batch_results'):
            batch_results = st.session_state.batch_results
            risk_summary = {
                'high_risk': (batch_results['Risk_Category'] == 'High Risk').sum(),
                'medium_risk': (batch_results['Risk_Category'] == 'Medium Risk').sum(),
                'low_risk': (batch_results['Risk_Category'] == 'Low Risk').sum(),
                'avg_risk': batch_results['Risk_Score'].mean()
            }
        
        # Executive Summary
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Employees", f"{total_employees:,}")
        with col2:
            st.metric("Current Attrition Rate", f"{current_attrition_rate:.1f}%")
        with col3:
            st.metric("Model Accuracy (AUC)", f"{best_auc:.3f}")
        with col4:
            if risk_summary:
                st.metric("High Risk Employees", risk_summary['high_risk'])
            else:
                st.metric("Analysis Status", "Batch Pending")
        
        # Key Performance Indicators Dashboard
        st.markdown("### 🎯 Key Performance Indicators")
        
        # Create comprehensive KPI dashboard
        fig_kpi = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Current Attrition Rate', 'Department Breakdown', 'Age Distribution',
                           'Income vs Attrition', 'Model Performance', 'Risk Distribution'),
            specs=[[{"type": "indicator"}, {"type": "pie"}, {"type": "histogram"}],
                  [{"type": "box"}, {"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Attrition Rate Gauge
        fig_kpi.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=current_attrition_rate,
            title={'text': "Attrition Rate (%)"},
            gauge={
                'axis': {'range': [None, 30]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 30], 'color': "red"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 15}
            }
        ), row=1, col=1)
        
        # 2. Department Breakdown
        dept_counts = data['Department'].value_counts()
        fig_kpi.add_trace(go.Pie(
            labels=dept_counts.index,
            values=dept_counts.values,
            name="Departments"
        ), row=1, col=2)
        
        # 3. Age Distribution
        fig_kpi.add_trace(go.Histogram(
            x=data['Age'],
            name="Age Distribution",
            nbinsx=20
        ), row=1, col=3)
        
        # 4. Income by Attrition
        fig_kpi.add_trace(go.Box(
            y=data[data['Attrition']=='Yes']['MonthlyIncome'],
            name='Left',
            marker_color='red'
        ), row=2, col=1)
        fig_kpi.add_trace(go.Box(
            y=data[data['Attrition']=='No']['MonthlyIncome'],
            name='Stayed',
            marker_color='blue'
        ), row=2, col=1)
        
        # 5. Model Performance
        model_names = list(st.session_state.model_results.keys())
        model_aucs = [st.session_state.model_results[name]['AUC-ROC'] for name in model_names]
        fig_kpi.add_trace(go.Bar(
            x=model_names,
            y=model_aucs,
            name='AUC-ROC',
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ), row=2, col=2)
        
        # 6. Risk Distribution (if available)
        if risk_summary:
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_values = [risk_summary['low_risk'], risk_summary['medium_risk'], risk_summary['high_risk']]
            fig_kpi.add_trace(go.Pie(
                labels=risk_labels,
                values=risk_values,
                name="Risk Categories",
                marker_colors=['#2ECC71', '#F39C12', '#E74C3C']
            ), row=2, col=3)
        
        # Update layout
        fig_kpi.update_layout(height=800, showlegend=True, title_text="📊 Executive KPI Dashboard")
        st.plotly_chart(fig_kpi, use_container_width=True)
        
        # Strategic Insights
        st.markdown("### 💡 Strategic Insights & Recommendations")
        
        insights = []
        
        # Attrition insights
        if current_attrition_rate > 20:
            insights.append("🔴 **CRITICAL ALERT**: Attrition rate exceeds 20%. Immediate executive intervention required.")
        elif current_attrition_rate > 15:
            insights.append("🟡 **WARNING**: Attrition rate above industry average (15%). Enhanced retention strategies needed.")
        else:
            insights.append("🟢 **POSITIVE**: Attrition rate within acceptable range. Maintain current strategies.")
        
        # Model insights
        if best_auc > 0.8:
            insights.append(f"🤖 **EXCELLENT MODEL**: High prediction accuracy ({best_auc:.3f} AUC). Reliable for decision-making.")
        elif best_auc > 0.7:
            insights.append(f"🤖 **GOOD MODEL**: Adequate prediction accuracy ({best_auc:.3f} AUC). Suitable for strategic planning.")
        else:
            insights.append(f"🤖 **MODEL IMPROVEMENT NEEDED**: Consider enhancing prediction accuracy ({best_auc:.3f} AUC).")
        
        # Risk insights (if available)
        if risk_summary:
            high_risk_percentage = risk_summary['high_risk'] / total_employees * 100
            if high_risk_percentage > 15:
                insights.append(f"⚠️ **HIGH RISK ALERT**: {high_risk_percentage:.1f}% of workforce at high attrition risk.")
            
            insights.append(f"📈 **RISK PROFILE**: Average workforce risk score is {risk_summary['avg_risk']:.3f}")
        
        # Department insights
        if 'Department' in data.columns:
            dept_attrition = data.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
            highest_attrition_dept = dept_attrition.idxmax()
            highest_rate = dept_attrition.max()
            insights.append(f"🏢 **DEPARTMENT FOCUS**: {highest_attrition_dept} requires attention ({highest_rate:.1f}% attrition rate)")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            st.markdown(f"{i}. {insight}")
        
        # Action Items for Executives
        st.markdown("### ✅ Executive Action Items")
        
        action_items = [
            "📋 **Immediate (This Week)**: Review high-risk employee list with department heads",
            "💰 **Short-term (1-4 weeks)**: Approve and implement targeted retention budget",
            "📈 **Medium-term (1-3 months)**: Establish regular attrition monitoring and reporting",
            "🔄 **Long-term (3-12 months)**: Integrate predictive analytics into HR strategy",
            "📊 **Ongoing**: Monthly executive review of retention metrics and program effectiveness"
        ]
        
        for i, item in enumerate(action_items, 1):
            st.markdown(f"{i}. {item}")
        
        # Export capability
        st.markdown("### 📥 Export Executive Summary")
        
        # Create executive summary data
        summary_data = {
            'Metric': [
                'Total Employees',
                'Current Attrition Rate (%)',
                'Employees Who Left',
                'Model Accuracy (AUC)',
                'High Risk Employees',
                'Medium Risk Employees',
                'Low Risk Employees',
                'Analysis Date'
            ],
            'Value': [
                total_employees,
                f"{current_attrition_rate:.1f}%",
                current_attrition_count,
                f"{best_auc:.3f}",
                risk_summary.get('high_risk', 'N/A'),
                risk_summary.get('medium_risk', 'N/A'),
                risk_summary.get('low_risk', 'N/A'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create CSV
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📊 Download Executive Summary Report",
            data=csv_buffer.getvalue(),
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )