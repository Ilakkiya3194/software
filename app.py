import ast
import math
import pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from radon.complexity import cc_visit
except ImportError:
    cc_visit = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Code Inspector AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# SIMPLE STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}

.metric-val {
    font-size: 22px;
    font-weight: bold;
    color: #58a6ff;
}

.metric-lbl {
    font-size: 11px;
    color: #8b949e;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "LOC",
    "CYCLO",
    "LENGTH",
    "VOLUME",
    "DIFFICULTY",
    "INT_FAN_IN",
    "INT_FAN_OUT",
    "NUM_OPERATORS",
    "NUM_OPERANDS",
    "BRANCH_COUNT"
]


# ============================================================
# LOAD DATASET AND TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = pd.read_csv("SoftwareDefectDataset.csv")

    X = data[FEATURES]
    y = data["DEFECT_LABEL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, confusion_matrix(y_test, y_pred)


# ============================================================
# TRAIN MODEL
# ============================================================

try:

    model, accuracy, cm = train_model()

except FileNotFoundError:

    st.error(
        "SoftwareDefectDataset.csv not found. "
        "Please keep the CSV file in the same folder as this app."
    )

    st.stop()

except Exception as e:

    st.error(f"Model training error: {e}")
    st.stop()


# ============================================================
# SAMPLE CODE
# ============================================================

SAMPLES = {

    "Write / Paste Custom Code": "",

    "Nested Loop (High Risk)": """
def process_data(items, flags, limit, mode):

    result = []

    if items:

        for item in items:

            if item.is_active:

                for flag in flags:

                    if flag.valid and len(result) < limit:

                        if mode == "A":
                            result.append(item.value * 2)

                        elif mode == "B":
                            result.append(item.value + 1)

    return result
""",

    "Clean Function (Low Risk)": """
def calculate_total(items):

    return sum(
        item.price
        for item in items
        if item.is_valid
    )
"""
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(code_str):

    tree = ast.parse(code_str)

    # LOC
    loc = len([
        line
        for line in code_str.splitlines()
        if line.strip()
    ])

    # FUNCTIONS
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
    ]

    # BRANCHES
    branches = sum(
        1
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.If, ast.For, ast.While, ast.Try)
        )
    )

    # LOOPS
    loops = sum(
        1
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.For, ast.While)
        )
    )

    # CYCLOMATIC COMPLEXITY
    if cc_visit:

        try:

            blocks = cc_visit(code_str)

            cyclo = sum(
                block.complexity
                for block in blocks
            )

        except Exception:

            cyclo = max(
                1,
                branches + len(functions)
            )

    else:

        cyclo = max(
            1,
            branches + len(functions)
        )

    # OPERATORS AND OPERANDS
    operators = 0
    operands = 0

    for node in ast.walk(tree):

        if isinstance(
            node,
            (
                ast.BinOp,
                ast.BoolOp,
                ast.Compare
            )
        ):
            operators += 1

        if isinstance(
            node,
            (
                ast.Name,
                ast.Constant
            )
        ):
            operands += 1

    # METRICS
    length = operators + operands

    volume = (
        length * math.log2(length + 1)
        if length > 0
        else 0
    )

    difficulty = (
        (operators / 2) *
        (operands / (operators + 1))
    )

    # FAN IN / FAN OUT
    fan_in = len(functions)

    fan_out = len([
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ])

    # DATAFRAME
    features = pd.DataFrame([{

        "LOC": loc,
        "CYCLO": cyclo,
        "LENGTH": length,
        "VOLUME": volume,
        "DIFFICULTY": difficulty,
        "INT_FAN_IN": fan_in,
        "INT_FAN_OUT": fan_out,
        "NUM_OPERATORS": operators,
        "NUM_OPERANDS": operands,
        "BRANCH_COUNT": branches

    }])

    return features


# ============================================================
# ANALYZE CODE
# ============================================================

def analyze_code(code_str):

    if not code_str.strip():

        return None, (
            "Please enter Python code "
            "and click Predict Defect Risk."
        )

    # Check syntax
    try:

        ast.parse(code_str)

    except SyntaxError as e:

        return None, (
            f"Syntax Error "
            f"(Line {e.lineno}): {e.msg}"
        )

    try:

        # Extract features
        features = extract_features(code_str)

        # Prediction
        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        # Find probability of defect class
        if 1 in model.classes_:

            defect_index = list(model.classes_).index(1)

            defect_probability = (
                probabilities[defect_index] * 100
            )

        else:

            defect_probability = 0

        # Metrics
        metrics = {

            "LOC": int(features["LOC"][0]),

            "Complexity": float(
                features["CYCLO"][0]
            ),

            "Functions": int(
                features["INT_FAN_IN"][0]
            ),

            "Branches": int(
                features["BRANCH_COUNT"][0]
            ),

            "Risk": defect_probability,

            "Prediction": prediction

        }

        return metrics, None

    except Exception as e:

        return None, str(e)


# ============================================================
# SESSION STATE
# ============================================================

if "code_text" not in st.session_state:

    st.session_state.code_text = (
        SAMPLES["Nested Loop (High Risk)"]
    )


if "analyzed_results" not in st.session_state:

    st.session_state.analyzed_results = None


# ============================================================
# LOAD SAMPLE
# ============================================================

def load_sample():

    selected = st.session_state.preset_selection

    st.session_state.code_text = SAMPLES[selected]


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction():

    metrics, error = analyze_code(
        st.session_state.code_text
    )

    st.session_state.analyzed_results = (
        metrics,
        error
    )


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Code Inspector AI")

st.caption(
    "Machine Learning Based Python Software Defect Prediction"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.success(
    f"Random Forest Model Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# TWO COLUMNS
# ============================================================

col_left, col_right = st.columns(
    [1, 1],
    gap="medium"
)


# ============================================================
# LEFT SIDE
# ============================================================

with col_left:

    st.selectbox(
        "Load Preset Example:",
        options=list(SAMPLES.keys()),
        key="preset_selection",
        on_change=load_sample
    )

    st.text_area(
        "Source Code Editor",
        key="code_text",
        height=320,
        placeholder="Type or paste your Python code here..."
    )

    st.button(
        "⚡ Predict Defect Risk",
        on_click=run_prediction,
        use_container_width=True
    )


# ============================================================
# RIGHT SIDE
# ============================================================

with col_right:

    if st.session_state.analyzed_results is None:

        st.info(
            "👈 Enter or paste Python code "
            "and click Predict Defect Risk."
        )

    else:

        metrics, error = (
            st.session_state.analyzed_results
        )

        if error:

            st.error(error)

        elif metrics:

            probability = metrics["Risk"]

            # =================================================
            # RISK CATEGORY
            # =================================================

            if probability < 35:

                risk_color = "#238636"
                risk_label = "LOW RISK"

            elif probability < 65:

                risk_color = "#d29922"
                risk_label = "MEDIUM RISK"

            else:

                risk_color = "#da3633"
                risk_label = "HIGH RISK"


            # =================================================
            # PREDICTION
            # =================================================

            if metrics["Prediction"] == 1:

                prediction_text = "DEFECT DETECTED"

            else:

                prediction_text = "NO DEFECT DETECTED"


            st.markdown(
                f"""
                ### Defect Prediction:

                <span style='color:{risk_color};
                font-weight:bold;
                font-size:20px'>
                {prediction_text}
                </span>

                <br>

                <b>{risk_label}</b>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                **Estimated Defect Probability:
                {probability:.2f}%**
                """
            )


            # =================================================
            # METRIC CARDS
            # =================================================

            m1, m2, m3, m4 = st.columns(4)

            m1.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-val">
                {metrics["LOC"]}
                </div>
                <div class="metric-lbl">
                LOC
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            m2.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-val">
                {metrics["Complexity"]}
                </div>
                <div class="metric-lbl">
                Complexity
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            m3.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-val">
                {metrics["Branches"]}
                </div>
                <div class="metric-lbl">
                Branches
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            m4.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-val">
                {metrics["Functions"]}
                </div>
                <div class="metric-lbl">
                Functions
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # GAUGE
            # =================================================

            fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=probability,

                    number={
                        "suffix": "%"
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100]
                        },

                        "bar": {
                            "color": risk_color
                        },

                        "steps": [

                            {
                                "range": [0, 35],
                                "color":
                                "rgba(35,134,54,0.2)"
                            },

                            {
                                "range": [35, 65],
                                "color":
                                "rgba(210,153,34,0.2)"
                            },

                            {
                                "range": [65, 100],
                                "color":
                                "rgba(218,54,51,0.2)"
                            }

                        ]

                    }

                )
            )

            fig.update_layout(

                height=160,

                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                font={
                    "color": "white"
                }

            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )


            # =================================================
            # CODE ANALYSIS
            # =================================================

            st.markdown("##### Code Analysis")

            has_issues = False

            if metrics["Complexity"] > 5:

                st.warning(
                    "High cyclomatic complexity detected. "
                    "Consider simplifying the logic."
                )

                has_issues = True

            if metrics["Branches"] > 3:

                st.warning(
                    "High branching detected. "
                    "Consider breaking the code into smaller functions."
                )

                has_issues = True

            if metrics["LOC"] > 40:

                st.warning(
                    "Large code block detected. "
                    "Consider modularizing the code."
                )

                has_issues = True

            if not has_issues:

                st.success(
                    "Code structure looks relatively simple."
                )


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("📊 Model Evaluation"):

    st.write(
        f"**Random Forest Accuracy:** "
        f"{accuracy * 100:.2f}%"
    )

    st.write("**Confusion Matrix:**")

    st.write(cm)