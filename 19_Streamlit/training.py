import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Training a Model")
st.write("Let's train a Random Forest Classifier to predict the wine quality")

parameters = {
    "max_depth": [4, 32],
    "min_samples_leaf": [1, 8],
    "min_weight_fraction_leaf": [0.0, 0.0001, 0.001],
    "min_samples_split": [4, 32],
    "max_features": [2, 12],
    "max_leaf_nodes": [200, 1200],
    'min_impurity_decrease': [0.0, 0.0001, 0.001]
}


def get_slider_step(min_value, max_value, integer_values):
    if integer_values:
        return 1

    range_size = max_value - min_value
    if range_size == 0:
        return 0.01

    return max(range_size / 100, 0.000001)


selected_parameters = {}

with st.form("rf_hyperparameters_form"):
    for parameter_name, parameter_values in parameters.items():
        if all(value < 1 for value in parameter_values):
            selected_parameters[parameter_name] = st.selectbox(
                parameter_name,
                options=parameter_values,
            )
            continue

        min_value, max_value = parameter_values[0], parameter_values[1]
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        integer_values = isinstance(min_value, int) and isinstance(max_value, int)
        step = get_slider_step(min_value, max_value, integer_values)

        if integer_values:
            selected_parameters[parameter_name] = st.slider(
                parameter_name,
                min_value=int(min_value),
                max_value=int(max_value),
                value=int(min_value),
                step=int(step),
            )
        else:
            selected_parameters[parameter_name] = st.slider(
                parameter_name,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(min_value),
                step=float(step),
            )

    submit_button = st.form_submit_button("Train Model")

if submit_button:
    wine_df = pd.read_csv("wine-quality-white-and-red.csv")

    # Convert categorical columns (e.g. wine type) into numeric features.
    X = pd.get_dummies(wine_df.drop(columns=["quality"]), drop_first=True)
    y = wine_df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        **selected_parameters,
        warm_start=True,
        random_state=42,
    )

    st.success("RandomForestClassifier created successfully")
    st.write(model)

    st.subheader("Training Error vs Test Error")
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)

    error_history = []
    estimator_steps = list(range(10, 201, 10))

    for index, n_estimators in enumerate(estimator_steps, start=1):
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        train_error = 1 - model.score(X_train, y_train)
        test_error = 1 - model.score(X_test, y_test)

        error_history.append(
            {
                "n_estimators": n_estimators,
                "Training Error": train_error,
                "Test Error": test_error,
            }
        )

        error_df = pd.DataFrame(error_history).set_index("n_estimators")
        chart_placeholder.line_chart(error_df)
        progress_bar.progress(index / len(estimator_steps))

    st.caption("Real-time plot updated while increasing the number of trees.")

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
        .set_index("feature")
    )

    st.subheader("Top Feature Importances")
    st.bar_chart(importance_df)