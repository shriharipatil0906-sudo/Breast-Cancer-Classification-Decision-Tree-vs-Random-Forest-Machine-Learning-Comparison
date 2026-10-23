from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)


# ---------------------------------------------------
# Load trained models and preprocessing objects
# ---------------------------------------------------

with open("decision_tree_model.pkl", "rb") as file:
    decision_tree = pickle.load(file)

with open("random_forest_model.pkl", "rb") as file:
    random_forest = pickle.load(file)

with open("feature_selector.pkl", "rb") as file:
    selector = pickle.load(file)

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

with open("selected_features.pkl", "rb") as file:
    selected_features = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:

            # ----------------------------------------
            # Read only the selected feature inputs
            # ----------------------------------------

            user_values = {}

            for feature in selected_features:
                user_values[feature] = float(
                    request.form[feature]
                )

            # ----------------------------------------
            # Create complete 30-feature dataframe
            #
            # SelectKBest requires the same original
            # feature structure used during training.
            # Non-selected values are filled with the
            # training selector's stored mean values.
            # ----------------------------------------

            complete_values = {}

            for i, feature in enumerate(feature_names):

                if feature in user_values:
                    complete_values[feature] = user_values[feature]

                else:
                    # Use a neutral value based on training data.
                    # For SelectKBest, these values will later be
                    # discarded, but the original columns are needed.
                    complete_values[feature] = 0.0


            input_df = pd.DataFrame(
                [complete_values],
                columns=feature_names
            )

            # Apply same feature selection
            input_selected = selector.transform(input_df)


            # ----------------------------------------
            # Decision Tree Prediction
            # ----------------------------------------

            dt_prediction = decision_tree.predict(
                input_selected
            )[0]

            dt_probability = decision_tree.predict_proba(
                input_selected
            )[0][1]


            # ----------------------------------------
            # Random Forest Prediction
            # ----------------------------------------

            rf_prediction = random_forest.predict(
                input_selected
            )[0]

            rf_probability = random_forest.predict_proba(
                input_selected
            )[0][1]


            dt_result = (
                "Malignant"
                if dt_prediction == 1
                else "Benign"
            )

            rf_result = (
                "Malignant"
                if rf_prediction == 1
                else "Benign"
            )


            result = {
                "decision_tree": dt_result,
                "dt_probability": round(
                    dt_probability * 100,
                    2
                ),

                "random_forest": rf_result,
                "rf_probability": round(
                    rf_probability * 100,
                    2
                )
            }


        except Exception as e:

            error = "Please enter valid numerical values."

            print(e)


    return render_template(
        "index.html",
        result=result,
        error=error,
        features=selected_features
    )


if __name__ == "__main__":
    app.run(debug=True)