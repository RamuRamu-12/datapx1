import os
import json
import shutil
import joblib
import base64
import io
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from django.utils.safestring import mark_safe
import dateutil.parser
import pandas as pd
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
# For genai using plotly
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import dateutil.parser
import re
import pmdarima as pm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller
from plotly.graph_objects import Figure

# Create your views here.



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configure OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs('uploads', exist_ok=True)


def index(request):
    return HttpResponse("Hai")


@csrf_exempt
def gpt_response(request):
    try:
        prompt = request.POST.get('prompt')
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        processed_data = process_response(response)
        return HttpResponse(json.dumps({'result': processed_data}), content_type="application/json")
    except Exception as e:
        print(e)
        return str(e)


def process_response(response_data):
    processed_data = ""
    # Display generated content dynamically
    for choice in response_data.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        processed_data += chunk_message
    processed_data = processed_data.lower().replace('```python', '').replace('```', '')
    print(processed_data)
    return processed_data


@csrf_exempt
def uploadFile(request):
    """
    This method is used for uploading files
    @args: None
    returns: None
    """
    try:
        if request.method == 'POST':
            file = request.FILES.get('file')
            if not file:
                return HttpResponse('No files uploaded')
            if os.path.exists(os.path.join(os.getcwd(), "uploads")):
                shutil.rmtree(os.path.join(os.getcwd(), "uploads"))
                os.makedirs("uploads", exist_ok=True)
            if os.path.exists("models"):
                shutil.rmtree('models')
            with default_storage.open(os.path.join('uploads', f'{file.name.split(".")[0]}.csv'),
                                      'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            df = pd.read_csv(os.path.join('uploads', f'{file.name.split(".")[0]}.csv'))
            # df.to_csv("data.csv", index=False)
            new_df, html_df = process_missing_data(df.copy())
            new_df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)
            new_df.to_csv("data.csv", index=False)
            with open(os.path.join('mvt_data.json'), 'w') as fp:
                json.dump({'data': html_df}, fp, indent=4)
            return HttpResponse(json.dumps({'status': "Success", "data": df.to_json(), 'file_name': file.name}),
                                content_type="application/json")
    except Exception as e:
        return HttpResponse(str(e))


# Ensure JSON serialization by converting NumPy arrays to lists
def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


@csrf_exempt
def gpt_graphical(request):
    if request.method == "POST":
        try:
            # Load CSV
            csv_file_path = 'data.csv'
            df = pd.read_csv(csv_file_path)

            # Generate CSV metadata
            csv_metadata = {"columns": df.columns.tolist()}
            metadata_str = ", ".join(csv_metadata["columns"])

            # User's query
            query = request.POST.get("prompt", "")

            # Prompt engineering for AI
            prompt_eng = (
                f"""You are an AI specialized in data analytics and visualization.
                        
                        The data for analysis is stored in a CSV file named 'data.csv', with the following attributes: {metadata_str}. Consider 'data.csv' as the sole data source for any analysis.
                        
                        Based on the user's query, generate Python code using Plotly to create the requested type of graph (e.g., bar, pie, scatter, etc.). If the user does not specify a graph type, determine whether to generate a line or bar graph based on the context.
                        
                        Ensure the graph meets the following criteria:
                        
                        Includes a title, axis labels (if applicable), and appropriate colors for data visualization.
                        Has a white background for both the plot and the paper.
                        Is visually appealing and provides sufficient context for understanding.
                        
                        The generated code must:
                        
                        Output a Plotly 'Figure' object stored in a variable named 'fig'.
                        Include the 'data' and 'layout' dictionaries required for the graph.
                        Be fully compatible with React.
                        User query: {query}
                """

            )
            trials = 3
            try:
                # Call AI to generate the code
                chat = generate_code(prompt_eng)
                print("Generated code from AI:")
                print(chat)

                # Check for valid Plotly code in the AI response
                if 'import' in chat:
                    namespace = {}
                    try:
                        # Execute the generated code
                        exec(chat, namespace)

                        # Retrieve the Plotly figure from the namespace
                        fig = namespace.get("fig")

                        if fig and isinstance(fig, Figure):
                            # Convert the Plotly figure to JSON
                            chart_data = fig.to_plotly_json()

                            # Recursively process the chart_data
                            chart_data_serializable = make_serializable(chart_data)

                            # Return the structured response to the frontend
                            return JsonResponse({
                                "chartData": chart_data_serializable
                            }, status=200)
                        else:
                            print("No valid Plotly figure found.")
                            return JsonResponse({"message": "No valid Plotly figure found."}, status=200)
                    except Exception as e:
                        error_message = f"There was an error while executing the code: {str(e)}"
                        print(error_message)
                        return JsonResponse({"message": error_message}, status=500)
                else:
                    print("Invalid AI response.")
                    return JsonResponse({"message": "AI response does not contain valid code."}, status=400)
            except Exception as e:
                pass
            trials -= 1

        except Exception as e:
            # Handle general exceptions
            error_message = f"An unexpected error occurred: {str(e)}"
            print(error_message)
            return JsonResponse({"message": error_message}, status=500)

    # Return a fallback HttpResponse for invalid request methods
    return HttpResponse("Invalid request method", status=405)


# Function to generate code from OpenAI API
def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


def data_processing(request):
    if request.method == 'GET':
        if os.path.exists(os.path.join('data.csv')):
            df = pd.read_csv(os.path.join('data.csv'))
            df = updatedtypes(df)
            if df.shape[0] > 0:
                nullvalues = df.isnull().sum().to_dict()
                parameters = list(nullvalues.keys())
                Count = list(nullvalues.values())
                total_missing = df.isnull().sum().sum()
                # df, html_df = process_missing_data(df)
                # cache.set('dataframe', html_df)
                # df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)
                nor = df.shape[0]
                nof = df.shape[1]
                timestamp = 'N'
                boolean = 'N'
                categorical_vars = []
                boolean_vars = []
                numeric_vars = {}
                datetime_vars = []
                text_data = []
                td = None
                stationary = "NA"
                numfilter = ['25%', '50%', '75%']
                single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
                df.drop(single_value_columns, axis=1, inplace=True)
                for i, j in df.dtypes.items():
                    if str(j) in ["float64", "int64"]:
                        data = df[i].describe().to_dict()
                        temp = [data.pop(key) for key in numfilter]
                        numeric_vars[i] = data
                    elif str(j) in ["object"] and i not in ['Remark']:
                        categorical_vars.append({i: df[i].nunique()})
                    elif str(j) in ["datetime64[ns]"]:
                        if i.upper() in ['DATE', "TIME", "DATE_TIME"]:
                            td = i
                        datetime_vars.append(i)
                    elif str(j) in ["bool"]:
                        boolean_vars.append(i)
                request.session['TimeSeriesColumns'] = datetime_vars
                if 'Remark' in df.columns:
                    text_data.append('Remark')
                istextdata = 'Y' if len(text_data) > 0 else 'N'
                if len(datetime_vars) > 0:
                    timestamp = 'Y'
                if td:
                    stationary = adf_test(df, td)
                catvalues = [{'Parameter': list(data.keys())[0], 'Count': list(data.values())[0]} for data in
                             categorical_vars]
                sentiment = checkSentiment(df, categorical_vars)
                if len(catvalues) > 0:
                    catdf = pd.DataFrame(catvalues)
                else:
                    catdf = 'NA'
                if len(numeric_vars) > 0:
                    numdf = pd.DataFrame(numeric_vars).T
                    numdf.columns = ['Count', 'Mean', 'Std', 'Min', 'Max']
                    numdf = numdf
                else:
                    numdf = pd.DataFrame()
                if len(boolean_vars) > 0:
                    boolean = 'Y'

                missingvalue = pd.DataFrame({"Parameters": parameters, 'Missing Value Count': Count})

                duplicate_records = df[df.duplicated(keep='first')].shape[0]
                for d in ['bar', 'pie', 'wordCloud']:
                    if os.path.exists(os.path.join(os.getcwd(), f'static/plots/{d}/')):
                        for f in os.listdir(os.path.join(os.getcwd(), f'static/plots/{d}/')):
                            os.remove(os.path.join(os.path.join(os.getcwd(), f'static/plots/{d}/'), f))
                    else:
                        os.makedirs(os.path.join(os.getcwd(), f'static/plots/{d}/'), exist_ok=True)

                barplots = plot_numeric(numeric_vars, df)

                pieplots = plot_categorical(categorical_vars, df)

                wordColudPlots = plot_wordCloud(text_data, df)

                return JsonResponse(
                    {'nof_rows': str(nor), 'nof_columns': str(nof), 'timestamp': timestamp,
                     "single_value_columns": ",".join(single_value_columns) if len(
                         single_value_columns) > 0 else "NA",
                     "sentiment": sentiment,
                     "stationary": stationary,
                     'catdf': catdf.to_json(orient='records'),
                     'missing_data': str(total_missing),
                     'numdf': numdf.to_json(orient='records') if numdf.shape[0] > 0 else "No data", 'boolean': boolean,
                     'missingvalue': missingvalue.to_json(orient='records'),
                     'textdata': istextdata, 'duplicate_records': str(duplicate_records),
                     'barplots': barplots,
                     'pieplots': pieplots,
                     'wordCloudPlots': wordColudPlots,
                     })

            else:
                return HttpResponse("No data")
        else:
            return HttpResponse(json.dumps(
                {'msg': 'Please upload file'}))
    else:
        return HttpResponse('Invalid Request')


@csrf_exempt
def kpi_prompt(request):
    try:
        if request.method == "POST":
            global KPI_LOGICS, checks
            KPI_LOGICS = defaultdict()
            checks = []
            prompt = request.POST.get('prompt')
            if not os.path.exists('data.csv'):
                return HttpResponse('No files uploaded')
            else:
                df = pd.read_csv("data.csv")
                prompt_desc = (
                    f"You are analytics_bot. Analyse the data: {df.head()} and for the uer query {prompt}, "
                    f"generate kpis with response as KPI Name, Column and Logic. Response should be in python dictionary format  with kpi names as keys. In response dont add any other information just provide only the response dictionary"
                )
                n = 2
                while n > 0:
                    genai_res = generate_code(prompt_desc)
                    data_dict = json.loads(genai_res)
                    print("datadict", data_dict)
                    for key, value in data_dict.items():
                        value = {i.lower(): j for i, j in value.items()}
                        if 'kpi name' in value:
                            kpi_name = value['kpi name']
                        elif 'name' in value:
                            kpi_name = value["name"]
                        else:
                            kpi_name = key
                        KPI_LOGICS[key] = {
                            "KPI Name": kpi_name,
                            "Column": value["column"],
                            "Logic": value["logic"]
                        }
                    if KPI_LOGICS is not None:
                        if not os.path.exists('kpis.json'):
                            kpis_store = dict()
                        else:
                            with open('kpis.json', 'r') as fp:
                                kpis_store = json.load(fp)
                        with open('kpis.json', 'w') as fp:
                            kpis_store.update(KPI_LOGICS)
                            json.dump(kpis_store, fp)
                        break
                if os.path.exists(os.path.join('uploads', 'kpi_config.json')):
                    with open('uploads/kpi_config.json', 'r') as json_file:
                        kpis_dict = json.load(json_file)
                    for kpi in kpis_dict['Kpis']['kpi']:
                        KPI_LOGICS[kpi['KPI_Name']] = kpi
                        checks.append(kpi['KPI_Name'])
                return JsonResponse(
                    {
                        "kpis": KPI_LOGICS, "checks": checks}
                )
    except Exception as e:
        print(e)


@csrf_exempt
def mvt(request):
    with open(os.path.join('mvt_data.json'), 'r') as fp:
        data = json.load(fp)
    return JsonResponse(
        {
            "df": data['data']
        }
    )


@csrf_exempt
def kpi_code(request):
    try:
        if request.method == "POST":
            kpi_list = request.POST.getlist("kpi_names")
            paths, codes = generate_kpi_code(kpi_list)
            return JsonResponse({
                'plots': paths,
                'code': codes,
                "kpis": KPI_LOGICS,
                "checks": checks
            })
    except Exception as e:
        print(e)


@csrf_exempt
def models(request):
    print("model")
    try:
        if not os.path.exists(os.path.join("uploads", 'processed_data.csv')):
            mvt(request)
        df = pd.read_csv(os.path.join("uploads", 'processed_data.csv'))
        single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
        df.drop(single_value_columns, axis=1, inplace=True)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) < 1:
            return JsonResponse(
                {
                    "msg": "This dataset doesn't meet the modeling requirement "}
            )
        if request.method == 'POST':
            model_type = request.POST.get('model')
            col = request.POST.get('col')
            request.session['col_predict'] = col
            if model_type == 'RandomForest':
                stat, cols = random_forest(df, col)
                return JsonResponse(
                    {
                        'columns': list(df.columns),
                        "rf": True,
                        "status": stat,
                        "rf_cols": cols
                    })
            elif model_type == "K-Means":
                stat, clustered_data = kmeans_train(df)
                return JsonResponse(
                    {
                        'columns': list(df.columns),
                        "cluster": True,
                        "status": stat,
                        "clustered_data": clustered_data.to_json()
                    })
            elif model_type == "Arima":
                print("arima model")
                stat, img_data = arima_train(df, col)
                return JsonResponse(
                    {
                        'columns': list(df.columns),
                        "status": stat,
                        "arima": True,
                        "path": img_data,
                    })
            elif model_type == 'OutlierDetection':
                res = outliercheck(df, col)

                return JsonResponse(
                    {
                        'columns': list(df.columns),
                        "status": True,
                        "processed_data": res,
                        "OutlierDetection": True
                    })
        else:
            return JsonResponse(
                {
                    'columns': list(df.columns)}
            )
    except Exception as e:
        print(e)
        return JsonResponse(
            {
                "msg": str(e)}
        )


def outliercheck(df, column):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'detect outliers for  the following data {df[column]}'}
        ]
    )
    all_text = ""
    # Display generated content dynamically
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    return all_text


def random_forest(data, target_column):
    try:
        if not os.path.exists(os.path.join("models", "rf", target_column, 'deployment.json')):
            os.makedirs(os.path.join("models", "rf", target_column), exist_ok=True)
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if y.nunique() <= 5:  # Classification for few unique target values
                model_type = 'Classification'
                model = RandomForestClassifier(random_state=42)
            else:  # Regression for continuous target values
                model_type = 'Regression'
                model = RandomForestRegressor(random_state=42)

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            cv = min(5, len(X_test))

            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv)
            print(f"Model Performance (CV): {scores.mean():.4f} ± {scores.std():.4f}")

            # Save the pipeline
            joblib.dump(pipeline, os.path.join("models", "rf", target_column, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join("models", "rf", target_column, "pipeline.pkl")}')

            with open(os.path.join("models", "rf", target_column, "deployment.json"), "w") as fp:
                json.dump({"columns": list(X_train.columns), "model_type": model_type, "Target_column": target_column},
                          fp, indent=4)
            return True, list(X_train.columns)
        else:
            with open(os.path.join(os.getcwd(), "models", "rf", target_column, 'deployment.json'), "r") as fp:
                data = json.load(fp)
            return True, data['columns']
    except Exception as e:
        print(e)
        return False, []


@csrf_exempt
def model_predict(request):
    try:
        if request.POST.get('form_name') == 'rf':
            res = {}
            for col in request.POST:
                if col == "targetColumn":
                    targetcol = request.POST[col]
                    continue
                res.update({col: request.POST[col]})
            del res['form_name']
            df = pd.DataFrame([res])
            loaded_pipeline = load_pipeline(
                os.path.join("models", "rf", targetcol, "pipeline.pkl"))
            predictions = loaded_pipeline.predict(df)
            print(predictions)
            return JsonResponse(
                {
                    'columns': list(df.columns),
                    'rf_result': f"Predicted {targetcol} value is {round(predictions[0], 2)}"
                }
            )

    except Exception as e:
        print(e)
        return JsonResponse(
            {
                'columns': [],
                'rf_result': "NA"
            }
        )


def load_models(model_type, df, target_col):
    try:
        if model_type == 'rf':
            model = load_model(os.path.join(os.getcwd(), 'models', "rf", target_col, "model.h5"))
            with open(os.path.join(os.getcwd(), 'models', "rf", target_col, "deployment.json"), 'r') as fp:
                deployment_data = json.load(fp)
            for column in deployment_data["columns"]:
                if isinstance(deployment_data["columns"][column], list):
                    encoder_path = os.path.join(os.getcwd(), 'models', "rf", target_col,
                                                f'{column.replace(" ", "_")}_encoder.pkl')
                    df[column.replace("_", " ")] = joblib.load(encoder_path).fit_transform(df[column.replace("_", " ")])
                else:
                    df[column] = df[column].astype(float)
            res = model.predict(df.iloc[0, :].to_numpy().reshape(1, -1))
            model_type = deployment_data["model_type"]
            if model_type == 'classification':
                result = np.argmax(res, axis=-1)
                res = joblib.load(
                    os.path.join(os.getcwd(), 'models', "rf", target_col,
                                 f'{deployment_data["Target_column"].replace(" ", "_")}_encoder.pkl')).inverse_transform(
                    result)
            return res[0]

    except Exception as e:
        print(e)


def kmeans_train(data):
    try:
        # Identify categorical and numerical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Handle missing values (if any)
        imputer = SimpleImputer(strategy='mean')
        data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        joblib.dump(imputer, 'imputer.pkl')

        # Build a transformer for preprocessing: scaling numerical columns and encoding categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),  # Standard scaling for numerical columns
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                # One-Hot encoding for categorical columns
            ])

        # Apply preprocessing and fit KMeans
        X = preprocessor.fit_transform(data)

        # Find the optimal k using the elbow method with KMeans
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Determine the optimal k
        optimal_k = find_elbow_point(inertia)
        print('Optimal number of clusters (k) based on the Elbow Method:', optimal_k)

        # Initialize KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)

        # Fit KMeans to the preprocessed data
        kmeans.fit(X)

        # Save the trained model and preprocessor
        joblib.dump(kmeans, 'kmeans_model.pkl')  # Save KMeans model
        joblib.dump(preprocessor, 'preprocessor.pkl')  # Save Preprocessing pipeline

        # Add cluster labels to the original data
        data['Cluster'] = kmeans.labels_
        return True, data
    except Exception as e:
        print(e)
        return False, data


def arima_train(data, target_col):
    try:
        # Identify date column by checking for datetime type
        date_column = None
        if not os.path.exists(os.path.join("models", 'arima', target_col)):
            os.makedirs(os.path.join("models", 'arima', target_col), exist_ok=True)
            for col in data.columns:
                if data.dtypes[col] == 'object':
                    try:
                        # Attempt to convert column to datetime
                        pd.to_datetime(data[col])
                        date_column = col
                        break
                    except (ValueError, TypeError):
                        continue
            if not date_column:
                raise ValueError("No datetime column found in the dataset.")
            print(date_column)
            # Set the date column as index
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            # Identify forecast columns (numeric columns)
            forecast_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not forecast_columns:
                raise ValueError("No numeric columns found for forecasting in the dataset.")

            # Infer frequency of datetime index
            freq = pd.infer_freq(data.index)
            print(date_column, freq)
            if freq:
                # Determine m based on inferred frequency
                if freq == '15T':  # Quarter-hourly data (every 15 minutes)
                    m = 96  # Daily seasonality (96 intervals in a day)
                elif freq == '30T':  # Half-hourly data (every 30 minutes)
                    m = 48  # Daily seasonality (48 intervals in a day)
                elif freq == 'H':  # Hourly data
                    m = 24  # Daily seasonality (24 intervals in a day)
                elif freq == 'D':  # Daily data
                    m = 7  # Weekly seasonality (7 days in a week)
                elif freq == 'W':  # Weekly data
                    m = 52  # Yearly seasonality (52 weeks in a year)
                elif freq == 'M':  # Monthly data
                    m = 12  # Yearly seasonality (12 months in a year)
                elif freq == 'Q':  # Quarterly data
                    m = 4  # Yearly seasonality (4 quarters in a year)
                elif freq == 'A' or (freq and freq.startswith('A-')):  # Annual data (any month-end)
                    m = 1  # No further seasonality within a year
                else:
                    raise ValueError(f"Unsupported frequency '{freq}'. Ensure data is in a common time interval.")
                results = {}
                try:
                    data_actual = data[target_col].dropna()  # Remove NaNs if any

                    # Split data into train and test sets
                    train = data_actual.iloc[:-m]
                    test = data_actual.iloc[-m:]

                    # Auto ARIMA model selection
                    model = pm.auto_arima(train,
                                          m=m,  # frequency of seasonality
                                          seasonal=True,  # Enable seasonal ARIMA
                                          d=None,  # determine differencing
                                          test='adf',  # adf test for differencing
                                          start_p=0, start_q=0,
                                          max_p=12, max_q=12,
                                          D=None,  # let model determine seasonal differencing
                                          trace=True,
                                          error_action='ignore',
                                          suppress_warnings=True,
                                          stepwise=True)
                    # Forecast and calculate errors
                    fc, confint = model.predict(n_periods=m, return_conf_int=True)
                    # Save results to dictionary
                    results = {
                        "actual": {
                            "date": list(test.index.astype(str)),
                            "values": [float(val) if isinstance(val, np.float_) else int(val) for val in
                                       test.values]
                        },
                        "forecast": {
                            "date": list(test.index.astype(str)),
                            "values": [float(val) if isinstance(val, np.float_) else int(val) for val in fc]
                        }
                    }
                    if not os.path.exists(os.path.join("models", 'arima', target_col)):
                        os.makedirs(os.path.join("models", 'arima', target_col), exist_ok=True)
                    with open(os.path.join("models", 'arima', target_col, target_col + '_results.json'), 'w') as fp:
                        json.dump(results, fp)
                    result_graph = plot_graph(results, os.path.join('models', 'arima', target_col))

                    print(
                        f"Results saved to {os.path.join('models', 'arima', target_col, target_col + '_results.json')}")
                    return True, result_graph
                except Exception as e:
                    print(e)
                    return False, str(e)
            else:
                return False, "Data does not exhibit trends, seasonality, or shifts in variance"
        else:
            with open(os.path.join("models", 'arima', target_col, target_col + '_results.json'), 'r') as fp:
                results = json.load(fp)
            result_graph = plot_graph(results, os.path.join('models', 'arima', target_col))

            print(f"Results saved to {os.path.join('models', 'arima', target_col, target_col + '_results.json')}")
            return True, result_graph

    except Exception as e:
        print(e)
        return False


def load_pipeline(save_path="model_pipeline.pkl"):
    # Load the saved pipeline
    pipeline = joblib.load(save_path)
    print(f"Pipeline loaded from: {save_path}")
    return pipeline


def find_elbow_point(inertia_values):
    # Calculate the rate of change between successive inertia values
    changes = np.diff(inertia_values)
    # Identify the elbow as the point where change starts to decrease
    elbow_point = np.argmin(np.abs(np.diff(changes))) + 1
    return elbow_point


def plot_graph(data, file_path):
    try:
        col = file_path.split('\\')[-1]
        actual_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data["actual"]["date"]]
        forecast_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data["forecast"]["date"]]

        # Extract values
        actual_values = data["actual"]["values"]
        forecast_values = data["forecast"]["values"]

        # Create Plotly figure
        fig = go.Figure()

        # Actual Data Line
        fig.add_trace(go.Scatter(
            x=actual_dates, y=actual_values,
            mode='lines+markers', name='Actual',
            line=dict(color='blue'), marker=dict(symbol='circle')
        ))

        # Forecast Data Line
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_values,
            mode='lines+markers', name='Forecast',
            line=dict(color='orange', dash='dash'), marker=dict(symbol='x')
        ))

        # Layout Settings
        fig.update_layout(
            title=f'{col} Actual vs Forecast Values Over Time',
            xaxis_title='Date',
            yaxis_title='Values',
            xaxis=dict(tickangle=-45),
            template="plotly_white",
            width=1000, height=600
        )

        # Convert figure to Base64 Image
        return fig.to_json()

    except Exception as e:
        print(e)
        return str(e)


def generate_kpi_code(kpi_list):
    try:
        df = pd.read_csv("data.csv")
        df = updatedtypes(df)
        codes = {}
        paths = {}
        if os.path.exists(os.path.join(os.getcwd(), f'static/charts')):
            shutil.rmtree(os.path.join(os.getcwd(), f'static/charts'))
        os.makedirs(f'static/charts', exist_ok=True)
        for kpi in kpi_list:
            prompt_desc = (
                f"""You are ai_bot.Make sure to read the data from data.csv file with example data as {df.head()} and generate Python code with KPI details as {KPI_LOGICS[kpi]}. 
                    Ensure the result is stored in a variable named 'result'.  Use Plotly to generate a suitable interactive plot for the obtained result. The type of plot should be chosen based on the structure of 'result' (e.g., bar plot for categorical/numeric summaries, line plot for time series, scatter plot for correlations, etc.). 
                    Instead of saving the plot, **convert the Plotly figure to a JSON representation** using `fig.to_json()` and return it as the output.
                    If the length of 'result' is 1, use a thin bar width and strictly set the x-axis limit to [-0.5, 0.5]."""
            )
            code = ''

            try:
                temp, chart_data = generate_code2(prompt_desc)
                code += temp
                paths[kpi] = chart_data
            except Exception as e:
                print(e)
                code += f'Code generation failed for {kpi}'
            codes[kpi] = "<b>" + kpi.capitalize() + "</b>" + "\n" + mark_safe(code) + '\n'

        return paths, codes
    except Exception as e:
        print(e)


@csrf_exempt
def generate_code2(prompt_eng):
    trials = 2
    chart_data = {}
    try:
        while trials > 0:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_eng}
                ]
            )
            all_text = ""

            # Display generated content dynamically
            for choice in response.choices:
                print(f"Debug - choice structure: {choice}")  # Debugging line
                message = choice.message
                print(f"Debug - message structure: {message}")  # Debugging line
                chunk_message = message.content if message else ''
                all_text += chunk_message

            print(all_text)
            python_chuncks = all_text.count("```python")
            idx = 0
            code = ''
            for i in range(python_chuncks):
                code_start = all_text[idx:].find("```python") + 9
                code_end = all_text[idx:].find("```", code_start)
                code += all_text[idx:][code_start:code_end]
                idx = code_end
            print(code)
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                fig = local_vars.get("fig")

                if fig and isinstance(fig, Figure):
                    # Convert the Plotly figure to JSON
                    chart_data = fig.to_plotly_json()
                code += f"\n <b>Output: {local_vars['result']}</b> \n <hr>"
                return code, make_serializable(chart_data)
            except Exception as e:
                print(e)
                trials -= 1
    except Exception as e:
        print(e)


def plot_numeric(numeric_vars, dataframe):
    df = dataframe
    plots = {}

    for i in list(numeric_vars.keys()):
        # Create traces
        fig = go.Figure()

        # Add bar plot
        fig.add_trace(go.Bar(x=np.arange(len(df[i])), y=df[i], name=i, marker_color='blue'))

        # Add min, max, and median reference lines
        fig.add_hline(y=df[i].min(), line=dict(color='blue', dash='dash'), annotation_text='Min',
                      annotation_position="top left")
        fig.add_hline(y=df[i].max(), line=dict(color='red', dash='dash'), annotation_text='Max',
                      annotation_position="top left")
        fig.add_hline(y=df[i].median(), line=dict(color='green', dash='dash'), annotation_text='Median',
                      annotation_position="top left")

        # Customize layout
        fig.update_layout(
            title=i,
            xaxis_title="Index",
            yaxis_title=i,
            template="plotly_white",
            width=1000,  # Equivalent to figsize=(20, 10)
            height=500
        )

        # Convert figure to JSON for rendering in web applications
        plots[i] = make_serializable(fig.to_json())

    return plots


def plot_categorical(categorical_vars, dataframe):
    df = dataframe
    plots = {}

    for i in categorical_vars:
        name = [k[0] for k in df[list(i)].value_counts().index.tolist()]
        count = df[list(i)].value_counts().values.tolist()

        # Create a pie chart using Plotly
        fig = px.pie(
            names=name,
            values=count,
            title=list(i)[0],
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.3  # Adjust to create a donut chart if needed
        )

        # Convert figure to JSON for web rendering
        plots[list(i)[0]] = fig.to_json()

    return plots


def plot_wordCloud(text_data, dataframe):
    df = dataframe
    plots = {}

    for i in text_data:
        # Generate word cloud
        text = " ".join(cat for cat in df[i])
        wordcloud = WordCloud(collocations=False, background_color='white').generate(text)

        # Convert to an image
        img = io.BytesIO()
        wordcloud.to_image().save(img, format="PNG")
        img_base64 = base64.b64encode(img.getvalue()).decode("utf-8")

        # Create a Plotly figure with the word cloud image
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_base64}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                xanchor="center",
                yanchor="middle",
                layer="below"
            )
        )

        # Layout settings
        fig.update_layout(
            title=i,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=800, height=500,
        )

        # Convert figure to JSON for web rendering
        plots[i] = fig.to_json()

    return plots


def updatedtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                pass
    return df


def adf_test(df, kpi):
    df_t = df.set_index(kpi)

    for col in df_t.columns:
        # Check if the column name is not in the specified list and is numeric
        if col.upper() not in ['DATE', 'TIME', 'DATE_TIME'] and pd.api.types.is_numeric_dtype(df_t[col]):
            if df_t[col].nunique() > 1:
                dftest = adfuller(df_t[col], autolag='AIC')
                statistic_value = dftest[0]
                p_value = dftest[1]
                if (p_value > 0.5) and all([statistic_value > j for j in dftest[4].values()]):
                    return "Y"
            else:
                break
    return "N"


def checkSentiment(df, categorical):
    sentiment = 'N'
    for i in categorical:
        # print([j for j in df[i]])
        data = ' '.join([str(j) for j in df[list(i.keys())[0]]]).upper()
        if ('GOOD' in data) | ('BAD' in data) | ('Better' in data):
            sentiment = "Y"
    return sentiment


def handle_missing_data(df):
    try:
        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_time_cols = df.select_dtypes(include=['datetime64']).columns

        # Impute numeric columns and track which cells were imputed
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols).round(2)

        # Mark imputed cells (True if the original cell was NaN)
        imputed_flags = df[numeric_cols].isnull()
        imputed_flags = imputed_flags.applymap(lambda x: x if x else False)

        # Update DataFrame with imputed values
        df[numeric_cols] = imputed_numeric_df

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col]   = df[col].iloc[i - 1] + avg_diff
                    imputed_flags.loc[i, col] = True

            imputed_flags.fillna(False, inplace=True)

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []
        for _, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = {
                    "value": row[col].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[col], pd.Timestamp) else row[col],
                    "is_imputed": str(imputed_flags[col].get(_, False)) if col in imputed_flags else str(False)
                    # Check if cell was imputed
                }
            data.append(row_data)
        return df, data
    except Exception as e:
        print(e)


def detect_and_parse_date(value):
    """
    Detects and converts dates in multiple formats, including:
    - MM-DD-YYYY
    - DD-MM-YYYY
    - MM/DD/YYYY
    - DD/MM/YYYY
    - YYYY-MM-DD
    """
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return pd.NaT  # Handle missing values safely

    try:
        # Check if it's a date with hyphens or slashes
        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", value):
            day_first = False  # Assume MM-DD-YYYY first

            # Check for an ambiguous case (day > 12) → Must be DD-MM-YYYY
            parts = re.split(r"[-/]", value)
            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
            if day > 12:
                day_first = True  # Switch to DD-MM-YYYY

            # Parse with detected format
            return dateutil.parser.parse(value, dayfirst=day_first)

        # Otherwise, use default dateutil parsing
        return dateutil.parser.parse(value)

    except ValueError:
        return pd.NaT  # Return NaT if parsing fails


def convert_to_datetime(df):
    """
    Converts object (string) columns containing dates to datetime format.
    """
    for col in df.columns:
        if df[col].dtype == "object":  # Process only string columns
            if df[col].str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", na=False).any():
                df[col] = df[col].apply(detect_and_parse_date)

    return df


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df = handle_missing_data(df)
    return df, html_df


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")




##Visualisation updated for both text and graph responses:
from rest_framework.response import Response
from rest_framework import status


@csrf_exempt
def gen_ai_bot(request):
    if request.method == "POST":
        csv_file_path = 'data.csv'
        df = pd.read_csv(csv_file_path)

        # Generate CSV metadata
        csv_metadata = {"columns": df.columns.tolist()}
        metadata_str = ", ".join(csv_metadata["columns"])

        sample_data = df.head(2)

        system_prompt = f"""
            You are an AI specialized in data analytics and visualization. The data for analysis is stored in a CSV file named data.csv, with the following attributes: {metadata_str} and sample data as {sample_data}.
    
            Follow these rules while responding to user queries:
    
            1. Strictly use 'data.csv' as the data source without stating any limitations or disclaimers about file access.
    
            2.Data Analysis: If the query requires numerical or tabular insights, extract relevant data from data.csv, perform necessary calculations, and provide a concise summary store the result in text_output variable.
    
            3. Visualization (if applicable):
            3.1 If the query requires a graph, generate Python code using Plotly to create the requested chart type (e.g., bar, pie, scatter, etc.). If no graph type is specified, intelligently choose between a line or bar chart based on the context. Ensure the graph includes:
    
                A title, axis labels (if applicable), and appropriate colors.
                A white background for both the plot and the paper.
                A visually appealing design that provides sufficient context for understanding.
            3.2 If a graph is generated, the code must:
    
            Output a Plotly Figure object stored in a variable named fig.
            Include the data and layout dictionaries necessary for rendering the graph.
            Ensure full compatibility with React.
        """
        result = {}
        prompt = request.POST.get('prompt')
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        pre_code_text, post_code_text, code = process_response(response)
        result.update({
            'text_pre_code_response': pre_code_text,
            "code": code,
            'text_post_code_response': post_code_text
        })
        if 'import' in code:
            namespace = {}
            try:
                # Execute the generated code
                exec(code, namespace)

                result.update({
                    'text_output': namespace.get('text_output')
                })

                # Retrieve the Plotly figure from the namespace
                fig = namespace.get("fig")

                if fig and isinstance(fig, Figure):
                    # Convert the Plotly figure to JSON
                    chart_data = fig.to_plotly_json()
                    chart_data = make_serializable(chart_data)
                    result.update({
                        'chart_response': chart_data
                    })

                return JsonResponse(
                    result, status=status.HTTP_200_OK
                )
            except Exception as e:
                print(e)
                return JsonResponse({'message': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return JsonResponse(
            result, status=status.HTTP_200_OK
        )


def process_response(response):
    all_text = ""
    text_post_code = ''
    code_start = -1
    code_end = -1
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    text_pre_code = all_text[:code_start - 9]
    if code_start != -1:
        text_post_code = all_text[code_end:]
    return text_pre_code, text_post_code, code


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj
