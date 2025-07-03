from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import io
import os
import uuid
import base64
import seaborn as sns

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file:
        return "No file uploaded"

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    df = pd.read_csv(file_path)

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df)
    df["ml_anomaly"] = model.predict(df)

    def check_faults(row):
        faults, actions = [], []

        if row["motor_temp"] < 30 or row["motor_temp"] > 120:
            faults.append("Motor Overheat")
            actions.append("Inspect cooling system or motor windings")

        if row["brake_pressure"] < 5 or row["brake_pressure"] > 8:
            faults.append("Brake Pressure Issue")
            actions.append("Check compressor & pipe for leakage")

        if row["wheel_current_diff"] > 200:
            faults.append("Wheel Slip Detected")
            actions.append("Inspect traction motor & current diff relay")

        if row["aux_status"] not in [0, 1]:
            faults.append("Auxiliary Sensor Error")
            actions.append("Reset or replace the sensor")

        return (
            ", ".join(faults) if faults else "OK",
            ", ".join(actions) if actions else "All systems nominal"
        )

    df[["faulty_component", "recommended_action"]] = df.apply(lambda row: pd.Series(check_faults(row)), axis=1)
    df["final_status"] = df.apply(lambda r: "Anomaly" if r["ml_anomaly"] == -1 or r["faulty_component"] != "OK" else "Normal", axis=1)

    total = len(df)
    faulty = len(df[df['final_status'] == 'Anomaly'])
    fault_percentage = round((faulty / total) * 100, 2)

    engine_load_status = "Stable" if df["motor_temp"].std() < 10 else "Fluctuating"
    brake_balance = "Balanced" if df["brake_pressure"].std() < 1.5 else "Unstable"
    voltage_variance = f"{df["wheel_current_diff"].std():.2f} A"
    aux_status_summary = "Nominal" if df["aux_status"].isin([0, 1]).all() else "Irregular"

    result_file = os.path.join(RESULT_FOLDER, f"report_{uuid.uuid4().hex[:8]}.csv")
    df.to_csv(result_file, index=False)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["motor_temp", "brake_pressure", "wheel_current_diff"]
    titles = ["Motor Temp (°C)", "Brake Pressure (kg/cm²)", "Wheel Current Diff (A)"]
    colors = ["#ff9800", "#4caf50", "#2196f3"]

    for i, col in enumerate(labels):
        axs[i].plot(df[col], color=colors[i], marker='o', label=titles[i], zorder=3)
        axs[i].scatter(df[df["final_status"] == "Anomaly"].index,
                       df[df["final_status"] == "Anomaly"][col],
                       color="red", s=70, label="Anomaly", zorder=4, edgecolor='black')
        axs[i].legend(facecolor='lightsteelblue', edgecolor='black')
        axs[i].grid(True, color='#90a4ae', linestyle='--', linewidth=0.7, zorder=0)
        axs[i].set_facecolor('#d0dae5')
        axs[i].spines['top'].set_linewidth(2)
        axs[i].spines['top'].set_color('#546e7a')
        axs[i].spines['right'].set_linewidth(2)
        axs[i].spines['right'].set_color('#546e7a')
        axs[i].spines['left'].set_color('#263238')
        axs[i].spines['left'].set_linewidth(2)
        axs[i].spines['bottom'].set_color('#263238')
        axs[i].spines['bottom'].set_linewidth(2)

    fig.patch.set_facecolor('#b0bec5')
    fig.patch.set_alpha(1)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    fig.suptitle("Sensor Readings with Anomaly Highlights", fontsize=14, fontweight='bold', color='#102027')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_data = buf.read()
    buf.close()

    image_base64 = base64.b64encode(image_data).decode('utf-8')

    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df.index, y=df['faulty_component'], hue=df['final_status'], palette={"Anomaly": "red", "Normal": "green"})
    plt.title("Fault Timeline")
    plt.xlabel("Data Index")
    plt.ylabel("Faulty Component")
    plt.xticks(rotation=45)
    timeline_buf = io.BytesIO()
    plt.savefig(timeline_buf, format='png', bbox_inches='tight')
    timeline_buf.seek(0)
    timeline_data = base64.b64encode(timeline_buf.read()).decode('utf-8')
    timeline_buf.close()

    top_fault = df[df['final_status'] == 'Anomaly']['faulty_component'].value_counts().idxmax()
    top_count = df[df['faulty_component'] == top_fault].shape[0]
    summary_text = f"Most common fault: {top_fault} ({top_count} occurrences)"

    faults_data = df[df['final_status'] == 'Anomaly'][['faulty_component', 'recommended_action']].to_dict(orient='records')

    return render_template(
        'results.html',
        image_data=image_base64,
        faults=faults_data,
        file_name=os.path.basename(result_file),
        bolt_image='static/bolt.png',
        panel_labels={
            'graph': 'Sensor Data Graph',
            'faults': 'Detected Faults'
        },
        fault_percentage=fault_percentage,
        timeline_chart=timeline_data,
        summary_text=summary_text,
        engine_load_status=engine_load_status,
        brake_balance=brake_balance,
        voltage_variance=voltage_variance,
        aux_status_summary=aux_status_summary
    )

@app.route('/download/<file_name>')
def download(file_name):
    return send_file(os.path.join(RESULT_FOLDER, file_name), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)