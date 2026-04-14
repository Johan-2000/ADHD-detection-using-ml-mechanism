import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = 'behavior_classification.csv'
REPORT_OUTPUT_DIR = 'behavior_reports'

if not os.path.exists(REPORT_OUTPUT_DIR):
    os.makedirs(REPORT_OUTPUT_DIR)
    print(f"Created output directory: {REPORT_OUTPUT_DIR}")

def load_data(file_path):
    """Loads the CSV data, handling potential issues."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records from {file_path}")
        df['Second'] = pd.to_numeric(df['Second'], errors='coerce')
        df.dropna(subset=['Second'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}. Please ensure it exists.")
        return None
    except Exception as e:
        print(f"An error occurred while loading or processing the data: {e}")
        return None

def generate_graphs(df):
    """Generates and saves visual graphs based on the behavior data."""

    print("Generating Pie Chart for Overall Behavior Distribution...")
    behavior_counts = df['Behavior'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        behavior_counts,
        labels=behavior_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#FF9999', '#66B2FF'],
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title('Overall Distribution of ADHD vs. Non-ADHD Behaviors', fontsize=16)
    pie_path = os.path.join(REPORT_OUTPUT_DIR, '01_Behavior_Distribution_Pie.png')
    plt.savefig(pie_path)
    plt.close()
    print(f"Saved Pie Chart to: {pie_path}")

    print("Generating Bar Chart for Raw Gesture Frequency...")
    plt.figure(figsize=(10, 6))
    sns.countplot(
        x='Label',
        data=df,
        palette='viridis',
        order=df['Label'].value_counts().index
    )
    plt.title('Frequency of Detected Raw Gestures', fontsize=16)
    plt.xlabel('Gesture Label (focused, Disruptive, Turning)', fontsize=12)
    plt.ylabel('Detection Count (Frames/Detections)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    bar_path = os.path.join(REPORT_OUTPUT_DIR, '02_Gesture_Frequency_Bar.png')
    plt.savefig(bar_path)
    plt.close()
    print(f"Saved Bar Chart to: {bar_path}")

    print("Generating Scatter Plot for Behavioral Trend Over Time...")

    df['Behavior_Numeric'] = df['Behavior'].apply(lambda x: 1 if x == 'ADHD' else 0)

    plt.figure(figsize=(15, 6))
    sns.scatterplot(
        x='Second',
        y='Behavior_Numeric',
        hue='Behavior',
        data=df,
        palette={'ADHD': 'red', 'Non-ADHD': 'green'},
        s=100,
        alpha=0.6,
        marker='o'
    )

    plt.yticks([0, 1], ['Non-ADHD (0)', 'ADHD (1)'])
    plt.ylim(-0.2, 1.2)
    plt.title('Behavioral Classification Over Time (Timeline)', fontsize=16)
    plt.xlabel('Time in Seconds (Video Timeline)', fontsize=12)
    plt.ylabel('Behavior Class', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.legend(title='Behavior Type')

    scatter_path = os.path.join(REPORT_OUTPUT_DIR, '03_Behavior_Trend_Scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved Scatter Plot to: {scatter_path}")


if __name__ == "__main__":
    data_df = load_data(CSV_PATH)

    if data_df is not None and not data_df.empty:
        generate_graphs(data_df)
        print("\n Graph generation complete. Check the 'behavior_reports' directory for charts.")
    elif data_df is not None:
         print("The loaded DataFrame is empty. Cannot generate graphs.")