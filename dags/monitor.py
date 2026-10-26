from datetime import datetime
from airflow.sdk import dag, task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from src.monitor import load_predictions, drift_report, save_metrics, count_new_labels

@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False)
def monitor():
    @task
    def report():
        r = drift_report(load_predictions(hours=24))   
        save_metrics(r)                                
        return r["drift"]                              

    @task.short_circuit
    def should_retrain(drift):
        return drift or count_new_labels() >= 200

    retrain = TriggerDagRunOperator(task_id="retrain", trigger_dag_id="train_register")

    should_retrain(report()) >> retrain

monitor()