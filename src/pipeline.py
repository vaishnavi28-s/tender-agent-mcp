import sys
import os
import time
import schedule

sys.path.append(os.path.dirname(__file__))
from fetch_tenders import fetch_and_process, build_vector_store

_pipeline_running = False


def run_pipeline():
    global _pipeline_running

    if _pipeline_running:
        print("Pipeline already running; skipping this scheduled trigger to avoid overlap.")
        return

    _pipeline_running = True
    try:
        print("Starting Tender Pipeline...")

        print("Fetching and processing tenders...")
        fetch_and_process()

        print("Building vector database...")
        build_vector_store()

        print("Pipeline run complete.")
    except Exception as e:
        print(f"Pipeline run failed: {e}")
    finally:
        _pipeline_running = False


schedule.every(6).hours.do(run_pipeline)

if __name__ == "__main__":
    print("Tender pipeline scheduler active. Running every 6 hours.")
    run_pipeline()
    while True:
        schedule.run_pending()
        time.sleep(60)
