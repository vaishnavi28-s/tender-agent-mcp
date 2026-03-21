import schedule
import time
import subprocess

def run_pipeline():
    print("Starting Tender Pipeline...")

    
    print("Running fetch_tenders.py")
    subprocess.run(["python", "fetch_tenders.py"], check=True)

   
    print("Cleaning markdown files")
    subprocess.run(["python", "clean.py"], check=True)

    
    print("Building vector database")
    subprocess.run(["python", "build_vector_db.py"], check=True)

    print("Pipeline run complete.")


schedule.every(6).hours.do(run_pipeline)

if __name__ == "__main__":
    print("Tender pipeline scheduler active. Running every 6 hours.")
    run_pipeline()  
    while True:
        schedule.run_pending()
        time.sleep(60)
