import logging
import os
import time
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def keep_app_awake():
    """
    Pings the Streamlit application URL to prevent it from going to sleep.
    Streamlit Community Cloud puts apps to sleep after 7 days of inactivity.
    This creates simulated activity so the inactivity timer is constantly reset.
    """
    app_url = os.environ.get("STREAMLIT_APP_URL")
    
    if not app_url:
        logger.warning("STREAMLIT_APP_URL environment variable is missing!")
        logger.warning("Skipping wake-up ping. Please add it to your GitHub Secrets.")
        return

    logger.info(f"Pinging Streamlit App at: {app_url} ...")
    
    try:
        # We don't care about the 303 redirect or authentication failures. 
        # The mere act of the HTTP request hitting the server resets the activity timer!
        response = requests.get(app_url, timeout=15)
        logger.info(f"Ping completed! Server responded with HTTP {response.status_code}")
        
    except requests.exceptions.Timeout:
        logger.error("Ping timed out. The server might be asleep or starting up.")
    except Exception as e:
        logger.error(f"Failed to ping the app: {e}")

def main():
    logger.info("=== Starting Scheduled Tasks ===")
    start_time = time.time()
    
    try:
        keep_app_awake()
        
        # You can add future background analytical or cleanup tasks here
        
        elapsed_time = time.time() - start_time
        logger.info(f"=== Scheduled Tasks Completed Successfully in {elapsed_time:.2f} seconds ===")
        
    except Exception as e:
        logger.error(f"Scheduled Task Failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
