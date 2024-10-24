
import fcntl
import time

# Function to write a DataFrame with file locking
def append_dataframe_with_lock(file_path, dataframe, max_attempts=100, wait_time=1):
    """
    Append a DataFrame to a CSV file with file locking to ensure safe concurrent access.
    Parameters:
    file_path (str): The path to the CSV file where the DataFrame will be appended.
    dataframe (pandas.DataFrame): The DataFrame to append to the CSV file.
    max_attempts (int, optional): The maximum number of attempts to acquire the file lock. Default is 100.
    wait_time (int, optional): The time to wait (in seconds) between attempts to acquire the file lock. Default is 1.
    Returns:
    None
    Raises:
    BlockingIOError: If the file is locked by another process and the maximum number of attempts is reached.
    Notes:
    - This function uses non-blocking file locking to attempt to acquire the lock without waiting indefinitely.
    - If the file is empty, the DataFrame's header will be written; otherwise, only the data will be appended.
    - If the lock cannot be acquired after the specified number of attempts, the function will print an error message and exit.
    """

    attempts = 0
    while attempts < max_attempts:
        try:
            with open(file_path, 'a') as f:
                # Attempt to acquire the lock without blocking for too long
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # LOCK_NB makes the call non-blocking
                
                try:
                    # Check if the file is empty to write the header only once
                    is_empty = f.tell() == 0
                    
                    # Write the DataFrame to the file in append mode
                    dataframe.to_csv(f, header=is_empty, index=False)
                    print("DataFrame successfully written in append mode.")
                    return  # Exit the function if writing is successful
                
                finally:
                    # Release the lock
                    fcntl.flock(f, fcntl.LOCK_UN)
        
        except BlockingIOError:
            # The file is locked by another process
            attempts += 1
            print(f"File is locked, attempt {attempts} of {max_attempts}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # If unable to acquire the lock after all attempts
    print("Unable to acquire the lock after multiple attempts. Operation failed.")