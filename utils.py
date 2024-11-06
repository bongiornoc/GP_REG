
import fcntl
import time
import sympy
import pandas as pd

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

def SymbolicExpressionConverter(program,symb):
    """
    Converts a symbolic expression represented as a string into a sympy expression.
    Args:
        program (str): The symbolic expression in string format.
        symb (list of str): A list of variable names used in the symbolic expression.
    Returns:
        sympy.Expr: The sympy expression equivalent of the input symbolic expression.
    Example:
        >>> SymbolicExpressionConverter('add(X0, mul(X1, X2))', ['x', 'y', 'z'])
        x + y*z
    """
    
    var = {'X%d'%i : sympy.Symbol(s) for i,s in enumerate(symb)}

    converter = {
        'sub': lambda x, y : x - y,
        'div': lambda x, y : x/y,
        'mul': lambda x, y : x*y,
        'add': lambda x, y : x + y,
        'neg': lambda x    : -x,
        'pow': lambda x, y : x**y,
    }
    converter.update(var)
    return sympy.expand(sympy.sympify(program, locals=converter))

def verify_terms_in_program(filename, terms, symbols):
    """
    Verify the presence of specific terms in symbolic expressions within a CSV file.
    This function reads a CSV file containing symbolic expressions, converts these expressions
    using the provided symbols, and checks if the expressions contain and only contain the specified terms.
    Args:
        filename (str): The path to the CSV file containing the symbolic expressions.
        terms (list): A list of terms to check for in the symbolic expressions.
        symbols (dict): A dictionary of symbols used for converting the expressions.
    Returns:
        pandas.DataFrame: A DataFrame with two additional columns:
            - 'HAS terms': A boolean column indicating if each expression contains all the specified terms.
            - 'ONLY terms': A boolean column indicating if each expression contains only the specified terms.
    """
    
    df = pd.read_csv(filename)

    df['best_program'] = df['best_program'].apply(lambda x: SymbolicExpressionConverter(x,symbols))


    HAS, ONLY = [], []
    for expression in df['best_program']:
        has_terms = all(expression.has(term) for term in terms)
        HAS.append(has_terms)
        
        if not has_terms:
            ONLY.append(False)
            continue
        
        full_term = sum(expression.coeff(term) * term for term in terms)
        ONLY.append((expression - full_term) == 0)
    
    df['HAS terms'] = HAS
    df['ONLY terms'] = ONLY
    return df

