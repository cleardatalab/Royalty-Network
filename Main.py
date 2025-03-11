import os
from datetime import datetime, timedelta

def make_commit(start_date: str, end_date: str):
    # Convert input strings to datetime objects
    start_dt = datetime.strptime(start_date, "%d/%m/%Y")
    end_dt = datetime.strptime(end_date, "%d/%m/%Y")
    
    # Iterate through each day in the range
    current_date = start_dt
    while current_date <= end_dt:
        # Check if the month is in September, October, November, or December
        if current_date.month in {9, 10, 11, 12}:
            dates = current_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write to a file
            with open('data.txt', 'a') as file:
                file.write(f'{dates}\n')
            
            # Staging and committing with the specific date
            os.system('git add data.txt')
            os.system(f'git commit --date="{dates}" -m "Commit for {current_date.strftime("%B %d, %Y")}"')
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    # Push the commits
    os.system('git push')

# Example usage
make_commit("01/09/2023", "31/12/2023")
