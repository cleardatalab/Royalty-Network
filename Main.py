import os
from datetime import datetime, timedelta

def make_commit(days: int):
    if days < 1:
        # Push the commits
        return os.system('git push')
    else:
        # Calculate the date based on 'days ago'
        target_date = datetime.now() - timedelta(days=days)
        
        # Check if the month is in September, October, November, or December
        if target_date.month in {9, 10, 11, 12}:
            dates = target_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write to a file
            with open('data.txt', 'a') as file:
                file.write(f'{dates}\n')
            
            # Staging and committing with the specific date
            os.system('git add data.txt')
            os.system(f'git commit --date="{dates}" -m "Commit for {target_date.strftime("%B %d, %Y")}"')
        
        # Continue recursion
        return make_commit(days - 1)
    
# Start the commit process for the past 365 days
make_commit(365)
