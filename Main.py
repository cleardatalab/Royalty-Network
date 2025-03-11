import os
import random
from datetime import datetime, timedelta

def make_commit(days: int, start_date: datetime):
    if days < 1:
        # Push the commits after all commits are made
        return os.system('git push')
    else:
        # Calculate the target date
        target_date = start_date + timedelta(days=days - 1)

        # Check if the month is in the selected range
        if target_date.month in selected_months:
            num_commits = random.randint(3, 4)  # Randomly choose between 2, 3, or 4 commits per day

            for _ in range(num_commits):  # Make multiple commits for the same day
                dates = target_date.strftime('%Y-%m-%d %H:%M:%S')

                # Write to a file
                with open('data.txt', 'a') as file:
                    file.write(f'{dates}\n')

                # Staging and committing with the specific date
                os.system('git add data.txt')
                os.system(f'git commit --date="{dates}" -m "Commit for {target_date.strftime("%B %d, %Y")}"')

        # Continue recursion
        return make_commit(days - 1, start_date)

# Assign the start and end dates directly
start_date = datetime(2025, 1, 1)  # Start from January 1, 2025
end_date = datetime(2025, 3, 31)   # End on March 31, 2025

# Calculate the total number of days between start and end date
total_days = (end_date - start_date).days + 1

# Get selected months dynamically from the user-defined date range
selected_months = {start_date.month + i for i in range((end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1)}

# Start the commit process
make_commit(total_days, start_date)
