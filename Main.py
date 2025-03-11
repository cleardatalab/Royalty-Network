import os
from datetime import datetime, timedelta

def make_commit(days: int, start_date: datetime):
    if days < 1:
        # Push the commits
        return os.system('git push')
    else:
        # Calculate the target date from the start_date
        target_date = start_date + timedelta(days=days - 1)

        # Check if the month is in January, February, or March
        if target_date.month in {1, 2, 3}:
            dates = target_date.strftime('%Y-%m-%d %H:%M:%S')

            # Write to a file
            with open('data.txt', 'a') as file:
                file.write(f'{dates}\n')

            # Staging and committing with the specific date
            os.system('git add data.txt')
            os.system(f'git commit --date="{dates}" -m "Commit for {target_date.strftime("%B %d, %Y")}"')

        # Continue recursion
        return make_commit(days - 1, start_date)

# Start the commit process from January 1, 2025, for 90 days (till March 31, 2025)
start_date = datetime(2025, 1, 1)
make_commit(90, start_date)
