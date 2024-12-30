import os
from datetime import datetime, timedelta

def make_commit(start_date: str, end_date: str):
    # Parse the input date strings
    start = datetime.strptime(start_date, '%d-%m-%Y')
    end = datetime.strptime(end_date, '%d-%m-%Y')

    # Ensure the start date is before the end date
    if start > end:
        print("Error: Start date must be before end date.")
        return

    current_date = start

    while current_date <= end:
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

    # Push the commits after the loop
    os.system('git push')

# Example usage
make_commit('13-12-2024', '30-12-2024')
