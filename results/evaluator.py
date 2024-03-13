import pandas as pd
from tabulate import tabulate
import re

# Load the CSV into a pandas DataFrame
df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()

# Function to parse colors and counts
def parse_colors(colors_str):
    colors = re.findall(r'\d*\$?\w+', colors_str)
    color_count = {}
    for color in colors:
        if '$' in color:
            count, color_name = color.split('$')
            color_count[color_name] = int(count)
        else:
            color_count[color] = 1
    return color_count

# Apply the function to the 'piece_colors' column
df['parsed_colors'] = df['piece_colors'].apply(parse_colors)

# Drop the original 'piece_colors' column if needed
df.drop('piece_colors', axis=1, inplace=True)

# print(df)

def evaluate_guess(df, guessed_id, guessed_colors):
    # Get the actual colors for the guessed ID
    actual_colors = df.loc[df['id'] == guessed_id, 'parsed_colors'].iloc[0]
    
    # Initialize counters for correct and wrong guesses
    correct_guesses = 0
    wrong_guesses = 0
    
    # Construct the table data
    table_data = []
    
    # Compare guessed colors with actual colors
    for color, count in actual_colors.items():
        guessed_count = guessed_colors.get(color, 0)
        result = '✓' if guessed_count <= count else '✗'
        table_data.append([f"{count} {color}", f"{guessed_count} {color}", result])
        if result == '✓':
            correct_guesses += 1
        else:
            wrong_guesses += 1
    
    for color, count in guessed_colors.items():
        if color not in actual_colors:
            table_data.append([f"0 {color}", f"{count} {color}", '✗'])
            wrong_guesses += 1
    
    # Check if the number of pieces is guessed correctly
    guessed_piece_count = sum(guessed_colors.values())
    actual_piece_count = df.loc[df['id'] == guessed_id, 'piece_count'].iloc[0]
    if guessed_piece_count == actual_piece_count:
        piece_result = '✓'
    else:
        piece_result = '✗'
    
    # Add the final row to the table with dotted line separator
    table_data.append(['-' * 20, '-' * 20, '-' * 10])
    table_data.append([f"Guessed Piece Nº: {guessed_piece_count}", f"Actual Piece Nº: {actual_piece_count}", f"Result: {piece_result}"])
    
    print(tabulate(table_data, headers=["Actual Colors", "Guessed Colors", "Result"], tablefmt="pretty"))
    
    return correct_guesses, wrong_guesses

# Example usage:
guessed_id = 0
guessed_colors = {'yellow': 2, 'lightblue': 1}

correct, wrong = evaluate_guess(df, guessed_id, guessed_colors)
print(f"\nTotal correct guesses: {correct}")
print(f"Total wrong guesses: {wrong}")




























