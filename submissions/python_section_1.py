from typing import Dict, List, Any
import pandas as pd
from itertools import permutations
import re
from geopy.distance import geodesic


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    
    # Iterate over the list in steps of 'n'
    for i in range(0, len(lst), n):
        group = []
        
        # Reverse each group manually
        for j in range(i, min(i + n, len(lst))):
            group.insert(0, lst[j])  # Insert elements at the beginning to reverse them
        
        result.extend(group)
    
    return result



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    
    d = {}
    for string in lst:
        length = len(string)
        if length not in d:
            d[length] = []
        d[length].append(string)
        
    return dict(sorted(d.items()))



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    flattened = {}
    
    # Stack to hold the items to be processed, starting with the root dictionary
    stack = [(nested_dict, '')]

    while stack:
        current_dict, parent_key = stack.pop()  # Get the current dictionary and its associated parent key
        
        for key, value in current_dict.items():
            # Construct the new key by appending the current key to the parent key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):  # If the value is a dictionary, add it to the stack
                stack.append((value, new_key))
                
            elif isinstance(value, list):  # If the value is a list
                for index, item in enumerate(value):
                    # Handle each element of the list
                    if isinstance(item, dict):  # If the item is a dictionary, add it to the stack
                        stack.append((item, f"{new_key}[{index}]"))
                    else:  # For non-dict items in the list, directly flatten the item
                        flattened[f"{new_key}[{index}]"] = item
                        
            else:  # For non-dict and non-list values, add them directly to the flattened dict
                flattened[new_key] = value

    return flattened


def unique_permutations(nums: List[int]) -> List[List[int]]:
    # Generate all permutations and convert to a set to remove duplicates
    unique_perms = set(permutations(nums))
    
    # Convert the set back to a list of lists
    return [list(perm) for perm in unique_perms]


def find_all_dates(text: str) -> List[str]:
    # Combine all patterns into one regular expression
    pattern = r'\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2}'
    
    # Use re.findall to return all matching dates
    return re.findall(pattern, text)



def decode_polyline(polyline_str: str) -> list:
    
    coordinates = []
    index, length = 0, len(polyline_str)
    lat, lng = 0, 0

    while index < length:
        b, shift, result = 0, 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            if b < 0x20:
                break
            shift += 5
        dlat = ~(result >> 1) if result & 1 else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            if b < 0x20:
                break
            shift += 5
        dlng = ~(result >> 1) if result & 1 else (result >> 1)
        lng += dlng

        coordinates.append((lat / 1E5, lng / 1E5))

    return coordinates

def calculate_distance(coords: list) -> list:
    
    distances = [0]  # Distance for the first coordinate is 0
    for i in range(1, len(coords)):
        distances.append(geodesic(coords[i-1], coords[i]).meters)  # Store distances in meters
    return distances

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    #'u{~vFvyys@fS]' (polyline string)
    # Decode the polyline
    coords = decode_polyline(polyline_str)
    
    # Calculate distances
    distances = calculate_distance(coords)
    
    # Create a DataFrame
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    df['distance'] = distances
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Step 2: Create the final transformed matrix
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Calculate the row and column sum excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    # Define a mapping for days of the week to specific dates in a week (Monday to Sunday)
    day_mapping = {
        'Monday': '2023-01-02',
        'Tuesday': '2023-01-03',
        'Wednesday': '2023-01-04',
        'Thursday': '2023-01-05',
        'Friday': '2023-01-06',
        'Saturday': '2023-01-07',
        'Sunday': '2023-01-01'
    }

    # Map startDay and endDay to the specific dates
    df['startDay'] = df['startDay'].map(day_mapping)
    df['endDay'] = df['endDay'].map(day_mapping)

    # Combine startDay and startTime, endDay and endTime to form datetime
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Function to generate a full week's timestamps (one week Monday to Sunday)
    def full_week_coverage():
        full_week = []
        for day in range(7):  # From Monday (0) to Sunday (6)
            start_of_day = pd.Timestamp(f'2023-01-0{(day + 1) % 7 + 1} 00:00:00')
            end_of_day = pd.Timestamp(f'2023-01-0{(day + 1) % 7 + 1} 23:59:59')
            full_week.append((start_of_day, end_of_day))
        return full_week
    
    # Check if the time ranges for the group cover the full week
    def check_coverage(group):
        full_week = full_week_coverage()
        covered_times = [(row['start'], row['end']) for _, row in group.iterrows()]
        
        # Check if all 7 days are covered by the given time intervals
        for expected_start, expected_end in full_week:
            if not any(start <= expected_end and end >= expected_start for start, end in covered_times):
                return False
        return True
    
    result = df.groupby(['id', 'id_2'])[['start', 'end']].apply(lambda group: check_coverage(group))
    
    return result
