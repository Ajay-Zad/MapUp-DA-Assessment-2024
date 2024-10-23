import pandas as pd
import numpy as np
from datetime import time, timedelta
pd.set_option('display.max_columns', None)

def calculate_distance_matrix(df)->pd.DataFrame():
    
    # Write your logic here
    # Pivot the DataFrame to get a matrix of direct distances
    df_pivot = df.pivot(index='id_start', columns='id_end', values='distance')
    
    # Combine the matrix with its transpose to make it symmetric
    df_pivot_combined = df_pivot.combine_first(df_pivot.T)
    
    # Fill diagonal with 0 (distance from a node to itself is 0)
    np.fill_diagonal(df_pivot_combined.values, 0)
    
    # Create a copy for the distance matrix to apply the Floyd-Warshall algorithm
    distance_matrix = df_pivot_combined.copy()
    
    # Replace NaNs with a large number (infinity) to represent no direct route
    distance_matrix = distance_matrix.fillna(np.inf)
    
    # Implement Floyd-Warshall algorithm for all-pairs shortest path
    for k in distance_matrix.columns:
        for i in distance_matrix.index:
            for j in distance_matrix.columns:
                # Check if the path through `k` is shorter than the direct path between i and j
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], 
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    # Final output matrix should now have shortest paths between all pairs
    return distance_matrix
    
    

def unroll_distance_matrix(df)->pd.DataFrame():
    result = []
    
    # Iterate through rows in the DataFrame
    for start_id, row_data in df.iterrows():
        for end_id in df.columns:
            # Accessing the distance using .at
            dist_value = df.at[start_id, end_id]
            if start_id != end_id:
                result.append({'id_start': start_id, 'id_end': end_id, 'distance': dist_value})
    
    # Convert the list of dictionaries into a DataFrame
    return pd.DataFrame(result)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    # Step 1: Calculate average distance for the reference ID
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Step 2: Define 10% threshold bounds
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1

    # Step 3: Group the DataFrame by id_start and calculate the average distance for each group
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()

    # Step 4: Filter the IDs where the average distance is within the threshold
    within_threshold = avg_distances[
        (avg_distances['distance'] >= lower_bound) & 
        (avg_distances['distance'] <= upper_bound)
    ]['id_start']

    # Return the sorted list of IDs within the threshold
    return sorted(within_threshold.tolist())


def calculate_toll_rate(df)->pd.DataFrame():
    # Define the toll rates for each vehicle type
    toll_rates = {
        'moto': 0.8,  
        'car': 1.2,   
        'rv': 1.5,    
        'bus': 2.2,   
        'truck': 3.6  
    }
    
    # Calculate the toll for each vehicle type by multiplying the distance with the corresponding rate
    for vehicle, rate in toll_rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    # Define time intervals for discount factors on weekdays
    time_discounts = {
        (time(0, 0), time(10, 0)): 0.8,
        (time(10, 0), time(18, 0)): 1.2,
        (time(18, 0), time(23, 59, 59)): 0.8,
    }
    
    # Weekend discount factor
    weekend_discount = 0.7

    # Define the days of the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekend = ['Saturday', 'Sunday']

    # Function to determine the discount factor based on day and time
    def get_discount_factor(start_time, start_day):
        if start_day in weekend:
            return weekend_discount
        else:
            for (start, end), factor in time_discounts.items():
                if start <= start_time < end:
                    return factor
        return 1.0

    # Create columns for the resulting DataFrame
    result_rows = []

    # Loop over each row in the input DataFrame
    for _, row in df.iterrows():
        for day in weekdays + weekend:
            for time_range, discount_factor in time_discounts.items():
                # Extract start and end times for this time range
                start_time, end_time = time_range

                # Apply discount factor based on time and day
                if day in weekend:
                    discount = weekend_discount
                else:
                    discount = discount_factor

                # Apply the discount to each vehicle type
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = start_time
                new_row['end_time'] = end_time
                
                new_row['moto'] = row['moto'] * discount
                new_row['car'] = row['car'] * discount
                new_row['rv'] = row['rv'] * discount
                new_row['bus'] = row['bus'] * discount
                new_row['truck'] = row['truck'] * discount

                # Add the modified row to the result rows list
                result_rows.append(new_row)

    # Convert the list of rows into a new DataFrame
    result_df = pd.DataFrame(result_rows)

    # Ensure id_start and id_end are integers
    result_df['id_start'] = result_df['id_start'].astype(int)
    result_df['id_end'] = result_df['id_end'].astype(int)

    # Arrange columns in the required order
    result_df = result_df[['id_start', 'id_end', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']]

    return result_df
