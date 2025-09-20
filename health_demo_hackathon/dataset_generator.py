import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_health_dataset():
    """
    Creates a realistic dataset for water-borne disease prediction in rural Assam.
    Simulates seasonal variations with higher turbidity during monsoon months (June-September).
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate 5000 data points
    n_samples = 5000
    
    # Create date range spanning multiple years to capture seasonal patterns
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 1460)) for _ in range(n_samples)]  # 4 years of data
    
    data = []
    
    for date in dates:
        month = date.month
        
        # Seasonal turbidity patterns for Assam
        if month in [6, 7, 8, 9]:  # Monsoon season
            # Higher turbidity during monsoon (June-September)
            base_turbidity = np.random.normal(25, 12)  # Higher mean, more variation
            turbidity_multiplier = np.random.uniform(1.5, 3.0)  # Additional monsoon effect
        elif month in [10, 11]:  # Post-monsoon
            base_turbidity = np.random.normal(15, 8)
            turbidity_multiplier = np.random.uniform(1.0, 1.5)
        elif month in [12, 1, 2]:  # Winter
            base_turbidity = np.random.normal(8, 4)  # Lower turbidity
            turbidity_multiplier = np.random.uniform(0.7, 1.0)
        else:  # Pre-monsoon (March-May)
            base_turbidity = np.random.normal(12, 6)
            turbidity_multiplier = np.random.uniform(0.8, 1.2)
        
        # Calculate final turbidity with seasonal effects
        turbidity = max(0.5, base_turbidity * turbidity_multiplier)
        
        # Add some random spikes to simulate contamination events
        if random.random() < 0.05:  # 5% chance of contamination spike
            turbidity *= np.random.uniform(2.0, 4.0)
        
        # Round to 1 decimal place
        turbidity = round(turbidity, 1)
        
        # Generate diarrhea cases based on turbidity with some noise
        # Higher turbidity generally leads to more cases, but with realistic variation
        base_cases = 0.3 * turbidity + np.random.normal(0, 2)
        
        # Add seasonal disease patterns (monsoon has higher baseline disease risk)
        if month in [6, 7, 8, 9]:
            seasonal_multiplier = np.random.uniform(1.2, 2.0)
        elif month in [10, 11]:
            seasonal_multiplier = np.random.uniform(1.0, 1.3)
        else:
            seasonal_multiplier = np.random.uniform(0.7, 1.0)
        
        diarrhea_cases = max(0, int(base_cases * seasonal_multiplier + np.random.poisson(1)))
        
        # Occasionally add outbreak scenarios
        if random.random() < 0.02:  # 2% chance of outbreak
            diarrhea_cases += random.randint(10, 30)
        
        # Determine risk level based on turbidity and cases
        # Risk assessment criteria adapted for rural Assam conditions
        if turbidity >= 30 or diarrhea_cases >= 15:
            risk_level = "High"
        elif (turbidity >= 15 and diarrhea_cases >= 5) or turbidity >= 20 or diarrhea_cases >= 8:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Add some realistic edge cases and adjustments
        if turbidity > 50 and diarrhea_cases < 3:  # High turbidity but low cases (maybe recent contamination)
            if random.random() < 0.7:
                risk_level = "Medium"  # Precautionary
        
        if diarrhea_cases > 20:  # Always high risk with many cases
            risk_level = "High"
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Month': month,
            'Turbidity (NTU)': turbidity,
            'Diarrhea Cases': diarrhea_cases,
            'Risk Level': risk_level
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some additional realistic features that might be collected
    df['Temperature (°C)'] = np.random.normal(27, 5)  # Typical Assam temperature
    df['Rainfall (mm)'] = np.where(
        df['Month'].isin([6, 7, 8, 9]), 
        np.random.exponential(15),  # Higher rainfall in monsoon
        np.random.exponential(3)    # Lower rainfall in other months
    )
    
    # Round temperature to 1 decimal
    df['Temperature (°C)'] = df['Temperature (°C)'].round(1)
    df['Rainfall (mm)'] = df['Rainfall (mm)'].round(1)
    
    # Ensure realistic distributions
    risk_counts = df['Risk Level'].value_counts()
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Risk Level distribution:")
    print(f"  Low: {risk_counts.get('Low', 0)} ({risk_counts.get('Low', 0)/len(df)*100:.1f}%)")
    print(f"  Medium: {risk_counts.get('Medium', 0)} ({risk_counts.get('Medium', 0)/len(df)*100:.1f}%)")
    print(f"  High: {risk_counts.get('High', 0)} ({risk_counts.get('High', 0)/len(df)*100:.1f}%)")
    print(f"\nTurbidity range: {df['Turbidity (NTU)'].min():.1f} - {df['Turbidity (NTU)'].max():.1f} NTU")
    print(f"Diarrhea cases range: {df['Diarrhea Cases'].min()} - {df['Diarrhea Cases'].max()}")
    
    return df

if __name__ == "__main__":
    print("🏥 Creating ASHA-doot Health Dataset for Rural Assam...")
    print("=" * 60)
    
    # Generate the dataset
    health_data = create_health_dataset()
    
    # Save to CSV
    filename = 'custom_health_data.csv'
    health_data.to_csv(filename, index=False)
    
    print(f"\n✅ Dataset successfully created and saved as '{filename}'")
    print(f"📊 The dataset contains {len(health_data)} records with seasonal patterns")
    print("🌧️  Higher turbidity during monsoon months (June-September)")
    print("📈 Ready for machine learning model training!")
    
    # Display sample data
    print("\n📋 Sample data preview:")
    print(health_data[['Date', 'Turbidity (NTU)', 'Diarrhea Cases', 'Risk Level']].head(10).to_string(index=False))
