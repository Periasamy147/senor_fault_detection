import boto3
import pandas as pd

# AWS DynamoDB Configuration
dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
table = dynamodb.Table("TemperatureReadings")

# Fetch latest data from DynamoDB
def fetch_latest_data():
    response = table.scan()
    df = pd.DataFrame(response["Items"])
    
    # Ensure correct data types
    df["temperature"] = df["temperature"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Sort and return latest 10 readings
    return df.sort_values("timestamp").tail(10)

if __name__ == "__main__":
    print(fetch_latest_data())
