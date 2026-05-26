import pandas as pd

df = pd.read_csv('data/client_1_data.csv')
print('Client 1 - isFraud distribution:')
print(df['isFraud'].value_counts())
print(f'Total samples: {len(df)}')
print(f'Fraud percentage: {df["isFraud"].mean()*100:.2f}%')

# Check for all clients
for i in range(1, 6):
    try:
        df = pd.read_csv(f'data/client_{i}_data.csv')
        print(f'Client {i}: Total={len(df)}, Fraud%={df["isFraud"].mean()*100:.2f}%')
    except:
        print(f'Client {i}: Error loading data')