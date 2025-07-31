'''
    ds5020 Linear algebra
    andrew consilvio
    final project - iteration 06
'''
# Import all the tools we need
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import silhouette_score
import seaborn as sns

# Import Edited Kaggle Dataset on Car Sales
df_source = pd.read_csv("/Users/home/Documents/Align Program/" \
"NEU - Summer Semester/DS5020 Linear Algebra/final project stuff/" \
"car_prices_saleyear-edited.csv")
df_source.info()

# Drop entries without Body style, State, Sale date or year
''' Some cleaning was done in the csv itself for ease of converting 
data. I also had to adjust and create a sale year row that just had the 
four digit number instead of full timestamp. '''
df_simplify = df_source.dropna(subset=['body','state',
                                       'saledate','saleyear']).copy()
df = pd.DataFrame(df_simplify)

# Factoring for just sale date/year information.
''' The internet provided suggestions on how to deal with the multiple
ways the date/time could be separated.  '''
df_simplify['saledate'] = pd.to_datetime(df_simplify['saledate'],
                                         format='mixed', 
                                         errors='coerce', utc=True)
df_simplify['year_number'] = df_simplify['saledate'].dt.year

# Print out the body types from the list
''' This was so I could easily see what was being input as body types. 
Once I had this list, I could then convert them into less overall categories
of body types. '''
print("Body types:", df_simplify['body'].unique())
print()

# Create a new list of body styles and convert based on above list
convert_body_type = {
    'Sedan': 'Sedan',
    'sedan': 'Sedan',
    'Coupe': 'Coupe',
    'Convertible': 'Convertible',
    'SUV': 'SUV',
    'Minivan': 'Van',
    'MiniVan': 'Van',
    'Van': 'Van',
    'Hatchback': 'Wagon', # This could go sedan/wagon
    'Wagon': 'Wagon',
    'Crew Cab': 'Truck',
    'crew cab': 'Truck',
    'Double Cab': 'Truck',
    'CrewMax Cab': 'Truck',
    'Access Cab': 'Truck',
    'King Cab': 'Truck',
    'SuperCrew': 'Truck',
    'Extended Cab': 'Truck',
    'SuperCab': 'Truck',
    'Regular Cab': 'Truck',
    'Quad Cab': 'Truck',
    'CTS Coupe': 'Coupe', # Cadillac Model, likely errored category
    'CTS-V Coupe': 'Coupe', # Cadillac Model, likely shifted data in CSV
    'Genesis Coupe': 'Coupe', # Same as above
    'Elantra Coupe': 'Coupe', # Same as above
    'G Coupe': 'Coupe', 
    'G Sedan': 'Sedan', 
    'G Convertible': 'Convertible', 
    'E-Series Van': 'Van', # Ford work vans will go in with vans/minivans
    'Koup': 'Coupe', 
    'Cab Plus': 'Truck',
    'Beetle Convertible': 'Convertible',
    'TSX Sport Wagon': 'Wagon',
    'Promaster Cargo Van': 'Van',
    'GranTurismo Convertible': 'Convertible',
    'CTS-V Wagon': 'Wagon',
    'Ram Van': 'Van',
    'Mega Cab': 'Truck',
    'Club Cab': 'Truck',
    'Xtracab': 'Truck',
    'CTS Wagon': 'Wagon',
    'Q60 Convertible': 'Convertible'
}

# Simplifying Body Types (simplify to coupe, sedan, suv, truck, van, wagon)
num_body_types = df_simplify['body'].nunique()
print(f"How many body types: {num_body_types}") # Before we adjust
print()

# How many states do we have?
num_states = df_simplify['state'].nunique()
print(f"Number of states: {num_states}")
missing_states = df_simplify[~df_simplify['state'].isin(
    convert_body_type.keys())]['state'].unique()
print(f"List of states: {missing_states}")
print()

# Body types in the CSV viewing manually aren't showing up
unmapped = df_simplify[~df_simplify['body'].isin(
    convert_body_type.keys())]['body'].unique()
print("Body types missing from mapping:", unmapped)
''' when convert_body_type has all the body types, 
    there should be nothing printing for this line.'''
print()


# update the dataframe based on the new body list
df_simplify['body_condensed'] = df_simplify['body'].map(convert_body_type)
num_condensed_body = df_simplify['body_condensed'].nunique()
print(f"Updated Body Type Amount: {num_condensed_body}")
# This should be 7 (coupe, convertible, sedan, suv, truck, van, wagon)
print()

# Drop all items with NaN; drop missing a value from these categories
df_simplify = df_simplify.dropna(subset=['body_condensed'])
df_simplify = df_simplify.dropna(subset=['condition'])
df_simplify = df_simplify.dropna(subset=['year_number'])

# preparing to cluster data; changing body type to a number value
encoded_df = pd.get_dummies(df_simplify[['body_condensed', 'condition']])
cluster_data = pd.concat([encoded_df, df_simplify[['year_number']]], axis=1)

# Implement K-Means Clustering on the data
kmeans = KMeans(n_clusters=5, random_state=42)
df_simplify['cluster'] = kmeans.fit_predict(cluster_data)

# Rough accuracy score. Closer to 1 is better, closer to 0 is worse
score = silhouette_score(cluster_data, df_simplify['cluster'])
print(f"Silhouette Score: {score:.3f}")
print()

# Separate States and make individual cluster maps
for state in df_simplify['state'].unique():
    state_maps = df_simplify[df_simplify['state'] == state]
    if len(state_maps) < 100:
        continue
    plt.figure(figsize=(6,4))
    sns.countplot(data=state_maps, x='cluster', hue='body_condensed')
    plt.title(f"Body Type Sales by State - {state}")
    plt.xlabel("Cluster")
    plt.ylabel("Vehicles Sold")
    plt.legend(title="Body Type")
    plt.tight_layout()
    plt.show()

# Cluster map for the entire dataset
sns.countplot(data=df_simplify, x='cluster', hue='body_condensed')
plt.title("Vehicle Body Type by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.legend(title="Body Type")
plt.tight_layout()
plt.show()