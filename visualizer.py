import pandas as pd
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class DataInterface(ABC):
    @abstractmethod
    def fetch_data(self, query: str = None) -> pd.DataFrame:
        pass

class SQLDataSource(DataInterface):
    def fetch_data(self, query: str = None) -> pd.DataFrame:
        # Placeholder for real SQL implementation
        print(f"Executing SQL query: {query}")
        # In a real scenario, we'd use sqlalchemy or similar
        return pd.DataFrame() 

class FileDataSource(DataInterface):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch_data(self, query: str = None) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            print(f"File {self.file_path} not found.")
            return pd.DataFrame()
        
        if self.file_path.endswith('.csv'):
            return pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            return pd.read_json(self.file_path)
        else:
            print("Unsupported file format.")
            return pd.DataFrame()

class MockDataSource(DataInterface):
    def fetch_data(self, query: str = None) -> pd.DataFrame:
        # Generate some mock data based on the query or default
        data = {
            'Category': ['A', 'B', 'C', 'D', 'E'],
            'Values': [10, 25, 15, 30, 20],
            'Trend': [5, 12, 18, 24, 30]
        }
        return pd.DataFrame(data)

class Visualizer:
    def __init__(self, strategy: DataInterface):
        self.strategy = strategy

    def analyze_and_plot(self, query: str = None, output_path: str = "static/graphs/latest_plot.png"):
        df = self.strategy.fetch_data(query)
        
        if df.empty:
            return "No data found to visualize."

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        
        # Simple Logic for graph selection
        cols = df.columns
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(exclude=['number']).columns

        if len(num_cols) >= 1 and len(cat_cols) >= 1:
            # Bar chart for Category vs Value
            df.plot(kind='bar', x=cat_cols[0], y=num_cols[0], ax=plt.gca())
            graph_type = "Bar Chart"
        elif len(num_cols) >= 2:
            # Line chart for multiple numeric columns (e.g. trends)
            df.plot(kind='line', ax=plt.gca())
            graph_type = "Line Chart"
        else:
            # Fallback to pie or something else
            df.plot(kind='pie', y=num_cols[0] if len(num_cols) > 0 else cols[0], ax=plt.gca())
            graph_type = "Pie Chart"

        plt.title(f"Visualized Data ({graph_type})")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path

if __name__ == "__main__":
    # Test with Mock Data
    viz = Visualizer(MockDataSource())
    path = viz.analyze_and_plot()
    print(f"Graph saved to: {path}")
