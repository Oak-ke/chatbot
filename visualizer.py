import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, strategy: DataInterface):
        self.strategy = strategy

    def analyze_and_plot(self, query: str = None, output_path: str = "static/graphs/latest_plot.png"):
        df = self.strategy.fetch_data(query)
        
        if not isinstance(df, pd.DataFrame):
            logger.error("Data source returned invalid data type (expected DataFrame).")
            return None
            
        logger.info(f"Analyzing data for plotting. Dataframe shape: {df.shape}")
        
        if df.empty:
            logger.warning("Dataframe is empty. Skipping visualization.")
            return None

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.figure(figsize=(10, 6))
        
        try:
            # Simple Logic for graph selection
            cols = df.columns
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(exclude=['number']).columns

            if not cols.empty and len(num_cols) >= 1 and len(cat_cols) >= 1:
                # Bar chart for Category vs Value
                df.plot(kind='bar', x=cat_cols[0], y=num_cols[0], ax=plt.gca())
                graph_type = "Bar Chart"
            elif len(num_cols) >= 2:
                # Line chart for multiple numeric columns (e.g. trends)
                df.plot(kind='line', ax=plt.gca())
                graph_type = "Line Chart"
            elif not cols.empty:
                # Fallback to pie or something else
                target_col = num_cols[0] if not num_cols.empty else cols[0]
                df.plot(kind='pie', y=target_col, ax=plt.gca())
                graph_type = "Pie Chart"
            else:
                logger.error("No valid columns found for plotting.")
                plt.close()
                return None
        except Exception as e:
            logger.error(f"Error during plotting: {str(e)}")
            plt.close()
            return None

        plt.title(f"Visualized Data ({graph_type})")
        logger.info(f"Generated {graph_type} visualization.")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Graph saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test with Mock Data
    viz = Visualizer(MockDataSource())
    path = viz.analyze_and_plot()
    print(f"Graph saved to: {path}")
