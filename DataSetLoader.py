import numpy as np
import pandas as pd
from ColumnMap import Data, ColumnMap

class DataSetLoader:
    """Manages a dataset, providing callbacks for WireEKF."""
    
    def __init__(self, n_nodes: int, path: str, column_map: ColumnMap | None = None):
        self.n = n_nodes
        self.index = 0
        self.column_map = column_map if column_map is not None else ColumnMap()
        self.file = open(path, "r")
        self.file.readline()  # Skip header
        line = self.file.readline()
        self.prev_data = self.column_map.from_csv_line(line)
    
    
    def get_nodes(self) -> Data:
        """Retrieve next row as a Data object."""
        line = self.file.readline()
        if not line:
            raise IndexError("Dataset exhausted.")
        data = self.column_map.from_csv_line(line)
        dt = data.time - self.prev_data.time
        data.v = {
            i: (
                (data.p[i][0] - self.prev_data.p[i][0]) / dt,
                (data.p[i][1] - self.prev_data.p[i][1]) / dt,
                (data.p[i][2] - self.prev_data.p[i][2]) / dt,
            )
            for i in range(1, self.n + 1)
        }
        self.prev_data = data
        return data

    def reset(self):
        """Reset dataset indices."""
        self.model_index = 0
        self.index = 0

    def close(self):
        """Close the dataset file."""
        self.file.close()

    
    